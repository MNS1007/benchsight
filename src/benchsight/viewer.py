"""The live viewer: a small local web page with the camera feed and the agent's
annotations drawn on top. It is served on 127.0.0.1 only and runs only while open.

  /         the page
  /stream   the camera as MJPEG
  /overlay  the current annotations as JSON
  /events   annotation updates as server-sent events
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from .overlay import COLORS

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>benchsight</title>
<style>
  html,body{margin:0;height:100%;background:#111;color:#eee;font:15px/1.4 system-ui,Segoe UI,sans-serif}
  #wrap{position:relative;width:100vw;height:calc(100vh - 54px);display:flex;align-items:center;justify-content:center}
  #stage{position:relative;max-width:100%;max-height:100%}
  #cam{display:block;max-width:100vw;max-height:calc(100vh - 54px)}
  svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
  .new{animation:pop .5s ease-out}
  @keyframes pop{from{opacity:0}to{opacity:1}}
  #bar{height:54px;display:flex;align-items:center;gap:12px;padding:0 16px;background:#1b1b1b;border-top:1px solid #333}
  #caption{flex:1;font-size:17px}
  #dot{width:9px;height:9px;border-radius:50%;background:#e62828}
  #meta{color:#888;font-size:12px}
</style></head>
<body>
<div id="wrap"><div id="stage"><img id="cam" src="/stream" alt="camera"><svg id="ov" viewBox="0 0 1000 1000" preserveAspectRatio="none"></svg></div></div>
<div id="bar"><div id="dot" title="camera on"></div><div id="caption">Waiting for the agent.</div><div id="meta">benchsight, local only</div></div>
<script>
const COLORS = __COLORS__;
const ov = document.getElementById('ov'), cap = document.getElementById('caption');
const S = v => v * 1000;
const ns = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent){ const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent||ov).appendChild(e); return e; }
function label(x, y, text, color){
  if (!text) return;
  el('text', {x: S(x)+10, y: S(y)-10, fill: color, 'font-size': 34, 'font-weight': 700, stroke: '#000', 'stroke-width': 6, 'paint-order': 'stroke'}).textContent = text;
}
function draw(state){
  ov.innerHTML = '';
  const defs = el('defs', {});
  for (const name in COLORS){
    const m = el('marker', {id: 'h-'+name, markerWidth: 6, markerHeight: 6, refX: 5, refY: 3, orient: 'auto'}, defs);
    el('path', {d: 'M0,0 L6,3 L0,6 z', fill: COLORS[name]}, m);
  }
  const g = el('g', {class: 'new'});
  for (const s of state.shapes){
    const c = COLORS[s.color];
    const line = {stroke: c, 'stroke-width': 8, fill: 'none', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'};
    if (s.type === 'arrow'){ el('line', {...line, x1: S(s.from[0]), y1: S(s.from[1]), x2: S(s.to[0]), y2: S(s.to[1]), 'marker-end': 'url(#h-'+s.color+')'}, g); label(s.from[0], s.from[1], s.label, c); }
    if (s.type === 'circle'){ el('ellipse', {...line, cx: S(s.at[0]), cy: S(s.at[1]), rx: S(s.r), ry: S(s.r)}, g); label(s.at[0]+s.r, s.at[1], s.label, c); }
    if (s.type === 'box'){ el('rect', {...line, x: S(s.box[0]), y: S(s.box[1]), width: S(s.box[2]-s.box[0]), height: S(s.box[3]-s.box[1])}, g); label(s.box[0], s.box[1], s.label, c); }
    if (s.type === 'path'){ el('polyline', {...line, points: s.points.map(p => S(p[0])+','+S(p[1])).join(' ')}, g); const e = s.points[s.points.length-1]; label(e[0], e[1], s.label, c); }
    if (s.type === 'text'){ label(s.at[0], s.at[1], s.text, c); }
  }
  cap.textContent = state.caption || (state.shapes.length ? '' : 'Waiting for the agent.');
}
fetch('/overlay').then(r => r.json()).then(draw);
const ev = new EventSource('/events');
ev.onmessage = m => draw(JSON.parse(m.data));
</script></body></html>"""


class Viewer:
    def __init__(self, feed, overlay, host: str = "127.0.0.1", port: int = 0, quality: int = 80):
        self.feed, self.overlay, self.quality = feed, overlay, quality
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep stdio clean: it carries the MCP protocol
                pass

            def do_GET(self):
                try:
                    if self.path == "/":
                        body = PAGE.replace("__COLORS__", json.dumps({k: v[1] for k, v in COLORS.items()})).encode()
                        self._send(200, "text/html; charset=utf-8", body)
                    elif self.path == "/overlay":
                        self._send(200, "application/json", json.dumps(viewer.overlay.state()).encode())
                    elif self.path == "/stream":
                        viewer._stream(self)
                    elif self.path == "/events":
                        viewer._events(self)
                    else:
                        self._send(404, "text/plain", b"not found")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://{host}:{self.httpd.server_address[1]}/"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._stopped = threading.Event()

    def start(self) -> str:
        self._thread.start()
        return self.url

    def stop(self) -> None:
        self._stopped.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    def _stream(self, h):
        h.send_response(200)
        h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        period = 1.0 / max(1.0, self.feed.fps)
        while not self._stopped.is_set():
            frame = self.feed.latest()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok:
                    data = buf.tobytes()
                    h.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            time.sleep(period)

    def _events(self, h):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        seen = -1
        while not self._stopped.is_set():
            state = self.overlay.wait(seen, timeout=15)
            if state["version"] == seen:
                h.wfile.write(b": keepalive\n\n")  # also detects a closed tab
            else:
                seen = state["version"]
                h.wfile.write(b"data: " + json.dumps(state).encode() + b"\n\n")
            h.wfile.flush()
