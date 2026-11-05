"""benchsight command line.

  benchsight snap               save one photo and print its path (the fallback for agents
                                without MCP: the agent runs it, then opens the image file)
  benchsight cameras            list working camera indices
  benchsight events [--follow]  print recent wake-up events, or follow new ones
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import camera


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="benchsight")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snap", help="save one photo and print its path")
    s.add_argument("--cam", type=int, default=0)
    s.add_argument("--max-side", type=int, default=1280)
    sub.add_parser("cameras", help="list working camera indices")
    e = sub.add_parser("events", help="print recent tracker events (object_lost, object_found, scene_changed, still)")
    e.add_argument("-f", "--follow", action="store_true",
                   help="keep printing new events, one JSON line each (run it under a background monitor)")
    e.add_argument("-n", "--lines", type=int, default=None,
                   help="how many past events to print first (default 20; 0 with --follow)")
    e.add_argument("--dir", type=Path, default=None, help="the benchsight store (default BENCHSIGHT_DIR or ./.benchsight)")
    a = p.parse_args(argv)
    if a.cmd == "cameras":
        print(camera.list_cameras())
    elif a.cmd == "events":
        from . import events
        path = (a.dir or camera.STORE) / "events.jsonl"
        if a.follow:
            try:
                events.follow(path, sys.stdout, backlog=a.lines or 0)
            except KeyboardInterrupt:
                pass
        else:
            for ln in events.recent(path, 20 if a.lines is None else a.lines):
                print(ln)
    else:
        print(camera.save(camera.fit(camera.grab(a.cam), a.max_side), "snap").path)


if __name__ == "__main__":
    main()
