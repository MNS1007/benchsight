"""Keep photos from piling up on disk.

Every file this server process writes (photos, annotated renders, contact sheets, the
events file) is recorded, and deleted when the session ends: the MCP client closes stdio,
the process gets SIGTERM, SIGINT or SIGHUP, or it exits normally. Set BENCHSIGHT_KEEP=1 to
keep them. Files older than 24 hours are also deleted on the next capture, as a backstop
for sessions that were killed hard.

Deletion is deliberately narrow, because BENCHSIGHT_DIR can point anywhere: only regular
files directly in the store whose names benchsight makes (20261105-101320-123-look.jpg,
events.jsonl) are ever deleted, never symlinks, folders or anything else. In the agent's
own cache, only Claude Code's saved copies of benchsight images
(~/.claude/projects/.../tool-results/mcp-benchsight-blob-*) are touched.
"""
from __future__ import annotations

import atexit
import os
import re
import signal
import threading
import time
from pathlib import Path

PHOTO = re.compile(r"^\d{8}-\d{6}-\d{3}-[a-z0-9_-]+\.jpg$")  # camera.save's names
EVENTS = "events.jsonl"
BLOB = "mcp-benchsight-blob-"  # Claude Code's saved copies of this server's images

_lock = threading.RLock()  # reentrant: a signal handler may run while the main thread holds it
_files: dict[Path, Path] = {}  # file -> its store, both resolved
_installed = False


def ours(path: Path) -> bool:
    """Is this a name benchsight writes into its store?"""
    return path.name == EVENTS or bool(PHOTO.match(path.name))


def _size(path: Path) -> int | None:
    """Size of a regular file benchsight may delete, else None (missing, a link, a folder)."""
    try:
        if path.is_symlink() or not path.is_file():
            return None
        return path.stat().st_size
    except OSError:
        return None


def _remove(path: Path) -> int | None:
    size = _size(path)
    if size is None:
        return None
    try:
        path.unlink()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    return size


def record(path: Path | str, store: Path | str | None = None) -> None:
    """Remember a file this process wrote, to delete at session end. store defaults to
    the file's folder; only a benchsight-named file directly in it will be deleted."""
    path = Path(path)
    store = Path(store) if store is not None else path.parent
    with _lock:
        _files[path.resolve()] = store.resolve()


def session_files() -> list[Path]:
    with _lock:
        return list(_files)


def session_end() -> tuple[int, int]:
    """Delete this session's files now. Returns (files deleted, bytes freed). Safe to
    call again (the record is cleared) and if the store is gone."""
    if os.environ.get("BENCHSIGHT_KEEP", "") not in ("", "0"):
        return 0, 0
    with _lock:
        items = list(_files.items())
        _files.clear()
    n = freed = 0
    for path, store in items:
        if path.parent != store or not ours(path):  # never outside the store, never foreign names
            continue
        size = _remove(path)
        if size is not None:
            n, freed = n + 1, freed + size
    return n, freed


def store_files(store: Path | str, older_than_s: float | None = 24 * 3600, now: float | None = None) -> list[tuple[Path, int]]:
    """benchsight files in the store, older than older_than_s (None: all). [(path, bytes)]."""
    store = Path(store)
    now = time.time() if now is None else now
    out = []
    try:
        entries = list(store.iterdir())
    except OSError:
        return []
    for p in sorted(entries):
        if not ours(p):
            continue
        size = _size(p)
        if size is None:
            continue
        if older_than_s is not None:
            try:
                if now - p.stat().st_mtime <= older_than_s:
                    continue
            except OSError:
                continue
        out.append((p, size))
    return out


def agent_cache_files(older_than_s: float | None = None, now: float | None = None,
                      home: Path | None = None) -> list[tuple[Path, int]]:
    """Claude Code's saved copies of benchsight images: files named mcp-benchsight-blob-*
    in a tool-results folder under ~/.claude/projects (and CLAUDE_CONFIG_DIR if set)."""
    roots = [(home or Path.home()) / ".claude"]
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        roots.append(Path(os.environ["CLAUDE_CONFIG_DIR"]).expanduser())
    now = time.time() if now is None else now
    seen, out = set(), []
    for root in roots:
        projects = root / "projects"
        for pattern in (f"*/tool-results/{BLOB}*", f"*/*/tool-results/{BLOB}*"):
            for p in sorted(projects.glob(pattern)):
                if p.parent.name != "tool-results" or not p.name.startswith(BLOB):
                    continue
                key = p.resolve()
                size = _size(p)
                if size is None or key in seen:
                    continue
                if older_than_s is not None:
                    try:
                        if now - p.stat().st_mtime <= older_than_s:
                            continue
                    except OSError:
                        continue
                seen.add(key)
                out.append((p, size))
    return out


def delete(files: list[tuple[Path, int]]) -> tuple[int, int]:
    """Delete files from store_files or agent_cache_files. Returns (files, bytes)."""
    n = freed = 0
    for p, _ in files:
        size = _remove(p)
        if size is not None:
            n, freed = n + 1, freed + size
            with _lock:
                _files.pop(p.resolve(), None)
    return n, freed


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def install() -> None:
    """Delete this session's files at exit and on SIGTERM, SIGINT and SIGHUP. Call once,
    from the main thread (the MCP server's main)."""
    global _installed
    if _installed:
        return
    _installed = True
    atexit.register(session_end)
    if threading.current_thread() is not threading.main_thread():
        return

    def on_signal(sig, _frame):
        try:
            session_end()
        finally:  # then die as the signal asked
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, on_signal)
            except (OSError, ValueError):
                pass
