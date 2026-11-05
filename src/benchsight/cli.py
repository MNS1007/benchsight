"""benchsight command line.

  benchsight snap               save one photo and print its path (the fallback for agents
                                without MCP: the agent runs it, then opens the image file)
  benchsight cameras            list working camera indices
  benchsight events [--follow]  print recent wake-up events, or follow new ones
  benchsight clean              delete saved benchsight files older than 24 h
      [--all] [--older-than HOURS] [--dry-run] [--agent-cache] [--dir DIR]
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
    c = sub.add_parser("clean", help="delete saved benchsight photos and events (default: older than 24 h)")
    c.add_argument("--all", action="store_true", help="delete every benchsight file, whatever its age")
    c.add_argument("--older-than", type=float, default=24.0, metavar="HOURS",
                   help="delete files older than this many hours (default 24)")
    c.add_argument("--dry-run", action="store_true", help="only list what would be deleted")
    c.add_argument("--agent-cache", action="store_true",
                   help="also delete Claude Code's saved copies of benchsight images "
                        "(~/.claude/projects/.../tool-results/mcp-benchsight-blob-*, nothing else)")
    c.add_argument("--dir", type=Path, default=None, help="the benchsight store (default BENCHSIGHT_DIR or ./.benchsight)")
    a = p.parse_args(argv)
    if a.cmd == "clean":
        clean(a)
    elif a.cmd == "cameras":
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


def clean(a) -> None:
    from . import cleanup
    age = None if a.all else max(0.0, a.older_than) * 3600
    files = cleanup.store_files(a.dir or camera.STORE, age)
    if a.agent_cache:
        files += cleanup.agent_cache_files(age)
    if a.dry_run:
        for path, size in files:
            print(f"would delete {path} ({cleanup.human(size)})")
        total = sum(size for _, size in files)
        print(f"{len(files)} files, {cleanup.human(total)} would be freed (dry run, nothing deleted)")
        return
    n = freed = 0
    for path, size in files:
        k, b = cleanup.delete([(path, size)])
        if k:
            n, freed = n + 1, freed + b
            print(f"deleted {path} ({cleanup.human(b)})")
    print(f"{n} files deleted, {cleanup.human(freed)} freed")


if __name__ == "__main__":
    main()
