"""benchsight snap: save one photo and print its path. The fallback for agents without
MCP: the agent runs it, then opens the image file."""
import argparse

from . import camera


def main() -> None:
    p = argparse.ArgumentParser(prog="benchsight")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snap", help="save one photo and print its path")
    s.add_argument("--cam", type=int, default=0)
    s.add_argument("--max-side", type=int, default=1280)
    sub.add_parser("cameras", help="list working camera indices")
    a = p.parse_args()
    if a.cmd == "cameras":
        print(camera.list_cameras())
    else:
        print(camera.save(camera.fit(camera.grab(a.cam), a.max_side), "snap").path)


if __name__ == "__main__":
    main()
