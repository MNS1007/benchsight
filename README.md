# benchsight

Let your coding agent see your workbench.

When you're wiring up a robot arm or other hardware and get stuck, the usual loop is: take a photo with your phone, move it to the laptop, paste it into Claude Code or ChatGPT, explain what you're looking at, and repeat. benchsight cuts that loop. You point the laptop camera (or a USB webcam) at the bench, and the agent takes the photo itself. It does this in the same session that already has your code, your terminal errors and the setup docs.

benchsight is an [MCP](https://modelcontextprotocol.io) server, so it works with Claude Code, Claude Desktop, Cursor and any other MCP client.

> Status: early (v0.0.1). The image tools are tested; real-camera testing on different machines is still ongoing. Issues and reports are welcome.

---

## Contents

- [What you get](#what-you-get)
- [Requirements](#requirements)
- [Install](#install)
- [Connect it to your agent](#connect-it-to-your-agent)
  - [Claude Code](#claude-code)
  - [Claude Desktop](#claude-desktop)
  - [Cursor and other MCP clients](#cursor-and-other-mcp-clients)
- [Add the hardware-help skill (Claude Code)](#add-the-hardware-help-skill-claude-code)
- [Allow camera access](#allow-camera-access)
- [Use it](#use-it)
- [The live viewer](#the-live-viewer)
- [Settings](#settings)
- [Privacy](#privacy)
- [Troubleshooting](#troubleshooting)
- [Update and uninstall](#update-and-uninstall)
- [Develop](#develop)
- [Why this exists](#why-this-exists)
- [Roadmap](#roadmap)

---

## What you get

Tools the agent can call:

| Tool | What it does | Inputs |
|---|---|---|
| `cameras` | Lists the cameras that work on this machine | none |
| `look` | Takes one photo now. It can zoom on a region, like a connector or a servo label | `cam` (default 0), `max_side` (default 1280), and the zoom box `x1, y1, x2, y2` as fractions 0 to 1 (default: whole frame) |
| `watch` | Takes 2 to 9 photos over a few seconds and returns them as one numbered sheet, for something that moves (a servo sweep, a cable being seated) | `cam`, `seconds` (default 6), `frames` (default 6), `max_side` (default 1600) |
| `point` | Circles a spot on the last `look` photo with a short label, to show you exactly which pin or cable it means. If the live viewer is open, the circle shows there too | `x`, `y` as fractions 0 to 1, `label` |
| `open_viewer` | Opens the **live viewer**: a small local window with the camera feed, where the agent's drawings appear on top | `cam`, `open_browser` (default true) |
| `annotate` | Draws on the live viewer: arrows, circles, boxes, freehand strokes and text, plus a one-line instruction under the video | `shapes` (list), `caption`, `replace` (default true) |
| `clear_annotations` | Removes all drawings from the viewer | none |
| `close_viewer` | Closes the viewer and turns the camera off | none |

`look` also takes `with_annotations` (default false). With it, the agent gets the photo with its own drawings on it, so it can check they landed in the right place.

You also get:

- **`benchsight` CLI:** `benchsight cameras` lists cameras, and `benchsight snap` saves one photo and prints its path. Use it with agents that don't support MCP: ask the agent to run `benchsight snap` and then open the image.
- **The `hardware-help` skill for Claude Code:** it tells the agent how to use the camera well. It asks you to point the camera first, then works through these steps:
  1. take a wide shot, then zoom;
  2. check what it sees against the build docs;
  3. circle the exact part;
  4. say when a photo can't settle the question.
- **Playbooks:** short checklists for common builds. The first one is the LeRobot SO-101 arm.

---

## Requirements

- Python 3.10 or newer
- A camera: the built-in webcam or any USB webcam
- An MCP client, such as Claude Code, Claude Desktop or Cursor
- Windows, macOS or Linux

Check your Python version:

```bash
python --version      # Windows
python3 --version     # macOS / Linux
```

---

## Install

Pick one of three ways.

### Option A: pipx (recommended)

[pipx](https://pipx.pypa.io) installs the tool in its own environment and puts the commands on your PATH.

```bash
pipx install git+https://github.com/MNS1007/benchsight
```

Check that it installed:

```bash
benchsight --help
benchsight cameras        # should print something like [0] or [0, 1]
```

### Option B: uv, no install step

If you use [uv](https://docs.astral.sh/uv/), skip installing. Your MCP client runs benchsight straight from GitHub. See the `uvx` line under [Claude Code](#claude-code).

### Option C: from source, in a virtual environment

```bash
git clone https://github.com/MNS1007/benchsight
cd benchsight
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash)
source .venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate
```

Then install:

```bash
pip install -e .
```

With this option, the command lives inside the environment. Use its full path when you connect your agent:

- Windows: `C:\path\to\benchsight\.venv\Scripts\benchsight-mcp.exe`
- macOS / Linux: `/path/to/benchsight/.venv/bin/benchsight-mcp`

---

## Connect it to your agent

### Claude Code

If you installed with pipx (Option A):

```bash
claude mcp add benchsight -- benchsight-mcp
```

With uv and no install (Option B):

```bash
claude mcp add benchsight -- uvx --from git+https://github.com/MNS1007/benchsight benchsight-mcp
```

From source (Option C), use the full path to the command:

```bash
# macOS / Linux
claude mcp add benchsight -- /path/to/benchsight/.venv/bin/benchsight-mcp
# Windows
claude mcp add benchsight -- C:\path\to\benchsight\.venv\Scripts\benchsight-mcp.exe
```

By default, the server is added for the current project only. To make it available in every project, add `--scope user`:

```bash
claude mcp add --scope user benchsight -- benchsight-mcp
```

Check it:

```bash
claude mcp list           # benchsight should show as connected
```

Inside Claude Code, `/mcp` also shows the server and its tools.

### Claude Desktop

Open the config file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add benchsight under `mcpServers`. Use the full path to the command; find it with `which benchsight-mcp` on macOS/Linux, or `where benchsight-mcp` on Windows.

```json
{
  "mcpServers": {
    "benchsight": {
      "command": "/full/path/to/benchsight-mcp"
    }
  }
}
```

On Windows, write the path with double backslashes, like `"C:\\Users\\you\\.local\\bin\\benchsight-mcp.exe"`. Then restart Claude Desktop.

### Cursor and other MCP clients

Any MCP client that can start a local command over stdio works. In Cursor, add this to `.cursor/mcp.json` in your project, or to `~/.cursor/mcp.json` to use it everywhere:

```json
{
  "mcpServers": {
    "benchsight": { "command": "benchsight-mcp" }
  }
}
```

If the client can't find `benchsight-mcp`, use its full path.

---

## Add the hardware-help skill (Claude Code)

The skill is optional, but it makes the agent use the camera much better. Copy the folder into your skills directory.

For all projects:

```bash
# macOS / Linux
mkdir -p ~/.claude/skills && cp -r skills/hardware-help ~/.claude/skills/
# Windows (PowerShell)
New-Item -ItemType Directory -Force "$HOME\.claude\skills" | Out-Null
Copy-Item -Recurse skills\hardware-help "$HOME\.claude\skills\"
```

For one project only, copy it to `<your-project>/.claude/skills/hardware-help/` instead.

If you installed with pipx or uv, you don't have the source folder. Download it from GitHub first:

```bash
git clone --depth 1 https://github.com/MNS1007/benchsight /tmp/benchsight && cp -r /tmp/benchsight/skills/hardware-help ~/.claude/skills/
```

---

## Allow camera access

Your operating system has to let the program that runs the MCP server use the camera. That program is your terminal for Claude Code, or the app itself for Claude Desktop and Cursor.

- **macOS:** System Settings > Privacy & Security > Camera, then turn it on for your terminal app (Terminal, iTerm, VS Code and so on) or for Claude / Cursor. macOS asks the first time a photo is taken; if you missed that prompt, turn it on here and restart the app.
- **Windows:** Settings > Privacy & security > Camera. Turn on "Camera access" and "Let desktop apps access your camera".
- **Linux:** your user needs access to `/dev/video*`, usually by being in the `video` group: `sudo usermod -aG video $USER`, then log out and back in.

Test it outside the agent:

```bash
benchsight snap           # prints the path of a saved photo
```

Open the printed file. If it's black or the command fails, see [Troubleshooting](#troubleshooting).

---

## Use it

1. Point the camera at the hardware. A USB webcam on a small stand or gooseneck works best, because you can aim it at the bench and keep your hands free.
2. Ask the agent in plain words. For example:
   - "My SO-101 elbow servo won't respond. I'm pointing the camera at the board, take a look."
   - "Look at the cable on servo 3 and tell me if it's fully seated."
   - "Watch the gripper while I run the calibration and tell me where it stops."
   - "Which pin does the yellow wire go to? Point at it."
3. The agent calls `look`, `watch` or `point`. Claude Code asks your permission the first time it uses each tool.
4. Pair it with the terminal. If a script just failed, say so. The agent can read the error and check whether the hardware explains it: a wrong servo ID, an unplugged cable, the wrong port.

Tips:

- **Several cameras:** `benchsight cameras` lists them. Tell the agent which one to use ("use camera 1").
- **Small parts:** ask the agent to zoom. It can crop to any region, and the crop is taken from the full-resolution frame.
- **Dark benches:** add light. A blurry or dark photo is not evidence, and the skill tells the agent to ask for a better one rather than guess.

---

## The live viewer

For step-by-step guidance, the agent can open a live view of your bench and draw on it while you work. For example: "this bracket is the wrong one", "move it here", "route the cable like this".

![The agent's drawings, shown on a test image](docs/viewer-example.png)

*The drawings shown on a test image. On a real bench they appear over the live camera.*

**How it works:**
1. Ask for it: "Open the viewer and guide me through mounting the gripper."
2. The agent calls `open_viewer`. A browser tab opens at a local address (like `http://127.0.0.1:53817/`) with the live camera. Keep it beside your bench or on a second screen.
3. When you ask "what's wrong?", the agent takes a photo, finds the problem, and draws on the live view:
   - **red** for wrong;
   - **green** for right;
   - **yellow** for where to move something;
   - a short instruction under the video.
4. The drawings stay on screen until the agent changes them, so you can keep working with both hands.
5. When you're done, the agent calls `close_viewer` and the camera turns off. Closing the tab alone does not turn the camera off.

**What the agent can draw.** Coordinates are fractions 0 to 1 of the frame, and the color is optional.

```json
{"type": "arrow",  "from": [0.7, 0.4], "to": [0.35, 0.3], "label": "move it here", "color": "yellow"}
{"type": "circle", "at": [0.3, 0.55], "r": 0.04, "label": "screw 2"}
{"type": "box",    "box": [0.59, 0.41, 0.82, 0.63], "label": "wrong bracket"}
{"type": "path",   "points": [[0.2, 0.75], [0.3, 0.85], [0.45, 0.8]], "label": "cable route", "color": "green"}
{"type": "text",   "at": [0.6, 0.6], "text": "flip it over"}
```

Drawings the agent anchors to a tracked object (with the `track` tool) follow that object as it or the camera moves. Other drawings are fixed to the camera image: if you move the camera, ask the agent to look again and redraw.

**Privacy:** the viewer is served on 127.0.0.1 only, so other devices on your network can't see it. It streams only while open.

**CPU use of tracking.** While the viewer is open, a tracker follows the objects the agent asked it to track (`track`), on the CPU. It aims to stay under 0.15 of one core (`BENCHSIGHT_CPU`). When updating every object every frame would cost more, the objects take turns: each one still updates at least every 4th frame, and camera motion moves all the boxes in between. That keeps one VitTrack object at the target. With more objects, or with the offline MIL fallback, it uses more, as the table shows, but much less than without the budget. The cost of the budget: an object can wait up to 3 frames, so a hidden object is reported lost after up to about 0.8 s, and fast motion (over about 50 px/s at 640 px) is followed less closely, mostly with MIL. `BENCHSIGHT_CPU=0` updates every object every frame. Measured on an Apple M2 with OpenCV 5.0, a 640x480 feed at 15 fps, and one OpenCV thread (`python scripts/bench_tracking.py`):

| Tracker | Objects | fps | CPU, share of one core | With no budget (`--cpu 0`) |
|---|---|---|---|---|
| none tracked | 0 | 14 | 3% | 3% |
| VitTrack | 1 | 14 | 15% | 24% |
| VitTrack | 4 | 14 | 22% | 37% |
| VitTrack | 8 | 14 | 28% | 70% |
| MIL (offline fallback) | 1 | 13 | 18% | 32% |
| MIL (offline fallback) | 4 | 13 | 33% | 100%, 13.6 fps |
| MIL (offline fallback) | 8 | 10.5 | 45% | 100%, 6.8 fps |

VitTrack is used when its model file (under 1 MB) has been downloaded once into `~/.cache/benchsight/models`; offline, benchsight falls back to MIL.

---

## Settings

| Setting | Default | What it does |
|---|---|---|
| `BENCHSIGHT_DIR` (environment variable) | `.benchsight` in the folder the server starts in | Where photos are saved |
| `cam` (tool input) | `0` | Which camera to use |
| `BENCHSIGHT_PORT` (environment variable) | random free port | Port for the live viewer, if you want a fixed address |
| `BENCHSIGHT_TRACKER` (environment variable) | `auto` | Object tracker: `auto` (VitTrack if its model is available, else MIL), `vit` or `mil` |
| `BENCHSIGHT_CPU` (environment variable) | `0.15` | Share of one core tracking aims to stay under. `0` means no limit: every object updates every frame |
| `max_side` (tool input) | 1280 (`look`, `point`), 1600 (`watch`) | Longest side of the image sent to the agent. Smaller means faster and fewer tokens |

To set `BENCHSIGHT_DIR` in Claude Code:

```bash
claude mcp add benchsight -e BENCHSIGHT_DIR=/path/to/photos -- benchsight-mcp
```

---

## Privacy

- **No background recording.** A photo is taken only when the agent calls a tool, and your client asks you before the first call.
- **Photos stay on your machine.** They're saved in the photo folder (see [Settings](#settings)) and deleted after 24 hours.
- **The only thing sent anywhere** is the photo the agent asked for, to the model you already use.
- Many webcams turn on a light while the camera is in use, so you can see when a photo is taken.

Add `.benchsight/` to your project's `.gitignore` so photos are never committed.

---

## Troubleshooting

**`benchsight-mcp: command not found`, or the client says the server failed to start**
The command isn't on the PATH the client sees. Use the full path (find it with `which benchsight-mcp` on macOS/Linux or `where benchsight-mcp` on Windows), or use the `uvx` setup.

**"camera 0 did not open"**
- Another app may be using the camera (Zoom, Teams, FaceTime, OBS). Close it.
- Camera permission may be off. See [Allow camera access](#allow-camera-access).
- The camera may have a different index. Run `benchsight cameras` and pass that number.

**The photo is black or very dark**
- Some cameras need a moment to adjust; benchsight drops the first few frames for this. Try again with more light.
- On a laptop with a privacy shutter or camera key, open the shutter or turn the camera on.

**`benchsight cameras` prints `[]`**
No camera answered. Check permissions and cables, and try unplugging and replugging a USB webcam.

**The tools don't appear in Claude Code**
Run `claude mcp list`. If benchsight shows as failed, run the same command it uses (for example `benchsight-mcp`) in a terminal to see the error. It should start and wait silently; press Ctrl+C to stop it.

**macOS: it worked in Terminal but not in Claude Desktop or Cursor**
Camera permission is per app. Turn it on for that app too.

---

## Update and uninstall

```bash
# update
pipx upgrade benchsight                 # pipx
git pull && pip install -e .            # from source

# uninstall
claude mcp remove benchsight
pipx uninstall benchsight               # pipx
rm -rf ~/.claude/skills/hardware-help   # the skill, if you copied it
```

---

## Develop

```bash
git clone https://github.com/MNS1007/benchsight
cd benchsight
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest
```

The tests use synthetic images and need no camera.

Project layout:

```
src/benchsight/camera.py   camera access, zoom, contact sheet, point marker, local store
src/benchsight/server.py   the MCP server and its tools
src/benchsight/feed.py     the live camera feed used by the viewer
src/benchsight/overlay.py  annotation shapes, checks and drawing
src/benchsight/viewer.py   the local live-viewer web page
src/benchsight/cli.py      the benchsight CLI (snap, cameras)
skills/hardware-help/      the Claude Code skill
playbooks/                 checklists for specific builds
tests/                     tests (no camera needed)
```

To try the server by hand, use the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector benchsight-mcp
```

Contributions are welcome. New playbooks for other robots and kits are especially useful.

---

## Why this exists

Several open-source MCP servers already take a webcam snapshot. benchsight isn't trying to be another one. It adds the workflow a hardware builder needs around the camera:

- **zoom and point**, so the agent can show you where, not just tell you;
- **short sequences** for moving parts;
- **a skill** that makes the agent check its view against the build docs before answering;
- **playbooks** for common builds, starting with the LeRobot SO-101 arm.

If that layer doesn't make the agent clearly more useful than a plain snapshot tool, this project should fold into an existing one.

## Roadmap

1. Real-world testing on SO-101 builds across Windows, macOS and Linux.
2. Let you click or draw on the viewer to point at something for the agent.
3. A watch mode with a stop condition, for example "tell me when the servo stops jittering".
4. A small eval set of real bench mistakes (wrong servo ID, reversed cable, missing screw, wrong calibration pose), run with and without benchsight.
5. More playbooks: other arms, 3D printer setup, breadboards.

## License

Not chosen yet. Until a license file is added, all rights are reserved.
