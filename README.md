# benchsight

Let your coding agent see your workbench.

When you are wiring up a robot arm or any other hardware and get stuck, the usual loop is: take a photo with your phone, move it to the laptop, paste it into Claude Code or ChatGPT, describe what you are looking at, and repeat. benchsight removes that loop. Point the laptop camera (or a USB webcam) at the bench, and the agent takes the photo itself, in the same session that already has your code, your terminal errors and the setup docs.

Status: early scaffold. Tested only on synthetic images so far.

## What it does

An MCP server with four tools:

| Tool | What it does |
|---|---|
| `cameras` | Lists the cameras that work on this machine |
| `look` | Takes one photo now. It can zoom on a region, for example a connector or a servo label |
| `watch` | Takes several photos over a few seconds and returns them as one numbered sheet, for something that moves (a servo sweep, a cable being seated) |
| `point` | Circles a spot on the last photo with a short label, to show you exactly which pin or cable it means |

There is also a CLI fallback, `benchsight snap`, for agents without MCP: it saves a photo and prints the path for the agent to open.

A Claude Code skill (`skills/hardware-help`) tells the agent how to use the camera well:
- ask you to point the camera first;
- take a wide shot, then zoom;
- check what it sees against the setup docs;
- point at the exact part;
- say when it can't tell from the photo.

## Privacy

- No background recording. A frame is taken only when the agent calls a tool, and Claude Code asks you before the first call.
- Photos stay on your machine in `./.benchsight/` and are deleted after 24 hours.
- Nothing is uploaded except the photo the agent asked for, to the model you already use.

## Install (Claude Code)

```
pip install -e .
claude mcp add benchsight -- benchsight-mcp
```

Copy `skills/hardware-help` into your project's `.claude/skills/`.

## Why this, given that webcam MCP servers exist

Several open-source MCP servers already take a webcam snapshot. benchsight is not trying to be another one. The aim is the hardware-builder workflow around the camera:

- zoom and point, so the agent can show you where, not just tell you;
- short sequences for moving parts;
- a skill that makes the agent check its view against the build docs before answering;
- playbooks for common builds, starting with the LeRobot SO-101 arm (servo IDs, cable routing, calibration poses).

If that layer doesn't make the agent clearly more useful than a plain snapshot tool, this project should fold into one of them.

## Roadmap

1. MVP: the four tools, the skill and an SO-101 playbook. Try it on a real SO-101 build.
2. Watch mode with a guard: a short live sequence, for example "tell me when the servo stops jittering".
3. A small eval set of real bench mistakes (wrong servo ID, reversed cable, missing screw, wrong calibration pose), with and without benchsight, to show whether it helps.
4. More playbooks: other arms, 3D printer setup, breadboards.

## License

To be chosen before the first public release.
