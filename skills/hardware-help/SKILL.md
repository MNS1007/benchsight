---
name: hardware-help
description: Use the bench camera (benchsight MCP tools) to see and debug physical hardware the user is building, like a robot arm, wiring or a 3D printer. Use when the user is stuck on a physical step, says "look at this", or an error may come from the hardware rather than the code.
---

# Helping with hardware through the bench camera

1. Ask the user to point the camera at the part. Then take a wide shot with `look`.
2. Zoom with `look` (x1, y1, x2, y2) on the part that matters: connector, label, screw, cable route. A blurry or dark photo is not evidence; ask for better light or a closer angle instead of guessing.
3. For anything that moves or happens over time (a servo sweep, a calibration pose, seating a cable), use `watch`.
4. Before saying something is wrong, check it against the build docs (a playbook in `playbooks/`, the project README, or the vendor guide) and quote the step.
5. Show the exact spot with `point` and say what to change in one short sentence.
6. Say plainly when the photo cannot settle it, and what would: another angle, a multimeter reading, or a terminal command.
7. Connect what you see to the terminal: if a script failed, read its error and check whether the hardware explains it (wrong ID, unplugged cable, wrong port).

Never take photos the user didn't ask for, and don't keep taking them in a loop.
