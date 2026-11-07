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
6. For a multi-step job (assembly, wiring, calibration), guide on the live view:
   1. `open_viewer`, then a wide `look` (no zoom).
   2. `track` the parts you will talk about, with boxes as fractions of that look and short labels. Keep the returned ids.
   3. `annotate` with `"anchor": "<id>"` on each shape so it follows its part: red for wrong, green for right, a yellow arrow for where to move something, one short caption. Check placement with `look(with_annotations=True)`.
   4. Optional, in Claude Code: while the user works, run the `watch` command that open_viewer returned (`benchsight events --follow --dir <store>`, with the absolute folder) with the background Monitor tool. Plain `benchsight events --follow` may read the wrong folder, because the server's `.benchsight` is relative to where it started. Each line is an event (`object_lost`, `object_found`, `scene_changed`, `still`); on one, `look` and redraw. A Monitor lasts 30 minutes at most; re-arm it if the job goes on.
   5. Use `objects()` for cheap position checks instead of new photos. A shape shown "(lost)" means its part is hidden or moved away; look again and `track` it again if it doesn't come back.
   6. At the end: stop the Monitor, `close_viewer` (camera off, and any recording stops), then `cleanup` to delete the session's saved images.
7. When the user says "start watching" (or "watch me do this", "record this"), record and review afterwards. You cannot watch live video, so never claim to be watching live; say you are recording and will look back when they stop.
   1. `open_viewer` if it isn't open (recording needs it), a wide `look`, then `start_recording(roi=[x1, y1, x2, y2])` with the roi around the workspace from that look. Tell them the red REC badge is on. Then wait; take no photos while they work.
   2. On "stop": `stop_recording`. Its activity stretches are only a hint and miss slow or small changes; you pick the moments.
   3. `review()` for an even overview of the whole run, then `review(start_s, end_s)` to zoom on the moments that matter (also the quiet ones if the overview leaves doubt).
   4. Explain what happened and when, in seconds, checked against the build docs, and say what the frames could not show.
   For "what happened?" questions about a stretch of work, prefer this over the events Monitor: events only say that something changed, a recording shows what. For a LeRobot dataset episode, use `review_video(path)` on its mp4 (under `~/.cache/huggingface/lerobot/<repo_id>/videos/...`) the same way.
8. Say plainly when the photo cannot settle it, and what would: another angle, a multimeter reading, or a terminal command.
9. Connect what you see to the terminal: if a script failed, read its error and check whether the hardware explains it (wrong ID, unplugged cable, wrong port).

Never take photos or start a recording the user didn't ask for, and don't keep taking photos in a loop: wake on events, or record and review.
