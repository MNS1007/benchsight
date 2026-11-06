"""Image helpers on synthetic frames; no camera needed."""
import numpy as np
import pytest

from benchsight import camera


def frame(h=480, w=640, v=0):
    return np.full((h, w, 3), v, dtype=np.uint8)


def test_fit_downscales_only():
    assert camera.fit(frame(), 320).shape[:2] == (240, 320)
    assert camera.fit(frame(100, 100), 320).shape[:2] == (100, 100)


def test_crop_by_fractions():
    assert camera.crop(frame(), (0.5, 0.5, 1, 1)).shape[:2] == (240, 320)
    with pytest.raises(ValueError):
        camera.crop(frame(), (0.6, 0.1, 0.4, 0.9))


def test_contact_sheet_grid():
    sheet = camera.contact_sheet([frame(v=i * 20) for i in range(5)], cols=3)
    assert sheet.shape[:2] == (480 * 2, 640 * 3)


def test_mark_draws_without_touching_the_original():
    img = frame()
    out = camera.mark(img, 0.5, 0.5, "pin 3")
    assert img.sum() == 0 and out.sum() > 0


def test_save_writes_a_jpeg(tmp_path):
    shot = camera.save(frame(v=128), "t", store=tmp_path)
    assert shot.path.exists() and shot.jpeg[:2] == b"\xff\xd8"


def test_server_registers_the_tools():
    import asyncio
    from benchsight import server
    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} == {"cameras", "look", "watch", "point", "open_viewer", "track", "objects", "untrack",
                                       "annotate", "clear_annotations", "close_viewer", "cleanup",
                                       "start_recording", "stop_recording", "review"}


def test_find_store_prefers_dir_then_env_then_local_then_active(tmp_path, monkeypatch):
    active, served = tmp_path / "active-store", tmp_path / "server" / ".benchsight"
    monkeypatch.setattr(camera, "ACTIVE", active)
    monkeypatch.delenv("BENCHSIGHT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # no ./.benchsight here
    assert camera.find_store() == camera.STORE  # nothing marked yet
    served.mkdir(parents=True)
    assert camera.mark_active(served) == served.resolve()
    assert camera.find_store() == served.resolve()
    assert camera.find_store(tmp_path / "x") == tmp_path / "x"
    (tmp_path / ".benchsight").mkdir()
    assert camera.find_store() == camera.STORE  # a local store wins over the marked one
    monkeypatch.setenv("BENCHSIGHT_DIR", str(tmp_path / "env"))
    assert camera.find_store() == camera.STORE


def test_cli_events_reads_the_active_store(tmp_path, monkeypatch, capsys):
    from benchsight import cli
    served = tmp_path / "server" / ".benchsight"
    served.mkdir(parents=True)
    (served / "events.jsonl").write_text('{"event": "still"}\n')
    monkeypatch.setattr(camera, "ACTIVE", tmp_path / "active-store")
    monkeypatch.delenv("BENCHSIGHT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    camera.mark_active(served)
    cli.main(["events"])
    assert '"still"' in capsys.readouterr().out
