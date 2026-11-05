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
    assert {t.name for t in tools} == {"cameras", "look", "watch", "point", "open_viewer", "annotate", "clear_annotations", "close_viewer"}
