"""Shapes anchored to tracked objects: they follow the object's box; static ones don't."""
import numpy as np
import pytest

from benchsight.overlay import Overlay, render, resolve

A = [0.2, 0.2, 0.4, 0.4]  # the anchor's box when the shapes were drawn
MOVED = [0.5, 0.3, 0.9, 0.7]  # moved right and down, and twice the size


def obj(box, status="tracking", oid="plate"):
    return {"id": oid, "label": "plate", "box": box, "status": status}


def drawn():
    ov = Overlay()
    ov.set([
        {"type": "box", "box": [0.2, 0.2, 0.4, 0.4], "anchor": "plate", "label": "this plate"},
        {"type": "circle", "at": [0.3, 0.3], "r": 0.05, "anchor": "plate"},
        {"type": "arrow", "from": [0.1, 0.8], "to": [0.3, 0.4], "anchor": "plate", "label": "push"},
        {"type": "path", "points": [[0.2, 0.5], [0.3, 0.6]], "anchor": "plate"},
        {"type": "text", "at": [0.35, 0.15], "text": "flip", "anchor": "plate"},
        {"type": "circle", "at": [0.8, 0.8], "r": 0.05, "label": "static"},
    ], anchors={"plate": A})
    return ov


def close(a, b):
    assert np.allclose(a, b, atol=1e-3), (a, b)


def test_box_and_circle_follow_and_scale():
    box, circle, *_ = resolve(drawn().state(), [obj(MOVED)])["shapes"]
    close(box["box"], MOVED)
    close(circle["at"], [0.7, 0.5])
    close(circle["r"], 0.1)
    assert box["lost"] is False and "rel" not in box


def test_arrow_head_follows_and_tail_keeps_its_offset():
    arrow = resolve(drawn().state(), [obj(MOVED)])["shapes"][2]
    close(arrow["to"], [0.7, 0.7])  # bottom middle of the moved box
    close(np.subtract(arrow["from"], arrow["to"]), [-0.2, 0.4])


def test_path_and_text_translate_without_scaling():
    _, _, _, path, text, _ = resolve(drawn().state(), [obj(MOVED)])["shapes"]
    shift = [0.4, 0.2]  # center 0.3,0.3 -> 0.7,0.5
    close(path["points"], [[0.6, 0.7], [0.7, 0.8]])
    close(text["at"], np.add([0.35, 0.15], shift))


def test_static_shapes_do_not_move():
    ov = drawn()
    before = ov.state()["shapes"][-1]
    after = resolve(ov.state(), [obj(MOVED)])["shapes"][-1]
    assert after == before == {"type": "circle", "color": "red", "label": "static", "at": [0.8, 0.8], "r": 0.05}
    assert "lost" not in after and "anchor" not in after


def test_same_box_gives_the_drawn_positions():
    ov = drawn()
    for raw, res in zip(ov.state()["shapes"], resolve(ov.state(), [obj(A)])["shapes"]):
        for k in ("box", "at", "from", "to", "points", "r"):
            if k in raw:
                close(res[k], raw[k])


@pytest.mark.parametrize("objects", [[obj(MOVED, "lost")], [obj(MOVED, "stale")], []])
def test_lost_stale_or_gone_anchors_mark_shapes_lost(objects):
    shapes = resolve(drawn().state(), objects)["shapes"]
    assert all(s["lost"] for s in shapes[:-1]) and "lost" not in shapes[-1]
    if not objects:  # gone: shown where it was drawn
        close(shapes[0]["box"], A)


def test_unknown_anchor_is_refused_with_the_tracked_ids():
    with pytest.raises(ValueError, match="tracked: plate"):
        Overlay().set([{"type": "circle", "at": [0.5, 0.5], "anchor": "screw"}], anchors={"plate": A})
    with pytest.raises(ValueError, match="call track first"):
        Overlay().set([{"type": "circle", "at": [0.5, 0.5], "anchor": "screw"}])


def test_drop_removes_only_shapes_on_those_anchors():
    ov = drawn()
    assert ov.anchored and ov.drop(["plate"]) == 5
    assert [s["label"] for s in ov.state()["shapes"]] == ["static"] and not ov.anchored


def test_render_draws_lost_shapes_dashed_and_faded():
    img = np.full((240, 320, 3), 40, np.uint8)
    ov = Overlay()
    ov.set([{"type": "box", "box": [0.2, 0.2, 0.6, 0.6], "anchor": "plate", "color": "green"}],
           anchors={"plate": [0.2, 0.2, 0.6, 0.6]})
    solid = render(img, resolve(ov.state(), [obj([0.2, 0.2, 0.6, 0.6])]))
    lost = render(img, resolve(ov.state(), [obj([0.2, 0.2, 0.6, 0.6], "lost")]))
    top = slice(46, 50), slice(70, 190)  # along the top edge of the box
    on_solid = (solid[top] != 40).any(axis=2).mean()
    on_lost = (lost[top] != 40).any(axis=2).mean()
    assert on_solid > 0.95 and 0.2 < on_lost < 0.9, "dashed edge has gaps"
    assert lost[top][..., 1].max() < solid[top][..., 1].max(), "faded"
    assert (img == 40).all()
