"""Global camera motion on synthetic frames."""
import cv2
import numpy as np

from benchsight import motion


def scene(seed=0, h=480, w=640):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 2)
    for _ in range(40):  # some shapes so ORB and corners have structure
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 40)), y + int(rng.integers(10, 40))), int(rng.integers(0, 255)), -1)
    return img


def shift(img, dx, dy):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_orb_and_lk_recover_a_translation():
    a = scene()
    b = shift(a, 7, -4)
    for m in (motion.orb_motion(a, b), motion.lk_motion(a, b)[0]):
        assert m.confidence > 0.5, m.method
        assert abs(m.M[0, 2] - 7) < 1 and abs(m.M[1, 2] + 4) < 1, (m.method, m.M)
        assert abs(m.magnitude - np.hypot(7, 4)) < 1 and abs(m.rotation) < 1 and abs(m.scale - 1) < 0.02


def test_unrelated_frames_give_low_confidence():
    m = motion.orb_motion(scene(1), scene(2))
    assert m.confidence < 0.5 or m.magnitude == 0


def test_camera_motion_runs_lk_and_orb_and_rescales_to_the_input():
    big = cv2.resize(scene(), (1280, 960))
    cm = motion.CameraMotion(orb_every=2)
    assert cm.update(big).magnitude == 0  # first frame: nothing to compare
    methods = []
    for k in range(1, 5):
        m = cm.update(shift(big, 6 * k, 0))
        methods.append(m.method)
        assert abs(m.M[0, 2] - 6) < 1.5 and abs(m.M[1, 2]) < 1.5, (m.method, m.M)
    assert "lk" in methods and "orb" in methods


def test_mask_leaves_out_a_moving_object():
    a = scene()
    b = a.copy()
    b[100:300, 100:300] = shift(a, 30, 0)[100:300, 100:300]  # only this patch moves
    mask = np.full(a.shape, 255, np.uint8)
    mask[80:320, 80:330] = 0
    m = motion.orb_motion(a, b, mask)
    assert m.magnitude < 1


def test_apply_moves_a_box():
    M = np.float64([[1, 0, 10], [0, 1, -5]])
    assert motion.apply(M, [0, 0, 20, 20]) == [10, -5, 30, 15]
