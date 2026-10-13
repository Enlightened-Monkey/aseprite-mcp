import numpy as np
import pytest

from aseprite_mcp.pixelart import anim as A
from aseprite_mcp.pixelart import lint as L
from test_style import DARK, _disc


def test_timing_helpers():
    assert A.units_to_ms([1, 2, 3], "game60") == [20, 30, 50]
    assert A.units_to_ms([2], "film24") == [80]
    assert A.ease_durations(6, 80, 160) == [160, 120, 80, 80, 120, 160]
    assert A.ease_durations(4, 80, 160, "out")[0] == 160
    assert A.ease_durations(4, 80, 160, "in")[-1] == 160
    assert "attack" in A.TIMING_PRESETS
    with pytest.raises(ValueError):
        A.units_to_ms([1], "pal50")


def test_easing_and_paths():
    assert A.ease(0.5, "linear") == 0.5
    assert A.ease(0.25, "ease_in_out") < 0.25
    back = [A.ease(t / 10, "back_out", 0.2) for t in range(11)]
    assert max(back) > 1.0 and back[-1] == pytest.approx(1.0)
    straight = A.path_positions((0, 0), (10, 0), 5, easing="linear")
    assert straight[2] == (5.0, 0.0)
    arc = A.path_positions((0, 0), (10, 0), 5, breakdown=(5, -4), easing="linear")
    assert arc[2] == pytest.approx((5.0, -4.0))
    over = A.path_positions((0, 0), (10, 0), 11, easing="back_out", overshoot_px=2)
    assert max(p[0] for p in over) == pytest.approx(12, abs=0.5) and over[-1][0] == pytest.approx(10)


def test_shift_and_subpixel():
    disc = _disc(size=20, r=6, outline=DARK)
    moved = A.shift(disc, 3, -2)
    assert A.centroid(moved)[0] == pytest.approx(A.centroid(disc)[0] + 3, abs=0.01)
    half = A.subpixel_shift(disc, 0.5, 0)
    # Only colours the art already has.
    keys = L._keys
    assert set(np.unique(keys(half))) <= set(np.unique(keys(disc)))
    assert A.centroid(disc)[0] <= A.centroid(half)[0] <= A.centroid(disc)[0] + 1
    assert (A.subpixel_shift(disc, 1, 0) == A.shift(disc, 1, 0)).all()


def test_squash_keeps_area_and_contact():
    disc = _disc(size=32, r=8)
    squashed = A.squash_stretch(disc, 0.25)
    a0, a1 = (disc[..., 3] > 0).sum(), (squashed[..., 3] > 0).sum()
    assert abs(int(a1) - int(a0)) < 0.15 * a0
    bottom = lambda im: np.nonzero(im[..., 3] > 0)[0].max()
    assert bottom(squashed) == bottom(disc)
    ys, xs = np.nonzero(squashed[..., 3] > 0)
    assert np.ptp(xs) > np.ptp(ys)
    with pytest.raises(ValueError):
        A.squash_stretch(disc, 0.9)


def _clip(n=6, step=1):
    base = _disc(size=24, r=5, outline=DARK)
    return [A.shift(base, i * step, 0) for i in range(n)]


def test_analyzer_patterns():
    frames = _clip(4)
    r = A.analyze(frames + [frames[2], frames[1]], [100] * 6)
    assert r["sequence"] == "0-1-2-3-2-1"
    assert r["transitions"][0]["shift"] == [1, 0] and r["transitions"][0]["residual"] == 0

    two = A.analyze([frames[0], frames[1]] * 2, [100] * 4)
    assert any(i["check"] == "two_frame_loop" for i in two["issues"])

    dup = A.analyze([frames[0], frames[0], frames[1], frames[2]], [100] * 4)
    assert any(i["check"] == "duplicate_frames" and i["frames"] == [2] for i in dup["issues"])

    mech = A.analyze(_clip(7), [100] * 7)
    assert any(i["check"] == "mechanical_spacing" for i in mech["issues"])
    eased = A.analyze(_clip(7), A.ease_durations(7))
    assert not any(i["check"] == "mechanical_spacing" for i in eased["issues"])

    timing = A.analyze(frames, [16, 100, 105, 100])
    assert any(i["check"] == "timing" and i["frames"] == [1, 3] for i in timing["issues"])
