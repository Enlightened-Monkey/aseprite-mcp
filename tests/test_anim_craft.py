"""Integration tests for the animation craft tools (real Aseprite)."""
import json

import numpy as np
from PIL import Image as PILImage

from aseprite_mcp.core import pixels
from aseprite_mcp.pixelart import anim as A
from aseprite_mcp.tools import anim_craft as T
from conftest import BASE, run
from test_style import DARK, _disc


def _sprite(name, frames=1, size=32):
    path = f"{BASE}/{name}.aseprite"
    pixels.create_sprite(path, np.zeros((size, size, 4), np.uint8), "bg")
    pixels.add_layer(path, "body")
    pixels.write_image(path, "body", 1, A.shift(_disc(size=size, r=5, outline=DARK), -6, 0))
    if frames > 1:
        pixels.ensure_frames(path, frames)
    return path


def _json(res):
    return json.loads(res[0] if isinstance(res, list) else res)


def test_path_timing_and_analysis():
    path = _sprite("anim_path")
    res = _json(run(T.animate_layer_path(path, "body", 1, 7, 12, 0, breakdown_dx=6, breakdown_dy=-4,
                                         easing="back_out", overshoot_px=1)))
    assert len(res["positions"]) == 7 and res["positions"][-1] == [12.0, 0.0]
    assert min(p[1] for p in res["positions"]) < -2          # arc through the breakdown
    assert max(p[0] for p in res["positions"]) > 12          # overshoot
    cs = [A.centroid(pixels.read_image(path, f, "body")) for f in range(1, 8)]
    assert cs[-1][0] - cs[0][0] == 12
    timing = _json(run(T.set_frame_timing(path, ease="in_out")))
    assert list(timing["durations"].values()) == A.ease_durations(7)
    assert run(T.set_frame_timing(path, preset="nope")).startswith("Unknown preset")
    report = _json(run(T.analyze_animation(path, layer_name="body")))
    assert report["frames"] == 7 and report["durations"] == A.ease_durations(7)


def test_analyze_gif_reference():
    frames = [A.shift(_disc(size=24, r=5, outline=DARK), i, 0) for i in range(3)]
    flat = []
    for f in frames:
        g = f.copy()
        g[g[..., 3] == 0] = (255, 0, 255, 255)
        flat.append(PILImage.fromarray(g, "RGBA").convert("RGB"))
    gif = f"{BASE}/ref.gif"
    flat[0].save(gif, save_all=True, append_images=flat[1:] + [flat[1]], duration=[100, 60, 100, 60], loop=0)
    report = _json(run(T.analyze_animation(gif)))
    assert report["sequence"] == "0-1-2-1" and report["durations"] == [100, 60, 100, 60]
    assert report["transitions"][0]["shift"] == [1, 0]


def test_squash_subpixel_overlap():
    path = _sprite("anim_moves", frames=4)
    res = _json(run(T.squash_stretch(path, "body", 1, 0.25, target_frame=2)))
    assert abs(res["pixels_after"] - res["pixels_before"]) < 0.2 * res["pixels_before"]
    _json(run(T.subpixel_move(path, "body", 1, 3, 0.5, 0)))
    body1, body3 = pixels.read_image(path, 1, "body"), pixels.read_image(path, 3, "body")
    assert not (body1 == body3).all()
    # Overlap: a follower that copies the leader rigidly should lag one frame.
    for f in range(1, 5):
        pixels.write_image(path, "body", f, A.shift(pixels.read_image(path, 1, "body"), 2 * (f - 1), 0))
    pixels.add_layer(path, "hair")
    hair = np.zeros((32, 32, 4), np.uint8)
    hair[8:10, 8:12] = (200, 160, 40, 255)
    for f in range(1, 5):
        pixels.write_image(path, "hair", f, A.shift(hair, 2 * (f - 1), 0))
    res = _json(run(T.apply_overlap(path, "body", ["hair"], delay_frames=1)))
    assert res["follower_offsets"]["hair"] == [[0, 0], [-2, 0], [-2, 0], [-2, 0]]


def test_build_loop_smear_slash():
    path = _sprite("anim_loop", frames=3)
    for f in (2, 3):
        pixels.write_image(path, "body", f, A.shift(pixels.read_image(path, 1, "body"), 3 * (f - 1), 0))
    pixels.set_durations(path, {1: 200, 2: 100, 3: 300})
    res = _json(run(T.build_loop(path, "1-3", pattern="abac", tag="idle")))
    assert res["sequence"] == [1, 2, 1, 3]
    assert pixels.get_durations(path) == [200, 100, 200, 300]
    assert "pingpong needs" in run(T.build_loop(path, "1-2"))
    res = _json(run(T.generate_smear(path, "body", 1, 4, 5)))
    smear = pixels.read_image(path, 5, "body")
    assert (smear[..., 3] > 0).sum() > (pixels.read_image(path, 1, "body")[..., 3] > 0).sum()
    res = _json(run(T.draw_slash_arc(path, "fx", 6, 16, 16, 12, 200, 340, frames=3)))
    assert res["frames"] == [6, 7, 8] and res["pixels"][0] > res["pixels"][2] > 0
