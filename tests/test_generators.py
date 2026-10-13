"""Integration tests for the rule-following generators (real Aseprite)."""
import json

import numpy as np

from aseprite_mcp.core import pixels
from aseprite_mcp.pixelart import lint as L
from aseprite_mcp.tools import animation, canvas, generators
from conftest import ok, run
from test_style import _disc


def _new(base_dir, name, w=40, h=40):
    path = f"{base_dir}/{name}.aseprite"
    ok(run(canvas.create_canvas(w, h, path)))
    return path


def test_clean_line_and_curve(base_dir):
    path = _new(base_dir, "gen_lines")
    res = json.loads(run(generators.draw_clean_line(path, "line", 1, 1, 35, 9, "#2A1B2E")))
    assert res["end"] == [35, 9]
    snap = json.loads(run(generators.draw_clean_line(path, "snap", 1, 20, 35, 33, "#2A1B2E", mode="snap")))
    assert snap["end"][0] in (34, 35)
    ok(run(generators.draw_smooth_curve(path, "curve", [[2, 38], [2, 14], [30, 14]], "#2A1B2E")))
    for layer in ("line", "snap", "curve"):
        img = pixels.read_image(path, layer=layer)
        assert L.check_l_corners(img, ["#2A1B2E"]) == [], layer
        assert L.check_jaggies(img, ["#2A1B2E"]) == [], layer
    assert run(generators.draw_clean_line(path, "line", 0, 0, 5, 5, "#000", mode="nope")).startswith("mode")


def test_outline_aa_and_reduce(base_dir):
    path = f"{base_dir}/gen_outline.aseprite"
    pixels.create_sprite(path, _disc(size=32, r=12), "body")
    res = json.loads(run(generators.outline_sprite(path, "body", style="selective", light="top-left")))
    assert res["pixels"] > 0 and res["outline_color"] != "#000000"
    img = pixels.read_image(path, layer="body")
    before = len(np.unique(L._keys(img)))
    aa = json.loads(run(generators.antialias_lines(path, "body", res["outline_color"], shades=1)))
    assert aa["pixels_changed"] >= 0
    red = json.loads(run(generators.reduce_colors(path, 4)))
    assert red["colors_after"] <= 4
    assert len(np.unique(L._keys(pixels.read_image(path, layer="body")))) <= 5  # + transparent


def test_outline_to_separate_layer(base_dir):
    path = f"{base_dir}/gen_outline2.aseprite"
    pixels.create_sprite(path, _disc(size=24, r=9), "body")
    ok(run(generators.outline_sprite(path, "body", style="black", output_layer="outline")))
    body = pixels.read_image(path, layer="body")
    line = pixels.read_image(path, layer="outline")
    assert ((line[..., 3] > 0) & (body[..., 3] > 0)).sum() == 0
    assert (line[..., 3] > 0).sum() > 30


def test_dither_and_gradient_map(base_dir):
    path = _new(base_dir, "gen_dither", 16, 32)
    ramp = ["#1A1C2C", "#3B5DC9", "#73EFF7"]
    res = json.loads(run(generators.dither_fill(path, "sky", 0, 0, 16, 32, ramp)))
    assert res["pixels"] == 16 * 32
    sky = pixels.read_image(path, layer="sky")
    assert len(np.unique(sky.reshape(-1, 4), axis=0)) == 3
    ok(run(canvas.add_frame(path)))
    res = json.loads(run(generators.gradient_map(path, "sky", ["#300A3C", "#F0C090"])))
    assert res["frames"] == [1, 2]
    mapped = pixels.read_image(path, layer="sky")
    assert {tuple(c) for c in mapped.reshape(-1, 4)[:, :3]} <= {(0x30, 0x0A, 0x3C), (0xF0, 0xC0, 0x90)}


def test_reduce_colors_shared_across_frames(base_dir):
    path = _new(base_dir, "gen_reduce", 8, 8)
    a = np.zeros((8, 8, 4), np.uint8)
    a[:4] = (200, 80, 80, 255)
    a[4:] = (40, 40, 200, 255)
    pixels.add_layer(path, "art")
    pixels.write_image(path, "art", 1, a)
    ok(run(animation.add_frames(path, 1)))
    b = a.copy()
    b[4:] = (44, 44, 205, 255)       # near-identical blue in frame 2
    pixels.write_image(path, "art", 2, b)
    res = json.loads(run(generators.reduce_colors(path, 2, layer_name="art")))
    assert res["colors_after"] == 2
    f1, f2 = pixels.read_image(path, 1, "art"), pixels.read_image(path, 2, "art")
    assert (f1 == f2).all()
