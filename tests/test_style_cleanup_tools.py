"""Integration tests for analyze_style and cleanup_sprite (real Aseprite)."""
import json
import os

import numpy as np
from PIL import Image as PILImage

from aseprite_mcp.core import pixels
from aseprite_mcp.tools import canvas, drawing, pixelart
from conftest import BASE, ok, run
from test_cleanup import _messy
from test_style import DARK, _disc


def _report(result):
    assert isinstance(result, list), result
    return json.loads(result[0])


def test_analyze_style_on_png_with_background():
    base = _disc(size=24, r=10, outline=DARK)
    flat = base.copy()
    flat[flat[..., 3] == 0] = (255, 0, 255, 255)   # opaque magenta background
    path = f"{BASE}/style_bg.png"
    PILImage.fromarray(flat, "RGBA").save(path)
    profile = json.loads(run(pixelart.analyze_style(path)))
    assert "Background colour #FF00FF" in profile["note"]
    assert profile["outline"]["style"] == "black_contour"
    assert profile["lighting"]["from"] == "top-left"


def test_cleanup_recovers_messy_png():
    base = _disc(size=32, r=13, outline=DARK)
    src = f"{BASE}/messy.png"
    PILImage.fromarray(_messy(base, 3.3), "RGBA").save(src)
    report = _report(run(pixelart.cleanup_sprite(src)))
    out = report["output"]["file"]
    assert out == f"{BASE}/messy_clean.aseprite" and os.path.exists(out)
    assert report["size"]["after"] == [32, 32]
    clean = pixels.read_image(out)
    assert ((clean[..., 3] > 0) == (base[..., 3] > 0)).mean() > 0.95


def test_cleanup_aseprite_adds_clean_layer(base_dir):
    path = f"{base_dir}/noisy.aseprite"
    ok(run(canvas.create_canvas(24, 24, path)))
    ok(run(canvas.add_layer(path, "art")))
    ok(run(drawing.draw_rectangle_at(path, "art", 1, 4, 4, 16, 16, "#C85050", True)))
    noise = [{"x": 1, "y": 1, "color": "#C85050"},     # floating pixel
             {"x": 9, "y": 9, "color": "#CB5352"}]     # near-identical speck
    ok(run(drawing.draw_pixels_at(path, "art", 1, noise)))
    report = _report(run(pixelart.cleanup_sprite(path, layer_name="art")))
    assert report["output"]["layer"] == "art clean"
    assert "art" in report["output"]["hidden_layers"]
    comp = pixels.read_image(path)
    assert comp[1, 1, 3] == 0
    assert tuple(comp[9, 9]) == (0xC8, 0x50, 0x50, 255)
    assert tuple(pixels.read_image(path, layer="art")[1, 1]) == (0xC8, 0x50, 0x50, 255)  # original kept


def test_cleanup_in_place_and_errors(base_dir):
    path = f"{base_dir}/inplace.aseprite"
    ok(run(canvas.create_canvas(16, 16, path)))
    ok(run(canvas.add_layer(path, "art")))
    ok(run(drawing.draw_pixels_at(path, "art", 1, [{"x": 2, "y": 2, "color": "#FF0000"}])))
    assert "in_place needs layer_name" in run(pixelart.cleanup_sprite(path, in_place=True))
    report = _report(run(pixelart.cleanup_sprite(path, layer_name="art", in_place=True, steps="orphans")))
    assert report["output"]["layer"] == "art"
    assert pixels.read_image(path, layer="art")[2, 2, 3] == 0
    assert run(pixelart.cleanup_sprite(path, steps="nope")).startswith("Failed")
    assert run(pixelart.cleanup_sprite(f"{base_dir}/missing.png")).startswith("Failed")


def test_lint_respects_black_outline_style(base_dir):
    path = f"{base_dir}/black_outline.aseprite"
    pixels.create_sprite(path, _disc(size=24, r=10, outline=(0, 0, 0, 255)), "art")
    plain = run(pixelart.lint_pixel_art(path, checks="colors"))
    plain = json.loads(plain if isinstance(plain, str) else plain[0])
    assert any(i["check"] == "pure_black" for i in plain["issues"])
    styled = run(pixelart.lint_pixel_art(path, respect_style=True))
    styled = json.loads(styled if isinstance(styled, str) else styled[0])
    assert not any(i["check"] == "pure_black" for i in styled["issues"])
    assert "colors.pure_black" in styled["skipped_for_style"]
