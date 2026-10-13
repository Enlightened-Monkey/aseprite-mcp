"""Integration tests for the pixel-art craft tools (real Aseprite)."""
import json
import os

from mcp.server.fastmcp import Image

from aseprite_mcp.tools import drawing, palette, pixelart
from conftest import BASE, ok, run


def _report(result):
    assert isinstance(result, list), result
    text, image = result
    assert isinstance(image, Image)
    return json.loads(text)


def test_lint_finds_floating_pixel_and_l_corner(sprite):
    # conftest: 16x16 #D04648 square at (8,8). Add a stray pixel and a 1-px
    # outline-coloured staircase with doubled corners.
    ok(run(drawing.draw_pixels_at(sprite, "body", 1, [{"x": 1, "y": 1, "color": "#D04648"}])))
    stair = [(26, 2), (27, 2), (27, 3), (28, 3), (28, 4), (29, 4), (29, 5), (30, 5)]
    ok(run(drawing.draw_pixels_at(sprite, "body", 1, [{"x": x, "y": y, "color": "#2A1B2E"} for x, y in stair])))
    out = f"{BASE}/lint_overlay.png"
    report = _report(run(pixelart.lint_pixel_art(sprite, line_colors="#2A1B2E", overlay_path=out)))
    checks = {i["check"] for i in report["issues"]}
    assert "orphans" in checks
    assert "l_corners" in checks
    assert os.path.exists(out)
    only = json.loads(run(pixelart.lint_pixel_art(sprite, checks="dither")))
    assert only["issues"] == []


def test_lint_rejects_unknown_check(sprite):
    assert run(pixelart.lint_pixel_art(sprite, checks="nope")).startswith("Failed")


def test_readability_preview(sprite):
    out = f"{BASE}/readability.png"
    text, image = run(pixelart.readability_preview(sprite, output_path=out))
    assert "Readability" in text and isinstance(image, Image)
    assert os.path.exists(out)


def test_analyze_palette_and_outline(sprite):
    report = json.loads(run(pixelart.analyze_palette(sprite)))
    assert report["unique_colors"] >= 2
    assert "budget" in report
    outline = json.loads(run(pixelart.suggest_outline_color(sprite)))
    assert outline["color"].startswith("#") and outline["v"] <= 25


def test_color_helpers():
    shades = json.loads(run(pixelart.shade_color("#D04648", "shadow", 2)))
    assert len(shades) == 2
    ramp = json.loads(run(palette.generate_color_ramp("#E08020", 5, details=True)))
    assert ramp[2]["hex"] == "#E08020"
    assert ramp[0]["h"] < 30 or ramp[0]["h"] > 300   # warm shadow goes red/magenta


def test_guide():
    assert "core rules" in run(pixelart.pixel_art_guide()).lower()
    assert "Unknown topic" in run(pixelart.pixel_art_guide("nope"))
    for topic in ("workflow", "lineart", "color", "animation"):
        assert len(run(pixelart.pixel_art_guide(topic))) > 500


def test_missing_file():
    assert run(pixelart.lint_pixel_art("/tmp/ase-pytest/none.aseprite")).startswith("Failed")
