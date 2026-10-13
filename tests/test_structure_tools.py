"""Integration tests for the structure tools (real Aseprite)."""
import json

import numpy as np

from aseprite_mcp.core import pixels
from aseprite_mcp.tools import structure_tools as T
from conftest import BASE, run
from test_style import DARK, _disc


def test_analyze_extract_recolor():
    path = f"{BASE}/structure.aseprite"
    pixels.create_sprite(path, _disc(size=24, r=9, outline=DARK), "art")
    summary, image = run(T.analyze_structure(path))
    s = json.loads(summary)
    assert s["has_outline"] and s["line_width"] == "1px" and s["part_count"] >= 1
    res = json.loads(run(T.extract_lineart(path, "art")))
    assert res["line_pixels"] > 30
    line = pixels.read_image(path, layer="lineart")
    art = pixels.read_image(path, layer="art")
    assert ((line[..., 3] > 0) & (art[..., :3] == DARK[:3]).all(-1)).sum() == 0
    # Recolour on a fresh copy of the outlined disc.
    pixels.write_image(path, "art", 1, _disc(size=24, r=9, outline=DARK))
    res = json.loads(run(T.recolor_lineart(path, "art", mode="tinted")))
    assert res["pixels_recoloured"] > 30
    assert run(T.recolor_lineart(path, "art", mode="neon")).startswith("Failed")
