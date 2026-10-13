"""Integration tests for the projection tools (real Aseprite)."""
import json

import numpy as np

from aseprite_mcp.core import pixels
from aseprite_mcp.tools import generators
from aseprite_mcp.tools import projection_tools as T
from conftest import BASE, run


def _canvas(name, w=96, h=64):
    path = f"{BASE}/{name}.aseprite"
    pixels.create_sprite(path, np.zeros((h, w, 4), np.uint8), "bg")
    return path


def test_iso_scene():
    path = _canvas("iso_scene")
    res = json.loads(run(T.draw_iso_floor(path, "floor", 3, 3, 32, origin_x=32, origin_y=8)))
    assert res["pixels"] > 0
    box = json.loads(run(T.draw_iso_box(path, "box", 48, 50, 6, 4, 14, color="#B08A5A")))
    assert box["left"] == 8 * 14 and box["right"] == 12 * 14
    ok = json.loads(run(generators.outline_sprite(path, "box", style="black", placement="inside",
                                                  color="#281830")))
    assert ok["pixels"] > 0
    check = json.loads(run(T.check_projection_lines(path, "iso", layer_name="box", line_colors=["#281830"])))
    assert check["issues"] == []
    assert "projection must be" in run(T.draw_iso_box(path, "box", 1, 1, 1, 1, 1, projection="fish"))


def test_lines_faces_blocks_grid():
    path = _canvas("proj_misc")
    end = json.loads(run(T.draw_projection_line(path, "lines", 2, 60, 20, "ne", "#FFFFFF")))
    assert end["end"] == [21, 51]
    tex = np.zeros((64, 96, 4), np.uint8)
    tex[2:18, 2:18] = (170, 70, 60, 255)
    pixels.add_layer(path, "texture")
    pixels.write_image(path, "texture", 1, tex)
    face = json.loads(run(T.skew_to_projection(path, "texture", "right", "wall", 40, 10)))
    assert face["size"] == [14, 22]
    wall = pixels.read_image(path, layer="wall")
    assert (wall[..., 3] > 0).sum() == 14 * 16
    ob = json.loads(run(T.draw_oblique_box(path, "props", 60, 40, 12, 10, 4, color="#7A5A3A")))
    assert ob["front"] == 120
    td = json.loads(run(T.draw_topdown_block(path, "props", 5, 5, 16, 8, 8)))
    assert td["camera_ratio_top_to_front"] == 1.0
    grid = json.loads(run(T.draw_projection_grid(path, "iso", 32)))
    assert grid["layer"] == "guides" and grid["pixels"] > 0
