import numpy as np
import pytest

from aseprite_mcp.pixelart import lint as L
from aseprite_mcp.pixelart import projection as P

INK = (40, 24, 48)


def test_diamond_rows_follow_the_tile_rules():
    rows = P.diamond_rows(16)
    assert [r for _, r in rows] == [2, 6, 10, 14, 14, 10, 6, 2]
    assert rows[0][0] == 7                       # 2-px tip centred in a 16-px cell
    with pytest.raises(ValueError):
        P.diamond_rows(18)


def test_floor_tiles_seamlessly():
    img = np.zeros((40, 64, 4), np.uint8)
    P.iso_floor(img, 3, 3, 16, [(10, 120, 10), (20, 160, 20)], origin=(24, 4))
    # The centre tile's whole cell is covered: its diamond plus the tips of
    # its alternate-colour neighbours fill the gaps.
    sx, sy = P.iso_to_screen(1, 1, 16, (24, 4))
    cell = img[sy:sy + 8, sx:sx + 16]
    assert (cell[..., 3] == 255).all()
    assert len({tuple(c) for c in cell.reshape(-1, 4)[:, :3]}) == 2


def test_projection_line_has_exact_runs():
    pts = P.projection_line(0, 20, 16, "ne", 2)
    ys = [y for _, y in pts]
    assert all(ys.count(v) == 2 for v in set(ys))
    assert pts[-1] == (15, 13)
    assert P.projection_line(0, 0, 4, "se", 1) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_iso_box_faces_and_edges():
    img = np.zeros((40, 40, 4), np.uint8)
    counts = P.iso_box(img, 20, 34, 4, 4, 10, (230, 200, 150), (170, 130, 90), (120, 85, 60))
    assert counts["left"] == counts["right"] == 8 * 10
    assert counts["top"] > 0
    # Light from the top-left: top lightest, then left, then right.
    from aseprite_mcp.pixelart.color import lightness
    assert float(lightness((230, 200, 150))) > float(lightness((170, 130, 90))) > float(lightness((120, 85, 60)))
    # The box's outer outline follows a strict 2:1 rhythm.
    from aseprite_mcp.pixelart.draw import outline
    lined, _, _ = outline(img, "black", color=INK, placement="inside")
    assert P.check_lines(lined, "iso", ["#281830"]) == []


def test_check_lines_flags_broken_iso_edge():
    img = np.zeros((20, 40, 4), np.uint8)
    pts = P.projection_line(0, 15, 12, "ne", 2) + [(12, 9), (13, 9), (14, 9)] + \
        [(15 + i, 8 - i // 2) for i in range(8)]
    for x, y in pts:
        img[y, x] = (*INK, 255)
    issues = P.check_lines(img, "iso", ["#281830"])
    assert issues and issues[0]["check"] == "iso_lines"


def test_skew_faces():
    tex = np.zeros((8, 8, 4), np.uint8)
    tex[...] = (200, 80, 80, 255)
    right = P.skew_face(tex, "right")
    assert right.shape[1] == 7 and right.shape[0] == 8 + 3
    top = P.skew_face(tex, "top")
    assert top.shape == (8, 16, 4)
    filled = (top[..., 3] > 0).sum()
    assert 0.4 * 128 < filled < 0.6 * 128          # a diamond fills half its box
    with pytest.raises(ValueError):
        P.skew_face(tex, "back")


def test_oblique_and_topdown():
    img = np.zeros((30, 30, 4), np.uint8)
    c = P.oblique_box(img, 5, 12, 10, 8, 4, (200, 80, 80), (240, 160, 140), (120, 50, 60))
    assert c["front"] == 80 and c["top"] == 40 and c["side"] == 32
    img2 = np.zeros((20, 20, 4), np.uint8)
    P.topdown_block(img2, 2, 2, 8, 4, 6, (240, 200, 160), (150, 100, 80))
    assert tuple(img2[3, 3, :3]) == (240, 200, 160) and tuple(img2[8, 3, :3]) == (150, 100, 80)


def test_grids():
    img = np.zeros((32, 32, 4), np.uint8)
    assert P.grid(img, "topdown", 8, (255, 255, 255)) > 0
    iso = np.zeros((32, 64, 4), np.uint8)
    assert P.grid(iso, "iso", 16, (255, 255, 255)) > 0
    with pytest.raises(ValueError):
        P.grid(img, "fisheye", 8, (0, 0, 0))
