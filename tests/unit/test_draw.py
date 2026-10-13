import numpy as np
import pytest

from aseprite_mcp.pixelart import draw as D
from aseprite_mcp.pixelart import lint as L
from aseprite_mcp.pixelart.color import lightness
from helpers import img
from test_style import DARK, _disc

INK = (40, 24, 48)


def _canvas(points, size=(40, 40)):
    a = np.zeros((*size, 4), np.uint8)
    for x, y in points:
        a[y, x] = (*INK, 255)
    return a


def _runs(points):
    keys = L._keys(_canvas(points))
    (chain,) = L._thin_chains(keys != -1)
    return [r["len"] for r in L.chain_runs(chain)]


@pytest.mark.parametrize("end", [(30, 7), (30, 19), (11, 30), (29, 29), (30, 13)])
def test_lines_are_clean(end):
    pts = D.line_points(0, 0, *end)
    assert pts[0] == (0, 0) and pts[-1] == end
    a = _canvas(pts)
    assert L.check_l_corners(a, ["#281830"]) == []
    assert L.check_jaggies(a, ["#281830"]) == []


def test_snapped_line_has_equal_steps():
    pts, end = D.snapped_line(0, 0, 30, 14)        # ~2.1:1 -> 2:1
    assert pts[0] == (0, 0) and 29 <= end[0] <= 30  # end moves to a whole step
    assert set(_runs(pts)) == {2}


def test_curve_is_pixel_perfect_and_smooth():
    pts = D.curve_points([(2, 30), (2, 2), (34, 2)])
    a = _canvas(pts)
    assert L.check_l_corners(a, ["#281830"]) == []
    assert L.check_jaggies(a, ["#281830"]) == []
    assert (2, 30) in pts and (34, 2) in pts
    with pytest.raises(ValueError):
        D.curve_points([(1, 1)])


def test_antialias_corners_inside_only():
    a = img("""
####ffffffff
....####ffff
........####
""")
    out, changed, new = D.antialias_edges(a, INK)
    assert changed
    for x, y in changed:
        assert a[y, x, 3] == 255                       # never onto transparency
        l = float(lightness(out[y, x, :3]))
        assert float(lightness(INK)) < l < float(lightness(a[y, x, :3]))
    assert len(changed) <= 4                            # about half of each 4-px step
    assert L.check_banding(out) == []


def test_antialias_two_shades_darker_at_corner():
    a = img("""
########ffffffffffff
........############
""")
    out, changed, _ = D.antialias_edges(a, INK, shades=2)
    row = sorted((x, y) for x, y in changed if y == 0)
    ls = [float(lightness(out[y, x, :3])) for x, y in row]
    assert ls == sorted(ls)                             # lighter going away from the corner


def test_outline_styles():
    base = _disc(size=24, r=9)
    for style in ("black", "colored", "shaded", "selective"):
        out, changed, info = D.outline(base, style=style)
        assert changed
        assert all(base[y, x, 3] == 0 for x, y in changed)  # outside placement
    black, _, info = D.outline(base, "black")
    assert info["dark"] != (0, 0, 0)
    sel, ch_sel, _ = D.outline(base, "selective", light="top-left")
    _, ch_all, _ = D.outline(base, "shaded", light="top-left")
    assert len(ch_sel) < len(ch_all)
    top_left = [p for p in ch_all if p not in ch_sel]
    assert np.mean([x for x, _ in top_left]) < 12 and np.mean([y for _, y in top_left]) < 12
    inside, ch_in, _ = D.outline(base, "black", placement="inside")
    assert all(base[y, x, 3] > 0 for x, y in ch_in)
    with pytest.raises(ValueError):
        D.outline(base, "fancy")


def test_dither_fill_ramp_and_alignment():
    ramp = [(20, 20, 60), (60, 60, 140), (140, 160, 220)]
    a = np.zeros((32, 8, 4), np.uint8)
    out, n = D.dither_fill(a, (0, 0, 8, 32), ramp)
    assert n == 256
    assert tuple(out[0, 0, :3]) == ramp[0] and tuple(out[31, 0, :3]) == ramp[2]
    # Only neighbouring ramp colours are mixed in any row.
    for y in range(32):
        cols = {tuple(c) for c in out[y, :, :3]}
        idx = sorted(ramp.index(c) for c in cols)
        assert idx[-1] - idx[0] <= 1
    # Two adjacent fills tile seamlessly (pattern aligned to the canvas).
    b = np.zeros((32, 16, 4), np.uint8)
    b, _ = D.dither_fill(b, (0, 0, 16, 32), ramp)
    c = np.zeros((32, 16, 4), np.uint8)
    c, _ = D.dither_fill(c, (0, 0, 8, 32), ramp)
    c, _ = D.dither_fill(c, (8, 0, 8, 32), ramp)
    assert (b == c).all()


def test_dither_fill_mask_and_levels():
    shape = _disc(size=16, r=6)
    out, n = D.dither_fill(shape, (0, 0, 16, 16), [(0, 0, 0), (255, 255, 255)], mask="opaque")
    assert n == (shape[..., 3] > 0).sum()
    assert (out[..., 3] == shape[..., 3]).all()
    hard, _ = D.dither_fill(np.zeros((8, 4, 4), np.uint8), (0, 0, 4, 8), [(0, 0, 0), (255, 255, 255)], levels=2)
    for y in range(8):
        assert len({tuple(c) for c in hard[y, :, :3]}) == 1   # 2 levels = hard bands


def test_gradient_map():
    grey = np.zeros((4, 4, 4), np.uint8)
    for x in range(4):
        grey[:, x] = (x * 80, x * 80, x * 80, 255)
    ramp = [(30, 10, 60), (120, 40, 90), (220, 120, 100), (250, 230, 170)]
    out = D.gradient_map(grey, ramp)
    assert [tuple(out[0, x, :3]) for x in range(4)] == ramp
    dithered = D.gradient_map(grey, ramp[:2], dither=True)
    assert {tuple(c) for c in dithered[:, 1, :3]} <= set(ramp[:2])
