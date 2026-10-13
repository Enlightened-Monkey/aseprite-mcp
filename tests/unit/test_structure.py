import numpy as np

from aseprite_mcp.pixelart import lint as L
from aseprite_mcp.pixelart import structure as ST
from helpers import img
from test_style import DARK, _disc


def _roles(a):
    r = ST.analyze(a)
    return r, {name: ST.role_mask(r, name) for name in ST.ROLES[1:]}


def test_outline_and_inner_line_found():
    a = img("""
..........
.########.
.#ffffff#.
.#ffffff#.
.########.
.#ffffff#.
.#ffffff#.
.########.
..........
""")
    r, m = _roles(a)
    assert m["outline"][1, 1] and m["outline"][7, 5]
    assert m["inner_line"][4, 4]                 # the separating line
    assert m["base"][2, 3]
    assert r["summary"]["has_outline"] and r["summary"]["outline_coverage"] == 1.0
    assert r["summary"]["line_width"] == "1px"


def test_dark_shading_band_is_not_line_art():
    # A dark shade used as a big area AND a thin band: shading, not lines.
    a = np.zeros((16, 16, 4), np.uint8)
    a[2:14, 2:14] = (200, 80, 80, 255)
    a[2:14, 9:14] = (120, 50, 60, 255)            # large shadow area
    a[5, 2:8] = (120, 50, 60, 255)                # thin band of the same shade
    r, m = _roles(a)
    assert not m["inner_line"][5, 4]


def test_lineart_on_empty_layer_and_outline_colors():
    a = np.zeros((20, 20, 4), np.uint8)
    for i in range(14):
        a[3 + i // 2, 2 + i] = (*DARK[:3], 255)
    r, m = _roles(a)
    assert (m["outline"] | m["inner_line"]).sum() == 14


def test_coloured_outline_detected_by_shape():
    disc = _disc(size=24, r=9)
    from aseprite_mcp.pixelart.draw import outline
    coloured, _, _ = outline(disc, "colored", placement="inside")
    r, _ = _roles(coloured)
    assert r["summary"]["has_outline"]


def test_split_and_recolor():
    disc = _disc(size=24, r=9, outline=DARK)
    lineart, rest, res = ST.split_lineart(disc)
    assert (lineart[..., 3] > 0).sum() == res["lines"].sum()
    assert (rest[..., 3] > 0).sum() <= (disc[..., 3] > 0).sum()
    # Filled interior keeps its shape; former line pixels now carry fill colours.
    assert not ((rest[..., :3] == DARK[:3]).all(-1) & (rest[..., 3] > 0)).any()
    out, n = ST.recolor_lineart(disc, "colored")
    assert n > 0 and not ((out[..., :3] == DARK[:3]).all(-1) & (out[..., 3] > 0)).any()
    out, n = ST.recolor_lineart(disc, "single", (10, 200, 10), which="outline")
    assert n > 0 and ((out[..., :3] == (10, 200, 10)).all(-1)).sum() == n


def test_lint_uses_detected_lines():
    # An L-corner in a dark shading cluster is not reported; on the outline it is.
    a = img("""
########
#ffffff#
#faafff#
#ffaaff#
#ffffff#
########
""")
    assert L.check_l_corners(a) == []
