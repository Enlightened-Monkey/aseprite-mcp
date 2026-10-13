import numpy as np
import pytest

from aseprite_mcp.pixelart import lint as L
from helpers import checks, img


def _px(issues):
    return sorted(tuple(p) for i in issues for p in i["pixels"])


# ------------------------------------------------------------- orphans

def test_floating_pixel_is_warning():
    issues = L.check_orphans(img("""
.....
.#...
.....
...ff
...ff
"""))
    assert [i["severity"] for i in issues] == ["warning"]
    assert _px(issues) == [(1, 1)]


def test_aa_single_is_not_orphan_but_noise_is():
    # 'a' sits between outline and fill in lightness -> AA, skipped.
    # 'w' inside the fill is brighter than every neighbour -> noise.
    issues = L.check_orphans(img("""
######
#ffaf#
#ffff#
#fwff#
######
"""))
    assert _px(issues) == [(2, 3)]
    assert issues[0]["severity"] == "info"


# ----------------------------------------------------------- L-corners

def test_l_corners_on_staircase():
    issues = L.check_l_corners(img("""
##....
.##...
..##..
...##.
"""))
    # Greedy: each step has one redundant corner pixel.
    assert len(_px(issues)) == 3


def test_deliberate_box_corner_and_fills_are_not_l_corners():
    assert L.check_l_corners(img("""
#####
#....
#....
""")) == []
    assert L.check_l_corners(img("""
fff
fff
fff
""")) == []


def test_t_junction_is_not_l_corner():
    assert L.check_l_corners(img("""
#######
...#...
...#...
""")) == []


def test_shading_cluster_corners_are_ignored():
    # 'a' forms an L-shaped shading cluster inside an outlined shape: not a line colour.
    assert L.check_l_corners(img("""
########
#ffffff#
#faafff#
#ffaaff#
#ffffff#
########
""")) == []


def test_clean_diagonal_has_no_l_corners():
    assert L.check_l_corners(img("""
#...
.#..
..#.
...#
""")) == []


# ------------------------------------------------------------- jaggies

def _runs_to_chain(seq):
    """Pixel chain whose horizontal runs follow seq, stepping down diagonally."""
    pts, x = [], 0
    for y, n in enumerate(seq):
        for i in range(n):
            pts.append((x + i, y))
        x += n
    return pts


def _jaggy_lengths(seq):
    runs = L.chain_runs(_runs_to_chain(seq))
    return [runs[i]["len"] for i in L.jaggy_runs(runs)]


def test_uniform_and_periodic_straight_lines_are_clean():
    assert _jaggy_lengths([2, 2, 2, 2, 2, 2]) == []
    assert _jaggy_lengths([3, 2, 3, 2, 3, 2, 3]) == []


def test_outlier_step_in_straight_line():
    assert _jaggy_lengths([2, 2, 2, 1, 2, 2, 2]) == [1]


def test_monotone_curve_is_clean():
    assert _jaggy_lengths([5, 4, 3, 2, 1, 1, 1]) == []


def test_bumpy_curve_is_flagged():
    flagged = _jaggy_lengths([5, 4, 3, 2, 1, 2, 1, 3, 2, 3, 6])
    assert flagged and set(flagged) <= {1, 2, 3}


def test_check_jaggies_end_to_end():
    art = np.zeros((12, 40, 4), np.uint8)
    for x, y in _runs_to_chain([2, 2, 2, 1, 2, 2, 2, 2]):
        art[y, x] = (40, 24, 48, 255)
    issues = L.check_jaggies(art)
    assert issues and (6, 3) in [tuple(p) for p in issues[0]["pixels"]]


# ------------------------------------------------------------- banding

def test_aa_copying_every_step_is_banding():
    issues = L.check_banding(img("""
###aaffffffff
...###aafffff
......###aaff
.........###a
"""))
    assert issues
    assert (3, 0) in [tuple(p) for p in issues[0]["pixels"]]


def test_partial_aa_is_not_banding():
    assert L.check_banding(img("""
###afffffffff
...###fffffff
......###afff
.........####
""")) == []


def test_stripe_between_parallel_lines_is_not_banding():
    assert L.check_banding(img("""
##ff##.......
..##ff##.....
....##ff##...
......##ff##.
""")) == []


# -------------------------------------------------------------- pillow

def _rings(light_dir=False):
    size = 16
    out = np.zeros((size, size, 4), np.uint8)
    shades = [(40, 24, 48), (100, 40, 50), (160, 64, 70), (220, 120, 110), (240, 170, 150)]
    for y in range(size):
        for x in range(size):
            d = min(x, y, size - 1 - x, size - 1 - y)
            s = min(d, 3)
            if light_dir and d > 0:
                s = min(4, s + (1 if x < size // 2 else 0))
            out[y, x] = (*shades[s], 255)
    return out


def test_pillow_shading_detected():
    assert checks(L.check_pillow_shading(_rings()), "pillow_shading")


def test_directional_shading_is_not_pillow():
    assert L.check_pillow_shading(_rings(light_dir=True)) == []


# ------------------------------------------------------------- colours

def test_near_duplicates_and_black():
    art = np.zeros((4, 4, 4), np.uint8)
    art[0:2] = (200, 80, 80, 255)
    art[2, :] = (201, 81, 80, 255)
    art[3, :] = (0, 0, 0, 255)
    issues = L.check_colors(art)
    assert checks(issues, "near_duplicates")
    assert checks(issues, "pure_black")


def test_color_budget():
    art = np.zeros((16, 16, 4), np.uint8)
    for i in range(16):
        art[i, :] = (i * 15, 255 - i * 15, (i * 70) % 256, 255)
    assert checks(L.check_colors(art), "color_count")


# -------------------------------------------------------------- dither

def test_dither_on_small_sprite():
    art = np.zeros((8, 8, 4), np.uint8)
    for y in range(8):
        for x in range(8):
            art[y, x] = (200, 80, 80, 255) if (x + y) % 2 else (120, 50, 60, 255)
    assert checks(L.check_dither(art), "dither_small_sprite")


# ------------------------------------------------------------ outer AA

def test_aa_outside_outline():
    issues = L.check_outer_aa(img("""
........
.######.
.#ffff#b
.#ffff#.
.######.
........
"""))
    assert _px(issues) == [(7, 2)]


def test_lint_runner():
    issues = L.lint(img("""
.....
.#...
.....
"""))
    assert issues[0]["severity"] == "warning"
    with pytest.raises(ValueError):
        L.lint(img("."), ["nope"])


# ----------------------------------------------------- shading pairs

def test_shading_pairs_verdicts():
    from aseprite_mcp.pixelart.color import generate_ramp, hex_to_rgb

    ramp = [hex_to_rgb(c["hex"]) for c in generate_ramp(hex_to_rgb("#D04648"), 3)]
    art = np.zeros((6, 6, 4), np.uint8)
    art[0:2] = (*ramp[0], 255)
    art[2:4] = (*ramp[1], 255)
    art[4:6] = (*ramp[2], 255)
    assert {p["verdict"] for p in L.shading_pairs(art)} == {"ok"}

    flat = np.zeros((6, 6, 4), np.uint8)
    flat[0:3] = (100, 30, 30, 255)
    flat[3:6] = (200, 60, 60, 255)
    assert [p["verdict"] for p in L.shading_pairs(flat)] == ["flat"]

    wrong = np.zeros((6, 6, 4), np.uint8)
    wrong[0:3] = (90, 80, 10, 255)   # shadow pushed toward yellow
    wrong[3:6] = (230, 60, 40, 255)
    assert [p["verdict"] for p in L.shading_pairs(wrong)] == ["reversed"]


def test_detect_scale():
    base = np.zeros((16, 16, 4), np.uint8)
    rng = np.random.default_rng(1)
    base[..., :3] = rng.integers(0, 4, (16, 16, 1)) * 60
    base[..., 3] = 255
    assert L.detect_scale(base) == 1
    big = base.repeat(3, axis=0).repeat(3, axis=1)
    assert L.detect_scale(big) == 3
    assert (L.downscale(big, 3) == base).all()


def test_simple_block_art_is_not_treated_as_upscaled():
    art = np.zeros((32, 32, 4), np.uint8)
    art[8:24, 8:24] = (200, 80, 80, 255)
    assert L.detect_scale(art) == 1


def test_half_step_aa_on_every_step_is_fine():
    # AA about half the step long on every corner is correct anti-aliasing.
    assert L.check_banding(img("""
####aaffffffffff
....####aaffffff
........####aaff
............####
""")) == []
