import numpy as np
from PIL import Image

from aseprite_mcp.pixelart import cleanup as CL
from aseprite_mcp.pixelart import lint as L
from helpers import img
from test_style import DARK, _disc


def _messy(base: np.ndarray, factor: float, seed=0) -> np.ndarray:
    """Upscale by a non-integer factor with bilinear blur, then add noise."""
    h, w = base.shape[:2]
    im = Image.fromarray(base, "RGBA").resize((round(w * factor), round(h * factor)), Image.BILINEAR)
    a = np.array(im).astype(np.int16)
    rng = np.random.default_rng(seed)
    a[..., :3] += rng.integers(-6, 7, a[..., :3].shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def test_estimate_grid_recovers_non_integer_scale():
    base = _disc(size=32, r=13, outline=DARK)
    grid = CL.estimate_grid(_messy(base, 3.3))
    assert abs(grid["scale_x"] - 3.3) < 0.1 and abs(grid["scale_y"] - 3.3) < 0.1


def test_native_art_keeps_its_grid():
    base = _disc(size=32, r=13, outline=DARK)
    assert CL.estimate_grid(base)["scale_x"] == 1.0


def test_cleanup_restores_messy_upscale():
    base = _disc(size=32, r=13, outline=DARK)
    out, log = CL.cleanup(_messy(base, 3.3))
    assert out.shape[:2] == base.shape[:2], log
    n_cols = len(np.unique(L._keys(out)))
    assert n_cols <= 8, log
    # Compare silhouettes and colour layout with the original.
    same_alpha = ((out[..., 3] > 0) == (base[..., 3] > 0)).mean()
    assert same_alpha > 0.95
    both = (out[..., 3] > 0) & (base[..., 3] > 0)
    close = np.abs(out[both, :3].astype(int) - base[both, :3].astype(int)).max(axis=1) < 40
    assert close.mean() > 0.85


def test_reduce_colors_merges_near_duplicates_first():
    a = np.zeros((4, 4, 4), np.uint8)
    a[:2] = (200, 80, 80, 255)
    a[2, :] = (201, 81, 80, 255)
    a[3, :] = (40, 40, 200, 255)
    out, merges = CL.reduce_colors(a)
    assert len(merges) == 1 and merges[0]["to"] == "#C85050"
    out, merges = CL.reduce_colors(a, target=1)
    assert len(np.unique(L._keys(out))) == 1


def test_binarize_alpha():
    a = np.zeros((2, 2, 4), np.uint8)
    a[0, 0] = (10, 10, 10, 200)
    a[0, 1] = (10, 10, 10, 60)
    out, n = CL.binarize_alpha(a)
    assert n == 2 and out[0, 0, 3] == 255 and out[0, 1, 3] == 0


def test_pixel_fixes():
    messy = img("""
..........
.#........
..........
..##......
...##.....
....##....
.....##...
""")
    out, log = CL.cleanup(messy, steps=("orphans", "corners"), line_colors=["#281830"])
    assert out[1, 1, 3] == 0                       # floating pixel removed
    assert L.check_l_corners(out, ["#281830"]) == []  # staircase now pixel-perfect
    assert {s["step"] for s in log["steps"]} == {"orphans", "corners"}


def test_despeckle_keeps_distinct_details():
    a = img("""
ffffff
ffffff
ffwfff
ffffff
""")
    a[1, 4] = (205, 84, 82, 255)  # a near-identical speck
    out, changed, kept = CL.despeckle(a)
    assert (4, 1) in changed and tuple(out[1, 4]) == (200, 80, 80, 255)
    assert (2, 2) in kept


def _line_img(runs, size=(14, 40)):
    from test_lint import _runs_to_chain
    a = np.zeros((*size, 4), np.uint8)
    for x, y in _runs_to_chain(runs):
        a[y, x] = DARK
    return a


def _runs_of(a):
    keys = L._keys(a)
    (chain,) = L._thin_chains(keys == L._keys(a[a[..., 3] > 0][:1].reshape(1, 1, 4))[0, 0])
    return [r["len"] for r in L.chain_runs(chain)]


def test_fix_jaggies_straight_outlier_keeps_end_points():
    a = _line_img([2, 2, 2, 1, 2, 2, 2, 2])
    key = int(L._keys(a)[0, 0])
    out, changed = CL.fix_jaggies(a, [key])
    assert changed
    assert L.check_jaggies(out, ["#281830"]) == []
    ends = lambda img: (tuple(np.argwhere(img[..., 3] > 0)[0]), tuple(np.argwhere(img[..., 3] > 0)[-1]))
    assert ends(out) == ends(a)
    assert (out[..., 3] > 0).sum() == (a[..., 3] > 0).sum()


def test_fix_jaggies_bumpy_curve():
    a = _line_img([5, 4, 3, 2, 1, 2, 1, 3, 2, 3, 6], size=(14, 40))
    key = int(L._keys(a)[0, 0])
    assert L.check_jaggies(a, ["#281830"])
    out, _ = CL.fix_jaggies(a, [key])
    assert len(L.check_jaggies(out, ["#281830"])[0]["pixels"] if L.check_jaggies(out, ["#281830"]) else []) < \
        len(L.check_jaggies(a, ["#281830"])[0]["pixels"])


def test_break_banding_halves_aa():
    a = img("""
###aaafffffffff
...###aaaffffff
......###aaafff
.........###aaa
""")
    assert L.check_banding(a)
    out, changed = CL.break_banding(a)
    assert L.check_banding(out) == []
    assert changed
    # The AA pixel next to each corner stays.
    assert tuple(out[0, 3]) == tuple(a[0, 3])


def test_dithered_areas_are_left_alone():
    a = np.zeros((16, 16, 4), np.uint8)
    for y in range(16):
        for x in range(16):
            a[y, x] = (200, 80, 80, 255) if (x + y) % 2 else (190, 76, 76, 255)
    out, log = CL.cleanup(a, steps=("speckles", "corners", "jaggies", "banding"))
    assert CL.changed_pixels(a, out) == []
