"""Automatic clean-up of rough, resized or noisy pixel art.

The pipeline only makes changes that are safe to do without judgement and
leaves the rest to the artist (reported by the linter afterwards):

1. alpha     - soft (semi-transparent) edges become fully opaque or clear
2. grid      - recover the true pixel grid of art that was upscaled, also by
               non-integer factors or with blur, and sample one colour per cell
3. palette   - merge near-identical colours; reduce to a colour budget when
               the art has far too many colours (blur, resampling, JPEG)
4. orphans   - delete pixels floating alone in transparency
5. speckles  - single noise pixels inside the sprite take the colour of their
               surroundings, unless they read as AA or are high-contrast
               details (eye, shine) that may be deliberate
6. corners   - pixel-perfect pass on line colours (remove L-corner doubles)
7. outer AA  - lighter pixels outside the outline become transparent
8. outline   - close gaps in a single-colour outline (only when the sprite
               uses one dark contour colour on >= 60% of its silhouette edge)

Every step returns the new image and the pixels it changed.
"""
from collections import Counter

import numpy as np
from PIL import Image

from . import lint as L
from .color import delta_e, oklab, palette_budget

_N4 = [(0, -1), (-1, 0), (1, 0), (0, 1)]
_N8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


# ------------------------------------------------------------- alpha

def binarize_alpha(img: np.ndarray, threshold: int = 128) -> tuple[np.ndarray, int]:
    out = img.copy()
    a = out[..., 3]
    soft = (a > 0) & (a < 255)
    out[..., 3] = np.where(a >= threshold, 255, 0)
    out[out[..., 3] == 0] = 0
    return out, int(soft.sum())


# -------------------------------------------------------------- grid

def _profile(img: np.ndarray, axis: int) -> np.ndarray:
    """Colour change between neighbouring columns (axis=1) or rows (axis=0)."""
    f = img.astype(np.float64)
    d = np.abs(np.diff(f, axis=axis)).sum(axis=2)
    return d.sum(axis=0 if axis == 1 else 1)


def _spectral_period(profile: np.ndarray, max_period: float) -> tuple[float, float]:
    """Candidate grid period of an edge profile from its power spectrum.

    Boundaries every `p` pixels put power at 1/p and its multiples, even
    when blur smears each edge. The frequency whose harmonics carry the most
    power on average is returned as a period, with its strength (mean
    harmonic power over the median power of the band; native art ~1-5).
    The caller still has to decide between this period and its multiples.
    """
    x = profile - profile.mean()
    n = len(x)
    size = 1 << int(np.ceil(np.log2(max(n * 16, 1024))))
    power = np.abs(np.fft.rfft(x * np.hanning(n), size)) ** 2
    freqs = np.fft.rfftfreq(size)
    band = (freqs >= 1 / max_period) & (freqs <= 1 / 2.0)
    if not band.any():
        return 1.0, 0.0
    idx = np.nonzero(band)[0]
    median = np.median(power[idx]) + 1e-9
    best_i, best_score = idx[0], -1.0
    for i in idx:
        harmonics = [power[k * i] for k in range(1, 5) if k * i < len(power) and freqs[k * i] <= 0.5]
        score = float(np.mean(harmonics))
        if score > best_score:
            best_i, best_score = i, score
    return float(1 / freqs[best_i]), best_score / median


def _refine(profile: np.ndarray, period: float) -> tuple[float, float, float]:
    """Fine-tune period (+-3%) and pick the phase whose boundaries hit most change."""
    n = len(profile)
    mean = profile.mean() + 1e-9
    best = (period, 0.0, 0.0)
    for p in np.linspace(period * 0.97, period * 1.03, 25):
        for ph in np.arange(0, p, 0.1):
            pos = np.floor(ph + np.arange(0, int(n / p) + 2) * p).astype(int) - 1
            pos = pos[(pos >= 0) & (pos < n)]
            if len(pos) >= 4:
                score = float(profile[pos].mean() / mean)
                if score > best[2]:
                    best = (float(p), float(ph), score)
    return best


def estimate_grid(img: np.ndarray) -> dict:
    """Estimate the pixel-art grid of an image (period and phase per axis).

    Returns {"scale_x", "scale_y", "offset_x", "offset_y", "confidence"};
    scale 1 means the image is already at native resolution.
    """
    h, w = img.shape[:2]
    native = {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0}
    exact = L.detect_scale(img)
    if exact > 1:
        return {"scale_x": float(exact), "scale_y": float(exact), "offset_x": 0.0, "offset_y": 0.0,
                "confidence": 1.0}
    if min(h, w) < 24:
        return {**native, "confidence": 0.0}
    max_p = min(32.0, min(h, w) / 6)
    hard, _ = binarize_alpha(img)  # soft edges blur the boundaries
    prof_x, prof_y = _profile(hard, 1), _profile(hard, 0)
    px, strength_x = _spectral_period(prof_x, max_p)
    py, strength_y = _spectral_period(prof_y, max_p)
    # Pixels are square: when the axes disagree, trust the stronger one.
    if abs(px - py) > 0.08 * max(px, py):
        px = py = px if strength_x >= strength_y else py
    p = (px + py) / 2
    px, ox, sx = _refine(prof_x, p)
    py, oy, sy = _refine(prof_y, p)
    strength = min(strength_x, strength_y)
    # Native art scores ~1-7 on strength; even blurred upscales score 10+.
    if strength < 10 or min(sx, sy) < 1.3:
        return {**native, "confidence": round(min(sx, sy), 2), "spectral_strength": round(float(strength), 1)}
    s = (px + py) / 2
    return {"scale_x": round(s, 3), "scale_y": round(s, 3), "offset_x": round(ox, 2),
            "offset_y": round(oy, 2), "confidence": round(min(sx, sy), 2),
            "spectral_strength": round(float(strength), 1)}


def _prequantize(img: np.ndarray, colors: int = 64) -> np.ndarray:
    """Median-cut the opaque colours so blurred cells share exact colours."""
    opaque = img[..., 3] > 0
    if not opaque.any():
        return img
    px = img[opaque][:, :3]
    if len(np.unique(px, axis=0)) <= colors:
        return img
    strip = Image.fromarray(px.reshape(1, -1, 3).astype(np.uint8), "RGB")
    q = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    out = img.copy()
    out[opaque, :3] = np.array(q).reshape(-1, 3)
    return out


def resample_grid(img: np.ndarray, grid: dict) -> np.ndarray:
    """Sample one colour per grid cell: the most common colour in its centre."""
    sx, sy = grid["scale_x"], grid["scale_y"]
    if sx <= 1 and sy <= 1:
        return img
    src = _prequantize(img)
    h, w = img.shape[:2]

    def cells(n, period, offset):
        start = offset - period * np.ceil(offset / period)
        bounds = []
        b = start
        while b < n:
            lo, hi = b, b + period
            # keep cells that are at least half inside the image
            if min(hi, n) - max(lo, 0) >= period / 2:
                bounds.append((lo, hi))
            b += period
        return bounds

    cols = cells(w, sx, grid["offset_x"])
    rows = cells(h, sy, grid["offset_y"])
    out = np.zeros((len(rows), len(cols), 4), np.uint8)
    for j, (y0, y1) in enumerate(rows):
        cy0 = int(np.clip(np.floor(y0 + 0.25 * sy), 0, h - 1))
        cy1 = int(np.clip(np.ceil(y1 - 0.25 * sy), cy0 + 1, h))
        for i, (x0, x1) in enumerate(cols):
            cx0 = int(np.clip(np.floor(x0 + 0.25 * sx), 0, w - 1))
            cx1 = int(np.clip(np.ceil(x1 - 0.25 * sx), cx0 + 1, w))
            block = src[cy0:cy1, cx0:cx1].reshape(-1, 4)
            if (block[:, 3] == 0).sum() * 2 >= len(block):
                continue
            opaque = block[block[:, 3] > 0]
            vals, counts = np.unique(opaque, axis=0, return_counts=True)
            out[j, i] = vals[int(np.argmax(counts))]
    return out


# ----------------------------------------------------------- palette

def _merge(img: np.ndarray, mapping: dict) -> np.ndarray:
    out = img.copy()
    keys = L._keys(img)
    for src, dst in mapping.items():
        out[keys == src] = dst
    return out


def reduce_colors(img: np.ndarray, target: int | None = None, min_delta: float = 3.0) -> tuple[np.ndarray, list]:
    """Merge colours: always pairs closer than min_delta (OKLab x100), then
    the cheapest pairs (distance x smaller usage) until `target` remain.

    The less-used colour of a pair takes the more-used one, so existing ramps
    stay and stray shades disappear. Returns the image and merges done.
    """
    keys = L._keys(img)
    counts = Counter(int(k) for k in keys[keys != -1])
    cols = list(counts)
    if len(cols) < 2:
        return img, []
    lab = {k: oklab(L._rgb(k)) * 100 for k in cols}
    alias = {k: k for k in cols}
    merges = []
    alive = set(cols)

    def find(k):
        while alias[k] != k:
            k = alias[k]
        return k

    while len(alive) > 1:
        al = list(alive)
        pairs = []
        for i in range(len(al)):
            for j in range(i + 1, len(al)):
                a, b = al[i], al[j]
                d = float(np.linalg.norm(lab[a] - lab[b]))
                pairs.append((d, d * min(counts[a], counts[b]) ** 0.5, a, b))
        dupes = [p for p in pairs if p[0] < min_delta]
        if dupes:
            d, _, a, b = min(dupes)
        elif target is not None and len(alive) > target:
            d, _, a, b = min(pairs, key=lambda p: p[1])
        else:
            break
        keep, drop = (a, b) if counts[a] >= counts[b] else (b, a)
        alias[drop] = keep
        counts[keep] += counts[drop]
        alive.discard(drop)
        merges.append({"from": L.rgb_to_hex(L._rgb(drop)), "to": L.rgb_to_hex(L._rgb(keep)),
                       "distance": round(d, 1)})
    mapping = {k: (*L._rgb(find(k)), k & 255) for k in cols if find(k) != k}
    return _merge(img, mapping), merges


# ------------------------------------------------------ pixel fixes

def remove_floating(img: np.ndarray) -> tuple[np.ndarray, list]:
    out = img.copy()
    px = []
    for issue in L.check_orphans(img):
        if issue["severity"] == "warning":
            for x, y in issue["pixels"]:
                out[y, x] = 0
                px.append((x, y))
    return out, px


def despeckle(img: np.ndarray, max_delta: float = 15.0) -> tuple[np.ndarray, list, list]:
    """Recolour single noise pixels with the colour that surrounds them.

    Only singles that are not plausible AA and are close (OKLab distance <
    max_delta) to the dominant neighbour colour are changed. Distinct singles
    (pupils, shines, accents) are kept and returned for review.
    """
    out = img.copy()
    keys = L._keys(img)
    h, w = keys.shape
    pad = np.pad(keys, 1, constant_values=-1)
    changed, kept = [], []
    for issue in L.check_orphans(img):
        if issue["severity"] != "info":
            continue
        for x, y in issue["pixels"]:
            nbs = [int(pad[1 + y + dy, 1 + x + dx]) for dx, dy in _N8]
            opaque = [k for k in nbs if k != -1]
            if not opaque:
                continue
            major, n = Counter(opaque).most_common(1)[0]
            if n < 4:
                kept.append((x, y))
                continue
            if delta_e(L._rgb(int(keys[y, x])), L._rgb(major)) < max_delta:
                out[y, x] = (*L._rgb(major), major & 255)
                changed.append((x, y))
            else:
                kept.append((x, y))
    return out, changed, kept


def fix_l_corners(img: np.ndarray, line_keys: list[int]) -> tuple[np.ndarray, list]:
    """Pixel-perfect pass: remove doubled corner pixels of line colours.

    The freed pixel takes the most common non-line colour among its four
    neighbours (transparent outside the sprite, the fill inside).
    """
    out = img.copy()
    keys = L._keys(img)
    h, w = keys.shape
    line = np.isin(keys, line_keys)
    changed = []
    for k in line_keys:
        for x, y in L.find_l_corners(keys == k):
            cands = []
            for dx, dy in _N4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if not line[ny, nx]:
                        cands.append(int(keys[ny, nx]))
                else:
                    cands.append(-1)
            if not cands:
                continue
            new = Counter(cands).most_common(1)[0][0]
            out[y, x] = 0 if new == -1 else (*L._rgb(new), new & 255)
            keys[y, x] = new
            line[y, x] = False
            changed.append((x, y))
    return out, changed


def remove_outer_aa(img: np.ndarray) -> tuple[np.ndarray, list]:
    out = img.copy()
    px = []
    for issue in L.check_outer_aa(img):
        for x, y in issue["pixels"]:
            out[y, x] = 0
            px.append((x, y))
    return out, px


def close_outline(img: np.ndarray) -> tuple[np.ndarray, list]:
    """Fill gaps in a single-colour outline along the silhouette edge.

    Applies only when one dark colour covers >= 60% of the edge pixels.
    An edge pixel of another colour becomes the outline colour when at
    least two of its 8 neighbours are outline-coloured edge pixels.
    """
    keys = L._keys(img)
    h, w = keys.shape
    pad = np.pad(keys, 1, constant_values=-1)
    edge = np.zeros((h, w), dtype=bool)
    for dx, dy in _N4:
        edge |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] == -1
    edge &= keys != -1
    counts = Counter(int(k) for k in keys[edge])
    if not counts:
        return img, []
    outline, n = counts.most_common(1)[0]
    if n < 0.6 * edge.sum() or float(L.lightness(L._rgb(outline))) >= 0.35:
        return img, []
    out = img.copy()
    epad = np.pad(edge, 1)
    changed = []
    for y, x in zip(*np.nonzero(edge & (keys != outline))):
        near = sum(1 for dx, dy in _N8 if pad[1 + y + dy, 1 + x + dx] == outline and epad[1 + y + dy, 1 + x + dx])
        if near >= 2:
            changed.append((int(x), int(y)))
    for x, y in changed:
        out[y, x] = (*L._rgb(outline), outline & 255)
    return out, changed


# ------------------------------------------------------------ runner

STEPS = ("alpha", "grid", "palette", "orphans", "speckles", "outline", "corners", "outer_aa")


def cleanup(img: np.ndarray, steps=STEPS, max_colors: int = 0, keep_details: bool = True,
            line_colors: list[str] | None = None, style: dict | None = None) -> tuple[np.ndarray, dict]:
    """Run the clean-up pipeline. Returns (clean image, log).

    max_colors: 0 = automatic (reduce only when there are far more colours
    than the size budget), -1 = never reduce (near-duplicates still merge),
    N = reduce to N colours. keep_details=False also recolours high-contrast
    single pixels. `style` (from analyze_style) keeps deliberate traits:
    a selective outline keeps its light edge pixels.
    """
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"Unknown step(s): {', '.join(unknown)}. Available: {', '.join(STEPS)}")
    log = {"steps": []}
    cur = img

    def note(step, **info):
        log["steps"].append({"step": step, **info})

    if "alpha" in steps:
        cur, n = binarize_alpha(cur)
        note("alpha", soft_pixels_fixed=n)
    keys0 = L._keys(cur)
    filtered = len(np.unique(keys0[keys0 != -1])) > 64  # blur / resampling / JPEG noise
    if "grid" in steps:
        grid = estimate_grid(cur)
        if grid["scale_x"] > 1:
            filtered = True
            before = cur.shape
            cur = resample_grid(cur, grid)
            note("grid", **grid, from_size=[before[1], before[0]], to_size=[cur.shape[1], cur.shape[0]])
        else:
            note("grid", scale_x=1.0, scale_y=1.0, message="already at native resolution")
    if "palette" in steps:
        keys = L._keys(cur)
        n = len(np.unique(keys[keys != -1]))
        rows = np.where((cur[..., 3] > 0).any(axis=1))[0]
        height = int(rows[-1] - rows[0] + 1) if rows.size else 0
        budget = palette_budget(height)["max_colors"]
        target = None
        if max_colors > 0:
            target = max_colors
        elif max_colors == 0 and n > 2 * budget:
            target = budget
        if target is not None and n > 64:
            cur = _prequantize(cur, max(target * 2, 32))
        # Resampled or filtered art carries colour noise: merge wider.
        cur, merges = reduce_colors(cur, target, min_delta=8.0 if filtered else 3.0)
        keys = L._keys(cur)
        note("palette", colors_before=n, merge_distance=8.0 if filtered else 3.0, colors_after=len(np.unique(keys[keys != -1])),
             target=target, merges=merges[:20], merge_count=len(merges))
    if "orphans" in steps:
        cur, px = remove_floating(cur)
        note("orphans", removed=len(px), pixels=px[:40])
    if "speckles" in steps:
        cur, px, kept = despeckle(cur, 15.0 if keep_details else 100.0)
        note("speckles", recoloured=len(px), pixels=px[:40], kept_for_review=kept[:40])
    if "outline" in steps:
        cur, px = close_outline(cur)
        note("outline", closed=len(px), pixels=px[:40])
    if "corners" in steps:
        keys = L._keys(cur)
        lines = [k for k, _ in L._resolve_line_colors(cur, line_colors)]
        cur, px = fix_l_corners(cur, lines)
        note("corners", fixed=len(px), pixels=px[:40])
    if "outer_aa" in steps:
        if style and style.get("outline", {}).get("style") == "selective":
            note("outer_aa", skipped="selective outline lightens edges on purpose")
        else:
            cur, px = remove_outer_aa(cur)
            note("outer_aa", removed=len(px), pixels=px[:40])
    return cur, log
