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
7. jaggies   - even out the staircase rhythm of lines (end points kept)
8. banding   - shorten bands that copy a staircase to about half a step
9. outer AA  - lighter pixels outside the outline become transparent
10. outline  - close gaps in a single-colour outline (only when the sprite
               uses one dark contour colour on >= 60% of its silhouette edge)

Every step returns the new image and the pixels it changed. Pixel-level
fixes (speckles, corners, jaggies, banding) never touch dithered areas.
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
    # Pixels are square: when the axes disagree (a regular staircase along
    # one axis can out-shout the grid there), trust the stronger axis and
    # judge confidence on it; the round trip below still guards the result.
    agree = abs(px - py) <= 0.08 * max(px, py)
    if not agree:
        px = py = px if strength_x >= strength_y else py
    p = (px + py) / 2
    px, ox, sx = _refine(prof_x, p)
    py, oy, sy = _refine(prof_y, p)
    strength = min(strength_x, strength_y) if agree else max(strength_x, strength_y)
    confidence = min(sx, sy) if agree else max(sx, sy)
    # Native art scores ~1-7 on strength; even blurred upscales score 10+.
    if strength < 10 or confidence < 1.3:
        return {**native, "confidence": round(confidence, 2), "spectral_strength": round(float(strength), 1)}
    s = (px + py) / 2
    grid = {"scale_x": round(s, 3), "scale_y": round(s, 3), "offset_x": round(ox, 2),
            "offset_y": round(oy, 2), "confidence": round(confidence, 2),
            "spectral_strength": round(float(strength), 1)}
    # A regular pattern in native art (stripes, a long 45-degree edge,
    # tiles) also has a strong period. A real upscale survives a round trip:
    # sampled down to the grid and back up, it still matches itself. Measured
    # on the defect benchmark: real upscales 0.88-1.00, native art <= 0.83.
    match, edge_match = _round_trip_match(hard, grid)
    grid["round_trip_match"] = round(match, 2)
    grid["round_trip_edges"] = round(edge_match, 2)
    if match < 0.85:
        return {**native, "confidence": grid["confidence"], "spectral_strength": grid["spectral_strength"],
                "round_trip_match": grid["round_trip_match"], "round_trip_edges": grid["round_trip_edges"],
                "message": "periodic pattern, but not an upscale (round trip does not match)"}
    return grid


def _cell_index(n: int, period: float, offset: float) -> np.ndarray:
    """For each of n pixels, the index of the grid cell holding its centre,
    matching resample_grid's cells (partial edge cells under half dropped)."""
    start = offset - period * np.ceil(offset / period)
    bounds = []
    b = start
    while b < n:
        lo, hi = b, b + period
        if min(hi, n) - max(lo, 0) >= period / 2:
            bounds.append((lo, hi))
        b += period
    idx = np.full(n, -1)
    for i, (lo, hi) in enumerate(bounds):
        sel = (np.arange(n) + 0.5 >= lo) & (np.arange(n) + 0.5 < hi)
        idx[sel] = i
    return idx


def _round_trip_match(img: np.ndarray, grid: dict, tolerance: float = 10.0) -> tuple[float, float]:
    """(all pixels, edge pixels): shares whose colour survives grid
    down/up-sampling. The edge share (pixels at real colour changes) is
    reported for diagnosis; blur makes it unreliable as a gate."""
    small = resample_grid(img, grid)
    h, w = img.shape[:2]
    iy = _cell_index(h, grid["scale_y"], grid["offset_y"])
    ix = _cell_index(w, grid["scale_x"], grid["offset_x"])
    ok = (iy[:, None] >= 0) & (ix[None, :] >= 0) & (img[..., 3] > 0)
    if not ok.any():
        return 0.0, 0.0
    # Edges = real colour changes (not noise): a perceptual jump above 10.
    lab = oklab(img[..., :3]) * 100
    edge = np.zeros((h, w), dtype=bool)
    jump_x = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=-1) > 10
    jump_y = np.linalg.norm(lab[1:] - lab[:-1], axis=-1) > 10
    edge[:, 1:] |= jump_x
    edge[:, :-1] |= jump_x
    edge[1:, :] |= jump_y
    edge[:-1, :] |= jump_y

    def share(mask):
        yy, xx = np.nonzero(mask)
        if not len(yy):
            return 0.0
        recon = small[iy[yy], ix[xx]]
        d = np.linalg.norm(oklab(img[yy, xx, :3]) - oklab(recon[:, :3]), axis=-1) * 100
        return float(((d < tolerance) & (recon[:, 3] > 0)).mean())

    return share(ok), share(ok & edge)


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
    members = {k: [k] for k in cols}  # original colours now drawn as k

    def drift_ok(a, b):
        """Merging must not move any original colour further than min_delta."""
        keep = a if counts[a] >= counts[b] else b
        return all(float(np.linalg.norm(lab[m] - lab[keep])) < min_delta for m in members[a] + members[b])

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
        dupes = [p for p in pairs if p[0] < min_delta and drift_ok(p[2], p[3])]
        if dupes:
            d, _, a, b = min(dupes)
        elif target is not None and len(alive) > target:
            d, _, a, b = min(pairs, key=lambda p: p[1])
        else:
            break
        keep, drop = (a, b) if counts[a] >= counts[b] else (b, a)
        alias[drop] = keep
        counts[keep] += counts[drop]
        members[keep] += members.pop(drop)
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
    (pupils, shines, accents) are kept and returned for review. A colour that
    appears as a single pixel many times (lint.PARTICLE_REPEATS or more) is a
    deliberate particle or texture - snow, stars, rain, sparkle, fur flecks -
    and all its singles are kept.
    """
    out = img.copy()
    keys = L._keys(img)
    h, w = keys.shape
    pad = np.pad(keys, 1, constant_values=-1)
    changed, kept = [], []
    # check_orphans already sets particles (repeated single-pixel colours) apart.
    for issue in L.check_orphans(img):
        if issue["check"] != "orphans" or issue["severity"] != "info":
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


def _as_masks(keys: np.ndarray, line_keys: list) -> list[tuple[int, np.ndarray]]:
    """Accept colour keys (whole colour is line) or (key, mask) pairs."""
    out = []
    for item in line_keys:
        if isinstance(item, tuple):
            out.append((int(item[0]), item[1]))
        else:
            out.append((int(item), keys == item))
    return out


def fix_l_corners(img: np.ndarray, line_keys: list) -> tuple[np.ndarray, list]:
    """Pixel-perfect pass: remove doubled corner pixels of line colours.

    The freed pixel takes the most common non-line colour among its four
    neighbours (transparent outside the sprite, the fill inside).
    """
    out = img.copy()
    keys = L._keys(img)
    h, w = keys.shape
    masks = _as_masks(keys, line_keys)
    line = np.zeros(keys.shape, dtype=bool)
    for _, m in masks:
        line |= m
    changed = []
    for k, m in masks:
        for x, y in L.find_l_corners(m):
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
    """Fill gaps in a single-colour outline along the silhouette edge
    (see lint.find_outline_gaps for what counts as a gap)."""
    gaps, outline = L.find_outline_gaps(img)
    out = img.copy()
    for x, y in gaps:
        out[y, x] = (*L._rgb(outline), outline & 255)
    return out, gaps


def thin_lines(img: np.ndarray) -> tuple[np.ndarray, list]:
    """Give the inner layer of 2-px thickenings back to the fill."""
    from .structure import analyze as _structure
    px = L.find_thick_lines(img)
    if not px:
        return img, []
    out = img.copy()
    keys = L._keys(img)
    line = _structure(img)["lines"].copy()
    for x, y in px:
        line[y, x] = False
    for x, y in px:
        _set(out, keys, x, y, _fill_colour(keys, line, x, y))
    return out, px


def trim_aa_bands(img: np.ndarray) -> tuple[np.ndarray, list]:
    """Shorten AA bands that double a line: keep the pixels in the corners
    of the line's steps, give the rest the colour on the band's far side."""
    from .structure import analyze as _structure
    bands = L.find_aa_bands(img)
    if not bands:
        return img, []
    out = img.copy()
    keys = L._keys(img)
    h, w = keys.shape
    lines = _structure(img)["lines"]
    changed = []
    for band in bands:
        keep = set(band["keep"])
        for x, y in band["pixels"]:
            if (x, y) in keep:
                continue
            for dx, dy in _N4:
                lx, ly, fx, fy = x - dx, y - dy, x + dx, y + dy
                if 0 <= lx < w and 0 <= ly < h and 0 <= fx < w and 0 <= fy < h and lines[ly, lx] \
                        and not lines[fy, fx] and keys[fy, fx] != -1:
                    out[y, x] = img[fy, fx]
                    changed.append((x, y))
                    break
    return out, changed


def _fill_colour(keys: np.ndarray, line: np.ndarray, x: int, y: int) -> int:
    """Most common non-line colour among the 4 neighbours (-1 = transparent)."""
    h, w = keys.shape
    cands = []
    for dx, dy in _N4:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            if not line[ny, nx]:
                cands.append(int(keys[ny, nx]))
        else:
            cands.append(-1)
    return Counter(cands).most_common(1)[0][0] if cands else -1


def _set(out: np.ndarray, keys: np.ndarray, x: int, y: int, key: int) -> None:
    out[y, x] = 0 if key == -1 else (*L._rgb(key), key & 255)
    keys[y, x] = key


def _window_order(runs: list[dict], bad: list[int]) -> list[tuple[int, int, list[tuple]]]:
    """Group jaggy runs into windows and re-order them into a clean rhythm.

    Returns (first, last, new_runs) per window, new_runs being (axis, len)
    pairs. Re-ordering keeps the same runs, so the chain's end points do not
    move. Inside a trend (a curve, possibly through 45 degrees) runs are
    sorted by steepness to follow the trend; on a straight stretch odd runs
    move to the end of the stretch.
    """
    out, groups = [], []
    for i in bad:
        if groups and i <= groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    n = len(runs)
    desc = lambda k: (runs[k]["axis"], runs[k]["len"])
    for g in groups:
        lo, hi = max(1, g[0] - 1), min(n - 2, g[-1] + 1)
        if any(runs[k]["break"] for k in range(lo + 1, hi + 1)):
            continue
        before = L._slope_value(runs[lo - 1]) if lo - 1 >= 0 else None
        after = L._slope_value(runs[hi + 1]) if hi + 1 < n else None
        cur = [desc(k) for k in range(lo, hi + 1)]
        slope = lambda r: L._slope_value({"axis": r[0], "len": r[1]})
        if before is not None and after is not None and before != after:
            new = sorted(cur, key=slope, reverse=before > after)
        else:
            common = Counter(cur).most_common(1)[0][0]
            while lo - 1 >= 1 and desc(lo - 1) == common and not runs[lo]["break"]:
                lo -= 1
            while hi + 1 <= n - 2 and desc(hi + 1) == common and not runs[hi + 1]["break"]:
                hi += 1
            cur = [desc(k) for k in range(lo, hi + 1)]
            odd = [r for r in cur if r != common]
            new = [common] * (len(cur) - len(odd)) + sorted(odd, key=slope, reverse=True)
        if [slope(r) for r in new] != [slope(r) for r in cur]:
            out.append((lo, hi, new))
    return out


def fix_jaggies(img: np.ndarray, line_keys: list) -> tuple[np.ndarray, list]:
    """Even out the staircase rhythm of line colours without moving end points.

    Uses the linter's run analysis; each flagged window of runs is re-laid in
    a clean order (see _window_order), keeping every run's own direction.
    Pixels a line leaves take the surrounding non-line colour; pixels it
    enters take the line colour.
    """
    out = img.copy()
    keys = L._keys(img)
    changed = []
    for key, m in _as_masks(keys, line_keys):
        for chain in L._thin_chains(m):
            runs = L.chain_runs(chain)
            for lo, hi, new in _window_order(runs, L.jaggy_runs(runs)):
                diags = {(runs[k + 1]["pixels"][0][0] - runs[k]["pixels"][-1][0],
                          runs[k + 1]["pixels"][0][1] - runs[k]["pixels"][-1][1]) for k in range(lo, hi)}
                if len(diags) != 1:
                    continue
                diag = diags.pop()
                if not (diag[0] and diag[1]):
                    continue
                old = [p for k in range(lo, hi + 1) for p in runs[k]["pixels"]]
                pts, (x, y) = [], old[0]
                for i, (axis, length) in enumerate(new):
                    step = (diag[0], 0) if axis == "h" else (0, diag[1])
                    for j in range(length):
                        pts.append((x, y))
                        if j < length - 1:
                            x, y = x + step[0], y + step[1]
                    if i < len(new) - 1:
                        x, y = x + diag[0], y + diag[1]
                if pts[-1] != old[-1]:
                    continue
                for (px, py) in set(pts) - set(old):
                    if keys[py, px] != key:
                        _set(out, keys, px, py, key)
                        changed.append((px, py))
                line = keys == key
                for (px, py) in set(old) - set(pts):
                    line[py, px] = False
                    _set(out, keys, px, py, _fill_colour(keys, line, px, py))
                    changed.append((px, py))
    return out, changed


def break_banding(img: np.ndarray) -> tuple[np.ndarray, list]:
    """Shorten bands that copy a line's staircase: anti-aliasing should
    cover about half a step, starting at the corner. Thin 1-px bands along
    45-degree lines are dropped on every other step. Freed pixels take the
    colour on the band's far side.

    Only AA-like bands are touched: the staircase they hug is detected line
    art and the band's tone lies between the line and what is beyond it.
    Parallel tone strips that model a form (a tusk, a rope, a pipe) are
    shading, not banding, and are left alone (the linter still lists them).
    """
    from .structure import analyze as _structure
    out = img.copy()
    changed = []
    lines = _structure(img)["lines"]
    lum = L.lightness(img[..., :3])
    for transposed in (False, True):
        keys = L._keys(out)
        k = keys.T if transposed else keys
        lmask = lines.T if transposed else lines
        lm = lum.T if transposed else lum
        n_cols = k.shape[1]
        for chain in L.band_chains(k):
            if not L.band_is_aa(chain, k, lmask, lm):
                continue
            for idx, st in enumerate(chain):
                (a0, a1), (b0, b1), side, row = st["a"], st["b"], st["side"], st["row"]
                la, lb = a1 - a0 + 1, b1 - b0 + 1
                if la == 1 and lb == 1:
                    drop = [b0] if idx % 2 else []
                else:
                    keep = max(1, la // 2)
                    if lb <= keep:
                        continue
                    drop = list(range(b0 + keep, b1 + 1)) if side == 1 else list(range(b0, b1 - keep + 1))
                far = b1 + 1 if side == 1 else b0 - 1
                fill = int(k[row, far]) if 0 <= far < n_cols else -1
                for c in drop:
                    x, y = (row, c) if transposed else (c, row)
                    out[y, x] = 0 if fill == -1 else (*L._rgb(fill), fill & 255)
                    changed.append((x, y))
    return out, changed


def dither_mask(img: np.ndarray) -> np.ndarray:
    """Pixels that belong to checkerboard dithering (plus a 1-px margin).

    Pixel-level fixes leave these alone: a dither pattern is made of
    "single" pixels and short diagonal strips on purpose.
    """
    keys = L._keys(img)
    a, b = keys[:-1, :-1], keys[:-1, 1:]
    c, d = keys[1:, :-1], keys[1:, 1:]
    checker = (a == d) & (b == c) & (a != b) & (a != -1) & (b != -1)
    mask = np.zeros(keys.shape, dtype=bool)
    for dy in (0, 1):
        for dx in (0, 1):
            mask[dy:dy + checker.shape[0], dx:dx + checker.shape[1]] |= checker
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


# ------------------------------------------------------------ runner

STEPS = ("alpha", "grid", "palette", "orphans", "speckles", "outer_aa", "outline", "thickness", "corners",
         "jaggies", "banding", "aa_band")


def cleanup(img: np.ndarray, steps=STEPS, max_colors: int = 0, keep_details: bool = True,
            line_colors: list[str] | None = None, style: dict | None = None) -> tuple[np.ndarray, dict]:
    """Run the clean-up pipeline. Returns (clean image, log).

    max_colors: 0 = automatic (reduce to the size budget only for blurred or
    resampled input with far too many colours), -1 = never reduce (near-duplicates still merge),
    N = reduce to N colours. keep_details=True (default) only fixes what the
    linter rates as a warning: lines on the outline, low-contrast specks;
    inner line art, textures and high-contrast single pixels are reported
    but kept. keep_details=False fixes those too. `style` (from analyze_style) keeps deliberate traits:
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
        elif max_colors == 0 and filtered and n > 2 * budget:
            # Only blurred/resampled art gets cut to the budget; a clean
            # sprite's palette is the artist's choice.
            target = budget
        if target is not None and n > 64:
            cur = _prequantize(cur, max(target * 2, 32))
        # Resampled or filtered art carries colour noise: merge wider.
        cur, merges = reduce_colors(cur, target, min_delta=8.0 if filtered else 3.0)
        keys = L._keys(cur)
        note("palette", colors_before=n, merge_distance=8.0 if filtered else 3.0, colors_after=len(np.unique(keys[keys != -1])),
             target=target, merges=merges[:20], merge_count=len(merges))
    def protect(before, after, px):
        """Undo changes inside dithered areas; returns the kept change list."""
        mask = dither_mask(before)
        if not mask.any():
            return after, px
        after = after.copy()
        after[mask] = before[mask]
        return after, [(x, y) for x, y in px if not mask[y, x]]

    if "orphans" in steps:
        cur, px = remove_floating(cur)
        note("orphans", removed=len(px), pixels=px[:40])
    if "speckles" in steps:
        prev = cur
        cur, px, kept = despeckle(cur, 15.0 if keep_details else 100.0)
        cur, px = protect(prev, cur, px)
        note("speckles", recoloured=len(px), pixels=px[:40], kept_for_review=kept[:40])
    # Outer AA goes before closing the outline, or a light pixel touching two
    # outline pixels would be turned into outline and thicken the line.
    if "outer_aa" in steps:
        if style and style.get("outline", {}).get("style") == "selective":
            note("outer_aa", skipped="selective outline lightens edges on purpose")
        else:
            cur, px = remove_outer_aa(cur)
            note("outer_aa", removed=len(px), pixels=px[:40])
    if "outline" in steps:
        cur, px = close_outline(cur)
        note("outline", closed=len(px), pixels=px[:40])
    # Line weight and AA bands are often a deliberate style in good art; they
    # are reported always, but only fixed in the thorough mode.
    if "thickness" in steps and not keep_details:
        prev = cur
        cur, px = thin_lines(cur)
        cur, px = protect(prev, cur, px)
        note("thickness", thinned=len(px), pixels=px[:40])
    if "corners" in steps:
        lines = [(k, m) for k, m, sure in L.line_masks(cur, line_colors) if sure or not keep_details]
        prev = cur
        cur, px = fix_l_corners(cur, lines)
        cur, px = protect(prev, cur, px)
        note("corners", fixed=len(px), pixels=px[:40])
    if "jaggies" in steps:
        lines = [(k, m) for k, m, sure in L.line_masks(cur, line_colors) if sure or not keep_details]
        prev = cur
        cur, px = fix_jaggies(cur, lines)
        cur, px = protect(prev, cur, px)
        note("jaggies", changed=len(px), pixels=px[:40])
    if "banding" in steps:
        prev = cur
        cur, px = break_banding(cur)
        cur, px = protect(prev, cur, px)
        note("banding", changed=len(px), pixels=px[:40])
    if "aa_band" in steps and not keep_details:
        prev = cur
        cur, px = trim_aa_bands(cur)
        cur, px = protect(prev, cur, px)
        note("aa_band", trimmed=len(px), pixels=px[:40])
    return cur, log


def changed_pixels(before: np.ndarray, after: np.ndarray) -> list[tuple[int, int]]:
    """Pixels whose visible value differs (transparent pixels are all equal)."""
    a = np.where(before[..., 3:4] > 0, before, 0)
    b = np.where(after[..., 3:4] > 0, after, 0)
    return [(int(x), int(y)) for y, x in zip(*np.nonzero((a != b).any(axis=-1)))]
