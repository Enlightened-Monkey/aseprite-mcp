"""Pixel-art linter: detects common craft mistakes in an RGBA image.

Every check takes an RGBA uint8 array (H, W, 4) and returns a list of
issues: {"check", "severity", "message", "pixels": [[x, y], ...]}.
Severities: "warning" (very likely a mistake) and "info" (worth a look;
may be intentional, e.g. a single highlight pixel).

Pixels with alpha 0 are transparent; everything else counts as opaque.
"""
from collections import Counter

import numpy as np

from .color import delta_e, hue_delta, lightness, palette_budget, rgb_to_hex, rgb_to_hsv

_N8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
_N4 = [(0, -1), (-1, 0), (1, 0), (0, 1)]


def _keys(img: np.ndarray) -> np.ndarray:
    """Pack RGBA into one int per pixel; transparent pixels become -1."""
    a = img.astype(np.int64)
    k = (a[..., 0] << 24) | (a[..., 1] << 16) | (a[..., 2] << 8) | a[..., 3]
    return np.where(img[..., 3] > 0, k, -1)


def _rgb(key: int) -> tuple[int, int, int]:
    return (key >> 24) & 255, (key >> 16) & 255, (key >> 8) & 255


def _issue(check, severity, message, pixels):
    return {"check": check, "severity": severity, "message": message,
            "pixels": [[int(x), int(y)] for x, y in pixels]}


def _opaque_bbox_height(img: np.ndarray) -> int:
    rows = np.where((img[..., 3] > 0).any(axis=1))[0]
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


# ---------------------------------------------------------------- orphans

def check_orphans(img: np.ndarray) -> list[dict]:
    """Single pixels that share their colour with none of their 8 neighbours.

    A lone opaque pixel floating in transparency is a warning. Inside the
    sprite, singles whose lightness lies between two neighbours are treated
    as deliberate anti-aliasing and skipped; the rest are reported as info
    because a single pixel can be an intentional eye or highlight.
    """
    keys = _keys(img)
    h, w = keys.shape
    pad = np.pad(keys, 1, constant_values=-1)
    same = np.zeros((h, w), dtype=np.int8)
    transparent_nb = np.zeros((h, w), dtype=np.int8)
    for dx, dy in _N8:
        nb = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        same += (nb == keys)
        transparent_nb += (nb == -1)
    singles = (keys != -1) & (same == 0)
    floating, noise = [], []
    light = lightness(img[..., :3])
    lpad = np.pad(light, 1, constant_values=np.nan)
    for y, x in zip(*np.nonzero(singles)):
        if transparent_nb[y, x] == 8:
            floating.append((x, y))
            continue
        nbs = [lpad[1 + y + dy, 1 + x + dx] for dx, dy in _N8 if pad[1 + y + dy, 1 + x + dx] != -1]
        lo, hi = min(nbs), max(nbs)
        if lo + 0.02 < light[y, x] < hi - 0.02:
            continue  # sits between neighbour values: reads as AA
        noise.append((x, y))
    issues = []
    if floating:
        issues.append(_issue("orphans", "warning",
            f"{len(floating)} isolated pixel(s) floating in transparency; delete them or connect them to the sprite.",
            floating))
    if noise:
        issues.append(_issue("orphans", "info",
            f"{len(noise)} single pixel(s) differ from all neighbours and are not between neighbour values. "
            "Keep only deliberate ones (eye, shine); stray dots read as noise.", noise))
    return issues


# ---------------------------------------------------------- line colours

def _thin_ratio(mask: np.ndarray) -> float:
    """Share of a mask's pixels that are not part of any 2x2 block."""
    block = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    fat = np.zeros_like(mask)
    for dy in (0, 1):
        for dx in (0, 1):
            fat[dy:dy + block.shape[0], dx:dx + block.shape[1]] |= block
    n = mask.sum()
    return float((mask & ~fat).sum() / n) if n else 0.0


def line_colors(img: np.ndarray) -> list[tuple[int, bool]]:
    """Colours that are used as lines (outline / line art).

    The colour dominating the silhouette edge (>= 25% of edge pixels), plus
    any dark colour (in the darkest third) whose pixels are almost all 1 px
    thin. Line rules (L-corners, staircase rhythm) apply to these colours
    only; shading clusters have their own logic. Returns (colour key,
    is_outline) pairs; findings on non-outline colours are less certain.
    """
    keys = _keys(img)
    h, w = keys.shape
    pad = np.pad(keys, 1, constant_values=-1)
    edge = np.zeros((h, w), dtype=bool)
    for dx, dy in _N4:
        edge |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] == -1
    edge &= keys != -1
    out = set()
    edge_counts = Counter(int(k) for k in keys[edge])
    total = sum(edge_counts.values())
    outline = {k for k, n in edge_counts.items() if n >= 0.25 * total}
    out |= outline
    opaque = keys != -1
    if opaque.any():
        light = lightness(img[..., :3])
        dark_cut = float(np.percentile(light[opaque], 33))
        for k in np.unique(keys[opaque]):
            k = int(k)
            mask = keys == k
            if mask.sum() >= 12 and float(lightness(_rgb(k))) <= dark_cut and _thin_ratio(mask) >= 0.85:
                out.add(k)
    return [(k, k in outline) for k in sorted(out)]


def _resolve_line_colors(img, colors):
    """(key, certain) pairs: auto-detected, or exactly the given hex colours."""
    if colors is None:
        return line_colors(img)
    from .color import hex_to_rgb
    wanted = {hex_to_rgb(c) for c in colors}
    return [(int(k), True) for k in np.unique(_keys(img)) if k != -1 and _rgb(int(k)) in wanted]


def _split_by_certainty(check, found, warn_msg):
    sure = [p for p, c in found if c]
    unsure = [p for p, c in found if not c]
    issues = []
    if sure:
        issues.append(_issue(check, "warning", warn_msg(len(sure)), sure))
    if unsure:
        issues.append(_issue(check, "info", warn_msg(len(unsure)) +
                             " (On a dark inner line colour; ignore if it is shading, not a line.)", unsure))
    return issues


# ---------------------------------------------------------- L-corners

def _run_len(mask: np.ndarray, x: int, y: int, dx: int, dy: int) -> int:
    h, w = mask.shape
    n = 0
    while 0 <= x < w and 0 <= y < h and mask[y, x]:
        n += 1
        x += dx
        y += dy
    return n


def find_l_corners(mask: np.ndarray) -> list[tuple[int, int]]:
    """Redundant corner pixels of 1-px strokes ("pixel-perfect" violations).

    A pixel P is redundant when it has a horizontal neighbour H and a
    vertical neighbour V in the stroke, the cell diagonal between H and V is
    empty (so this is a thin line, not a filled area), every other stroke
    neighbour of P stays connected to H or V without P, and the corner is not
    a deliberate right angle (both arms at least 3 px long). Detection is
    greedy, so of two overlapping candidates only one is reported.
    """
    m = mask.copy()
    h, w = m.shape
    found = []

    def at(x, y):
        return 0 <= x < w and 0 <= y < h and m[y, x]

    for y in range(h):
        for x in range(w):
            if not m[y, x]:
                continue
            if (at(x - 1, y) and at(x + 1, y)) or (at(x, y - 1) and at(x, y + 1)):
                continue  # middle of a straight run or a T-junction
            for hx in (-1, 1):
                for vy in (-1, 1):
                    if not (at(x + hx, y) and at(x, y + vy)) or at(x + hx, y + vy):
                        continue
                    others = [(x + dx, y + dy) for dx, dy in _N8
                              if (dx, dy) not in ((hx, 0), (0, vy)) and at(x + dx, y + dy)]
                    hx_, vy_ = (x + hx, y), (x, y + vy)
                    if any(max(abs(ox - hx_[0]), abs(oy - hx_[1])) > 1 and
                           max(abs(ox - vy_[0]), abs(oy - vy_[1])) > 1 for ox, oy in others):
                        continue
                    if _run_len(m, x, y, hx, 0) >= 3 and _run_len(m, x, y, 0, vy) >= 3:
                        continue
                    found.append((x, y))
                    m[y, x] = False
                    break
                if not m[y, x]:
                    break
    return found


def check_l_corners(img: np.ndarray, colors: list[str] | None = None) -> list[dict]:
    keys = _keys(img)
    found = []
    for key, certain in _resolve_line_colors(img, colors):
        found += [(p, certain) for p in find_l_corners(keys == key)]
    return _split_by_certainty("l_corners", found, lambda n:
        f"{n} doubled corner pixel(s) thicken 1-px lines into L-shapes; remove them so the "
        "line steps diagonally (pixel-perfect).")


# ------------------------------------------------------------ jaggies

def _thin_chains(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Ordered pixel chains of the 1-px-wide parts of a mask.

    Pixels inside any 2x2 block are fills, not lines, and are dropped; so
    are redundant L-corners. Junction pixels (3+ neighbours) split chains.
    """
    m = mask.copy()
    block = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
    fat = np.zeros_like(m)
    for dy in (0, 1):
        for dx in (0, 1):
            fat[dy:dy + block.shape[0], dx:dx + block.shape[1]] |= block
    m &= ~fat
    for x, y in find_l_corners(m):
        m[y, x] = False
    h, w = m.shape

    def nbs(x, y):
        return [(x + dx, y + dy) for dx, dy in _N8
                if 0 <= x + dx < w and 0 <= y + dy < h and m[y + dy, x + dx]]

    pts = list(zip(*np.nonzero(m)))
    degree = {(x, y): len(nbs(x, y)) for y, x in pts}
    ok = {p for p, d in degree.items() if d <= 2}
    seen, chains = set(), []
    starts = [p for p in ok if degree[p] <= 1 or any(q not in ok for q in nbs(*p))]
    starts += sorted(ok)  # loops have no endpoint; any pixel will do
    for s in starts:
        if s in seen or s not in ok:
            continue
        chain, cur = [s], s
        seen.add(s)
        while True:
            nxt = [q for q in nbs(*cur) if q in ok and q not in seen]
            if not nxt:
                break
            # Prefer orthogonal steps so diagonal shortcuts do not skip pixels.
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            cur = nxt[0]
            seen.add(cur)
            chain.append(cur)
        if len(chain) >= 4:
            chains.append(chain)
    return chains


def chain_runs(chain: list[tuple[int, int]]) -> list[dict]:
    """Split an ordered chain into runs (steps) between diagonal moves.

    Each run is {"axis": "h"|"v"|None, "len": n, "pixels": [...]}; runs of
    one pixel have no axis. A turn without a diagonal (a deliberate corner)
    starts a new segment, marked by "break": True on the next run.
    """
    runs = [{"axis": None, "len": 1, "pixels": [chain[0]], "break": False}]
    for a, b in zip(chain, chain[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        cur = runs[-1]
        if dx and dy:
            runs.append({"axis": None, "len": 1, "pixels": [b], "break": False})
            continue
        axis = "h" if dy == 0 else "v"
        if cur["axis"] in (None, axis):
            cur["axis"] = axis
            cur["len"] += 1
            cur["pixels"].append(b)
        else:
            runs.append({"axis": axis, "len": 1, "pixels": [b], "break": True})
    return runs


def _slope_value(run) -> float:
    """Order runs by steepness: long horizontal > ... > 45 deg > ... > long vertical."""
    if run["axis"] == "v":
        return -float(run["len"] - 1)
    return float(run["len"] - 1)


def jaggy_runs(runs: list[dict]) -> list[int]:
    """Indices of runs that break the staircase rhythm.

    Interior runs only (the first and last are truncated by the chain ends).
    Plateaus are collapsed, then a run is a jaggy when it is a local extreme
    that sits right next to another local extreme (a zig-zag such as 1-2-1
    or 3-2-3 inside a trend), or an outlier between two equal neighbours
    (2-2-1-2-2). Steady alternation of two neighbouring lengths over 4+ runs
    (2-3-2-3) is how non-integer slopes look and is accepted.
    """
    segs, cur = [], []
    for i, r in enumerate(runs):
        if r["break"] and cur:
            segs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        segs.append(cur)

    bad = set()
    for seg in segs:
        idx = seg[1:-1]
        if len(idx) < 3:
            continue
        vals = [_slope_value(runs[i]) for i in idx]
        # Collapse plateaus to (value, [indices]).
        groups = []
        for i, v in zip(idx, vals):
            if groups and groups[-1][0] == v:
                groups[-1][1].append(i)
            else:
                groups.append((v, [i]))
        ext = []
        for g in range(1, len(groups) - 1):
            a, b, c = groups[g - 1][0], groups[g][0], groups[g + 1][0]
            if (b > a and b > c) or (b < a and b < c):
                ext.append(g)
        # Windows of consecutive extremes = zig-zags.
        g = 0
        while g < len(ext):
            j = g
            while j + 1 < len(ext) and ext[j + 1] == ext[j] + 1:
                j += 1
            window = list(range(ext[g] - 1, ext[j] + 2))
            wvals = {groups[k][0] for k in window}
            periodic = len(wvals) == 2 and abs(max(wvals) - min(wvals)) == 1 and len(window) >= 4
            if j > g and not periodic:
                for k in range(ext[g], ext[j] + 1):
                    bad.update(groups[k][1])
            elif j == g:
                k = ext[g]
                single = len(groups[k][1]) == 1
                same_sides = groups[k - 1][0] == groups[k + 1][0]
                plateau_sides = len(groups[k - 1][1]) >= 2 or len(groups[k + 1][1]) >= 2
                if single and same_sides and plateau_sides:
                    bad.update(groups[k][1])
            g = j + 1
    return sorted(bad)


def check_jaggies(img: np.ndarray, colors: list[str] | None = None) -> list[dict]:
    keys = _keys(img)
    found = []
    for key, certain in _resolve_line_colors(img, colors):
        for chain in _thin_chains(keys == key):
            runs = chain_runs(chain)
            for i in jaggy_runs(runs):
                found += [(p, certain) for p in runs[i]["pixels"]]
    return _split_by_certainty("jaggies", found, lambda n:
        f"{n} pixel(s) in line steps that break the staircase rhythm. Straight lines need equal "
        "steps; curves need step lengths that shrink then grow smoothly (e.g. 4-3-2-1-1-2-3-4, not 3-1-3 "
        "or 1-2-1).")


# ------------------------------------------------------------ banding

def _row_runs(keys_row: np.ndarray) -> list[tuple[int, int, int]]:
    runs, start = [], 0
    for x in range(1, len(keys_row) + 1):
        if x == len(keys_row) or keys_row[x] != keys_row[start]:
            if keys_row[start] != -1:
                runs.append((int(keys_row[start]), start, x - 1))
            start = x
    return runs


def _banding_pass(keys: np.ndarray, min_rows: int) -> list[list[tuple[int, int]]]:
    """Find runs of colour B that follow a staircase of colour A step for step.

    Works row by row: in each row an A-run is directly followed (or preceded)
    by a B-run; across consecutive rows the A-runs form a staircase and the
    B-runs keep the same length. Returns pixel lists in (row, col) order.
    """
    rows = [_row_runs(keys[y]) for y in range(keys.shape[0])]
    pairs = []  # per row: list of (a, b, a_run, b_run, side)
    for runs in rows:
        p = []
        for i, (ca, a0, a1) in enumerate(runs):
            for j, side in ((i + 1, 1), (i - 1, -1)):
                if 0 <= j < len(runs):
                    cb, b0, b1 = runs[j]
                    if cb == ca or not (b0 == a1 + 1 if side == 1 else b1 == a0 - 1):
                        continue
                    # A-B-A or B-A-B is a stripe between parallel lines, not a band.
                    k, m = j + side, i - side
                    if 0 <= k < len(runs) and runs[k][0] == ca and (
                            runs[k][1] == b1 + 1 if side == 1 else runs[k][2] == b0 - 1):
                        continue
                    if 0 <= m < len(runs) and runs[m][0] == cb and (
                            runs[m][2] == a0 - 1 if side == 1 else runs[m][1] == a1 + 1):
                        continue
                    p.append((ca, cb, (a0, a1), (b0, b1), side))
        pairs.append(p)

    bands, used = [], set()
    for y in range(len(pairs)):
        for start in pairs[y]:
            if (y, start) in used:
                continue
            chain = [(y, start)]
            cy, cur = y, start
            while cy + 1 < len(pairs):
                ca, cb, (a0, a1), (b0, b1), side = cur
                nxt = None
                for cand in pairs[cy + 1]:
                    na, nb, (c0, c1), (d0, d1), nside = cand
                    stair = c0 == a1 + 1 or c1 == a0 - 1
                    if (na, nb, nside) == (ca, cb, side) and stair and d1 - d0 == b1 - b0:
                        nxt = cand
                        break
                if not nxt:
                    break
                cy, cur = cy + 1, nxt
                chain.append((cy, cur))
            thin = all(b1 == b0 for _, (_, _, _, (b0, b1), _) in chain)
            if len(chain) >= (min_rows + 1 if thin else min_rows):
                used.update(chain)
                px = []
                for ry, (_, _, _, (b0, b1), _) in chain:
                    px += [(ry, x) for x in range(b0, b1 + 1)]
                bands.append(px)
    return bands


def check_banding(img: np.ndarray, min_rows: int = 3) -> list[dict]:
    keys = _keys(img)
    pixels = []
    for band in _banding_pass(keys, min_rows):
        pixels += [(x, y) for y, x in band]
    for band in _banding_pass(keys.T, min_rows):
        pixels += [(x, y) for x, y in band]  # transposed: (row=x, col=y)
    if not pixels:
        return []
    return [_issue("banding", "warning",
        f"{len(pixels)} pixel(s) form bands that copy a neighbouring staircase step for step. Break the "
        "alignment: shorten or drop the band on some steps, or use a darker shade so it merges with the line.",
        pixels)]


# ------------------------------------------------------------- pillow

def check_pillow_shading(img: np.ndarray) -> list[dict]:
    """Shading that brightens toward the centre equally from every side.

    Rings are measured by distance from the silhouette edge; a uniform
    outline ring (one colour) is skipped. Pillow shading shows lightness
    rising ring after ring while the first shaded ring is about equally
    dark on all four sides (no light direction).
    """
    opaque = img[..., 3] > 0
    if opaque.sum() < 64:
        return []
    keys = _keys(img)
    light = lightness(img[..., :3])
    depth = np.zeros(opaque.shape, dtype=np.int32)
    cur = opaque.copy()
    d = 0
    while cur.any() and d < 8:
        d += 1
        depth[cur] = d
        pad = np.pad(cur, 1)
        cur = cur & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    first = 1
    ring1 = Counter(int(k) for k in keys[depth == 1])
    if ring1 and ring1.most_common(1)[0][1] >= 0.7 * sum(ring1.values()):
        first = 2  # skip a solid outline
    rings = [k for k in range(first, first + 3) if (depth == k).sum() >= 8]
    if len(rings) < 3:
        return []
    means = [light[depth == k].mean() for k in rings]
    if not (means[0] + 0.02 < means[1] and means[1] + 0.02 < means[2]):
        return []
    ring = depth == rings[0]
    ys, xs = np.nonzero(ring)
    cy, cx = ys.mean(), xs.mean()
    row = np.arange(opaque.shape[0])[:, None]
    col = np.arange(opaque.shape[1])[None, :]
    sides = [light[ring & (row < cy)].mean(), light[ring & (row >= cy)].mean(),
             light[ring & (col < cx)].mean(), light[ring & (col >= cx)].mean()]
    if max(sides) - min(sides) > 0.04:
        return []
    return [_issue("pillow_shading", "warning",
        "Shading gets lighter toward the centre equally from every edge (pillow shading). Pick one light "
        "direction: lit side lighter, opposite side darker, and let shadows follow the form.",
        [(x, y) for y, x in zip(*np.nonzero(ring))])]


# ------------------------------------------------------------ colours

def check_colors(img: np.ndarray) -> list[dict]:
    keys = _keys(img)
    counts = Counter(int(k) for k in keys[keys != -1])
    issues = []
    height = _opaque_bbox_height(img)
    budget = palette_budget(height)
    n = len(counts)
    if n > budget["max_colors"]:
        issues.append(_issue("color_count", "warning" if n > budget["max_colors"] * 1.5 else "info",
            f"{n} colours for a sprite {height} px tall; around {budget['max_colors']} is a typical ceiling "
            f"({budget['steps_per_ramp']} shades per ramp). Merge near-identical shades.", []))
    cols = list(counts)
    dupes = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if delta_e(_rgb(cols[i]), _rgb(cols[j])) < 3.0:
                dupes.append((cols[i], cols[j]))
    if dupes:
        desc = ", ".join(f"{rgb_to_hex(_rgb(a))}~{rgb_to_hex(_rgb(b))}" for a, b in dupes[:8])
        minor = {a if counts[a] < counts[b] else b for a, b in dupes}
        px = [(x, y) for y, x in zip(*np.nonzero(np.isin(keys, list(minor))))]
        issues.append(_issue("near_duplicates", "warning",
            f"{len(dupes)} pair(s) of colours are nearly indistinguishable ({desc}). Merge each pair.", px))
    black = [k for k in cols if _rgb(k) == (0, 0, 0)]
    if black:
        share = counts[black[0]] / max(1, sum(counts.values()))
        if share > 0.02:
            issues.append(_issue("pure_black", "info",
                f"Pure black covers {share:.0%} of the sprite. A tinted near-black (dark plum, brown or teal, "
                "value 8-25%) usually looks richer; see suggest_outline_color.", []))
    return issues


# ------------------------------------------------------------- dither

def check_dither(img: np.ndarray) -> list[dict]:
    keys = _keys(img)
    a, b = keys[:-1, :-1], keys[:-1, 1:]
    c, d = keys[1:, :-1], keys[1:, 1:]
    checker = (a == d) & (b == c) & (a != b) & (a != -1) & (b != -1)
    if not checker.any():
        return []
    mark = np.zeros(keys.shape, dtype=bool)
    for dy in (0, 1):
        for dx in (0, 1):
            mark[dy:dy + checker.shape[0], dx:dx + checker.shape[1]] |= checker
    height = _opaque_bbox_height(img)
    share = mark.sum() / max(1, (keys != -1).sum())
    if height >= 48 or share < 0.05:
        return []
    return [_issue("dither_small_sprite", "warning",
        f"{share:.0%} of a {height}px-tall sprite is checker-dithered. Dithering makes small sprites noisy; "
        "use flat shading here and keep dither for large static areas.",
        [(x, y) for y, x in zip(*np.nonzero(mark))])]


# --------------------------------------------------- AA outside outline

def check_outer_aa(img: np.ndarray) -> list[dict]:
    """Lighter pixels placed outside a dark outline, against transparency."""
    keys = _keys(img)
    h, w = keys.shape
    pad = np.pad(keys, 1, constant_values=-1)
    edge = np.zeros((h, w), dtype=bool)
    for dx, dy in _N4:
        edge |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] == -1
    edge &= keys != -1
    edge_counts = Counter(int(k) for k in keys[edge])
    if not edge_counts:
        return []
    outline, count = edge_counts.most_common(1)[0]
    if count < 0.6 * edge.sum():
        return []
    out_l = float(lightness(_rgb(outline)))
    pixels = []
    for y, x in zip(*np.nonzero(edge & (keys != outline))):
        k = int(keys[y, x])
        if float(lightness(_rgb(k))) <= out_l + 0.05:
            continue
        nb = [pad[1 + y + dy, 1 + x + dx] for dx, dy in _N4]
        if outline in nb and -1 in nb:
            # Outside the outline if the outline neighbour is opposite the transparent one.
            for (dx, dy) in _N4:
                if pad[1 + y + dy, 1 + x + dx] == outline and pad[1 + y - dy, 1 + x - dx] == -1:
                    pixels.append((x, y))
                    break
    if not pixels:
        return []
    return [_issue("aa_outside_outline", "warning",
        f"{len(pixels)} lighter pixel(s) sit outside the outline against transparency. Anti-aliasing "
        "belongs inside the shape; outside it looks spotty on unknown backgrounds.", pixels)]


# -------------------------------------------------------------- runner

CHECKS = {
    "orphans": check_orphans,
    "l_corners": check_l_corners,
    "jaggies": check_jaggies,
    "banding": check_banding,
    "pillow_shading": check_pillow_shading,
    "colors": check_colors,
    "dither": check_dither,
    "outer_aa": check_outer_aa,
}


def lint(img: np.ndarray, checks: list[str] | None = None,
         line_colors: list[str] | None = None) -> list[dict]:
    """Run the named checks (all by default). line_colors (hex) overrides
    the automatic detection of which colours are lines."""
    names = checks or list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        raise ValueError(f"Unknown check(s): {', '.join(unknown)}. Available: {', '.join(CHECKS)}")
    issues = []
    for n in names:
        if n in ("l_corners", "jaggies"):
            issues += CHECKS[n](img, line_colors)
        else:
            issues += CHECKS[n](img)
    issues.sort(key=lambda i: i["severity"] != "warning")
    return issues


# ------------------------------------------------- palette analysis

def shading_pairs(img: np.ndarray, min_contacts: int = 3) -> list[dict]:
    """Colour pairs that touch in the image and look like shading steps.

    Both colours saturated (S >= 15), hues within 60 deg, clearly different
    lightness. For each pair reports the hue rotation going from the darker
    to the lighter colour and whether it heads toward yellow (expected).
    """
    keys = _keys(img)
    contacts = Counter()
    for a, b in ((keys[:, :-1], keys[:, 1:]), (keys[:-1, :], keys[1:, :])):
        m = (a != b) & (a != -1) & (b != -1)
        for p, q in zip(a[m].tolist(), b[m].tolist()):
            contacts[(min(p, q), max(p, q))] += 1
    out = []
    for (p, q), n in contacts.items():
        if n < min_contacts:
            continue
        rp, rq = _rgb(p), _rgb(q)
        lp, lq = float(lightness(rp)), float(lightness(rq))
        if abs(lp - lq) < 0.06:
            continue
        dark, light = (rp, rq) if lp < lq else (rq, rp)
        hd, sd, _ = rgb_to_hsv(dark)
        hl, sl, _ = rgb_to_hsv(light)
        if sd < 15 or sl < 15 or abs(hue_delta(hd, hl)) > 60:
            continue
        shift = hue_delta(hd, hl)
        toward_light = hue_delta(hd, 60.0)
        if abs(shift) < 4:
            verdict = "flat"
        elif (shift > 0) == (toward_light > 0) or abs(toward_light) < 4:
            verdict = "ok"
        else:
            verdict = "reversed"
        out.append({"dark": rgb_to_hex(dark), "light": rgb_to_hex(light), "contacts": n,
                    "hue_shift": round(shift, 1), "verdict": verdict})
    out.sort(key=lambda r: -r["contacts"])
    return out


# -------------------------------------------------------- pixel scale

def detect_scale(img: np.ndarray) -> int:
    """Integer factor the art was upscaled by (1 for native pixel art).

    Every same-colour run, horizontal and vertical, of an image upscaled k
    times starts and ends on multiples of k; the largest such k wins. Simple
    images are ambiguous (a 16x16 square at (8,8) is "8x"), so a factor only
    counts when the 1x result is at least 16 px each way with 3+ colours.
    """
    keys = _keys(img)
    h, w = keys.shape
    if len(np.unique(keys)) < 3:
        return 1
    best = 1
    for k in range(2, min(h, w) // 16 + 1):
        if h % k or w % k:
            continue
        cols = keys[:, ::k]
        rows = keys[::k, :]
        if (np.repeat(cols, k, axis=1)[:, :w] == keys).all() and \
           (np.repeat(rows, k, axis=0)[:h, :] == keys).all():
            best = k
    return best


def downscale(img: np.ndarray, k: int) -> np.ndarray:
    return img[::k, ::k].copy() if k > 1 else img
