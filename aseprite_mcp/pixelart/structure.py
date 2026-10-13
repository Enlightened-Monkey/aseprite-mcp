"""Structure of a sprite: which pixels are line art, fills, shading, AA...

Every opaque pixel gets a role:

    outline     - line art on the silhouette edge (touches transparency)
    inner_line  - line art inside the sprite (separations, folds, details)
    aa          - in-between tones next to lines (anti-aliasing)
    base        - the main colour of a part
    shadow      - darker tones of a part
    highlight   - lighter tones of a part
    dither      - checkerboard pixels
    detail      - tiny isolated marks (eyes, shines, specks)

Line art is found by shape first: a 1-2 px strip that is clearly darker
than what lies on both sides of it (or transparency on one side), forming
an elongated stroke. On the silhouette edge that is enough, so black,
coloured and shaded outlines are all found. Inside the sprite the strip's
colour must also be a line colour (used mostly as thin strips), so a dark
shading band is not mistaken for line art.
"""
from collections import Counter

import numpy as np

from . import lint as L
from .color import hue_delta, lightness, rgb_to_hsv

ROLES = ("transparent", "outline", "inner_line", "aa", "base", "shadow", "highlight", "dither", "detail")
ROLE_COLORS = {
    "outline": (230, 30, 30), "inner_line": (255, 140, 0), "aa": (255, 230, 60), "base": (120, 170, 240),
    "shadow": (50, 80, 170), "highlight": (190, 230, 255), "dither": (180, 80, 220), "detail": (40, 200, 90),
}
_DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]
_N4 = [(0, -1), (-1, 0), (1, 0), (0, 1)]
_N8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def _shifted(a: np.ndarray, dx: int, dy: int, fill) -> np.ndarray:
    """a moved so that out[y, x] == a[y + dy, x + dx] (fill outside)."""
    h, w = a.shape
    out = np.full_like(a, fill)
    ys = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, -dx), min(w, w - dx))
    ys2 = slice(max(0, dy), min(h, h + dy))
    xs2 = slice(max(0, dx), min(w, w + dx))
    out[ys, xs] = a[ys2, xs2]
    return out


def _components(mask: np.ndarray, conn8: bool = True) -> list[list[tuple[int, int]]]:
    nbrs = _N8 if conn8 else _N4
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    comps = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, comp = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dx, dy in nbrs:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(comp)
    return comps


def line_candidates(img: np.ndarray, contrast: float = 0.1, edge_contrast: float = 0.05) -> np.ndarray:
    """Pixels of 1-2 px strips darker than both sides by `contrast` (OKLab L).

    Transparency counts as "lighter", so silhouette outlines (with the lower
    edge_contrast) and strokes on an otherwise empty line-art layer are
    found. Plain edges of a dark fill are not lines: the fill continues on
    the inner side.
    """
    opaque = img[..., 3] > 0
    lum = lightness(img[..., :3]).astype(np.float64)
    LIGHTER = 9.0  # transparency: lighter than anything
    lum_t = np.where(opaque, lum, LIGHTER)
    cand = np.zeros(opaque.shape, dtype=bool)
    # Outside the canvas is unknown, not transparent: repeat the edge, so a
    # full scene does not get a fake outline along the canvas border.
    edge_pad = np.pad(lum_t, 2, mode="edge")
    h, w = lum.shape

    def side(dx, dy):
        """Lightness of the neighbour at (+dx, +dy)."""
        return edge_pad[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]

    for dx, dy in _DIRS:
        a = side(dx, dy)       # one side
        b = side(-dx, -dy)     # other side
        # Against transparency a lower contrast suffices: an outline next to
        # the shadow side of a part is only slightly darker than it.
        need = np.where((a >= LIGHTER) | (b >= LIGHTER), edge_contrast, contrast)
        w1 = (a > lum + need) & (b > lum + need)
        # Two-pixel strips: P and its neighbour Q (in +d) are similar and
        # both darker than what lies beyond them.
        q = side(dx, dy)
        beyond_q = side(2 * dx, 2 * dy)
        pair = (np.abs(q - lum) < 0.06) & (q < LIGHTER)
        need2 = np.where((b >= LIGHTER) | (beyond_q >= LIGHTER), edge_contrast, contrast)
        w2 = pair & (b > lum + need2) & (beyond_q > np.maximum(lum, q) + need2)
        w2_partner = _shifted(w2, -dx, -dy, False)  # mark Q as well
        cand |= w1 | w2 | w2_partner
    return cand & opaque


def analyze(img: np.ndarray, contrast: float = 0.1) -> dict:
    """Classify every opaque pixel by role and summarise the structure.

    Returns {"roles": int array (index into ROLES), "summary": {...}}.
    """
    h, w = img.shape[:2]
    keys = L._keys(img)
    opaque = keys != -1
    lum = lightness(img[..., :3])
    roles = np.zeros((h, w), dtype=np.int8)

    # --- line art: candidate strips that form elongated strokes
    cand = line_candidates(img, contrast)
    pad = np.pad(keys, 1, mode="edge")  # the canvas border is not air
    near_air = np.zeros((h, w), dtype=bool)
    for dx, dy in _N4:
        near_air |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] == -1
    # Inside the sprite a strip only counts as line art when its colour is a
    # line colour: one used mostly as thin strips across the sprite. A
    # shading colour that also fills areas makes thin bands that are shading.
    colour_total = Counter(int(k) for k in keys[opaque])
    colour_strips = Counter(int(k) for k in keys[cand])
    line_keys = [k for k, n in colour_strips.items() if n >= 0.6 * colour_total[k]]
    cand &= near_air | np.isin(keys, line_keys)
    lines = np.zeros_like(cand)
    for comp in _components(cand):
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        if len(comp) >= 4 and max(max(ys) - min(ys), max(xs) - min(xs)) >= 3:
            for y, x in comp:
                lines[y, x] = True
    roles[lines & near_air] = ROLES.index("outline")
    roles[lines & ~near_air] = ROLES.index("inner_line")

    # --- dither (checkerboard) outside the lines
    a, b = keys[:-1, :-1], keys[:-1, 1:]
    c, d = keys[1:, :-1], keys[1:, 1:]
    chk = (a == d) & (b == c) & (a != b) & (a != -1) & (b != -1)
    dither = np.zeros((h, w), dtype=bool)
    for oy in (0, 1):
        for ox in (0, 1):
            dither[oy:oy + chk.shape[0], ox:ox + chk.shape[1]] |= chk
    dither &= ~lines
    roles[dither] = ROLES.index("dither")

    # --- anti-aliasing: short runs next to a line, between the line and the fill
    line_l = np.where(lines, lum, np.nan)
    for y, x in zip(*np.nonzero(opaque & ~lines & ~dither)):
        k = keys[y, x]
        same = sum(1 for dx, dy in _N4 if pad[1 + y + dy, 1 + x + dx] == k)
        if same > 1:
            continue
        nl = [line_l[y + dy, x + dx] for dx, dy in _N8
              if 0 <= y + dy < h and 0 <= x + dx < w and lines[y + dy, x + dx]]
        if not nl:
            continue
        fills = [lum[y + dy, x + dx] for dx, dy in _N8
                 if 0 <= y + dy < h and 0 <= x + dx < w and opaque[y + dy, x + dx]
                 and not lines[y + dy, x + dx] and keys[y + dy, x + dx] != k]
        if fills and min(nl) + 0.03 < lum[y, x] < max(fills) - 0.03:
            roles[y, x] = ROLES.index("aa")

    # --- parts: same-hue regions separated by lines; base / shadow / highlight
    rest = opaque & (roles == 0)
    hsv = {}

    def family(k):
        if k not in hsv:
            hh, ss, vv = rgb_to_hsv(L._rgb(int(k)))
            hsv[k] = (hh, ss < 12 or vv < 12)
        return hsv[k]

    parts = []
    seen = np.zeros((h, w), dtype=bool)
    for y, x in zip(*np.nonzero(rest)):
        if seen[y, x]:
            continue
        h0, grey0 = family(keys[y, x])
        stack, comp = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dx, dy in _N4:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and rest[ny, nx] and not seen[ny, nx]:
                    hh, grey = family(keys[ny, nx])
                    if grey == grey0 and (grey or abs(hue_delta(h0, hh)) <= 35):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        parts.append(comp)

    part_info = []
    for comp in parts:
        if len(comp) <= 2:
            for y, x in comp:
                roles[y, x] = ROLES.index("detail")
            continue
        counts = Counter(int(keys[y, x]) for y, x in comp)
        base = counts.most_common(1)[0][0]
        base_l = float(lightness(L._rgb(base)))
        for y, x in comp:
            k = int(keys[y, x])
            if k == base:
                roles[y, x] = ROLES.index("base")
            elif float(lum[y, x]) < base_l - 0.02:
                roles[y, x] = ROLES.index("shadow")
            elif float(lum[y, x]) > base_l + 0.02:
                roles[y, x] = ROLES.index("highlight")
            else:
                roles[y, x] = ROLES.index("base")
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        part_info.append({"base": L.rgb_to_hex(L._rgb(base)), "pixels": len(comp), "tones": len(counts),
                          "bbox": [int(min(xs)), int(min(ys)), int(max(xs) - min(xs) + 1),
                                   int(max(ys) - min(ys) + 1)]})
    part_info.sort(key=lambda p: -p["pixels"])

    # --- summary
    edge = opaque & near_air
    n_edge = int(edge.sum())
    outline_mask = roles == ROLES.index("outline")
    line_px = int(lines.sum())
    coverage = float((edge & outline_mask).sum() / n_edge) if n_edge else 0.0
    line_cols = Counter(int(k) for k in keys[lines])
    # Line width: share of line pixels not inside a 2x2 block of line pixels.
    width1 = L._thin_ratio(lines) if line_px else 0.0
    gaps = [(int(x), int(y)) for y, x in zip(*np.nonzero(edge & ~outline_mask))] if coverage >= 0.6 else []
    counts = {r: int((roles == i).sum()) for i, r in enumerate(ROLES) if i}
    total = max(1, int(opaque.sum()))
    summary = {
        "has_outline": coverage >= 0.6,
        "outline_coverage": round(coverage, 2),
        "outline_gaps": len(gaps),
        "inner_line_share": round(counts["inner_line"] / total, 3),
        "line_width": "1px" if width1 >= 0.85 else "mixed 1-2px" if width1 >= 0.5 else "2px+",
        "line_width_1px_share": round(width1, 2),
        "line_colors": [{"color": L.rgb_to_hex(L._rgb(k)), "pixels": n} for k, n in line_cols.most_common(6)],
        "role_pixels": counts,
        "parts": part_info[:12],
        "part_count": len(part_info),
    }
    return {"roles": roles, "lines": lines, "gaps": gaps, "summary": summary}


def role_mask(result: dict, *names: str) -> np.ndarray:
    idx = [ROLES.index(n) for n in names]
    return np.isin(result["roles"], idx)


def role_image(img: np.ndarray, result: dict) -> np.ndarray:
    """RGBA map with one flat colour per role (transparent stays clear)."""
    out = np.zeros_like(img)
    for i, name in enumerate(ROLES):
        if i == 0:
            continue
        out[result["roles"] == i] = (*ROLE_COLORS[name], 255)
    return out


# ------------------------------------------------------- operations

def _fill_from_neighbours(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels by the most common unmasked neighbour colour,
    growing inward so 2-px lines fill too. Pixels with only transparency
    around become transparent."""
    out = img.copy()
    todo = mask.copy()
    keys = L._keys(img)
    h, w = mask.shape
    for _ in range(4):
        if not todo.any():
            break
        done = []
        for y, x in zip(*np.nonzero(todo)):
            cands = [int(keys[y + dy, x + dx]) for dx, dy in _N4
                     if 0 <= y + dy < h and 0 <= x + dx < w and not todo[y + dy, x + dx]]
            opaque = [k for k in cands if k != -1]
            if opaque:
                k = Counter(opaque).most_common(1)[0][0]
                done.append((y, x, k))
            elif cands:
                done.append((y, x, -1))
        for y, x, k in done:
            out[y, x] = 0 if k == -1 else (*L._rgb(k), k & 255)
            keys[y, x] = k
            todo[y, x] = False
    out[todo] = 0
    return out


def split_lineart(img: np.ndarray, include_aa: bool = False):
    """(lineart, rest): line pixels on their own, and the art with the line
    pixels filled from the neighbouring colours (flats + shading)."""
    res = analyze(img)
    mask = res["lines"].copy()
    if include_aa:
        mask |= res["roles"] == ROLES.index("aa")
    lineart = np.zeros_like(img)
    lineart[mask] = img[mask]
    return lineart, _fill_from_neighbours(img, mask), res


def recolor_lineart(img: np.ndarray, mode: str = "colored", color=None, which: str = "all"):
    """Recolour detected line art.

    colored - each line pixel takes the darkest shade of the part it borders
              (coloured line art: softer, more modern)
    tinted  - one tinted near-black matched to the sprite instead of black
    single  - one given colour
    which: all, outline or inner.
    """
    from .color import suggest_outline
    from .draw import _part_dark
    if mode not in ("colored", "tinted", "single"):
        raise ValueError("mode must be colored, tinted or single")
    if which not in ("all", "outline", "inner"):
        raise ValueError("which must be all, outline or inner")
    res = analyze(img)
    names = {"all": ("outline", "inner_line"), "outline": ("outline",), "inner": ("inner_line",)}[which]
    mask = role_mask(res, *names)
    keys = L._keys(img)
    fill = (keys != -1) & ~res["lines"]
    palette = [L._rgb(int(k)) for k in np.unique(keys[fill])]
    counts = Counter(int(k) for k in keys[keys != -1])
    out = img.copy()
    h, w = mask.shape
    if mode == "single":
        if color is None:
            raise ValueError("single mode needs a colour")
        target = tuple(color[:3])
    elif mode == "tinted":
        target = suggest_outline([(L._rgb(k), n) for k, n in counts.items()])
    changed = 0
    for y, x in zip(*np.nonzero(mask)):
        if mode == "colored":
            near = Counter(int(keys[y + dy, x + dx]) for dx, dy in _N8
                           if 0 <= y + dy < h and 0 <= x + dx < w and fill[y + dy, x + dx])
            if not near:
                continue
            rgb = _part_dark(L._rgb(near.most_common(1)[0][0]), palette)
        else:
            rgb = target
        out[y, x, :3] = rgb
        changed += 1
    return out, changed
