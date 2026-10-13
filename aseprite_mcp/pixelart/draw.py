"""Generators that draw by the rules instead of fixing afterwards.

All functions take and return RGBA uint8 arrays (H, W, 4) in sprite
coordinates and report what they changed. Colours are (r, g, b) tuples.
"""
from collections import Counter

import numpy as np

from . import lint as L
from .color import lightness, oklab, rgb_to_hsv, shade, suggest_outline

_N4 = [(0, -1), (-1, 0), (1, 0), (0, 1)]
_N8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

# Light directions as unit-ish vectors pointing TOWARD the light.
LIGHT = {
    "top-left": (-1, -1), "top": (0, -1), "top-right": (1, -1), "right": (1, 0),
    "bottom-right": (1, 1), "bottom": (0, 1), "bottom-left": (-1, 1), "left": (-1, 0),
}


def _put(img, x, y, rgb, alpha=255):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        img[y, x] = (*rgb[:3], alpha)
        return True
    return False


# ------------------------------------------------------------- lines

def _run_lengths(total: int, runs: int) -> list[int]:
    """Split `total` pixels into `runs` steps with a clean rhythm.

    Equal steps when possible. With only one or two odd steps they go to
    the line's ends, where they read as the line starting or stopping;
    with more, short and long steps are spread evenly (a periodic 2-3-2-3
    rhythm reads as straight).
    """
    base, extra = divmod(total, runs)
    if extra == 0:
        return [base] * runs
    odd, odd_len, even_len = (extra, base + 1, base) if extra <= runs - extra else (runs - extra, base, base + 1)
    if odd <= 2:
        lens = [even_len] * runs
        lens[-1] = odd_len
        if odd == 2:
            lens[0] = odd_len
        return lens
    # Evenly spread: odd step k sits at round((k + 0.5) * runs / odd).
    lens = [even_len] * runs
    for k in range(odd):
        lens[min(runs - 1, int((k + 0.5) * runs / odd))] = odd_len
    return lens


def line_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """8-connected line between two points with a clean staircase.

    Steps are as equal as the slope allows: one or two odd steps are put at
    the ends, otherwise short and long steps alternate evenly. No L-corners.
    """
    dx, dy = x1 - x0, y1 - y0
    sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    major, minor = max(abs(dx), abs(dy)), min(abs(dx), abs(dy))
    if major == 0:
        return [(x0, y0)]
    x_major = abs(dx) >= abs(dy)
    pts, (x, y) = [], (x0, y0)
    lens = _run_lengths(major + 1, minor + 1)
    for i, n in enumerate(lens):
        for j in range(n):
            pts.append((x, y))
            if j < n - 1:
                x, y = (x + sx, y) if x_major else (x, y + sy)
        if i < len(lens) - 1:
            x, y = x + sx, y + sy
    return pts


def snapped_line(x0: int, y0: int, x1: int, y1: int, max_ratio: int = 6) -> tuple[list, tuple[int, int]]:
    """Line snapped to the nearest clean slope (1:1, 2:1, ... max_ratio:1).

    Every step has exactly the same length, so the line reads perfectly
    straight. Keeps the start point and direction; the end point moves to
    the nearest whole step. Returns (points, end).
    """
    dx, dy = x1 - x0, y1 - y0
    adx, ady = abs(dx), abs(dy)
    if adx == 0 or ady == 0:
        return line_points(x0, y0, x1, y1), (x1, y1)
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    if adx >= ady:
        n = min(max_ratio, max(1, round((adx + 1) / (ady + 1))))
        count = max(n, (adx + 1) // n * n)  # whole steps only
        pts = [(x0 + sx * i, y0 + sy * (i // n)) for i in range(count)]
    else:
        n = min(max_ratio, max(1, round((ady + 1) / (adx + 1))))
        count = max(n, (ady + 1) // n * n)
        pts = [(x0 + sx * (i // n), y0 + sy * i) for i in range(count)]
    return pts, pts[-1]


def _bezier(points: list[tuple[float, float]], t: np.ndarray) -> np.ndarray:
    """Points on a Bezier curve (Bernstein form), shape (len(t), 2)."""
    from math import comb
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts) - 1
    basis = np.stack([comb(n, i) * (1 - t) ** (n - i) * t ** i for i in range(n + 1)], axis=1)
    return basis @ pts


def curve_points(control: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Rasterise a Bezier curve (2 = line, 3 = quadratic, 4 = cubic, ...)
    as a clean 1-px stroke: dense sampling, gaps bridged, L-corners removed
    and the staircase rhythm evened out (end points kept)."""
    if len(control) < 2:
        raise ValueError("A curve needs at least 2 control points")
    length = sum(np.hypot(*np.subtract(control[i + 1], control[i])) for i in range(len(control) - 1))
    t = np.linspace(0, 1, max(8, int(length * 8)))
    xy = _bezier(control, t)
    raw = []
    for x, y in np.floor(xy + 0.5).astype(int):
        p = (int(x), int(y))
        if not raw or raw[-1] != p:
            if raw and max(abs(p[0] - raw[-1][0]), abs(p[1] - raw[-1][1])) > 1:
                raw += line_points(*raw[-1], *p)[1:]
            else:
                raw.append(p)
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    ox, oy = min(xs) - 1, min(ys) - 1
    mask = np.zeros((max(ys) - oy + 2, max(xs) - ox + 2), dtype=bool)
    for x, y in raw:
        mask[y - oy, x - ox] = True
    for x, y in L.find_l_corners(mask):
        mask[y, x] = False
    canvas = np.zeros((*mask.shape, 4), np.uint8)
    canvas[mask] = (0, 0, 0, 255)
    from .cleanup import fix_jaggies
    canvas, _ = fix_jaggies(canvas, [int(L._keys(canvas[mask][:1].reshape(1, 1, 4))[0, 0])])
    ys2, xs2 = np.nonzero(canvas[..., 3] > 0)
    return [(int(x) + ox, int(y) + oy) for x, y in zip(xs2, ys2)]


# ----------------------------------------------------- anti-aliasing

def _mix(a, b, t):
    """Colour t of the way from a to b, mixed in OKLab."""
    la, lb = oklab(a), oklab(b)
    lab = la * (1 - t) + lb * t
    # OKLab -> linear sRGB -> sRGB
    l_, m_, s_ = (lab @ np.array([[1, 1, 1], [0.3963377774, -0.1055613458, -0.0894841775],
                                   [0.2158037573, -0.0638541728, -1.2914855480]])) ** 3
    rgb = np.array([l_, m_, s_]) @ np.array([[4.0767416621, -1.2684380046, -0.0041960863],
                                              [-3.3077115913, 2.6097574011, -0.7034186147],
                                              [0.2309699292, -0.3413193965, 1.7076147010]])
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return tuple(int(round(c * 255)) for c in srgb)


def _snap(rgb, palette, max_delta=6.0):
    """Nearest palette colour if it is close enough, else the colour itself."""
    if not palette:
        return rgb
    lab = oklab(rgb)
    best = min(palette, key=lambda p: float(np.linalg.norm(oklab(p) - lab)))
    return best if float(np.linalg.norm(oklab(best) - lab)) * 100 < max_delta else rgb


def antialias_edges(img: np.ndarray, line_rgb, shades: int = 1, coverage: float = 0.5,
                    use_palette: bool = True) -> tuple[np.ndarray, list, list]:
    """Anti-alias the staircase steps of a line colour.

    At every step between two runs, the two concave corner cells get AA
    that runs alongside the neighbouring run for about `coverage` of its
    length (0.5 = half a step), starting at the corner. Colours lie between
    the line and the colour they replace (OKLab), 1 or 2 shades (darker
    next to the corner), snapped to existing colours when one is close.
    AA is only placed on opaque pixels, never onto transparency (it would
    look spotty on unknown backgrounds). Steps on 45-degree parts
    (runs of 1) are left alone.

    Returns (image, changed pixels, new colours introduced).
    """
    if shades not in (1, 2):
        raise ValueError("shades must be 1 or 2")
    line_rgb = tuple(line_rgb[:3])
    out = img.copy()
    keys = L._keys(img)
    h, w = keys.shape
    line_key = [k for k in np.unique(keys) if k != -1 and L._rgb(int(k)) == line_rgb]
    if not line_key:
        return out, [], []
    line_key = int(line_key[0])
    palette = [L._rgb(int(k)) for k in np.unique(keys) if k != -1] if use_palette else []
    plan = {}  # (x, y) -> mix fraction toward the replaced colour

    for chain in L._thin_chains(keys == line_key):
        runs = L.chain_runs(chain)
        for k in range(len(runs) - 1):
            a, b = runs[k], runs[k + 1]
            if b["break"]:
                continue
            last, first = a["pixels"][-1], b["pixels"][0]
            d = (first[0] - last[0], first[1] - last[1])
            if not (d[0] and d[1]):
                continue
            axis = a["axis"] or b["axis"]
            if axis is None:
                continue
            s = (d[0], 0) if axis == "h" else (0, d[1])
            # Corner 1 continues run a, alongside run b; corner 2 sits before
            # run b, alongside run a (going backwards).
            for start, step, along in (((last[0] + s[0], last[1] + s[1]), s, b),
                                       ((first[0] - s[0], first[1] - s[1]), (-s[0], -s[1]), a)):
                if along["len"] < 2:
                    continue
                n = max(1, int(round(along["len"] * coverage)))
                x, y = start
                if not (0 <= x < w and 0 <= y < h):
                    continue
                base = int(keys[y, x])
                if base in (line_key, -1):
                    continue
                for i in range(n):
                    cx, cy = x + step[0] * i, y + step[1] * i
                    if not (0 <= cx < w and 0 <= cy < h) or int(keys[cy, cx]) != base:
                        break
                    if shades == 1:
                        t = 0.5
                    else:
                        t = 0.35 if i < (n + 1) // 2 else 0.7
                    plan[(cx, cy)] = max(plan.get((cx, cy), 0), 1 - t)  # strength of line colour

    changed, new_colors = [], set()
    for (x, y), strength in plan.items():
        under = L._rgb(int(keys[y, x]))
        rgb = _snap(_mix(line_rgb, under, 1 - strength), palette)
        if rgb not in palette:
            new_colors.add(rgb)
        out[y, x] = (*rgb, 255)
        changed.append((x, y))
    return out, changed, sorted(new_colors)


# ------------------------------------------------------------ outline

def _part_dark(rgb, palette):
    """Darkest colour of the ramp (hue family) rgb belongs to, if it is
    clearly darker; otherwise a hue-shifted shadow two steps down."""
    h, s, v = rgb_to_hsv(rgb)
    fam = [p for p in palette if abs(((rgb_to_hsv(p)[0] - h + 180) % 360) - 180) <= 30
           and (rgb_to_hsv(p)[1] >= 12) == (s >= 12)]
    if fam:
        darkest = min(fam, key=lambda p: float(lightness(p)))
        if float(lightness(rgb)) - float(lightness(darkest)) >= 0.15:
            return darkest
    return shade(rgb, "shadow", 2)


def outline(img: np.ndarray, style: str = "black", color=None, light: str = "top-left",
            placement: str = "outside", corners: bool = False) -> tuple[np.ndarray, list, dict]:
    """Draw an outline around the opaque shape in one of the pixel-art styles.

    style:
      black    - one dark colour all round (default: a tinted near-black that
                 suits the sprite, or `color`)
      colored  - each stretch takes the darkest shade of the part it wraps
      shaded   - colored on the lit side, the dark colour on the shadow side
      selective- like shaded but the outline is left out where it faces the
                 light (sel-out; reads best on dark backgrounds)
    placement: "outside" grows the sprite by 1 px; "inside" recolours the
    sprite's own edge pixels. corners=True also fills diagonal corners.
    """
    if style not in ("black", "colored", "shaded", "selective"):
        raise ValueError("style must be black, colored, shaded or selective")
    if placement not in ("outside", "inside"):
        raise ValueError("placement must be 'outside' or 'inside'")
    if light not in LIGHT:
        raise ValueError(f"light must be one of {', '.join(LIGHT)}")
    out = img.copy()
    opaque = img[..., 3] > 0
    h, w = opaque.shape
    keys = L._keys(img)
    counts = Counter(int(k) for k in keys[keys != -1])
    palette = [L._rgb(k) for k in counts]
    dark = tuple(color[:3]) if color is not None else suggest_outline([(L._rgb(k), n) for k, n in counts.items()])
    nbrs = _N8 if corners else _N4
    lx, ly = LIGHT[light]
    norm = (lx * lx + ly * ly) ** 0.5

    targets = []
    for y in range(h):
        for x in range(w):
            inside = opaque[y, x]
            if (placement == "outside") == inside:
                continue
            touching = [(dx, dy) for dx, dy in nbrs
                        if 0 <= x + dx < w and 0 <= y + dy < h and opaque[y + dy, x + dx] != inside]
            if placement == "inside":
                touching = [(dx, dy) for dx, dy in nbrs
                            if not (0 <= x + dx < w and 0 <= y + dy < h) or not opaque[y + dy, x + dx]]
            if touching:
                targets.append((x, y, touching))

    changed = []
    for x, y, touching in targets:
        # Outward normal: away from the sprite.
        if placement == "outside":
            nx = -sum(dx for dx, _ in touching)
            ny = -sum(dy for _, dy in touching)
            part = Counter(int(keys[y + dy, x + dx]) for dx, dy in touching).most_common(1)[0][0]
        else:
            nx = sum(dx for dx, _ in touching)
            ny = sum(dy for _, dy in touching)
            part = int(keys[y, x])
        facing = (nx * lx + ny * ly) / ((nx * nx + ny * ny) ** 0.5 * norm + 1e-9)
        part_rgb = L._rgb(part)
        if style == "black":
            rgb = dark
        elif style == "colored":
            rgb = _part_dark(part_rgb, palette)
        else:
            if facing > 0.35:
                if style == "selective":
                    continue
                rgb = _part_dark(part_rgb, palette)
            elif facing < -0.35:
                rgb = dark
            else:
                rgb = _part_dark(part_rgb, palette) if style == "shaded" else dark
        out[y, x] = (*rgb, 255)
        changed.append((x, y))
    return out, changed, {"dark": dark}


# ------------------------------------------------------------- dither

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
# Densities (out of 16) that give clean, evenly spread patterns on a 4x4 cell.
CLEAN_LEVELS = (0, 1, 2, 4, 6, 8, 10, 12, 14, 15, 16)


def dither_fill(img: np.ndarray, rect: tuple[int, int, int, int], ramp: list, direction: str = "vertical",
                levels: int = 0, mask: str = "none", only_color=None) -> tuple[np.ndarray, int]:
    """Fill a rectangle with a dithered gradient across a ramp of colours.

    Only neighbouring ramp colours are mixed, with ordered (Bayer) patterns
    aligned to the canvas so neighbouring fills tile seamlessly. levels = 0
    uses the clean 4x4 density ladder (0,1,2,4,6,8,10,12,14,15,16 of 16);
    levels = N quantises each band to N steps (2 = hard bands, no dither).
    direction: vertical (first colour at the top), horizontal (left),
    or radial (centre outward). mask: "none" fills every pixel, "opaque"
    only existing pixels; only_color (rgb) restricts to pixels of that colour.

    Returns (image, pixels written).
    """
    x0, y0, rw, rh = rect
    if rw < 1 or rh < 1:
        raise ValueError("rectangle must be at least 1x1")
    if len(ramp) < 2:
        raise ValueError("ramp needs at least 2 colours")
    if direction not in ("vertical", "horizontal", "radial"):
        raise ValueError("direction must be vertical, horizontal or radial")
    if mask not in ("none", "opaque"):
        raise ValueError("mask must be 'none' or 'opaque'")
    out = img.copy()
    h, w = img.shape[:2]
    bands = len(ramp) - 1
    written = 0
    cx, cy = x0 + (rw - 1) / 2, y0 + (rh - 1) / 2
    rmax = max(((rw - 1) / 2) ** 2 + ((rh - 1) / 2) ** 2, 1e-9) ** 0.5
    for y in range(max(0, y0), min(h, y0 + rh)):
        for x in range(max(0, x0), min(w, x0 + rw)):
            if mask == "opaque" and img[y, x, 3] == 0:
                continue
            if only_color is not None and tuple(img[y, x, :3]) != tuple(only_color[:3]):
                continue
            if direction == "vertical":
                t = (y - y0) / max(1, rh - 1)
            elif direction == "horizontal":
                t = (x - x0) / max(1, rw - 1)
            else:
                t = min(1.0, ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / rmax)
            pos = t * bands
            i = min(bands - 1, int(pos))
            frac = pos - i
            if levels == 0:
                count = min(CLEAN_LEVELS, key=lambda c: abs(c / 16 - frac))
            else:
                steps = max(2, levels)
                count = round(round(frac * (steps - 1)) / (steps - 1) * 16)
            use_next = BAYER4[y % 4, x % 4] < count
            rgb = ramp[i + 1] if use_next else ramp[i]
            out[y, x] = (*rgb[:3], 255)
            written += 1
    return out, written


# ------------------------------------------------------ gradient map

def gradient_map(img: np.ndarray, ramp: list, dither: bool = False, normalize: bool = True) -> np.ndarray:
    """Recolour by value: each opaque pixel's lightness picks a ramp colour.

    ramp is ordered dark to light. normalize stretches the image's own
    lightness range over the whole ramp. dither=True adds a 50% checker
    step between neighbouring ramp colours (2n-1 levels), aligned to the
    canvas.
    """
    if len(ramp) < 2:
        raise ValueError("ramp needs at least 2 colours")
    out = img.copy()
    opaque = img[..., 3] > 0
    if not opaque.any():
        return out
    lum = lightness(img[..., :3])
    lo, hi = (lum[opaque].min(), lum[opaque].max()) if normalize else (0.0, 1.0)
    t = np.clip((lum - lo) / max(hi - lo, 1e-9), 0, 1)
    n = len(ramp)
    levels = 2 * n - 1 if dither else n
    q = np.minimum(levels - 1, np.floor(t * levels)).astype(int)
    ys, xs = np.nonzero(opaque)
    for y, x in zip(ys, xs):
        lvl = q[y, x]
        if dither:
            i = lvl // 2
            if lvl % 2 and (x + y) % 2:
                i += 1
        else:
            i = lvl
        out[y, x, :3] = ramp[min(i, n - 1)][:3]
    return out
