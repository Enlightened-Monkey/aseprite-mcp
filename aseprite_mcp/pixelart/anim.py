"""Animation craft: timing, easing, motion helpers and an animation analyzer.

Pure Python on lists of RGBA frames (H, W, 4) and per-frame durations (ms).
"""
from collections import Counter

import numpy as np

from . import lint as L
from .color import oklab

# ------------------------------------------------------------- timing

# Per-frame durations in ms for common actions (typical game practice).
TIMING_PRESETS = {
    "idle_breathing": {"frames": 4, "durations": [250, 250, 250, 250], "sequence": "A-B-A-C",
                       "note": "torso rises 1 px; arms/head follow a frame late"},
    "idle_pingpong": {"frames": 8, "durations": [200, 100, 100, 100, 200, 100, 100, 100],
                      "sequence": "0-1-2-3-4-3-2-1", "note": "hold the two extremes, quick in-betweens"},
    "walk_small": {"frames": 4, "durations": [70] * 4, "note": "tiny sprites"},
    "walk": {"frames": 6, "durations": [100] * 6, "note": "contact, down, passing, up per step"},
    "walk_smooth": {"frames": 8, "durations": [80] * 8, "note": "second half mirrors the first"},
    "run": {"frames": 8, "durations": [60] * 8, "note": "both feet off the ground in some frames"},
    "jump": {"frames": 6, "durations": [60, 80, 100, 160, 100, 60],
             "note": "crouch (anticipation), launch, rise, apex hold, fall, land"},
    "attack": {"frames": 6, "durations": [150, 80, 40, 120, 80, 350],
               "note": "anticipation, wind-up, strike/smear (short), contact, recovery, end hold"},
    "attack_fast": {"frames": 5, "durations": [100, 30, 30, 80, 300],
                    "note": "player input: minimal anticipation so it feels instant"},
    "hit_react": {"frames": 3, "durations": [60, 120, 160], "note": "snap away, hold, recover"},
    "smear": {"frames": 1, "durations": [40], "note": "felt, not seen: 40-50 ms"},
    "impact_fx": {"frames": 8, "durations": [40] * 8, "note": "expand, break up, erode"},
    "ui_loop": {"frames": 4, "durations": [80] * 4, "note": "shine sweep, glint, pulse"},
    "ui_pop": {"frames": 3, "durations": [40, 40, 40], "note": "small, oversized, final size; then hold"},
    "blink": {"frames": 3, "durations": [2000, 60, 60], "note": "long open hold, half, closed"},
}

FPS_BASE = {"game60": 1000 / 60, "game30": 1000 / 30, "film24": 1000 / 24}


def units_to_ms(units: list[int], base: str = "game60") -> list[int]:
    """Hold lengths in base frames ("ones", "twos"...) to milliseconds."""
    if base not in FPS_BASE:
        raise ValueError(f"base must be one of {', '.join(FPS_BASE)}")
    return [max(10, int(round(u * FPS_BASE[base] / 10.0)) * 10) for u in units]


def ease_durations(n: int, mid_ms: int = 80, key_ms: int = 160, profile: str = "in_out") -> list[int]:
    """Easing by timing: frames near the keys last longer, mid-motion shorter.

    profile in_out: long at both ends (key_ms), mid_ms in the middle, with a
    step between; "in": slows into the last key; "out": leaves the first key
    slowly.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    step = (key_ms + mid_ms) // 2
    if profile == "in_out":
        prof = [key_ms, step] + [mid_ms] * max(0, n - 4) + [step, key_ms]
        prof = prof[:n] if n >= 4 else [key_ms] + [mid_ms] * max(0, n - 2) + ([key_ms] if n > 1 else [])
    elif profile == "out":
        prof = ([key_ms, step] + [mid_ms] * n)[:n]
    elif profile == "in":
        prof = ([mid_ms] * n + [step, key_ms])[-n:]
    else:
        raise ValueError("profile must be in_out, in or out")
    return [int(v) for v in prof]


# ------------------------------------------------------------- easing

def ease(t: float, kind: str = "ease_in_out", overshoot: float = 0.0) -> float:
    """Easing curves. back_out overshoots by `overshoot` (fraction of the move)
    before settling at 1."""
    t = min(1.0, max(0.0, t))
    if kind == "linear":
        return t
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1 - (1 - t) ** 2
    if kind == "ease_in_out":
        return 3 * t * t - 2 * t * t * t
    if kind == "back_out":
        # Reaches 1 + overshoot around t~0.7, then settles to 1.
        o = overshoot if overshoot > 0 else 0.1
        if t < 0.7:
            return (1 + o) * (1 - (1 - t / 0.7) ** 2)
        return 1 + o - o * ((t - 0.7) / 0.3) ** 2
    raise ValueError("easing must be linear, ease_in, ease_out, ease_in_out or back_out")


def path_positions(start, end, frames: int, breakdown=None, easing: str = "ease_in_out",
                   overshoot_px: float = 0.0) -> list[tuple[float, float]]:
    """Positions along a straight path or an arc through a breakdown point.

    With a breakdown the path is a quadratic Bezier that passes through it
    at the middle, so motion follows an arc instead of a straight line.
    overshoot_px (with back_out easing) is the distance travelled past the
    end before settling.
    """
    if frames < 2:
        raise ValueError("frames must be >= 2")
    p0, p2 = np.array(start, float), np.array(end, float)
    dist = float(np.linalg.norm(p2 - p0)) or 1.0
    o = overshoot_px / dist
    out = []
    if breakdown is not None:
        p1 = 2 * np.array(breakdown, float) - (p0 + p2) / 2  # control so B(0.5) = breakdown
    for i in range(frames):
        s = ease(i / (frames - 1), easing, o)
        if breakdown is None:
            p = p0 + (p2 - p0) * s
        else:
            sc = min(1.0, s)
            p = (1 - sc) ** 2 * p0 + 2 * (1 - sc) * sc * p1 + sc * sc * p2
            if s > 1:  # overshoot continues along the end tangent
                tangent = (p2 - p1) / (np.linalg.norm(p2 - p1) or 1.0)
                p = p2 + tangent * (s - 1) * dist
        out.append((float(p[0]), float(p[1])))
    return out


# --------------------------------------------------- pixel transforms

def shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate by whole pixels; uncovered area is transparent."""
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    xs, xd = (0, dx) if dx >= 0 else (-dx, 0)
    ys, yd = (0, dy) if dy >= 0 else (-dy, 0)
    cw, ch = w - abs(dx), h - abs(dy)
    if cw > 0 and ch > 0:
        out[yd:yd + ch, xd:xd + cw] = img[ys:ys + ch, xs:xs + cw]
    return out


def _palette(img):
    keys = L._keys(img)
    return [L._rgb(int(k)) for k in np.unique(keys[keys != -1])]


def subpixel_shift(img: np.ndarray, dx: float, dy: float, protect=None, scale: int = 8) -> np.ndarray:
    """Move art by a fraction of a pixel using only colours it already has.

    The image is supersampled, moved, and each output pixel is resolved:
    a colour covering >= 80% wins; otherwise the existing colour closest to
    the coverage-weighted mix of the two main colours is used, so edges
    slide by tone instead of by whole pixels. Against transparency there is
    no in-between tone, so silhouette edges snap at 50% coverage. Pixels in
    `protect` (bool mask, e.g. a face) only move by whole pixels.
    """
    h, w = img.shape[:2]
    ix, iy = int(round(dx)), int(round(dy))
    whole = shift(img, ix, iy)
    if abs(dx - ix) < 1e-6 and abs(dy - iy) < 1e-6:
        return whole
    keys = L._keys(img)
    palette_keys = [int(k) for k in np.unique(keys) if k != -1]
    pal_lab = {k: oklab(L._rgb(k)) for k in palette_keys}
    big = np.repeat(np.repeat(keys, scale, axis=0), scale, axis=1)
    ox, oy = int(round(dx * scale)), int(round(dy * scale))
    moved = np.full_like(big, -1)
    H, W = big.shape
    xs, xd = (0, ox) if ox >= 0 else (-ox, 0)
    ys, yd = (0, oy) if oy >= 0 else (-oy, 0)
    if W - abs(ox) > 0 and H - abs(oy) > 0:
        moved[yd:yd + H - abs(oy), xd:xd + W - abs(ox)] = big[ys:ys + H - abs(oy), xs:xs + W - abs(ox)]
    cells = moved.reshape(h, scale, w, scale).transpose(0, 2, 1, 3).reshape(h, w, scale * scale)
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            cnt = Counter(cells[y, x].tolist())
            (k1, n1), *rest = cnt.most_common(2)
            total = scale * scale
            if n1 >= 0.8 * total or not rest:
                k = k1
            else:
                k2, n2 = rest[0]
                if -1 in (k1, k2):
                    opaque_k = k2 if k1 == -1 else k1
                    opaque_n = n2 if k1 == -1 else n1
                    k = opaque_k if opaque_n >= 0.5 * total else -1
                else:
                    mix = (pal_lab[k1] * n1 + pal_lab[k2] * n2) / (n1 + n2)
                    lo, hi = sorted((float(pal_lab[k1][0]), float(pal_lab[k2][0])))
                    between = [p for p in palette_keys if lo - 1e-6 <= float(pal_lab[p][0]) <= hi + 1e-6]
                    k = min(between or [k1, k2], key=lambda p: float(np.linalg.norm(pal_lab[p] - mix)))
            if k != -1:
                out[y, x] = (*L._rgb(k), k & 255)
    if protect is not None:
        moved_mask = shift(protect.astype(np.uint8)[..., None].repeat(4, axis=2), ix, iy)[..., 0] > 0
        out[moved_mask] = whole[moved_mask]
    return out


def squash_stretch(img: np.ndarray, amount: float, axis: str = "y", anchor: str = "bottom") -> np.ndarray:
    """Squash (amount > 0) or stretch (amount < 0) keeping the area.

    Along `axis` the size scales by (1 - amount), across it by 1 / (1 - amount),
    so the silhouette keeps about the same number of pixels. Nearest-neighbour,
    anchored at the bottom/top/centre of the art's bounding box (the contact
    point should stay put). Expect a clean-up pass afterwards.
    """
    if not -0.6 < amount < 0.6:
        raise ValueError("amount must be between -0.6 and 0.6")
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    if anchor not in ("bottom", "top", "center", "left", "right"):
        raise ValueError("anchor must be bottom, top, center, left or right")
    opaque = img[..., 3] > 0
    if not opaque.any():
        return img.copy()
    ys, xs = np.nonzero(opaque)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    along = 1 - amount
    sx, sy = (1 / along, along) if axis == "y" else (along, 1 / along)
    cx = {"left": x0, "right": x1}.get(anchor, (x0 + x1) / 2)
    cy = {"bottom": y1, "top": y0}.get(anchor, (y0 + y1) / 2)
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    yy, xx = np.mgrid[0:h, 0:w]
    srcx = np.floor((xx + 0.5 - cx) / sx + cx).astype(int)
    srcy = np.floor((yy + 0.5 - cy) / sy + cy).astype(int)
    ok = (srcx >= 0) & (srcx < w) & (srcy >= 0) & (srcy < h)
    out[ok] = img[srcy[ok], srcx[ok]]
    return out


def centroid(img: np.ndarray):
    ys, xs = np.nonzero(img[..., 3] > 0)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


# ------------------------------------------------------------ analyzer

def _norm(img):
    return np.where(img[..., 3:4] > 0, img, 0)


def _best_shift(a: np.ndarray, b: np.ndarray, radius: int = 4):
    """Whole-pixel shift of a that best matches b, and the pixels still
    differing after it (the part of the change that is not plain motion)."""
    ka, kb = L._keys(a), L._keys(b)
    best = (0, 0, int((ka != kb).sum()))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == dy == 0:
                continue
            s = L._keys(shift(a, dx, dy))
            diff = int((s != kb).sum())
            if diff < best[2] or (diff == best[2] and abs(dx) + abs(dy) < abs(best[0]) + abs(best[1])):
                best = (dx, dy, diff)
    return best


def analyze(frames: list[np.ndarray], durations: list[int]) -> dict:
    """Review an animation for timing and motion problems.

    Reports the frame sequence (re-used drawings, ping-pong), holds and
    duplicate frames, per-transition motion (best whole-pixel shift plus the
    "residual": pixels redrawn beyond plain movement - high residual on parts
    that should stand still means line boil; this cannot be told apart from
    deliberate frame-by-frame motion automatically, so it is reported, not
    flagged), and issues:
    - two_frame_loop: only two distinct drawings (reads as flicker)
    - duplicate_frames: identical neighbours (merge into one longer frame)
    - odd_frame: one frame differs far more from both neighbours than the rest
    - mechanical_spacing: 4+ steps of equal movement at equal timing
    - volume_drift: pixel count jumps (squash/stretch not keeping area)
    - one_frame_colors: colours used in a single frame of a loop
    - timing: durations under 20 ms, or not multiples of 10 (GIF rounding)
    """
    n = len(frames)
    if n == 0:
        raise ValueError("no frames")
    if len(durations) != n:
        raise ValueError("durations must match frames")
    norm = [_norm(f) for f in frames]
    ids, seen = [], {}
    for f in norm:
        key = f.tobytes()
        ids.append(seen.setdefault(key, len(seen)))
    issues = []

    def issue(check, severity, message, frames_=()):
        issues.append({"check": check, "severity": severity, "message": message,
                       "frames": [int(i) + 1 for i in frames_]})

    unique = len(seen)
    if n >= 2 and unique == 2:
        issue("two_frame_loop", "warning", "Only two distinct drawings: two-frame loops read as flicker. "
              "Use at least 3 (e.g. A-B-A-C, a ping-pong, or move a part around a small circle).")
    dup = [i for i in range(1, n) if ids[i] == ids[i - 1]]
    if dup:
        issue("duplicate_frames", "info", "Identical consecutive frames: merge each run into one frame with a "
              "longer duration.", dup)

    counts = [int((f[..., 3] > 0).sum()) for f in norm]
    cents = [centroid(f) for f in norm]
    trans = []
    for i in range(n - 1 if n > 1 else 0):
        a, b = norm[i], norm[i + 1]
        changed = int((L._keys(a) != L._keys(b)).sum())
        dx, dy, rest = _best_shift(a, b) if changed else (0, 0, 0)
        trans.append({"from": i + 1, "to": i + 2, "changed": changed, "shift": [dx, dy], "residual": rest})

    area = max(1, int(np.median(counts)))
    if len(trans) >= 3:
        res = np.array([t["residual"] for t in trans], float)
        med = float(np.median(res)) or 1.0
        odd = []
        for i in range(1, n - 1):
            if res[i - 1] > 2.5 * med and res[i] > 2.5 * med and res[i - 1] > 0.05 * area:
                odd.append(i)
        if odd:
            issue("odd_frame", "info", "A frame differs from both neighbours much more than the rest. Right for "
                  "smears, impacts and pops; otherwise one odd frame can spoil a loop - redraw it to sit between "
                  "its neighbours.", odd)

    run, mech = [], []
    for i, t in enumerate(trans):
        if run and t["shift"] == trans[run[-1]]["shift"] and t["shift"] != [0, 0] \
                and durations[i] == durations[run[-1]] and i == run[-1] + 1:
            run.append(i)
        else:
            if len(run) >= 4:
                mech += run
            run = [i] if t["shift"] != [0, 0] else []
    if len(run) >= 4:
        mech += run
    if mech:
        issue("mechanical_spacing", "info", "Movement of identical size at identical timing over 4+ frames looks "
              "mechanical. Ease it: bunch positions near the keys or lengthen frames near the keys "
              "(ease_durations).", mech)

    drift = [i for i in range(n) if abs(counts[i] - area) > 0.15 * area]
    if drift and len(drift) < n:
        issue("volume_drift", "info", "The amount of pixels changes by more than 15% in some frames. Fine for "
              "squash & stretch if area is kept; otherwise the shape is growing or shrinking.", drift)

    colour_frames = Counter()
    for f in norm:
        kk = L._keys(f)
        for c in np.unique(kk[kk != -1]):
            colour_frames[int(c)] += 1
    if n >= 3:
        rare = [L.rgb_to_hex(L._rgb(c)) for c, m in colour_frames.items() if m == 1]
        if rare:
            issue("one_frame_colors", "info", f"{len(rare)} colour(s) appear in only one frame ({', '.join(rare[:6])})"
                  " - they flash during playback unless intended (FX, glint).")

    bad_t = [i for i, d in enumerate(durations) if d < 20 or d % 10]
    if bad_t:
        issue("timing", "info", "Durations under 20 ms or not multiples of 10 ms get rounded in GIFs and some "
              "engines.", bad_t)

    pattern = "-".join(str(i) for i in ids)
    total = int(sum(durations))
    return {
        "frames": n, "unique_drawings": unique, "sequence": pattern, "durations": [int(d) for d in durations],
        "loop_ms": total, "holds": [i + 1 for i, d in enumerate(durations) if d >= 2 * np.median(durations)],
        "transitions": trans, "pixel_counts": counts,
        "centroids": [None if c is None else [round(c[0], 2), round(c[1], 2)] for c in cents],
        "issues": issues,
    }


# ---------------------------------------------------------- effects

def _lighter_map(img: np.ndarray, steps: int = 1) -> dict:
    """Map each colour to the colour `steps` places lighter in the image's
    own palette (by lightness), staying inside the palette."""
    pal = sorted(_palette(img), key=lambda c: float(oklab(c)[0]))
    return {c: pal[min(len(pal) - 1, i + steps)] for i, c in enumerate(pal)}


def smear(frame_a: np.ndarray, frame_b: np.ndarray, mode: str = "elongated", copies: int = 2) -> np.ndarray:
    """In-between that fakes motion blur from pose A to pose B.

    elongated: A is stamped along the path to B at every pixel step and B is
    drawn on top, so the shape stretches with a crisp leading edge.
    multiples: `copies` after-images of A along the path, one palette step
    lighter, under B. Uses only the art's own colours; show it for 40-50 ms.
    """
    if mode not in ("elongated", "multiples"):
        raise ValueError("mode must be 'elongated' or 'multiples'")
    ca, cb = centroid(frame_a), centroid(frame_b)
    if ca is None or cb is None:
        raise ValueError("both frames need some pixels")
    dx, dy = cb[0] - ca[0], cb[1] - ca[1]
    out = np.zeros_like(frame_a)

    def stamp(img):
        m = img[..., 3] > 0
        out[m] = img[m]

    if mode == "elongated":
        steps = max(1, int(round(max(abs(dx), abs(dy)))))
        for i in range(steps):
            t = i / steps
            stamp(shift(frame_a, int(round(dx * t)), int(round(dy * t))))
    else:
        light = _lighter_map(frame_a)
        for i in range(copies, 0, -1):
            t = i / (copies + 1)
            ghost = shift(frame_a, int(round(dx * t)), int(round(dy * t)))
            m = ghost[..., 3] > 0
            for y, x in zip(*np.nonzero(m)):
                ghost[y, x, :3] = light[tuple(int(v) for v in ghost[y, x, :3])]
            stamp(ghost)
    stamp(frame_b)
    return out


def slash_arc(size: tuple[int, int], center, radius: float, start_deg: float, end_deg: float,
              thickness: float, colors: list, frames: int = 3) -> list[np.ndarray]:
    """Weapon-trail arc frames: a full bright crescent, then the body colour
    fading from the start of the sweep, then a thin tail at its end. Angles in degrees, 0 = right,
    90 = down (screen coordinates); the sweep goes from start to end.

    colors: [core (brightest), body, tail]; fewer colours are reused.
    The crescent is thickest in the middle of its sweep and tapers to a
    point at both ends.
    """
    if frames < 1:
        raise ValueError("frames must be >= 1")
    w, h = size
    cx, cy = center
    cols = [tuple(c[:3]) for c in colors] or [(255, 255, 255)]
    while len(cols) < 3:
        cols.append(cols[-1])
    sweep = end_deg - start_deg
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    ang = np.degrees(np.arctan2(yy + 0.5 - cy, xx + 0.5 - cx))
    out = []
    for f in range(frames):
        # Frame 1 is the full crescent; later frames vanish from the start
        # of the sweep and get thinner, leaving a tail at the weapon's end.
        head = 1.0
        tail = 0.0 if frames == 1 else 0.7 * f / (frames - 1)
        thick = thickness * (1.0 - 0.45 * f / max(1, frames - 1))
        rel = ((ang - start_deg) % 360) / abs(sweep) if sweep > 0 else ((start_deg - ang) % 360) / abs(sweep)
        inside = (rel >= tail) & (rel <= head)
        span = max(head - tail, 1e-6)
        u = np.clip((rel - tail) / span, 0, 1)
        local = thick * np.sin(np.pi * u)
        band = inside & (dist <= radius) & (dist >= radius - local)
        img = np.zeros((h, w, 4), np.uint8)
        color = cols[0] if f == 0 else cols[1] if f < frames - 1 else cols[2]
        img[band] = (*color, 255)
        if f == 0 and frames > 1:
            core = inside & (dist <= radius) & (dist >= radius - local * 0.45)
            img[core] = (*cols[0], 255)
            outer = band & ~core
            img[outer] = (*cols[1], 255)
        out.append(img)
    return out
