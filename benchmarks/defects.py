"""Defect benchmark: clean pixel-art bases + injected, known defects.

Each injector returns (corrupted, mask) where mask marks exactly the pixels
it changed, so detection (did the linter flag them?) and repair (did
cleanup restore them?) can be measured against ground truth.
"""
import numpy as np
from PIL import Image

from aseprite_mcp.pixelart import lint as L

INK = (34, 24, 46)          # outline
FILL = [(160, 52, 66), (214, 86, 82), (246, 166, 128)]   # shadow, base, light
FILL2 = [(52, 88, 150), (78, 132, 200), (150, 196, 236)]
EYE = (250, 244, 230)
N4 = ((0, -1), (-1, 0), (1, 0), (0, 1))
N8 = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


# ------------------------------------------------------------------ bases

def _shade(mask: np.ndarray, ramp, light=(-1, -1)) -> np.ndarray:
    """Cel-shade a mask: 3 tones by position along the light direction."""
    img = np.zeros((*mask.shape, 4), np.uint8)
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    r = max(np.ptp(xs), np.ptp(ys)) / 2 + 1e-9
    for y, x in zip(ys, xs):
        t = ((x - cx) * light[0] + (y - cy) * light[1]) / r
        img[y, x] = (*ramp[0 if t < -0.25 else 1 if t < 0.45 else 2], 255)
    return img


def _outline_inside(img: np.ndarray, mask: np.ndarray, colour=INK) -> np.ndarray:
    """1-px pixel-perfect outline on the shape's own edge pixels."""
    h, w = mask.shape
    pad = np.pad(mask, 1)
    edge = np.zeros_like(mask)
    for dx, dy in N4:
        edge |= ~pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    edge &= mask
    out = img.copy()
    for x, y in L.find_l_corners(edge):
        edge[y, x] = False
        # A corner pixel on the silhouette becomes air, so the line steps
        # diagonally instead of leaving a stray fill pixel outside it.
        if any(not (0 <= x + dx < w and 0 <= y + dy < h) or not mask[y + dy, x + dx] for dx, dy in N4):
            out[y, x] = 0
    out[edge] = (*colour, 255)
    return out


def _disc(h, w, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _polygon(h, w, pts):
    from PIL import ImageDraw
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.array(im).astype(bool)


def base_orb() -> np.ndarray:
    m = _disc(36, 36, 17.5, 17.5, 13.5, 13.5)
    return _outline_inside(_shade(m, FILL), m)


def base_shapes() -> np.ndarray:
    # Edges at 1:1, 2:1 and 3:1 plus straight sides.
    pts = [(4, 34), (4, 22), (16, 10), (40, 4), (52, 4), (52, 14), (40, 26), (22, 34)]
    m = _polygon(40, 58, pts)
    return _outline_inside(_shade(m, FILL2), m)


def base_bean() -> np.ndarray:
    h, w = 48, 40
    head = _disc(h, w, 19.5, 15.5, 11.5, 10.5)
    body = _disc(h, w, 19.5, 34.5, 13.5, 11.5) & ~head
    img = _shade(body, FILL2)
    img[head] = _shade(head, FILL)[head]
    img = _outline_inside(img, head | body)
    # Inner line between head and body (pixel-perfect: one row of the head edge).
    pad = np.pad(body, 1)
    seam = head & (pad[2:, 1:-1])
    img[seam] = (*INK, 255)
    for ex in (14, 23):
        img[13:15, ex:ex + 2] = (*EYE, 255)
        img[14, ex + 1] = (*INK, 255)
    return img


BASES = {"orb": base_orb, "shapes": base_shapes, "bean": base_bean}


# ---------------------------------------------------------------- helpers

def _ink_mask(img):
    return (img[..., :3] == INK).all(-1) & (img[..., 3] > 0)


def _opaque(img):
    return img[..., 3] > 0


def _pick(rng, cands, n):
    cands = list(cands)
    if not cands:
        return []
    idx = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [cands[i] for i in sorted(idx)]


def _spread(rng, cands, n, min_dist=4):
    """Pick up to n candidates at least min_dist apart."""
    cands = list(cands)
    rng.shuffle(cands)
    out = []
    for c in cands:
        if all(abs(c[0] - o[0]) + abs(c[1] - o[1]) >= min_dist for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out


def _diff_mask(a, b):
    return (a != b).any(-1)


def _mix(a, b, t=0.5):
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b))


def _inner_neighbour(img, ink, x, y):
    h, w = ink.shape
    for dx, dy in N4:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and img[ny, nx, 3] and not ink[ny, nx]:
            return nx, ny
    return None


# --------------------------------------------------------------- injectors

def l_corners(img, rng, n=6):
    """Doubled corner pixels on the outline (pixel-perfect broken)."""
    out = img.copy()
    ink = _ink_mask(img)
    h, w = ink.shape
    cands = []
    at = lambda x, y: 0 <= x < w and 0 <= y < h and ink[y, x]
    for y in range(h):
        for x in range(w):
            if ink[y, x]:
                continue
            # P becomes the doubled corner of a diagonal step H-V; the cell
            # diagonal across from P must stay empty (no filled 2x2).
            if any(at(x + hx, y) and at(x, y + vy) and not at(x + hx, y + vy)
                   for hx in (-1, 1) for vy in (-1, 1)):
                cands.append((x, y))
    for x, y in _spread(rng, set(cands), n):
        out[y, x] = (*INK, 255)
    return out, _diff_mask(img, out)


def gaps(img, rng, n=4):
    """Holes in the outline: line pixels replaced by what is inside."""
    out = img.copy()
    ink = _ink_mask(img)
    for x, y in _spread(rng, zip(*np.nonzero(ink.T)), n, 6):
        nb = _inner_neighbour(img, ink, x, y)
        out[y, x] = img[nb[1], nb[0]] if nb else 0
    return out, _diff_mask(img, out)


def thick_line(img, rng, n=3, length=4):
    """Outline doubled into 2-px blobs over short stretches."""
    out = img.copy()
    ink = _ink_mask(img)
    starts = _spread(rng, zip(*np.nonzero(ink.T)), n, 10)
    for sx, sy in starts:
        stack, seen = [(sx, sy)], set()
        while stack and len(seen) < length:
            x, y = stack.pop()
            if (x, y) in seen:
                continue
            seen.add((x, y))
            nb = _inner_neighbour(img, ink, x, y)
            if nb:
                out[nb[1], nb[0]] = (*INK, 255)
            for dx, dy in N8:
                if 0 <= x + dx < ink.shape[1] and 0 <= y + dy < ink.shape[0] and ink[y + dy, x + dx]:
                    stack.append((x + dx, y + dy))
    return out, _diff_mask(img, out)


def stray_pixels(img, rng, n_float=4, n_inside=4):
    """Floating pixels in transparency and contrasting specks inside."""
    out = img.copy()
    op = _opaque(img)
    h, w = op.shape
    pad = np.pad(op, 3)
    near = np.zeros_like(op)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            near |= pad[3 + dy:3 + dy + h, 3 + dx:3 + dx + w]
    air = [(x, y) for y, x in zip(*np.nonzero(near & ~op))
           if not any(0 <= x + dx < w and 0 <= y + dy < h and op[y + dy, x + dx] for dx, dy in N8)]
    for x, y in _spread(rng, air, n_float):
        out[y, x] = (*FILL[1], 255)
    ink = _ink_mask(img)
    inside = [(x, y) for y, x in zip(*np.nonzero(op & ~ink))
              if all(0 <= x + dx < w and 0 <= y + dy < h and op[y + dy, x + dx] and not ink[y + dy, x + dx]
                     for dx, dy in N8)]
    for x, y in _spread(rng, inside, n_inside, 6):
        c = img[y, x, :3].astype(int)
        out[y, x, :3] = np.clip(c + rng.choice([-1, 1]) * 26, 0, 255)   # a slightly off shade
    return out, _diff_mask(img, out)


def jaggies(img, rng, n=3):
    """Break the staircase rhythm: one line pixel moved one step sideways."""
    out = img.copy()
    ink = _ink_mask(img)
    h, w = ink.shape
    chains = L._thin_chains(ink)
    cands = []
    for ch in chains:
        runs = L.chain_runs(ch)
        for k in range(1, len(runs) - 1):
            r = runs[k]
            if r["len"] >= 2 and not r["break"] and r["axis"] in ("h", "v"):
                cands.append(r["pixels"][-1])
    for x, y in _spread(rng, cands, n, 8):
        nb = _inner_neighbour(img, ink, x, y)
        if not nb:
            continue
        # The run's end pixel drops one step inward: the run shortens and
        # a one-pixel step appears.
        out[y, x] = img[nb[1], nb[0]]
        out[nb[1], nb[0]] = (*INK, 255)
    return out, _diff_mask(img, out)


def full_step_aa(img, rng, length=14):
    """AA tone hugging the outline step for step along an arc (banding)."""
    out = img.copy()
    ink = _ink_mask(img)
    chain = max(L._thin_chains(ink), key=len)
    start = int(rng.integers(0, max(1, len(chain) - length)))
    for x, y in chain[start:start + length]:
        nb = _inner_neighbour(img, ink, x, y)
        if nb:
            out[nb[1], nb[0], :3] = _mix(INK, img[nb[1], nb[0], :3], 0.55)
    return out, _diff_mask(img, out)


def outer_aa(img, rng, n=6):
    """Light AA pixels outside the outline, against transparency."""
    out = img.copy()
    ink = _ink_mask(img)
    op = _opaque(img)
    h, w = op.shape
    cands = [(x, y) for y in range(h) for x in range(w) if not op[y, x]
             and sum(1 for dx, dy in N4 if 0 <= x + dx < w and 0 <= y + dy < h and ink[y + dy, x + dx]) >= 2]
    for x, y in _spread(rng, cands, n):
        out[y, x] = (*_mix(INK, FILL[2], 0.6), 255)
    return out, _diff_mask(img, out)


def mixed_blacks(img, rng, share=0.4):
    """Outline drawn with three near-identical darks instead of one."""
    out = img.copy()
    alts = [(26, 20, 40), (44, 30, 54)]
    for y, x in zip(*np.nonzero(_ink_mask(img))):
        if rng.random() < share:
            out[y, x, :3] = alts[int(rng.integers(0, 2))]
    return out, _diff_mask(img, out)


def pillow(img, rng):
    """Reshade the inside by distance from the edge (no light direction)."""
    out = img.copy()
    ink = _ink_mask(img)
    inner = _opaque(img) & ~ink
    ramp = [(160, 52, 66), (200, 76, 78), (232, 120, 104), (246, 166, 128)]
    depth = np.zeros(inner.shape, int)
    cur = inner.copy()
    d = 0
    while cur.any():
        depth[cur] = d
        pad = np.pad(cur, 1)
        cur = cur & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        d += 1
    for y, x in zip(*np.nonzero(inner)):
        out[y, x, :3] = ramp[min(3, depth[y, x] // 2)]
    return out, _diff_mask(img, out)


def dither_small(img, rng):
    """Checker dither between two shades on a small sprite."""
    out = img.copy()
    inner = _opaque(img) & ~_ink_mask(img)
    ys, xs = np.nonzero(inner)
    cy = int(np.median(ys))
    for y, x in zip(ys, xs):
        if abs(y - cy) <= 4 and (x + y) % 2:
            out[y, x, :3] = FILL[0] if tuple(img[y, x, :3]) != FILL[0] else FILL[1]
    return out, _diff_mask(img, out)


def soft_alpha(img, rng):
    """Semi-transparent fringe around the sprite (exported with smoothing)."""
    out = img.copy()
    op = _opaque(img)
    h, w = op.shape
    for y in range(h):
        for x in range(w):
            if not op[y, x] and any(0 <= x + dx < w and 0 <= y + dy < h and op[y + dy, x + dx] for dx, dy in N4):
                out[y, x] = (*INK, 70)
    edge = op & np.array([[any(not (0 <= x + dx < w and 0 <= y + dy < h) or not op[y + dy, x + dx]
                               for dx, dy in N4) for x in range(w)] for y in range(h)])
    out[edge, 3] = 200
    return out, _diff_mask(img, out)


def upscale_blur(img, rng, factor=3.3):
    """Resized by a non-integer factor with bilinear blur, plus colour noise."""
    h, w = img.shape[:2]
    im = Image.fromarray(img, "RGBA").resize((round(w * factor), round(h * factor)), Image.BILINEAR)
    a = np.array(im).astype(int)
    a[..., :3] += rng.integers(-5, 6, a[..., :3].shape)
    return np.clip(a, 0, 255).astype(np.uint8), None


def colour_noise(img, rng, amount=7):
    """Every opaque pixel nudged by a few RGB levels (lossy export)."""
    out = img.astype(int)
    op = _opaque(img)
    out[op, :3] += rng.integers(-amount, amount + 1, (int(op.sum()), 3))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, _diff_mask(img, out)


DEFECTS = {
    "l_corners": (l_corners, ["l_corners"]),
    "jaggies": (jaggies, ["jaggies", "l_corners"]),
    "gaps": (gaps, ["outline_gaps"]),
    "thick_line": (thick_line, ["line_width"]),
    "stray_pixels": (stray_pixels, ["orphans"]),
    "full_step_aa": (full_step_aa, ["aa_band", "banding"]),
    "outer_aa": (outer_aa, ["aa_outside_outline"]),
    "mixed_blacks": (mixed_blacks, ["near_duplicates"]),
    "pillow": (pillow, ["pillow_shading"]),
    "dither_small": (dither_small, ["dither_small_sprite"]),
    "soft_alpha": (soft_alpha, []),
    "upscale_blur": (upscale_blur, []),
    "colour_noise": (colour_noise, ["near_duplicates"]),
}
