"""Segmentation of pixel art into elements: edges, regions, objects, background.

Classical, dependency-free algorithms tuned for palette images (few exact
colours, hard edges), plus helpers to bring AI masks (from models that see
an upscaled, smoothed world) back onto the pixel grid:

- edge_strength: perceptual colour difference across every pixel border
- felzenszwalb: graph-based region segmentation (merges regions whose inner
  variation exceeds the contrast of the border between them)
- merge_regions: regions -> objects, by weak shared borders
- background_mask: what belongs to the backdrop (touches the frame, low
  saliency against the border colours)
- snap_labels: re-assign each small same-colour cluster to one label, so
  label borders follow the art's own colour borders
"""
from collections import Counter, defaultdict

import numpy as np

from . import lint as L
from .color import oklab


def lab_image(img: np.ndarray) -> np.ndarray:
    return oklab(img[..., :3]) * 100


def edge_strength(img: np.ndarray) -> np.ndarray:
    """Per-pixel edge strength: the largest OKLab distance (x100) to a
    4-neighbour; transparent/opaque borders count as strong edges."""
    lab = lab_image(img)
    opaque = img[..., 3] > 0
    h, w = opaque.shape
    out = np.zeros((h, w))
    for dy, dx in ((0, 1), (1, 0)):
        a = lab[: h - dy, : w - dx]
        b = lab[dy:, dx:]
        d = np.linalg.norm(a - b, axis=-1)
        alpha_edge = opaque[: h - dy, : w - dx] != opaque[dy:, dx:]
        d = np.where(alpha_edge, 100.0, d)
        out[: h - dy, : w - dx] = np.maximum(out[: h - dy, : w - dx], d)
        out[dy:, dx:] = np.maximum(out[dy:, dx:], d)
    return out


def edge_map(img: np.ndarray, low: float = 8.0, high: float = 20.0) -> np.ndarray:
    """Binary edges with hysteresis: strong edges, plus weaker ones that
    connect to a strong edge (like Canny, on colour distance)."""
    s = edge_strength(img)
    strong = s >= high
    weak = s >= low
    out = strong.copy()
    h, w = s.shape
    stack = list(zip(*np.nonzero(strong)))
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and weak[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    stack.append((ny, nx))
    return out


class _DSU:
    def __init__(self, n):
        self.p = np.arange(n)
        self.size = np.ones(n, dtype=np.int64)
        self.inner = np.zeros(n)  # largest edge inside the component

    def find(self, a):
        p = self.p
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:
            p[a], a = root, p[a]
        return root

    def union(self, a, b, w):
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.p[b] = a
        self.size[a] += self.size[b]
        self.inner[a] = max(self.inner[a], self.inner[b], w)
        return a


def felzenszwalb(img: np.ndarray, k: float = 60.0, min_size: int = 12, conn8: bool = True) -> np.ndarray:
    """Graph-based segmentation (Felzenszwalb & Huttenlocher) on OKLab.

    Pixels are graph nodes, neighbours are joined by edges weighted by
    colour distance. Edges are processed from weakest; two regions merge when
    the edge is no stronger than each region's own variation plus k/size.
    Larger k gives larger regions. Regions under min_size pixels are merged
    into their most similar neighbour. Transparent pixels get label -1.
    """
    h, w = img.shape[:2]
    lab = lab_image(img)
    opaque = (img[..., 3] > 0).ravel()
    idx = np.arange(h * w).reshape(h, w)
    offs = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if conn8 else [])
    us, vs, ws = [], [], []
    for dy, dx in offs:
        ys = slice(0, h - dy)
        if dx >= 0:
            xa, xb = slice(0, w - dx), slice(dx, w)
        else:
            xa, xb = slice(-dx, w), slice(0, w + dx)
        a = idx[ys, xa].ravel()
        b = idx[dy:, xb].ravel()
        d = np.linalg.norm(lab[ys, xa] - lab[dy:, xb], axis=-1).ravel()
        if dx and dy:
            d = d * 1.2  # diagonal neighbours are a little further apart
        keep = opaque[a] & opaque[b]
        us.append(a[keep]); vs.append(b[keep]); ws.append(d[keep])
    u = np.concatenate(us); v = np.concatenate(vs); wgt = np.concatenate(ws)
    order = np.argsort(wgt, kind="stable")
    dsu = _DSU(h * w)
    for e in order:
        a, b = dsu.find(u[e]), dsu.find(v[e])
        if a == b:
            continue
        we = wgt[e]
        if we <= min(dsu.inner[a] + k / dsu.size[a], dsu.inner[b] + k / dsu.size[b]):
            dsu.union(a, b, we)
    for e in order:  # small regions join their closest neighbour
        a, b = dsu.find(u[e]), dsu.find(v[e])
        if a != b and (dsu.size[a] < min_size or dsu.size[b] < min_size):
            dsu.union(a, b, wgt[e])
    roots = np.array([dsu.find(i) for i in range(h * w)])
    labels = np.full(h * w, -1)
    _, compact = np.unique(roots[opaque], return_inverse=True)
    labels[opaque] = compact
    return labels.reshape(h, w)


def region_stats(img: np.ndarray, labels: np.ndarray) -> dict:
    """Per label: pixel count, mean OKLab colour, bounding box, frame contact."""
    lab = lab_image(img)
    h, w = labels.shape
    stats = {}
    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        m = labels == lbl
        ys, xs = np.nonzero(m)
        stats[int(lbl)] = {
            "pixels": int(m.sum()), "mean": lab[m].mean(axis=0),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "frame": int(m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum()),
        }
    return stats


def boundaries(img: np.ndarray, labels: np.ndarray) -> dict:
    """{(a, b): [count, summed colour distance]} over shared label borders."""
    lab = lab_image(img)
    h, w = labels.shape
    out = defaultdict(lambda: [0, 0.0])
    for dy, dx in ((0, 1), (1, 0)):
        a = labels[: h - dy, : w - dx]
        b = labels[dy:, dx:]
        d = np.linalg.norm(lab[: h - dy, : w - dx] - lab[dy:, dx:], axis=-1)
        m = (a != b) & (a >= 0) & (b >= 0)
        for la, lb, dd in zip(a[m].tolist(), b[m].tolist(), d[m].tolist()):
            key = (la, lb) if la < lb else (lb, la)
            out[key][0] += 1
            out[key][1] += dd
    return out


def merge_regions(img: np.ndarray, labels: np.ndarray, border_delta: float = 12.0,
                  colour_delta: float = 25.0, max_objects: int = 0) -> np.ndarray:
    """Merge regions into objects: neighbours whose shared border is weak
    (mean colour jump < border_delta) and whose mean colours are not far
    apart (< colour_delta) belong together. With max_objects, keep merging
    the weakest pairs until at most that many remain.

    Region statistics and borders are computed once and merged as regions
    join, so this stays fast with hundreds of regions.
    """
    lab = lab_image(img)
    ids = [int(v) for v in np.unique(labels) if v >= 0]
    total = {i: lab[labels == i].sum(axis=0) for i in ids}
    count = {i: int((labels == i).sum()) for i in ids}
    bnd = {k: list(v) for k, v in boundaries(img, labels).items()}
    nbrs = defaultdict(set)
    for a, b in bnd:
        nbrs[a].add(b)
        nbrs[b].add(a)
    parent = {i: i for i in ids}
    alive = set(ids)

    def mean(i):
        return total[i] / count[i]

    while len(alive) > 1:
        best = None
        for (a, b), (n, s) in bnd.items():
            jump = s / n
            colour = float(np.linalg.norm(mean(a) - mean(b)))
            if (jump < border_delta and colour < colour_delta) or (max_objects and len(alive) > max_objects):
                score = jump + 0.3 * colour
                if best is None or score < best[0]:
                    best = (score, a, b)
        if best is None:
            break
        _, a, b = best
        keep, drop = (a, b) if count[a] >= count[b] else (b, a)
        parent[drop] = keep
        total[keep] = total[keep] + total.pop(drop)
        count[keep] += count.pop(drop)
        alive.discard(drop)
        # Re-key the dropped region's borders onto the kept one.
        for c in nbrs.pop(drop):
            key = (min(drop, c), max(drop, c))
            n, s = bnd.pop(key)
            nbrs[c].discard(drop)
            if c == keep:
                continue
            nk = (min(keep, c), max(keep, c))
            if nk in bnd:
                bnd[nk][0] += n
                bnd[nk][1] += s
            else:
                bnd[nk] = [n, s]
                nbrs[keep].add(c)
                nbrs[c].add(keep)

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    out = np.full(labels.shape, -1)
    order = {r: n for n, r in enumerate(sorted(alive))}
    for i in ids:
        out[labels == i] = order[root(i)]
    return out


def background_mask(img: np.ndarray, labels: np.ndarray, frame_share: float = 0.1) -> np.ndarray:
    """Regions forming the backdrop: they touch the frame substantially and
    their colour is close to the frame's colours (low saliency), or they are
    transparent. Everything else is foreground."""
    h, w = labels.shape
    lab = lab_image(img)
    frame = np.zeros((h, w), dtype=bool)
    frame[0] = frame[-1] = True
    frame[:, 0] = frame[:, -1] = True
    border_cols = lab[frame & (labels >= 0)]
    stats = region_stats(img, labels)
    perimeter = 2 * (h + w)
    bg = labels < 0
    for lbl, st in stats.items():
        touches = st["frame"] / perimeter
        if not len(border_cols):
            break
        dist = float(np.min(np.linalg.norm(border_cols - st["mean"], axis=-1)))
        if touches >= frame_share or (touches > 0 and dist < 8):
            bg |= labels == lbl
    return bg


def snap_labels(img: np.ndarray, labels: np.ndarray, max_cluster: int = 64) -> np.ndarray:
    """Make label borders follow the art's colour borders.

    Each same-colour 4-connected cluster of at most max_cluster pixels is
    given to the label that covers most of it. AI masks are drawn on a
    smoothed, upscaled image and cut through clusters; pixel art objects
    are made of whole clusters.
    """
    keys = L._keys(img)
    out = labels.copy()
    h, w = keys.shape
    seen = np.zeros((h, w), dtype=bool)
    for y, x in zip(*np.nonzero(keys != -1)):
        if seen[y, x]:
            continue
        k = keys[y, x]
        stack, comp = [(y, x)], []
        seen[y, x] = True
        while stack and len(comp) <= max_cluster:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and keys[ny, nx] == k:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if stack or len(comp) > max_cluster:
            continue  # big region: keep the per-pixel labels
        votes = Counter(int(labels[p]) for p in comp)
        winner = votes.most_common(1)[0][0]
        for p in comp:
            out[p] = winner
    return out


def region_vote(labels: np.ndarray, regions: np.ndarray, majority: float = 0.25,
                max_share: float = 0.5, keep: set | None = None) -> np.ndarray:
    """Give fine colour regions to the element that covers enough of them.

    A region joins the label covering most of it when that label covers at
    least `majority` of the region's pixels, the region is not bigger than
    `max_share` of the element (a few stray mask pixels must not claim a
    large backdrop region) and it does not run along the frame (backdrop).
    Fills holes in AI masks and aligns borders to real colour edges. Labels
    in `keep` are never spread (e.g. particles).
    """
    out = labels.copy()
    h, w = labels.shape
    sizes = Counter(labels[labels >= 0].tolist())
    frame = np.zeros((h, w), dtype=bool)
    frame[0] = frame[-1] = True
    frame[:, 0] = frame[:, -1] = True
    for r in np.unique(regions):
        if r < 0:
            continue
        m = regions == r
        total = int(m.sum())
        vals = labels[m]
        vals = vals[vals >= 0]
        if keep:
            vals = vals[~np.isin(vals, list(keep))]
        if not len(vals):
            continue
        winner, n = Counter(vals.tolist()).most_common(1)[0]
        on_frame = int((m & frame).sum())
        if n >= majority * total and total <= max_share * sizes[winner] and on_frame <= 0.05 * total:
            target = m if not keep else m & ~np.isin(labels, list(keep))
            out[target] = winner
    return out


def particle_mask(img: np.ndarray, max_blob: int = 4, repeats: int = L.PARTICLE_REPEATS) -> np.ndarray:
    """Particles: small one-colour blobs (up to max_blob px, 8-connected)
    sitting on a plain surround (at most two colours around them), whose
    colour forms such blobs many times - snow flakes, stars, rain drops,
    sparks. Textured details (rocks, fur) have busier surroundings."""
    keys = L._keys(img)
    h, w = keys.shape
    seen = np.zeros((h, w), dtype=bool)
    blobs = defaultdict(list)
    for y, x in zip(*np.nonzero(keys != -1)):
        if seen[y, x]:
            continue
        k = keys[y, x]
        stack, comp = [(y, x)], []
        seen[y, x] = True
        while stack and len(comp) <= max_blob:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and keys[ny, nx] == k:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if stack or len(comp) > max_blob:
            continue
        # The blob must stand out: lighter or darker than everything around it.
        ring = set()
        for cy, cx in comp:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and keys[ny, nx] != k:
                        ring.add(int(keys[ny, nx]))
        if ring and -1 not in ring and len(ring) <= 2:
            blobs[int(k)].append(comp)
    out = np.zeros((h, w), dtype=bool)
    for k, comps in blobs.items():
        if len(comps) >= repeats:
            for comp in comps:
                for p in comp:
                    out[p] = True
    return out


def resolve_by_regions(masks: list[np.ndarray], regions: np.ndarray, own: float = 0.5,
                       backdrop: float = 0.25) -> np.ndarray:
    """Turn overlapping masks into labels region by region.

    A fine colour region goes whole to a mask covering at least `own` of
    it (the smallest such mask when several do: a tusk inside the body's
    mask), so where masks overlap the one that really owns the region wins. A large region on the frame
    that no mask covers at least `backdrop` of is backdrop (-1) even if
    masks reach into it (sky enclosed by a curled tusk). Other pixels fall
    back to "smaller mask wins".
    """
    h, w = regions.shape
    labels = masks_to_labels(masks, (h, w))
    sizes = [int(mk.sum()) for mk in masks]
    frame = np.zeros((h, w), dtype=bool)
    frame[0] = frame[-1] = True
    frame[:, 0] = frame[:, -1] = True
    for r in np.unique(regions):
        if r < 0:
            continue
        m = regions == r
        total = int(m.sum())
        cover = [int((mk & m).sum()) / total for mk in masks]
        best = int(np.argmax(cover)) if cover else -1
        owners = [i for i, c in enumerate(cover) if c >= own]
        if owners:
            # Several masks own it (a tusk inside the mammoth's mask): the
            # smaller mask is the detail on top.
            labels[m] = min(owners, key=lambda i: sizes[i])
        elif (m & frame).any() and total > 0.05 * h * w and (not cover or cover[best] < backdrop):
            labels[m] = -1
    return labels


def masks_to_labels(masks: list[np.ndarray], shape, min_pixels: int = 4) -> np.ndarray:
    """Combine possibly overlapping boolean masks into one label image:
    smaller masks win (they are the details on top of larger ones)."""
    labels = np.full(shape, -1)
    for i, m in sorted(enumerate(masks), key=lambda im: -int(im[1].sum())):
        if m.sum() >= min_pixels:
            labels[m] = i
    return labels


def downsample_mask(mask_hr: np.ndarray, scale: int) -> np.ndarray:
    """High-resolution mask -> pixel grid by majority vote per pixel."""
    h, w = mask_hr.shape[0] // scale, mask_hr.shape[1] // scale
    m = mask_hr[: h * scale, : w * scale].reshape(h, scale, w, scale)
    return m.mean(axis=(1, 3)) >= 0.5


# -------------------------------------------------------------- rendering

def label_colours(n: int) -> list[tuple[int, int, int]]:
    """n well separated, saturated colours (golden-angle hues)."""
    import colorsys
    out = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        sat = 0.65 + 0.3 * ((i // 7) % 2)
        val = 0.95 - 0.25 * ((i // 3) % 2)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def render_labels(labels: np.ndarray, colours=None) -> np.ndarray:
    n = int(labels.max()) + 1 if labels.size else 0
    cols = colours or label_colours(max(n, 1))
    out = np.zeros((*labels.shape, 4), np.uint8)
    for i in range(n):
        out[labels == i] = (*cols[i % len(cols)], 255)
    return out


def render_overlay(img: np.ndarray, labels: np.ndarray, alpha: float = 0.55, colours=None) -> np.ndarray:
    flat = render_labels(labels, colours).astype(np.float64)
    base = img.astype(np.float64)
    out = base.copy()
    m = labels >= 0
    out[m, :3] = base[m, :3] * (1 - alpha) + flat[m, :3] * alpha
    out[..., 3] = np.maximum(img[..., 3], np.where(m, 255, 0))
    return out.astype(np.uint8)


def render_contours(img: np.ndarray, labels: np.ndarray, colour=(255, 0, 255)) -> np.ndarray:
    out = img.copy()
    h, w = labels.shape
    edge = np.zeros((h, w), dtype=bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    out[edge] = (*colour, 255)
    return out
