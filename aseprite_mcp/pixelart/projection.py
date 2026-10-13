"""Game projections: isometric (2:1), 45-degree dimetric, oblique, top-down.

Pixel isometric uses 2:1 lines (two pixels across, one up) instead of true
30 degrees. Tiles are W wide and W/2 tall; a lineless filled diamond has
2-pixel tips and grows 2 px per side per row, so diamonds in two colours
tile like a checkerboard without chunky outline crossings.
"""
import numpy as np

from . import lint as L

# Pixels across per pixel up/down for the receding edges of each projection.
RATIOS = {"iso": 2, "dimetric45": 1, "oblique": 1, "oblique2": 2}


def _put(img, x, y, rgb):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        img[y, x] = (*rgb[:3], 255)


def projection_line(x: int, y: int, length: int, direction: str, ratio: int = 2) -> list[tuple[int, int]]:
    """A receding edge with exact runs of `ratio` pixels per 1 px step.

    direction: ne (up-right), nw (up-left), se (down-right), sw (down-left).
    length counts pixels along x.
    """
    if direction not in ("ne", "nw", "se", "sw"):
        raise ValueError("direction must be ne, nw, se or sw")
    sx = 1 if direction in ("ne", "se") else -1
    sy = -1 if direction in ("ne", "nw") else 1
    return [(x + sx * i, y + sy * (i // ratio)) for i in range(length)]


def diamond_rows(width: int) -> list[tuple[int, int]]:
    """(x_start, run) per row of a lineless iso diamond in a W x W/2 cell.

    Tips are 2 px; each row outward adds 2 px per side. Width must be a
    multiple of 4.
    """
    if width < 4 or width % 4:
        raise ValueError("tile width must be a multiple of 4 (e.g. 16, 32, 64)")
    half = width // 4
    top = [(width // 2 - (2 * r + 1), 4 * r + 2) for r in range(half)]
    return top + top[::-1]


def iso_to_screen(col: float, row: float, tile_w: int, origin=(0, 0), height: float = 0):
    """Top-left corner of the W x W/2 cell of tile (col, row), lifted by
    `height` px. Columns go down-right, rows go down-left."""
    ox, oy = origin
    return ox + (col - row) * tile_w // 2, oy + (col + row) * tile_w // 4 - height


def iso_floor(img: np.ndarray, cols: int, rows: int, tile_w: int, colors: list, origin=(0, 0)) -> int:
    """Checkerboard of lineless iso tiles; origin is the top-left corner of
    tile (0, 0)'s cell. Returns pixels drawn."""
    rows_spec = diamond_rows(tile_w)
    n = 0
    for r in range(rows):
        for c in range(cols):
            sx, sy = iso_to_screen(c, r, tile_w, origin)
            rgb = colors[(c + r) % len(colors)]
            for dy, (x0, run) in enumerate(rows_spec):
                for dx in range(run):
                    _put(img, sx + x0 + dx, sy + dy, rgb)
                    n += 1
    return n


def iso_box(img: np.ndarray, x: int, y: int, right: int, left: int, height: int,
            top_rgb, left_rgb, right_rgb, ratio: int = 2) -> dict:
    """Draw a box standing on the ground, anchored at its front-bottom corner.

    (x, y) is the right one of the two bottom-tip pixels (tips are 2 px); `right` and `left` are
    the footprint lengths in steps along each receding axis (each step =
    `ratio` px across, 1 px up), height in px. Faces: top, left, right; pick
    tones so the face toward the light is lighter (typically top lightest).
    ratio 2 = isometric, 1 = 45-degree dimetric.
    Returns the face pixel counts.
    """
    counts = {"top": 0, "left": 0, "right": 0}
    if min(right, left, height) < 1:
        raise ValueError("right, left and height must be >= 1")
    wr, wl = right * ratio, left * ratio

    def front_right(i):  # bottom y of column x + i
        return y - (i + 1) // ratio

    def front_left(i):   # bottom y of column x - 1 - i
        return y - (i + 1) // ratio

    lower, upper = {}, {}
    for i in range(wr):
        lower[x + i] = front_right(i)
    for i in range(wl):
        lower[x - 1 - i] = front_left(i)
    # Back edges are the front edges moved along the other axis.
    for i in range(wr):
        upper[x - wl + i] = front_right(i) - left
    for i in range(wl):
        upper[x + wr - 1 - i] = front_left(i) - right
    # Side faces hang below the front edges.
    for i in range(wr):
        for k in range(height):
            _put(img, x + i, front_right(i) - k, right_rgb)
            counts["right"] += 1
    for i in range(wl):
        for k in range(height):
            _put(img, x - 1 - i, front_left(i) - k, left_rgb)
            counts["left"] += 1
    # Top face: the footprint lifted by the height.
    for cx, lo in lower.items():
        for yy in range(upper[cx] - height + 1, lo - height + 1):
            _put(img, cx, yy, top_rgb)
            counts["top"] += 1
    return counts


def oblique_box(img: np.ndarray, x: int, y: int, w: int, h: int, depth: int, front_rgb, top_rgb, side_rgb,
                direction: str = "ne", ratio: int = 1) -> dict:
    """Oblique box: true-shape front rectangle (top-left at x, y) with the top
    and one side receding at 1:1 (or 2:1) toward `direction` (ne or nw)."""
    if direction not in ("ne", "nw"):
        raise ValueError("direction must be ne or nw")
    if min(w, h, depth) < 1:
        raise ValueError("w, h and depth must be >= 1")
    sx = 1 if direction == "ne" else -1
    counts = {"front": 0, "top": 0, "side": 0}
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            _put(img, xx, yy, front_rgb)
            counts["front"] += 1
    for d in range(1, depth + 1):
        off_x, off_y = sx * d * ratio, -d
        for k in range(ratio):
            # top: the front's top edge shifted back
            for xx in range(x, x + w):
                _put(img, xx + off_x - sx * k, y + off_y, top_rgb)
            # side: the front's side edge shifted back
            edge = x + w - 1 if sx > 0 else x
            for yy in range(y, y + h):
                _put(img, edge + off_x - sx * k, yy + off_y, side_rgb)
        counts["top"] += w * ratio
        counts["side"] += h * ratio
    return counts


def topdown_block(img: np.ndarray, x: int, y: int, w: int, top_h: int, front_h: int, top_rgb, front_rgb) -> None:
    """Top-down (3/4 view) block: a top face over a front face. Keep the
    top:front ratio the same for every block in a scene (one camera)."""
    for yy in range(y, y + top_h):
        for xx in range(x, x + w):
            _put(img, xx, yy, top_rgb)
    for yy in range(y + top_h, y + top_h + front_h):
        for xx in range(x, x + w):
            _put(img, xx, yy, front_rgb)


def skew_face(texture: np.ndarray, face: str, scale_x: float = 0.866) -> np.ndarray:
    """Map a flat, front-facing texture onto an iso face (nearest neighbour).

    left / right: the texture is narrowed to ~87% (cos 30 deg) and sheared
    2:1 so it lies on the left or right wall. top: the texture is turned
    45 degrees and squashed to half height (a 2:1 diamond floor/roof).
    Clean up afterwards; the result is a starting point.
    """
    h, w = texture.shape[:2]
    if face in ("left", "right"):
        ow = max(1, int(round(w * scale_x)))
        rise = (ow - 1) // 2
        out = np.zeros((h + rise, ow, 4), np.uint8)
        for u in range(ow):
            sx = min(w - 1, int(u / scale_x))
            shift_y = (u // 2) if face == "left" else (rise - u // 2)
            for yy in range(h):
                out[yy + shift_y, u] = texture[yy, sx]
        return out
    if face == "top":
        n = max(w, h)
        out = np.zeros((n, 2 * n, 4), np.uint8)
        for v in range(n):
            for u in range(2 * n):
                # Inverse of u = x - y + n, v = (x + y) / 2.
                fx = (u - n + 2 * v) / 2
                fy = (2 * v - (u - n)) / 2
                ix, iy = int(np.floor(fx)), int(np.floor(fy))
                if 0 <= ix < w and 0 <= iy < h:
                    out[v, u] = texture[iy, ix]
        return out
    raise ValueError("face must be left, right or top")


def grid(img: np.ndarray, projection: str, cell: int, rgb) -> int:
    """Guide grid for a projection: square cells (side, topdown, top,
    oblique) or diagonal lines at the projection's slope (iso, dimetric45)."""
    h, w = img.shape[:2]
    n = 0
    if projection in ("side", "topdown", "top", "oblique"):
        for yy in range(0, h, cell):
            for xx in range(w):
                _put(img, xx, yy, rgb)
                n += 1
        for xx in range(0, w, cell):
            for yy in range(h):
                _put(img, xx, yy, rgb)
                n += 1
        return n
    if projection not in ("iso", "dimetric45"):
        raise ValueError("projection must be side, topdown, top, oblique, iso or dimetric45")
    ratio = RATIOS[projection]
    step = cell // ratio if projection == "iso" else cell
    for start in range(-w, h + w, max(1, step)):
        for xx in range(w):
            for yy in (start + xx // ratio, start - xx // ratio + w // ratio):
                if 0 <= yy < h:
                    _put(img, xx, yy, rgb)
                    n += 1
    return n


def check_lines(img: np.ndarray, projection: str, line_colors=None) -> list[dict]:
    """Receding edges that break the projection's rhythm.

    Traces line-colour strokes (as the linter does) and looks at straight
    diagonal stretches of 3+ steps: in iso every step must be exactly 2 px
    (2:1), in 45-degree projections exactly 1 px.
    """
    if projection not in ("iso", "dimetric45", "oblique"):
        raise ValueError("projection must be iso, dimetric45 or oblique")
    want = 2 if projection == "iso" else 1
    keys = L._keys(img)
    bad = []
    for key, mask, _ in L.line_masks(img, line_colors):
        for chain in L._thin_chains(mask):
            runs = L.chain_runs(chain)
            # Split into straight stretches: same axis family and the same
            # diagonal direction between runs. Runs at the ends of a stretch
            # are corners/tips and may be shorter.
            stretch = []

            def flush():
                inner = stretch[1:-1]
                if len(inner) >= 2:
                    lens = [r["len"] for r in inner]
                    if any(v != want for v in lens) and sum(v == want for v in lens) >= len(lens) / 2:
                        for r in inner:
                            if r["len"] != want:
                                bad.extend(r["pixels"])

            prev_dir = None
            for k, r in enumerate(runs):
                d = None
                if k + 1 < len(runs) and not runs[k + 1]["break"]:
                    a, b = r["pixels"][-1], runs[k + 1]["pixels"][0]
                    d = (b[0] - a[0], b[1] - a[1])
                horizontal = r["axis"] in ("h", None) and r["len"] <= 4
                if r["break"] or not horizontal or (d is not None and prev_dir is not None and d != prev_dir):
                    if horizontal and not r["break"] and d is not None and prev_dir is not None and d != prev_dir:
                        stretch.append(r)  # the turning run closes this stretch
                        flush()
                        stretch = [r]
                    else:
                        flush()
                        stretch = [r] if horizontal else []
                else:
                    stretch.append(r)
                if d is not None:
                    prev_dir = d
            flush()
    if not bad:
        return []
    return [{"check": f"{projection}_lines", "severity": "warning",
             "message": f"{len(bad)} pixel(s) on receding edges break the {want}:1 rhythm of the {projection} "
                        f"projection. Every step of a receding edge must be exactly {want} px.",
             "pixels": [[int(x), int(y)] for x, y in bad]}]
