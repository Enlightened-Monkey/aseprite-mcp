"""Images for the agent to look at: lint overlays and readability sheets."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .color import lightness

CHECKER = ((236, 236, 236), (204, 204, 204))
MARK_COLORS = {
    "orphans": (255, 0, 255),
    "l_corners": (255, 40, 40),
    "jaggies": (255, 140, 0),
    "banding": (0, 160, 255),
    "pillow_shading": (255, 220, 0),
    "near_duplicates": (0, 220, 120),
    "dither_small_sprite": (160, 80, 255),
    "aa_outside_outline": (0, 255, 255),
}


def display_scale(w: int, h: int, target: int = 192) -> int:
    return max(1, min(16, target // max(w, h, 1)))


def _on_checker(img: np.ndarray, cell: int) -> Image.Image:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.where((((xx // cell) + (yy // cell)) % 2)[..., None] == 0, CHECKER[0], CHECKER[1])
    a = img[..., 3:4].astype(np.float64) / 255
    rgb = img[..., :3] * a + bg * (1 - a)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def _on_color(img: np.ndarray, color) -> Image.Image:
    a = img[..., 3:4].astype(np.float64) / 255
    rgb = img[..., :3] * a + np.array(color[:3]) * (1 - a)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def _up(im: Image.Image, s: int) -> Image.Image:
    return im.resize((im.width * s, im.height * s), Image.NEAREST) if s > 1 else im


def lint_overlay(img: np.ndarray, issues: list[dict], scale: int | None = None) -> Image.Image:
    """Upscaled sprite with a coloured square around each flagged pixel."""
    h, w = img.shape[:2]
    s = scale or max(1, min(12, 768 // max(w, h, 1)))
    base = _up(_on_checker(img, 1), s)
    draw = ImageDraw.Draw(base)
    for issue in issues:
        color = MARK_COLORS.get(issue["check"], (255, 0, 0))
        for x, y in issue["pixels"]:
            draw.rectangle([x * s, y * s, x * s + s - 1, y * s + s - 1], outline=color, width=max(1, s // 6))
    return base


def readability_sheet(img: np.ndarray, backgrounds=("#E8E8E8", "#808080", "#181820")) -> Image.Image:
    """Contact sheet of views that expose readability problems.

    1x and 2x (true size), zoomed, blurred (as seen from afar), silhouette,
    value-only (lightness), and the sprite on light / mid / dark backgrounds.
    """
    h, w = img.shape[:2]
    s = display_scale(w, h)
    panels = []
    base = _on_checker(img, 4)
    panels.append(("1x", base))
    if max(w, h) <= 256:
        panels.append(("2x", _up(base, 2)))
    if s > 2:
        panels.append((f"{s}x", _up(base, s)))
    blurred = _on_color(img, (128, 128, 128)).filter(ImageFilter.GaussianBlur(radius=1))
    panels.append(("blurred", _up(blurred, s)))
    sil = np.where(img[..., 3:4] > 0, 0, 255).repeat(3, axis=2).astype(np.uint8)
    panels.append(("silhouette", _up(Image.fromarray(sil, "RGB"), s)))
    L = (np.clip(lightness(img[..., :3]), 0, 1) * 255).astype(np.uint8)
    val = np.where(img[..., 3] > 0, L, 255)
    panels.append(("value", _up(Image.fromarray(np.stack([val] * 3, -1), "RGB"), s)))
    for bg in backgrounds:
        rgb = tuple(int(bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        panels.append((f"on {bg}", _up(_on_color(img, rgb), s)))

    pad, label_h = 8, 14
    cols = 3
    cell_w = max(p.width for _, p in panels) + pad
    rows_of = [panels[i:i + cols] for i in range(0, len(panels), cols)]
    heights = [max(p.height for _, p in r) + label_h + pad for r in rows_of]
    sheet = Image.new("RGB", (cols * cell_w + pad, sum(heights) + pad), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    y = pad
    for r, rh in zip(rows_of, heights):
        x = pad
        for label, p in r:
            draw.text((x, y), label, fill=(0, 0, 0))
            sheet.paste(p, (x, y + label_h))
            x += cell_w
        y += rh
    return sheet


def before_after(before: np.ndarray, after: np.ndarray, changed: list | None = None) -> Image.Image:
    """Side-by-side comparison at a common display size, changed pixels boxed."""
    target = 384
    sb = max(1, target // max(before.shape[:2]))
    sa = max(1, target // max(after.shape[:2]))
    left = _up(_on_checker(before, 4 if sb < 4 else 1), sb)
    right = _up(_on_checker(after, 1), sa)
    if changed:
        draw = ImageDraw.Draw(right)
        for x, y in changed:
            draw.rectangle([x * sa, y * sa, x * sa + sa - 1, y * sa + sa - 1], outline=(255, 0, 255),
                           width=max(1, sa // 6))
    pad, label_h = 8, 14
    sheet = Image.new("RGB", (left.width + right.width + 3 * pad, max(left.height, right.height) + label_h + 2 * pad),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((pad, pad), f"before {before.shape[1]}x{before.shape[0]}", fill=(0, 0, 0))
    d.text((2 * pad + left.width, pad), f"after {after.shape[1]}x{after.shape[0]}", fill=(0, 0, 0))
    sheet.paste(left, (pad, pad + label_h))
    sheet.paste(right, (2 * pad + left.width, pad + label_h))
    return sheet
