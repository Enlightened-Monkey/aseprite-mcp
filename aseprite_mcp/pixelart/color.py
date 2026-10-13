"""Colour maths for pixel art: HSV ramps with hue shifting, OKLab distances.

Conventions: RGB components are ints 0-255; HSV is (h in degrees [0,360),
s in 0-100, v in 0-100). Perceptual distances use OKLab scaled by 100, so a
difference of ~2 is barely visible and ~10 is clearly different.
"""
import colorsys
import math

import numpy as np

# Shadows lean toward violet, highlights toward yellow, along the shorter arc.
DARK_POLE = 270.0
LIGHT_POLE = 60.0


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    h = value.strip().lstrip("#")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    if len(h) not in (6, 8):
        raise ValueError(f"Invalid color value: {value}")
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        raise ValueError(f"Invalid color value: {value}") from None


def rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(c) for c in rgb[:3]))


def rgb_to_hsv(rgb) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb[:3]))
    return h * 360, s * 100, v * 100


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    s = min(100.0, max(0.0, s))
    v = min(100.0, max(0.0, v))
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, s / 100, v / 100)
    return round(r * 255), round(g * 255), round(b * 255)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(rgb) -> np.ndarray:
    """Convert RGB (..., 3) ints 0-255 to OKLab (..., 3), L in 0-1."""
    c = _srgb_to_linear(np.asarray(rgb, dtype=np.float64)[..., :3] / 255.0)
    lms = c @ np.array([
        [0.4122214708, 0.2119034982, 0.0883024619],
        [0.5363325363, 0.6806995451, 0.2817188376],
        [0.0514459929, 0.1073969566, 0.6299787005],
    ])
    lms = np.cbrt(lms)
    return lms @ np.array([
        [0.2104542553, 1.9779984951, 0.0259040371],
        [0.7936177850, -2.4285922050, 0.7827717662],
        [-0.0040720468, 0.4505937099, -0.8086757660],
    ])


def lightness(rgb) -> np.ndarray:
    """Perceived lightness (OKLab L, 0-1)."""
    return oklab(rgb)[..., 0]


def delta_e(rgb_a, rgb_b) -> float:
    """Perceptual colour distance (OKLab euclidean x 100)."""
    return float(np.linalg.norm(oklab(rgb_a) - oklab(rgb_b)) * 100)


def hue_delta(a: float, b: float) -> float:
    """Signed shortest rotation from hue a to hue b, in (-180, 180]."""
    d = (b - a) % 360
    return d - 360 if d > 180 else d


def rotate_toward(h: float, pole: float, degrees: float) -> float:
    """Rotate hue h toward pole by up to `degrees`, never overshooting it."""
    d = hue_delta(h, pole)
    step = min(abs(d), degrees)
    return (h + math.copysign(step, d)) % 360


def generate_ramp(
    base_rgb,
    steps: int = 5,
    base_index: int | None = None,
    hue_shift_per_step: float = 16.0,
    darkest_value: float = 20.0,
    lightest_value: float = 97.0,
    saturation_curve: str = "arc",
) -> list[dict]:
    """Build a dark-to-light shading ramp around a base colour.

    Shadows rotate toward violet and highlights toward yellow (shortest arc),
    value moves roughly linearly, and saturation follows a curve:
    "arc" keeps the mids richest, dulls the darkest step a little and washes
    out the top toward a tinted near-white; "flat" keeps it constant;
    "sat_shadows" / "sat_highlights" push saturation to one end.

    Returns a list of {hex, h, s, v} ordered darkest to lightest; the base
    colour sits at base_index unchanged.
    """
    if not 2 <= steps <= 16:
        raise ValueError("steps must be between 2 and 16")
    if base_index is None:
        base_index = (steps - 1) // 2
    if not 0 <= base_index < steps:
        raise ValueError("base_index must be within the ramp")
    if saturation_curve not in ("arc", "flat", "sat_shadows", "sat_highlights"):
        raise ValueError("saturation_curve must be arc, flat, sat_shadows or sat_highlights")

    h0, s0, v0 = rgb_to_hsv(base_rgb)
    n_dark = base_index
    n_light = steps - 1 - base_index
    vstep_d = max(6.0, (v0 - darkest_value) / n_dark) if n_dark else 0.0
    vstep_l = max(5.0, (lightest_value - v0) / n_light) if n_light else 0.0

    def sat(k: int, n: int, dark: bool) -> float:
        t = k / n
        if saturation_curve == "flat":
            f = 1.0
        elif saturation_curve == "arc":
            f = 1 - 0.15 * t if dark else 1 - 0.5 * t ** 1.5
        elif saturation_curve == "sat_shadows":
            f = 1 + 0.15 * t if dark else 1 - 0.6 * t
        else:  # sat_highlights
            f = 1 - 0.35 * t if dark else 1 + 0.1 * t
        return s0 * f

    out = []
    for k in range(n_dark, 0, -1):
        h = rotate_toward(h0, DARK_POLE, k * hue_shift_per_step)
        out.append((h, sat(k, n_dark, True), max(4.0, v0 - k * vstep_d)))
    out.append((h0, s0, v0))
    for k in range(1, n_light + 1):
        h = rotate_toward(h0, LIGHT_POLE, k * hue_shift_per_step * 0.8)
        out.append((h, sat(k, n_light, False), min(100.0, v0 + k * vstep_l)))

    ramp = []
    for i, (h, s, v) in enumerate(out):
        rgb = tuple(base_rgb[:3]) if i == base_index else hsv_to_rgb(h, s, v)
        hh, ss, vv = rgb_to_hsv(rgb)
        ramp.append({"hex": rgb_to_hex(rgb), "h": round(hh), "s": round(ss), "v": round(vv)})
    return ramp


def shade(base_rgb, direction: str = "shadow", steps: int = 1) -> tuple[int, int, int]:
    """One (or more) shading steps away from a colour.

    shadow: hue -25 deg toward violet, saturation x0.8, value -22 per step.
    highlight: hue -15 deg toward yellow, saturation x0.85, value +15 per step.
    """
    h, s, v = rgb_to_hsv(base_rgb)
    for _ in range(steps):
        if direction == "shadow":
            h, s, v = rotate_toward(h, DARK_POLE, 25), s * 0.8, max(4.0, v - 22)
        elif direction == "highlight":
            h, s, v = rotate_toward(h, LIGHT_POLE, 15), s * 0.85, min(100.0, v + 15)
        else:
            raise ValueError("direction must be 'shadow' or 'highlight'")
    return hsv_to_rgb(h, s, v)


def suggest_outline(colors_with_counts: list[tuple[tuple[int, int, int], int]]) -> tuple[int, int, int]:
    """Tinted near-black outline colour that harmonises with a sprite.

    Takes the darkest quarter of the sprite's colours (by lightness), averages
    their hue weighted by usage and saturation, nudges it toward violet and
    returns a dark, moderately saturated colour (V ~16, S ~45). Falls back to
    a plum near-black for greyscale input.
    """
    if not colors_with_counts:
        return hsv_to_rgb(300, 35, 16)
    ordered = sorted(colors_with_counts, key=lambda c: float(lightness(c[0])))
    dark = ordered[: max(1, len(ordered) // 4)]
    x = y = 0.0
    for rgb, count in dark:
        h, s, _ = rgb_to_hsv(rgb)
        w = count * s
        x += w * math.cos(math.radians(h))
        y += w * math.sin(math.radians(h))
    if math.hypot(x, y) < 1e-6:
        return hsv_to_rgb(300, 35, 16)
    hue = math.degrees(math.atan2(y, x)) % 360
    return hsv_to_rgb(rotate_toward(hue, DARK_POLE, 20), 45, 16)


def palette_budget(sprite_height: int) -> dict:
    """Rule-of-thumb colour budget for a sprite of the given height (px)."""
    if sprite_height <= 16:
        return {"max_colors": 8, "steps_per_ramp": "2-3"}
    if sprite_height <= 32:
        return {"max_colors": 12, "steps_per_ramp": "3-4"}
    if sprite_height <= 64:
        return {"max_colors": 16, "steps_per_ramp": "4-5"}
    return {"max_colors": 32, "steps_per_ramp": "5-7"}
