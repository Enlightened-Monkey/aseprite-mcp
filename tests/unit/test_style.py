import numpy as np

from aseprite_mcp.pixelart.style import analyze_style

DARK = (40, 24, 48, 255)
BLACK = (0, 0, 0, 255)


def _disc(size=24, r=10, outline=None, shades=((90, 40, 50), (200, 80, 80), (240, 160, 140)),
          light=(-1, -1)):
    """A shaded disc lit from `light` (dx, dy), optionally outlined."""
    img = np.zeros((size, size, 4), np.uint8)
    c = (size - 1) / 2
    for y in range(size):
        for x in range(size):
            d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
            if d > r:
                continue
            t = ((x - c) * light[0] + (y - c) * light[1]) / r  # >0 toward the light
            s = shades[0] if t < -0.2 else shades[1] if t < 0.4 else shades[2]
            img[y, x] = (*s, 255)
            if outline is not None and d > r - 1:
                img[y, x] = outline
    return img


def test_black_contour_and_light_direction():
    p = analyze_style(_disc(outline=DARK))
    assert p["outline"]["style"] == "black_contour"
    assert p["lighting"]["from"] == "top-left"
    assert p["shading"] == "cel"
    assert p["antialiasing"]["level"] in ("none", "light")


def test_pure_black_outline_is_marked_as_style():
    p = analyze_style(_disc(outline=BLACK))
    assert p["outline"]["pure_black"]
    assert any("Pure black is part of this style" in r["rule"] for r in p["rules"])


def test_no_outline():
    p = analyze_style(_disc())
    assert p["outline"]["style"] == "none"
    assert any(r["trait"] == "outline: none" for r in p["rules"])


def test_flat_sprite():
    img = np.zeros((20, 20, 4), np.uint8)
    img[4:16, 4:16] = (200, 80, 80, 255)
    img[8:12, 8:12] = (60, 120, 200, 255)
    p = analyze_style(img)
    assert p["shading"] == "flat"
    assert p["palette"]["class"] == "1-bit"


def test_light_from_right():
    p = analyze_style(_disc(light=(1, 0)))
    assert p["lighting"]["from"] == "right"


def test_dithered_and_filtered():
    img = np.zeros((64, 64, 4), np.uint8)
    for y in range(64):
        for x in range(64):
            img[y, x] = (200, 80, 80, 255) if (x + y) % 2 else (120, 50, 60, 255)
    p = analyze_style(img)
    assert p["dithering"]["level"] == "heavy"
    assert "dither" not in p["lint"]["checks"]   # large, deliberately dithered

    blurred = np.zeros((32, 32, 4), np.uint8)
    blurred[..., :3] = np.arange(32 * 32 * 3).reshape(32, 32, 3) % 256
    blurred[..., 3] = 255
    p = analyze_style(blurred)
    assert p["cleanliness"]["filtered"]
    assert p["rules"][0]["topic"] == "cleanup"
