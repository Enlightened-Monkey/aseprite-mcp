"""Build RGBA test images from ASCII art."""
import numpy as np

PALETTE = {
    ".": (0, 0, 0, 0),
    "#": (40, 24, 48, 255),     # outline
    "f": (200, 80, 80, 255),    # fill
    "a": (120, 50, 60, 255),    # mid / AA
    "b": (160, 64, 70, 255),
    "w": (250, 240, 220, 255),
    "k": (0, 0, 0, 255),
}


def img(art: str, palette=PALETTE) -> np.ndarray:
    rows = [r for r in art.strip("\n").splitlines()]
    width = max(len(r) for r in rows)
    out = np.zeros((len(rows), width, 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            out[y, x] = palette[ch]
    return out


def checks(issues, name):
    return [i for i in issues if i["check"] == name]
