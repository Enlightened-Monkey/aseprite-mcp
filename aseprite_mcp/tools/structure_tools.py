"""Structure tools: find line art and other pixel roles, split or recolour
line art."""
import io
import json

import numpy as np
from mcp.server.fastmcp import Image

from .. import mcp
from ..core import pixels
from ..pixelart import cleanup as CL
from ..pixelart import structure as ST
from ..pixelart.color import hex_to_rgb
from .pixelart import _load_any


def _png(im) -> Image:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


def _side_by_side(img: np.ndarray, roles_img: np.ndarray):
    from PIL import Image as PILImage
    from ..pixelart.render import _on_checker, _up
    h, w = img.shape[:2]
    s = max(1, min(12, 360 // max(h, w)))
    left = _up(_on_checker(img, 1), s)
    right = _up(_on_checker(roles_img, 1), s)
    sheet = PILImage.new("RGB", (left.width * 2 + 24, left.height + 16), (255, 255, 255))
    sheet.paste(left, (8, 8))
    sheet.paste(right, (16 + left.width, 8))
    return sheet


@mcp.tool()
async def analyze_structure(
    filename: str,
    frame_index: int = 1,
    layer_name: str = "",
    remove_background: bool = True,
):
    """Find the line art and the role of every pixel in a sprite.

    Line art is detected by shape - 1-2 px strips clearly darker than both
    sides, forming strokes - so black, coloured and shaded outlines are all
    found, and dark shading bands are not mistaken for lines (inside the
    sprite a line's colour must be one used mostly as thin strips).
    Roles: outline (line art on the silhouette), inner_line, aa, base,
    shadow, highlight (per part), dither and detail (tiny marks: eyes,
    shines). Parts are same-hue regions separated by lines.

    Reports: whether there is an outline and how much of the silhouette it
    covers (with gap count), line width (1 px / mixed / thicker), share of
    inner line art, line colours, the parts with their base colours, and
    pixel counts per role. The image shows the sprite next to a role map:
    red outline, orange inner lines, yellow AA, light/mid/dark blue
    highlight/base/shadow, purple dither, green detail.
    Upscaled or resized input is brought back to its pixel grid first.

    Args:
        filename: .aseprite file or image (png, gif, jpg, bmp, webp)
        frame_index: Frame, starting at 1
        layer_name: Only this layer (Aseprite files)
        remove_background: For opaque images, make a colour shared by all
            four corners transparent (default true)
    """
    try:
        img, note = _load_any(filename, frame_index, layer_name, remove_background)
        grid = CL.estimate_grid(img)
        if grid["scale_x"] > 1:
            img = CL.resample_grid(CL.binarize_alpha(img)[0], grid)
        res = ST.analyze(img)
    except ValueError as e:
        return f"Failed to analyse structure: {e}"
    summary = res["summary"]
    summary["outline_gap_pixels"] = res["gaps"][:40]
    if grid["scale_x"] > 1:
        summary["note"] = f"Input was upscaled ~{grid['scale_x']}x; analysed at its native pixel grid."
    if note:
        summary["background"] = note
    return [json.dumps(summary), _png(_side_by_side(img, ST.role_image(img, res)))]


@mcp.tool()
async def extract_lineart(
    filename: str,
    layer_name: str,
    target_layer: str = "lineart",
    frame_index: int = 1,
    include_aa: bool = False,
    remove_from_source: bool = True,
) -> str:
    """Move the detected line art of a layer onto its own layer.

    The classic split: line art on top, flats and shading underneath. With
    remove_from_source (default), the line pixels on the source layer are
    filled with the neighbouring colours, so the colour layer works without
    lines (handy to recolour lines, try a lineless version, or edit shapes
    and lines separately). include_aa also moves the AA pixels that hug the
    lines.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer holding the art
        target_layer: Layer for the line art (created when missing)
        frame_index: Frame, starting at 1
        include_aa: Move line AA too (default false)
        remove_from_source: Fill the lines' place on the source (default true)
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
        lineart, rest, res = ST.split_lineart(img, include_aa)
        if target_layer not in pixels.list_layers(filename):
            pixels.add_layer(filename, target_layer)
        pixels.write_image(filename, target_layer, frame_index, lineart)
        if remove_from_source:
            pixels.write_image(filename, layer_name, frame_index, rest)
    except ValueError as e:
        return f"Failed to extract line art: {e}"
    return json.dumps({"line_pixels": int((lineart[..., 3] > 0).sum()), "target_layer": target_layer,
                       "source_filled": remove_from_source,
                       "outline_coverage": res["summary"]["outline_coverage"]})


@mcp.tool()
async def recolor_lineart(
    filename: str,
    layer_name: str,
    mode: str = "colored",
    color: str = "",
    which: str = "all",
    frame_index: int = 1,
) -> str:
    """Recolour the detected line art of a layer.

    - colored: every line pixel takes the darkest shade of the part it
      borders - coloured line art, softer and more modern than black
    - tinted: one tinted near-black matched to the sprite instead of pure black
    - single: one colour you choose
    which limits it to the outline (silhouette) or the inner lines, e.g.
    keep a dark outline for readability but colour the inner lines.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer holding the art
        mode: colored (default), tinted or single
        color: Hex colour for single mode
        which: all (default), outline or inner
        frame_index: Frame, starting at 1
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
        out, changed = ST.recolor_lineart(img, mode, hex_to_rgb(color) if color else None, which)
        if changed:
            pixels.write_image(filename, layer_name, frame_index, out)
    except ValueError as e:
        return f"Failed to recolour line art: {e}"
    return json.dumps({"pixels_recoloured": changed, "mode": mode, "which": which})
