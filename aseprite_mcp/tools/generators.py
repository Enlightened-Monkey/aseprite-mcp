"""Drawing generators that follow pixel-art rules by construction.

Each tool reads the target layer's pixels, runs an algorithm from
aseprite_mcp.pixelart.draw and writes the layer back (the layer is created
when missing).
"""
import json

import numpy as np

from .. import mcp
from ..core import pixels
from ..pixelart import cleanup as CL
from ..pixelart import draw as D
from ..pixelart import lint as L
from ..pixelart.color import hex_to_rgb, rgb_to_hex


def _layer_image(filename: str, layer: str, frame: int) -> np.ndarray:
    if layer not in pixels.list_layers(filename):
        pixels.add_layer(filename, layer)
    return pixels.read_image(filename, frame, layer)


def _colors(values) -> list[tuple[int, int, int]]:
    if isinstance(values, str):
        values = [v for v in values.replace(";", ",").split(",") if v.strip()]
    return [hex_to_rgb(v) for v in values]


def _paint(img: np.ndarray, pts, rgb) -> list:
    done = []
    for x, y in pts:
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x] = (*rgb, 255)
            done.append((x, y))
    return done


@mcp.tool()
async def draw_clean_line(
    filename: str,
    layer_name: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str,
    frame_index: int = 1,
    mode: str = "exact",
) -> str:
    """Draw a 1-px line with a clean pixel-art staircase.

    mode "exact": hits both end points; steps are as equal as the slope
    allows (one or two odd steps go to the ends, otherwise short and long
    steps alternate evenly). mode "snap": snaps to the nearest clean slope
    (1:1, 2:1, 3:1 ... 6:1) so every step is identical; the end point may
    move and is reported. Never produces L-corners. Prefer snap for
    straight edges of objects (crates, buildings, blades).

    Args:
        filename: Aseprite file to modify
        layer_name: Layer to draw on (created when missing)
        x1, y1, x2, y2: End points (sprite coordinates)
        color: Hex colour
        frame_index: Frame, starting at 1
        mode: "exact" (default) or "snap"
    """
    try:
        rgb = hex_to_rgb(color)
        if mode == "exact":
            pts, end = D.line_points(x1, y1, x2, y2), (x2, y2)
        elif mode == "snap":
            pts, end = D.snapped_line(x1, y1, x2, y2)
        else:
            return "mode must be 'exact' or 'snap'"
        img = _layer_image(filename, layer_name, frame_index)
        done = _paint(img, pts, rgb)
        pixels.write_image(filename, layer_name, frame_index, img)
    except ValueError as e:
        return f"Failed to draw line: {e}"
    return json.dumps({"pixels": len(done), "end": list(end)})


@mcp.tool()
async def draw_smooth_curve(
    filename: str,
    layer_name: str,
    points: list,
    color: str,
    frame_index: int = 1,
) -> str:
    """Draw a smooth 1-px curve through Bezier control points.

    The curve starts at the first point and ends at the last; the points in
    between pull it (3 points = quadratic, 4 = cubic). The result is
    pixel-perfect (no L-corners) and its step lengths change gradually, as a
    clean pixel-art curve should. Describe a curve by a few control points
    instead of placing pixels by hand; for longer shapes chain several
    curves end to end.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer to draw on (created when missing)
        points: Control points as [[x, y], ...] (2 or more)
        color: Hex colour
        frame_index: Frame, starting at 1
    """
    try:
        rgb = hex_to_rgb(color)
        control = [(float(p[0]), float(p[1])) for p in points]
        pts = D.curve_points(control)
        img = _layer_image(filename, layer_name, frame_index)
        done = _paint(img, pts, rgb)
        pixels.write_image(filename, layer_name, frame_index, img)
    except (ValueError, TypeError, IndexError) as e:
        return f"Failed to draw curve: {e}"
    return json.dumps({"pixels": len(done)})


@mcp.tool()
async def antialias_lines(
    filename: str,
    layer_name: str,
    line_color: str,
    frame_index: int = 1,
    shades: int = 1,
    coverage: float = 0.5,
    use_palette: bool = True,
) -> str:
    """Anti-alias the staircase steps of a line colour, the pixel-art way.

    Puts intermediate colours in the concave corners of every step, running
    alongside the neighbouring step for about half its length (coverage),
    darker next to the corner when 2 shades are used. AA goes only onto
    pixels inside the art, never onto transparency, and 45-degree parts are
    left crisp. Colours are mixed between the line and what they replace and
    snapped to existing palette colours when close. Skip AA on tiny sprites
    and crisp styles (see pixel_art_guide("antialiasing")).

    Args:
        filename: Aseprite file to modify
        layer_name: Layer holding the lines (and the fill next to them)
        line_color: Hex colour of the lines to smooth
        frame_index: Frame, starting at 1
        shades: 1 or 2 AA shades
        coverage: Fraction of each step covered by AA (0.25-0.75, default 0.5)
        use_palette: Reuse existing colours when one is close (default true)
    """
    if not 0.1 <= coverage <= 0.9:
        return "coverage must be between 0.1 and 0.9"
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
        out, changed, new = D.antialias_edges(img, hex_to_rgb(line_color), shades, coverage, use_palette)
        if changed:
            pixels.write_image(filename, layer_name, frame_index, out)
    except ValueError as e:
        return f"Failed to anti-alias: {e}"
    return json.dumps({"pixels_changed": len(changed), "new_colors": [rgb_to_hex(c) for c in new]})


@mcp.tool()
async def outline_sprite(
    filename: str,
    layer_name: str,
    style: str = "black",
    color: str = "",
    light: str = "top-left",
    placement: str = "outside",
    corners: bool = False,
    frame_index: int = 1,
    output_layer: str = "",
) -> str:
    """Outline the opaque shape of a layer in a chosen pixel-art style.

    Styles:
    - black: one dark colour all round; default is a tinted near-black
      matched to the sprite (not pure black), or pass color
    - colored: each stretch uses the darkest shade of the part it wraps
    - shaded: colored on the side facing the light, dark on the shadow side
    - selective (sel-out): like shaded but left out where it faces the
      light; reads best on dark backgrounds
    placement "outside" adds a 1-px ring around the shape; "inside" recolours
    the shape's own edge. corners=true also fills diagonal corner pixels
    (thicker look). With output_layer, only the outline is written there.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer whose shape is outlined
        style: black, colored, shaded or selective
        color: Hex colour for the dark outline (default: auto tinted dark)
        light: Direction the light comes from: top-left, top, top-right,
            right, bottom-right, bottom, bottom-left, left
        placement: "outside" (default) or "inside"
        corners: Include diagonal corners (default false)
        frame_index: Frame, starting at 1
        output_layer: Write the outline to this layer instead (created when missing)
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
        out, changed, info = D.outline(img, style, hex_to_rgb(color) if color else None, light,
                                       placement, corners)
        if output_layer:
            only = np.zeros_like(out)
            for x, y in changed:
                only[y, x] = out[y, x]
            base = _layer_image(filename, output_layer, frame_index)
            mask = only[..., 3] > 0
            base[mask] = only[mask]
            pixels.write_image(filename, output_layer, frame_index, base)
        else:
            pixels.write_image(filename, layer_name, frame_index, out)
    except ValueError as e:
        return f"Failed to outline: {e}"
    return json.dumps({"pixels": len(changed), "outline_color": rgb_to_hex(info["dark"]),
                       "layer": output_layer or layer_name})


@mcp.tool()
async def dither_fill(
    filename: str,
    layer_name: str,
    x: int,
    y: int,
    width: int,
    height: int,
    colors: list,
    direction: str = "vertical",
    levels: int = 0,
    mask: str = "none",
    only_color: str = "",
    frame_index: int = 1,
) -> str:
    """Fill an area with a dithered gradient through a colour ramp.

    Mixes only neighbouring ramp colours with ordered (Bayer) patterns that
    are aligned to the canvas, so separate fills tile seamlessly. Default
    levels use the clean 4x4 density ladder; levels=N gives N steps per
    colour pair (2 = hard bands, no dither). Use for skies, large static
    backgrounds and big surfaces; avoid on small or animated sprites.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer to paint (created when missing)
        x, y, width, height: Area to fill
        colors: Ramp as hex colours, first to last, e.g. ["#1A1C2C", "#3B5DC9", "#73EFF7"]
        direction: vertical (first colour at the top), horizontal (left) or radial (centre)
        levels: 0 = clean dither ladder (default), N = N steps per colour pair
        mask: "none" fills every pixel; "opaque" only pixels that already exist
        only_color: Only replace pixels of this hex colour (e.g. a flat sky)
        frame_index: Frame, starting at 1
    """
    try:
        ramp = _colors(colors)
        img = _layer_image(filename, layer_name, frame_index)
        out, n = D.dither_fill(img, (x, y, width, height), ramp, direction, levels, mask,
                               hex_to_rgb(only_color) if only_color else None)
        pixels.write_image(filename, layer_name, frame_index, out)
    except ValueError as e:
        return f"Failed to dither: {e}"
    return json.dumps({"pixels": n})


def _targets(filename: str, layer_name: str, frames: str):
    layers = [layer_name] if layer_name else pixels.list_layers(filename)
    return layers, pixels.parse_frames(filename, frames)


@mcp.tool()
async def reduce_colors(
    filename: str,
    max_colors: int,
    layer_name: str = "",
    frames: str = "all",
) -> str:
    """Reduce the colours used to at most max_colors, consistently.

    Near-identical colours merge first; then the cheapest merges (similar
    colours, little usage) until the target is reached, the rarer colour
    taking the more common one so ramps survive. One shared mapping is
    applied to every chosen layer and frame, so an animation does not
    flicker. Typical targets: 4-8 for 16 px sprites, 12 for 32 px, 16 for
    64 px characters.

    Args:
        filename: Aseprite file to modify
        max_colors: Colour budget (2-256)
        layer_name: Only this layer (default: all layers)
        frames: "all" (default), "3", "2-5" or "1,3,4"
    """
    if not 2 <= max_colors <= 256:
        return "max_colors must be between 2 and 256"
    try:
        layers, frame_list = _targets(filename, layer_name, frames)
        tiles = [(l, f, pixels.read_image(filename, f, l)) for l in layers for f in frame_list]
        stack = np.concatenate([t[2] for t in tiles], axis=0)
        before = len(np.unique(L._keys(stack)[L._keys(stack) != -1]))
        reduced, merges = CL.reduce_colors(stack, max_colors)
        h = tiles[0][2].shape[0]
        written = 0
        for i, (layer, frame, old) in enumerate(tiles):
            new = reduced[i * h:(i + 1) * h]
            if not (new == old).all():
                pixels.write_image(filename, layer, frame, new)
                written += 1
    except ValueError as e:
        return f"Failed to reduce colours: {e}"
    keys = L._keys(reduced)
    return json.dumps({"colors_before": before, "colors_after": len(np.unique(keys[keys != -1])),
                       "merges": merges[:30], "cels_updated": written})


@mcp.tool()
async def gradient_map(
    filename: str,
    layer_name: str,
    colors: list,
    frames: str = "all",
    dither: bool = False,
    normalize: bool = True,
) -> str:
    """Colour a value (greyscale) drawing with a ramp: value first, colour second.

    Each opaque pixel's lightness picks a ramp colour (dark to light). Paint
    the forms in a few greys first - easy to judge light and shadow - then
    map them onto a hue-shifted ramp from generate_color_ramp. normalize
    stretches the layer's own value range over the whole ramp; dither adds
    50% checker steps between neighbouring colours.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer to recolour
        colors: Ramp as hex colours, darkest first
        frames: "all" (default), "3", "2-5" or "1,3,4"
        dither: Add checker in-between steps (default false)
        normalize: Stretch the value range over the ramp (default true)
    """
    try:
        ramp = _colors(colors)
        frame_list = pixels.parse_frames(filename, frames)
        for f in frame_list:
            img = pixels.read_image(filename, f, layer_name)
            pixels.write_image(filename, layer_name, f, D.gradient_map(img, ramp, dither, normalize))
    except ValueError as e:
        return f"Failed to apply gradient map: {e}"
    return json.dumps({"frames": frame_list, "ramp": [rgb_to_hex(c) for c in ramp]})
