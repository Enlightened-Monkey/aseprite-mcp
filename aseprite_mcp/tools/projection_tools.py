"""Projection tools: isometric floors and boxes, oblique and top-down blocks,
projection lines, texture-to-face skewing, guide grids and line checks."""
import json

import numpy as np

from .. import mcp
from ..core import pixels
from ..pixelart import projection as P
from ..pixelart.color import hex_to_rgb, shade


def _layer(filename: str, layer: str, frame: int) -> np.ndarray:
    if layer not in pixels.list_layers(filename):
        pixels.add_layer(filename, layer)
    return pixels.read_image(filename, frame, layer)


def _faces(color: str, top: str, left: str, right: str, light: str):
    """Three face tones: explicit, or derived from one base colour so the
    top is lightest and the side facing the light is lighter."""
    if top and left and right:
        return hex_to_rgb(top), hex_to_rgb(left), hex_to_rgb(right)
    base = hex_to_rgb(color or "#B08A5A")
    lit, dark = shade(base, "highlight", 1), shade(base, "shadow", 1)
    if light in ("top-right", "right"):
        return lit, dark, base
    return lit, base, dark


@mcp.tool()
async def draw_iso_floor(
    filename: str,
    layer_name: str,
    cols: int,
    rows: int,
    tile_width: int = 32,
    colors: list | None = None,
    origin_x: int = 0,
    origin_y: int = 0,
    frame_index: int = 1,
) -> str:
    """Draw an isometric floor of lineless diamond tiles in a checkerboard.

    Tiles are tile_width x tile_width/2 (2:1). Each diamond has 2-px tips
    and grows 2 px per side per row; two alternating colours tile without
    the chunky crossings outlined tiles get. Columns run down-right, rows
    down-left. origin is the top-left corner of tile (0, 0)'s cell; tile
    (c, r) sits at origin + ((c - r) * W/2, (c + r) * W/4).

    Args:
        filename: Aseprite file to modify
        layer_name: Layer to draw on (created when missing)
        cols, rows: Floor size in tiles
        tile_width: Multiple of 4 (16, 32, 64...)
        colors: Two (or more) hex colours to alternate (default two greens)
        origin_x, origin_y: Top-left of tile (0, 0)'s cell
        frame_index: Frame, starting at 1
    """
    try:
        cols_rgb = [hex_to_rgb(c) for c in (colors or ["#466E46", "#5A8750"])]
        img = _layer(filename, layer_name, frame_index)
        n = P.iso_floor(img, cols, rows, tile_width, cols_rgb, (origin_x, origin_y))
        pixels.write_image(filename, layer_name, frame_index, img)
    except ValueError as e:
        return f"Failed to draw floor: {e}"
    return json.dumps({"pixels": n, "tile": [tile_width, tile_width // 2],
                       "place_tile": "x = origin_x + (c - r) * W/2, y = origin_y + (c + r) * W/4"})


@mcp.tool()
async def draw_iso_box(
    filename: str,
    layer_name: str,
    x: int,
    y: int,
    right: int,
    left: int,
    height: int,
    color: str = "",
    top_color: str = "",
    left_color: str = "",
    right_color: str = "",
    light: str = "top-left",
    projection: str = "iso",
    frame_index: int = 1,
) -> str:
    """Draw a box (crate, wall block, building mass) in isometric or 45-degree dimetric.

    Anchored at its bottom front tip: (x, y) is the right pixel of the 2-px
    bottom tip. right / left are the footprint lengths in steps along the
    two receding axes (iso: each step is 2 px across and 1 px up), height in
    px. Faces get three tones: pass top/left/right colours, or one base
    colour and they are derived (top lightest, the side toward the light
    lighter). All receding edges are exact 2:1 (or 1:1 for dimetric45).
    Draw boxes back to front; outline afterwards with outline_sprite.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer (created when missing)
        x, y: Bottom front tip
        right, left: Footprint steps along each axis
        height: Height in px
        color: Base colour to derive the three faces from
        top_color, left_color, right_color: Explicit face colours
        light: top-left (default) or top-right
        projection: iso (2:1) or dimetric45 (1:1)
        frame_index: Frame, starting at 1
    """
    if projection not in ("iso", "dimetric45"):
        return "projection must be iso or dimetric45"
    try:
        top, lft, rgt = _faces(color, top_color, left_color, right_color, light)
        img = _layer(filename, layer_name, frame_index)
        counts = P.iso_box(img, x, y, right, left, height, top, lft, rgt, P.RATIOS[projection])
        pixels.write_image(filename, layer_name, frame_index, img)
    except ValueError as e:
        return f"Failed to draw box: {e}"
    return json.dumps(counts)


@mcp.tool()
async def draw_projection_line(
    filename: str,
    layer_name: str,
    x: int,
    y: int,
    length: int,
    direction: str,
    color: str,
    projection: str = "iso",
    frame_index: int = 1,
) -> str:
    """Draw a receding edge with the exact rhythm of a projection.

    iso: every step 2 px across, 1 px up/down (never mixed 1- or 3-px steps);
    dimetric45 / oblique: 1:1; oblique2: 2:1. Vertical edges stay vertical -
    use draw_clean_line for those.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer (created when missing)
        x, y: Start pixel
        length: Length in pixels along x
        direction: ne, nw, se or sw
        color: Hex colour
        projection: iso, dimetric45, oblique or oblique2
        frame_index: Frame, starting at 1
    """
    if projection not in P.RATIOS:
        return f"projection must be one of {', '.join(P.RATIOS)}"
    try:
        pts = P.projection_line(x, y, length, direction, P.RATIOS[projection])
        img = _layer(filename, layer_name, frame_index)
        rgb = hex_to_rgb(color)
        for px, py in pts:
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                img[py, px] = (*rgb, 255)
        pixels.write_image(filename, layer_name, frame_index, img)
    except ValueError as e:
        return f"Failed to draw line: {e}"
    return json.dumps({"end": list(pts[-1])})


@mcp.tool()
async def skew_to_projection(
    filename: str,
    source_layer: str,
    face: str,
    target_layer: str,
    x: int,
    y: int,
    frame_index: int = 1,
    source_x: int = 0,
    source_y: int = 0,
    source_width: int = 0,
    source_height: int = 0,
) -> str:
    """Turn a flat, front-facing drawing into an isometric wall, floor or roof face.

    Draw a texture flat (bricks, a door, a window, planks), then map it:
    left / right faces are narrowed to ~87% and sheared 2:1 so they lie on
    an iso wall; top is turned 45 degrees and halved in height (a 2:1
    diamond). The result is pasted with its top-left at (x, y) on
    target_layer. Expect a short clean-up pass (cleanup_sprite steps
    "corners,jaggies") for stray pixels.

    Args:
        filename: Aseprite file to modify
        source_layer: Layer holding the flat drawing
        face: left, right or top
        target_layer: Layer to paste the face onto (created when missing)
        x, y: Top-left position of the pasted face
        frame_index: Frame, starting at 1
        source_x, source_y, source_width, source_height: Region of the
            source (default: the drawing's bounding box)
    """
    try:
        src = pixels.read_image(filename, frame_index, source_layer)
        if source_width and source_height:
            tex = src[source_y:source_y + source_height, source_x:source_x + source_width]
        else:
            ys, xs = np.nonzero(src[..., 3] > 0)
            if not len(xs):
                return "The source layer is empty"
            tex = src[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        out = P.skew_face(tex, face)
        img = _layer(filename, target_layer, frame_index)
        h, w = out.shape[:2]
        for dy in range(h):
            for dx in range(w):
                tx, ty = x + dx, y + dy
                if out[dy, dx, 3] and 0 <= tx < img.shape[1] and 0 <= ty < img.shape[0]:
                    img[ty, tx] = out[dy, dx]
        pixels.write_image(filename, target_layer, frame_index, img)
    except ValueError as e:
        return f"Failed to skew: {e}"
    return json.dumps({"face": face, "size": [int(w), int(h)], "at": [x, y]})


@mcp.tool()
async def draw_oblique_box(
    filename: str,
    layer_name: str,
    x: int,
    y: int,
    width: int,
    height: int,
    depth: int,
    color: str = "",
    front_color: str = "",
    top_color: str = "",
    side_color: str = "",
    direction: str = "ne",
    slope: int = 1,
    frame_index: int = 1,
) -> str:
    """Draw an oblique box: true-shape front, top and side receding at 45 degrees.

    The front rectangle keeps its real shape (top-left at x, y); the top and
    one side recede toward direction ne or nw at 1:1 (slope 1) or 2:1
    (slope 2), never converging. Easy and readable for towns, furniture and
    crates in side or RPG views.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer (created when missing)
        x, y: Top-left of the front face
        width, height: Front face size
        depth: Receding depth in steps
        color: Base colour to derive the faces from (front base, top lighter, side darker)
        front_color, top_color, side_color: Explicit face colours
        direction: ne (side on the right) or nw (side on the left)
        slope: 1 (45 deg) or 2 (2:1)
        frame_index: Frame, starting at 1
    """
    try:
        if front_color and top_color and side_color:
            front, top, side = hex_to_rgb(front_color), hex_to_rgb(top_color), hex_to_rgb(side_color)
        else:
            front = hex_to_rgb(color or "#B08A5A")
            top, side = shade(front, "highlight", 1), shade(front, "shadow", 1)
        img = _layer(filename, layer_name, frame_index)
        counts = P.oblique_box(img, x, y, width, height, depth, front, top, side, direction, slope)
        pixels.write_image(filename, layer_name, frame_index, img)
    except ValueError as e:
        return f"Failed to draw oblique box: {e}"
    return json.dumps(counts)


@mcp.tool()
async def draw_topdown_block(
    filename: str,
    layer_name: str,
    x: int,
    y: int,
    width: int,
    top_height: int,
    front_height: int,
    color: str = "",
    top_color: str = "",
    front_color: str = "",
    frame_index: int = 1,
) -> str:
    """Draw a top-down (3/4 view) block: top face over front face.

    The top:front ratio is the camera height - keep it identical for every
    block in a scene (front = top is easiest for tilesets; front > top is a
    low camera, top > front a high one).

    Args:
        filename: Aseprite file to modify
        layer_name: Layer (created when missing)
        x, y: Top-left of the block
        width: Width in px
        top_height, front_height: Heights of the two faces
        color: Base colour (top lighter, front darker), or explicit colours
        top_color, front_color: Explicit face colours
        frame_index: Frame, starting at 1
    """
    try:
        if top_color and front_color:
            top, front = hex_to_rgb(top_color), hex_to_rgb(front_color)
        else:
            base = hex_to_rgb(color or "#8A8A9A")
            top, front = shade(base, "highlight", 1), shade(base, "shadow", 1)
        img = _layer(filename, layer_name, frame_index)
        P.topdown_block(img, x, y, width, top_height, front_height, top, front)
        pixels.write_image(filename, layer_name, frame_index, img)
    except ValueError as e:
        return f"Failed to draw block: {e}"
    return json.dumps({"camera_ratio_top_to_front": round(top_height / front_height, 2)})


@mcp.tool()
async def draw_projection_grid(
    filename: str,
    projection: str,
    cell: int = 16,
    color: str = "#00C8FF",
    layer_name: str = "guides",
    frame_index: int = 1,
) -> str:
    """Draw a guide grid for a projection on its own layer.

    side / topdown / top / oblique: square cells; iso: 2:1 diagonals;
    dimetric45: 1:1 diagonals. Build on the grid, then hide the layer
    (set_layer_visibility) or lower its opacity (set_layer_opacity) - and
    exclude it from exports.

    Args:
        filename: Aseprite file to modify
        projection: side, topdown, top, oblique, iso or dimetric45
        cell: Grid spacing in px (tile width for iso)
        color: Line colour
        layer_name: Guide layer (default "guides", created when missing)
        frame_index: Frame, starting at 1
    """
    try:
        img = _layer(filename, layer_name, frame_index)
        n = P.grid(img, projection, cell, hex_to_rgb(color))
        pixels.write_image(filename, layer_name, frame_index, img)
    except ValueError as e:
        return f"Failed to draw grid: {e}"
    return json.dumps({"pixels": n, "layer": layer_name})


@mcp.tool()
async def check_projection_lines(
    filename: str,
    projection: str = "iso",
    frame_index: int = 1,
    layer_name: str = "",
    line_colors: list | None = None,
) -> str:
    """Check that receding edges keep the projection's exact rhythm.

    iso: every step of a slanted edge must be exactly 2 px (2:1);
    dimetric45 / oblique: exactly 1 px. Corner and tip steps are allowed to
    differ. Reports the offending pixels.

    Args:
        filename: Aseprite file to read
        projection: iso, dimetric45 or oblique
        frame_index: Frame, starting at 1
        layer_name: Only this layer (default: everything visible)
        line_colors: Hex colours of the lines (default: auto-detected)
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
        issues = P.check_lines(img, projection, line_colors)
    except ValueError as e:
        return f"Failed to check lines: {e}"
    if not issues:
        return json.dumps({"issues": [], "summary": f"All receding edges follow the {projection} rhythm."})
    return json.dumps({"issues": [{**i, "count": len(i["pixels"]), "pixels": i["pixels"][:40]} for i in issues]})
