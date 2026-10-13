"""Bridge between Aseprite documents and numpy pixel arrays.

Pixel-level algorithms (linting, anti-aliasing, ramps, dithering...) are
written in Python against plain RGBA arrays so they can be unit-tested
without Aseprite. Aseprite is only used for I/O: these helpers render a
frame or a layer to a temporary PNG and read it back, or write an array
into a cel.

Arrays are uint8 with shape (height, width, 4), RGBA, sprite-global
coordinates (cel offsets are already applied).
"""
import os
import tempfile

import numpy as np
from PIL import Image

from .commands import AsepriteCommand, lua_escape
from .lua import FIND_LAYER, NORMALIZE_CEL


def _lua_path(path: str) -> str:
    return lua_escape(os.path.abspath(path).replace("\\", "/"))


def _tmp_png() -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    return path


def read_image(filename: str, frame: int = 1, layer: str = "") -> np.ndarray:
    """Read a frame as an RGBA array.

    With layer empty, returns the composite of all visible layers (what
    the viewer sees). With a layer name (group paths allowed), returns
    only that layer's cel, canvas-sized; an empty cel yields a fully
    transparent array.

    Raises:
        ValueError: on a missing file, bad frame index or unknown layer.
    """
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} not found")
    out = _tmp_png()
    safe_layer = lua_escape(layer)
    script = f"""
    {FIND_LAYER}
    local spr = app.activeSprite
    if not spr then print("ERROR:No active sprite") return end
    local idx = {int(frame)}
    if idx < 1 or idx > #spr.frames then print("ERROR:Frame index out of range") return end
    local img = Image(spr.width, spr.height, ColorMode.RGB)
    if "{safe_layer}" == "" then
        img:drawSprite(spr, idx)
    else
        local layer = find_layer(spr, "{safe_layer}")
        if not layer then print("ERROR:Layer not found: {safe_layer}") return end
        if layer.isGroup then print("ERROR:Layer is a group: {safe_layer}") return end
        local cel = layer:cel(idx)
        if cel then
            -- Convert via a temporary sprite so indexed/gray cels render as RGB.
            local tmp = Sprite(spr.width, spr.height, spr.colorMode)
            tmp:setPalette(spr.palettes[1])
            tmp.cels[1].image = cel.image
            tmp.cels[1].position = cel.position
            img:drawSprite(tmp, 1)
            tmp:close()
        end
    end
    img:saveAs("{_lua_path(out)}")
    print("OK")
    """
    try:
        ok, output = AsepriteCommand.execute_lua_script_checked(script, filename)
        if not ok:
            raise ValueError(output.strip() or "Aseprite failed to render image")
        with Image.open(out) as im:
            return np.array(im.convert("RGBA"), dtype=np.uint8)
    finally:
        if os.path.exists(out):
            os.remove(out)


def write_image(filename: str, layer: str, frame: int, pixels: np.ndarray) -> None:
    """Replace a layer's cel at a frame with a canvas-sized RGBA array.

    The cel is created when missing. The sprite must be in RGB mode.

    Raises:
        ValueError: on a missing file/layer, bad frame, wrong array shape or
        a non-RGB sprite.
    """
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} not found")
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("pixels must have shape (height, width, 4)")
    src = _tmp_png()
    Image.fromarray(arr, "RGBA").save(src)
    safe_layer = lua_escape(layer)
    script = f"""
    {FIND_LAYER}
    {NORMALIZE_CEL}
    local spr = app.activeSprite
    if not spr then print("ERROR:No active sprite") return end
    if spr.colorMode ~= ColorMode.RGB then print("ERROR:Sprite must be in RGB color mode") return end
    local idx = {int(frame)}
    if idx < 1 or idx > #spr.frames then print("ERROR:Frame index out of range") return end
    local layer = find_layer(spr, "{safe_layer}")
    if not layer then print("ERROR:Layer not found: {safe_layer}") return end
    if layer.isGroup then print("ERROR:Layer is a group: {safe_layer}") return end
    local src = Image{{ fromFile = "{_lua_path(src)}" }}
    if src.width ~= spr.width or src.height ~= spr.height then
        print("ERROR:Image size " .. src.width .. "x" .. src.height ..
              " does not match sprite " .. spr.width .. "x" .. spr.height)
        return
    end
    app.transaction(function()
        local cel = normalize_cel(spr, layer, spr.frames[idx], true)
        cel.image = src
        cel.position = Point(0, 0)
    end)
    spr:saveAs(spr.filename)
    print("OK")
    """
    try:
        ok, output = AsepriteCommand.execute_lua_script_checked(script, filename)
        if not ok:
            raise ValueError(output.strip() or "Aseprite failed to write image")
    finally:
        if os.path.exists(src):
            os.remove(src)


def read_frames(filename: str, layer: str = "") -> list[np.ndarray]:
    """Read every frame (composite or one layer) as RGBA arrays."""
    return [read_image(filename, i, layer) for i in range(1, frame_count(filename) + 1)]


def frame_count(filename: str) -> int:
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} not found")
    ok, output = AsepriteCommand.execute_lua_script_checked(
        'local spr = app.activeSprite\n'
        'if not spr then print("ERROR:No active sprite") return end\n'
        'print("FRAMES:" .. #spr.frames)', filename)
    if ok:
        for line in output.splitlines():
            if line.startswith("FRAMES:"):
                return int(line[len("FRAMES:"):])
    raise ValueError(output.strip() or "Could not read frame count")


def save_png(pixels: np.ndarray, path: str, scale: int = 1) -> str:
    """Save an RGBA array as a PNG, optionally nearest-neighbour upscaled."""
    im = Image.fromarray(np.asarray(pixels, dtype=np.uint8), "RGBA")
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path)
    return path


def create_sprite(path: str, pixels: np.ndarray, layer: str = "Layer 1") -> None:
    """Create a new RGB .aseprite file holding one layer with these pixels."""
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("pixels must have shape (height, width, 4)")
    src = _tmp_png()
    Image.fromarray(arr, "RGBA").save(src)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    script = f"""
    local spr = Sprite{{ fromFile = "{_lua_path(src)}" }}
    if not spr then print("ERROR:Could not load image") return end
    if spr.colorMode ~= ColorMode.RGB then app.command.ChangePixelFormat{{ format = "rgb" }} end
    spr.layers[1].name = "{lua_escape(layer)}"
    spr:saveAs("{_lua_path(path)}")
    print("OK")
    """
    try:
        ok, output = AsepriteCommand.execute_lua_script_checked(script)
        if not ok or not os.path.exists(path):
            raise ValueError(output.strip() or "Aseprite failed to create the sprite")
    finally:
        if os.path.exists(src):
            os.remove(src)


def add_layer(filename: str, layer: str) -> None:
    """Add an empty top-level layer (no-op when it already exists)."""
    script = f"""
    local spr = app.activeSprite
    if not spr then print("ERROR:No active sprite") return end
    for _, l in ipairs(spr.layers) do
        if l.name == "{lua_escape(layer)}" then print("OK") return end
    end
    local l = spr:newLayer()
    l.name = "{lua_escape(layer)}"
    spr:saveAs(spr.filename)
    print("OK")
    """
    ok, output = AsepriteCommand.execute_lua_script_checked(script, filename)
    if not ok:
        raise ValueError(output.strip() or "Could not add layer")


def hide_layers_except(filename: str, keep: str) -> list[str]:
    """Hide every top-level layer except `keep`; returns the names hidden."""
    script = f"""
    local spr = app.activeSprite
    if not spr then print("ERROR:No active sprite") return end
    for _, l in ipairs(spr.layers) do
        if l.name ~= "{lua_escape(keep)}" and l.isVisible then
            l.isVisible = false
            print("HIDDEN:" .. l.name)
        end
    end
    spr:saveAs(spr.filename)
    """
    ok, output = AsepriteCommand.execute_lua_script_checked(script, filename)
    if not ok:
        raise ValueError(output.strip() or "Could not hide layers")
    return [line[len("HIDDEN:"):] for line in output.splitlines() if line.startswith("HIDDEN:")]
