"""Pixel-art craft tools: linting, readability views, colour helpers, guide.

The algorithms live in aseprite_mcp.pixelart (pure Python); these tools read
pixels out of Aseprite, run them, and hand back JSON plus an image the agent
can look at.
"""
import io
import json
import os
from collections import Counter

from mcp.server.fastmcp import Image

from .. import mcp
from ..core import pixels
from ..core.commands import reject_traversal
from ..pixelart import color as C
from ..pixelart import lint as L
from ..pixelart import render

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")


def _png(im) -> Image:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


def _save(im, path: str) -> str | None:
    """Save an image when a path is given; returns an error message or None."""
    if not path:
        return None
    err = reject_traversal(path)
    if err:
        return err
    if not path.lower().endswith(".png"):
        path += ".png"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path)
    return None


def _split(value: str) -> list[str] | None:
    items = [v.strip() for v in (value or "").split(",") if v.strip()]
    return items or None


@mcp.tool()
async def lint_pixel_art(
    filename: str,
    frame_index: int = 1,
    layer_name: str = "",
    checks: str = "",
    line_colors: str = "",
    overlay_path: str = "",
    max_pixels: int = 40,
    respect_style: bool = False,
):
    """Check a frame for common pixel-art mistakes and show where they are.

    Run after each drawing pass and fix what it reports. Checks:
    - orphans: stray single pixels (floating ones are warnings; single pixels
      inside the sprite are info, since an eye or shine pixel can be deliberate)
    - l_corners: doubled corner pixels that thicken 1-px lines (fix: delete them)
    - jaggies: line steps that break the staircase rhythm (straight lines need
      equal steps; curves need step lengths that change gradually)
    - banding: shading/AA bands that copy a neighbouring staircase step for step
    - pillow_shading: shading that brightens toward the centre from every edge
    - colors: too many colours for the size, near-duplicate colours, pure black
    - dither: checker dithering on a small sprite
    - outer_aa: lighter pixels outside the outline against transparency

    Line checks apply to line colours only (the outline colour and thin dark
    lines, auto-detected); pass line_colors to choose them yourself.
    Upscaled art (e.g. imported at 3x) is detected and analysed at 1x.

    Args:
        filename: Aseprite file to read
        frame_index: Frame to check, starting at 1
        layer_name: Check only this layer (default: everything visible)
        checks: Comma-separated subset of the checks above (default: all)
        line_colors: Comma-separated hex colours to treat as lines (default: auto)
        overlay_path: Optional path to also save the marked-up PNG
        max_pixels: Max coordinates listed per issue (default 40)
        respect_style: Recognise the style first (analyze_style) and skip
            findings that are part of it: pure black in a black-outline
            style, dither on a large deliberately dithered sprite, light
            edge pixels of a selective outline

    Returns:
        JSON report (pixel coordinates are sprite x,y) and an overlay image
        with each flagged pixel boxed (colour per check).
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
        scale = L.detect_scale(img)
        work = L.downscale(img, scale)
        names = _split(checks)
        skipped = {}
        if respect_style:
            from ..pixelart.style import analyze_style as _analyze
            style = _analyze(work)
            skipped = style["lint"]["skipped"]
            if names is None:
                names = style["lint"]["checks"]
        issues = L.lint(work, names, _split(line_colors))
        if "colors.pure_black" in skipped:
            issues = [i for i in issues if i["check"] != "pure_black"]
    except ValueError as e:
        return f"Failed to lint: {e}"

    report = {
        "size": [int(work.shape[1]), int(work.shape[0])],
        "warnings": sum(i["severity"] == "warning" for i in issues),
        "info": sum(i["severity"] == "info" for i in issues),
        "issues": [{
            "check": i["check"], "severity": i["severity"], "message": i["message"],
            "count": len(i["pixels"]), "pixels": i["pixels"][:max(0, max_pixels)],
        } for i in issues],
    }
    if scale > 1:
        report["note"] = f"Image is pixel art upscaled {scale}x; analysed at 1x (coordinates are 1x)."
    if skipped:
        report["skipped_for_style"] = skipped
    if not issues:
        report["summary"] = "No issues found."
        return json.dumps(report)
    overlay = render.lint_overlay(work, issues)
    err = _save(overlay, overlay_path)
    if err:
        return err
    if overlay_path:
        report["overlay_path"] = overlay_path
    return [json.dumps(report), _png(overlay)]


@mcp.tool()
async def readability_preview(
    filename: str,
    frame_index: int = 1,
    layer_name: str = "",
    output_path: str = "",
):
    """Show a frame the way viewers will see it, to judge readability.

    Returns one contact sheet: true size (1x, 2x), zoomed, blurred (as seen
    from a distance; blobs and banding show up here), silhouette (must read
    as the subject on its own), value-only greyscale (checks light/dark
    structure), and the sprite on light, mid and dark backgrounds. Look at it
    after every major pass: if something is unclear at 1x it is unclear.

    Args:
        filename: Aseprite file to read
        frame_index: Frame to show, starting at 1
        layer_name: Show only this layer (default: everything visible)
        output_path: Optional path to also save the sheet as PNG
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
    except ValueError as e:
        return f"Failed to render preview: {e}"
    img = L.downscale(img, L.detect_scale(img))
    sheet = render.readability_sheet(img)
    err = _save(sheet, output_path)
    if err:
        return err
    text = f"Readability sheet for frame {frame_index} ({img.shape[1]}x{img.shape[0]})"
    if output_path:
        text += f", saved to {output_path}"
    return [text, _png(sheet)]


@mcp.tool()
async def shade_color(color: str, direction: str = "shadow", steps: int = 1) -> str:
    """Get a shadow or highlight of a colour, with hue shifting.

    shadow: hue turns toward violet, saturation drops a little, value drops
    about 22 per step. highlight: hue turns toward yellow, value rises about
    15 per step. Use for quick shading decisions; for a full material use
    generate_color_ramp.

    Args:
        color: Hex colour, e.g. "#D04648"
        direction: "shadow" or "highlight"
        steps: How many steps away from the colour (1-6)

    Returns:
        JSON array of hex colours, one per step, moving away from the input.
    """
    if not 1 <= steps <= 6:
        return "steps must be between 1 and 6"
    try:
        rgb = C.hex_to_rgb(color)
        return json.dumps([C.rgb_to_hex(C.shade(rgb, direction, k)) for k in range(1, steps + 1)])
    except ValueError as e:
        return str(e)


def _colour_counts(img) -> list[tuple[tuple[int, int, int], int]]:
    opaque = img[..., 3] > 0
    counts = Counter(map(tuple, img[opaque][:, :3].tolist()))
    return [(tuple(int(v) for v in rgb), n) for rgb, n in counts.items()]


@mcp.tool()
async def suggest_outline_color(filename: str, frame_index: int = 1, layer_name: str = "") -> str:
    """Suggest a tinted near-black outline colour that suits the sprite.

    Pure black outlines look flat; a dark plum, brown or teal tuned to the
    sprite's own dark shades looks richer. Pass the result to outline_cel /
    outline_native or use it for line art.

    Args:
        filename: Aseprite file to read
        frame_index: Frame to sample, starting at 1
        layer_name: Sample only this layer (default: everything visible)
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
    except ValueError as e:
        return f"Failed to suggest outline colour: {e}"
    rgb = C.suggest_outline(_colour_counts(img))
    h, s, v = C.rgb_to_hsv(rgb)
    return json.dumps({"color": C.rgb_to_hex(rgb), "h": round(h), "s": round(s), "v": round(v)})


@mcp.tool()
async def analyze_palette(filename: str, frame_index: int = 1, layer_name: str = "") -> str:
    """Review the colours a frame actually uses.

    Reports colour count against a budget for the sprite's height,
    near-duplicate colours (merge them), pure black, and every pair of
    touching colours that looks like a shading step, with its hue shift:
    "ok" (lighter step turns toward yellow), "flat" (no hue shift: shading
    by brightness only, looks dull) or "reversed" (lighter step turns away
    from yellow, e.g. shadows pushed toward green/yellow).

    Args:
        filename: Aseprite file to read
        frame_index: Frame to analyse, starting at 1
        layer_name: Analyse only this layer (default: everything visible)
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
    except ValueError as e:
        return f"Failed to analyse palette: {e}"
    img = L.downscale(img, L.detect_scale(img))
    counts = _colour_counts(img)
    rows = (img[..., 3] > 0).any(axis=1).nonzero()[0]
    height = int(rows[-1] - rows[0] + 1) if rows.size else 0
    pairs = L.shading_pairs(img)
    verdicts = Counter(p["verdict"] for p in pairs)
    color_issues = L.check_colors(img)
    report = {
        "unique_colors": len(counts),
        "sprite_height": height,
        "budget": C.palette_budget(height),
        "colors": [{"color": C.rgb_to_hex(rgb), "count": n}
                   for rgb, n in sorted(counts, key=lambda c: -c[1])][:32],
        "shading_pairs": {"ok": verdicts["ok"], "flat": verdicts["flat"],
                          "reversed": verdicts["reversed"]},
        "problem_pairs": [p for p in pairs if p["verdict"] != "ok"][:12],
        "issues": [{"check": i["check"], "severity": i["severity"], "message": i["message"]}
                   for i in color_issues],
    }
    return json.dumps(report)


def _topics() -> list[str]:
    return sorted(f[:-3] for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".md"))


def read_guide(topic: str) -> str:
    topic = (topic or "core").strip().lower()
    if topic not in _topics():
        return f"Unknown topic '{topic}'. Available: {', '.join(_topics())}"
    with open(os.path.join(KNOWLEDGE_DIR, f"{topic}.md"), encoding="utf-8") as f:
        return f.read()


@mcp.tool()
async def pixel_art_guide(topic: str = "core") -> str:
    """Pixel-art technique guide: rules of thumb with concrete numbers.

    Read "core" first, then the topic you are about to work on. Topics:
    core, workflow, style, lineart, antialiasing, color, readability,
    dithering, perspective, cleanup, subpixel, animation.

    Args:
        topic: Guide topic (default "core")
    """
    return read_guide(topic)


@mcp.resource("pixelart://guide/{topic}")
def guide_resource(topic: str) -> str:
    """Pixel-art technique guide for one topic (see pixel_art_guide)."""
    return read_guide(topic)


# ------------------------------------------------- style and clean-up

_ASE_EXT = (".aseprite", ".ase")


def _load_any(filename: str, frame_index: int, layer_name: str, remove_background: bool):
    """Pixels from an Aseprite file or a plain image (png/gif/jpg/bmp/webp).

    For plain images without transparency whose four corners share one
    colour, that colour is treated as the background and made transparent
    (when remove_background is true).
    """
    if filename.lower().endswith(_ASE_EXT):
        return pixels.read_image(filename, frame_index, layer_name), None
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} not found")
    from PIL import Image as PILImage
    import numpy as np
    with PILImage.open(filename) as im:
        try:
            im.seek(max(0, frame_index - 1))
        except EOFError:
            raise ValueError("Frame index out of range") from None
        arr = np.array(im.convert("RGBA"))
    note = None
    if remove_background and (arr[..., 3] == 255).all():
        corners = {tuple(arr[0, 0]), tuple(arr[0, -1]), tuple(arr[-1, 0]), tuple(arr[-1, -1])}
        if len(corners) == 1:
            bg = arr[0, 0].copy()
            arr[(arr == bg).all(axis=-1)] = 0
            note = f"Background colour {C.rgb_to_hex(bg)} (all four corners) made transparent."
    return arr, note


@mcp.tool()
async def analyze_style(
    filename: str,
    frame_index: int = 1,
    layer_name: str = "",
    remove_background: bool = True,
) -> str:
    """Recognise a sprite's pixel-art style and file it under the matching rules.

    Measures: outline style (none, black_contour, black_inline, dark_contour,
    colored, shaded, selective), shading (flat, cel, soft, pillow, dithered),
    light direction, anti-aliasing / soft-edge level, dithering, palette size
    and tones per ramp, hue-shift direction, sprite size class, and whether
    the image is clean pixel art at all (blurred, resized or filtered images
    are flagged). Gives a style label (1-bit, minimal 4-shade, 8-bit retro,
    16-bit, modern hi-bit, not clean pixel art).

    Use it before editing or extending someone else's sprite: the "rules"
    list says which guide rules govern each trait so new pixels match the
    existing style, and "lint" gives style-aware settings for lint_pixel_art.

    Args:
        filename: .aseprite file or an image (png, gif, jpg, bmp, webp)
        frame_index: Frame to analyse, starting at 1
        layer_name: Analyse only this layer (Aseprite files only)
        remove_background: For opaque images, treat a colour shared by all
            four corners as background (default true)
    """
    from ..pixelart.style import analyze_style as _analyze

    try:
        img, note = _load_any(filename, frame_index, layer_name, remove_background)
    except ValueError as e:
        return f"Failed to analyse style: {e}"
    profile = _analyze(img)
    if note:
        profile["note"] = note
    return json.dumps(profile)


@mcp.tool()
async def cleanup_sprite(
    filename: str,
    frame_index: int = 1,
    layer_name: str = "",
    output: str = "",
    steps: str = "",
    max_colors: int = 0,
    keep_details: bool = True,
    line_colors: str = "",
    in_place: bool = False,
    remove_background: bool = True,
):
    """Finish a rough, resized or noisy sprite: the automatic clean-up pass.

    Works on an .aseprite file or a plain image a user supplied. It first
    recognises the style (see analyze_style) so deliberate traits are kept,
    then runs these steps (all by default, or a comma-separated subset):
    - alpha: semi-transparent edge pixels become fully opaque or clear
    - grid: art that was upscaled - also by non-integer factors, blurred or
      saved as JPEG - is resampled back to its true pixel grid
    - palette: near-identical colours merge; far too many colours (blur,
      JPEG) are reduced to a size-appropriate budget
    - orphans: pixels floating alone in transparency are deleted
    - speckles: single noise pixels take the surrounding colour (distinct
      single pixels such as pupils or shines are kept and listed for review)
    - outline: gaps in a single dark contour are closed
    - corners: pixel-perfect pass on line colours (L-corner doubles removed)
    - outer_aa: light pixels outside the outline are removed (skipped for a
      selective outline)
    Afterwards it lints the result and lists what still needs a human eye
    (jaggies, banding, pillow shading...) - fix those by hand.

    Output: for an Aseprite file of unchanged size, the result goes to a new
    layer "<layer> clean" (or "clean") and the other layers are hidden; with
    in_place=true and a layer_name, that layer is overwritten. When the size
    changes (grid recovered) or the input is an image, a new .aseprite (or
    .png when output ends with .png) is written to `output`, default
    "<name>_clean.aseprite" next to the input.

    Args:
        filename: .aseprite file or image (png, gif, jpg, bmp, webp)
        frame_index: Frame to clean, starting at 1
        layer_name: Clean only this layer (default: the visible composite)
        output: Output path for a new file (see above)
        steps: Comma-separated subset of: alpha, grid, palette, orphans,
            speckles, outline, corners, outer_aa (default: all)
        max_colors: 0 = automatic, -1 = keep colour count (still merges
            near-duplicates), N = reduce to N colours
        keep_details: Keep high-contrast single pixels (default true)
        line_colors: Comma-separated hex colours that are lines (default: auto)
        in_place: Overwrite layer_name instead of adding a layer
        remove_background: For opaque images, make a colour shared by all
            four corners transparent (default true)

    Returns:
        JSON log of every change, the style profile label, remaining lint
        issues, and a before/after image (changed pixels boxed when the size
        is unchanged).
    """
    import numpy as np

    from ..pixelart import cleanup as CL
    from ..pixelart.style import analyze_style as _analyze

    try:
        img, note = _load_any(filename, frame_index, layer_name, remove_background)
        style = _analyze(img)
        clean, log = CL.cleanup(
            img, _split(steps) or CL.STEPS, max_colors, keep_details, _split(line_colors), style)
    except ValueError as e:
        return f"Failed to clean up: {e}"

    is_ase = filename.lower().endswith(_ASE_EXT)
    same_size = clean.shape == img.shape
    base = os.path.splitext(filename)[0]
    try:
        if is_ase and same_size and not output:
            if in_place:
                if not layer_name:
                    return "in_place needs layer_name (the layer to overwrite)"
                target = layer_name
            else:
                target = f"{layer_name} clean" if layer_name else "clean"
                pixels.add_layer(filename, target)
            pixels.write_image(filename, target, frame_index, clean)
            hidden = [] if in_place else pixels.hide_layers_except(filename, target)
            where = {"file": filename, "layer": target, "hidden_layers": hidden}
        else:
            out = output or f"{base}_clean.aseprite"
            err = reject_traversal(out)
            if err:
                return err
            if out.lower().endswith(".png"):
                pixels.save_png(clean, out)
            else:
                pixels.create_sprite(out, clean, "clean")
            where = {"file": out}
    except ValueError as e:
        return f"Failed to save clean-up: {e}"

    changed = None
    if same_size:
        diff = (clean != img).any(axis=-1)
        changed = [(int(x), int(y)) for y, x in zip(*np.nonzero(diff))]
    remaining = L.lint(clean, style["lint"]["checks"], _split(line_colors))
    report = {
        "output": where,
        "style": {"label": style["label"], "outline": style["outline"]["style"],
                  "shading": style["shading"], "light_from": style["lighting"]["from"]},
        "size": {"before": [int(img.shape[1]), int(img.shape[0])],
                 "after": [int(clean.shape[1]), int(clean.shape[0])]},
        "pixels_changed": len(changed) if changed is not None else "size changed",
        "steps": log["steps"],
        "remaining_issues": [{"check": i["check"], "severity": i["severity"], "message": i["message"],
                              "count": len(i["pixels"]), "pixels": i["pixels"][:20]} for i in remaining],
    }
    if note:
        report["note"] = note
    if remaining:
        report["next"] = ("Automatic fixes are done. Review remaining_issues with lint_pixel_art on the "
                          "output and fix them by hand, then check readability_preview.")
    return [json.dumps(report), _png(render.before_after(img, clean, changed))]
