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

    Returns:
        JSON report (pixel coordinates are sprite x,y) and an overlay image
        with each flagged pixel boxed (colour per check).
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
        scale = L.detect_scale(img)
        work = L.downscale(img, scale)
        issues = L.lint(work, _split(checks), _split(line_colors))
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
    core, workflow, lineart, antialiasing, color, readability, dithering,
    perspective, cleanup, subpixel, animation.

    Args:
        topic: Guide topic (default "core")
    """
    return read_guide(topic)


@mcp.resource("pixelart://guide/{topic}")
def guide_resource(topic: str) -> str:
    """Pixel-art technique guide for one topic (see pixel_art_guide)."""
    return read_guide(topic)
