"""Animation craft tools: timing, motion with easing/arcs/overshoot,
squash & stretch, sub-pixel moves, overlap, loops, smears and an analyzer.
"""
import io
import json
import os

import numpy as np
from mcp.server.fastmcp import Image

from .. import mcp
from ..core import pixels
from ..pixelart import anim as A
from ..pixelart import lint as L
from ..pixelart.color import hex_to_rgb


def _png(im) -> Image:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


def _strip(frames: list[np.ndarray], durations: list[int]):
    """Contact strip of frames at a readable scale, with frame numbers."""
    from PIL import Image as PILImage, ImageDraw
    from ..pixelart.render import _on_checker, _up
    h, w = frames[0].shape[:2]
    s = max(1, min(8, 160 // max(h, w)))
    cols = min(len(frames), 8)
    rows = (len(frames) + cols - 1) // cols
    cw, ch = w * s + 6, h * s + 18
    sheet = PILImage.new("RGB", (cols * cw + 6, rows * ch + 6), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(frames):
        x, y = 6 + (i % cols) * cw, 6 + (i // cols) * ch
        d.text((x, y), f"{i + 1}: {durations[i]}ms", fill=(0, 0, 0))
        sheet.paste(_up(_on_checker(f, 1), s), (x, y + 12))
    return sheet


def _load_frames(filename: str, layer: str, frames: str):
    """Frames + durations from an Aseprite file or an animated image."""
    if filename.lower().endswith((".aseprite", ".ase")):
        idx = pixels.parse_frames(filename, frames)
        durs = pixels.get_durations(filename)
        return [pixels.read_image(filename, i, layer) for i in idx], [durs[i - 1] for i in idx], idx
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} not found")
    from PIL import Image as PILImage, ImageSequence
    out, durs = [], []
    with PILImage.open(filename) as im:
        for fr in ImageSequence.Iterator(im):
            out.append(np.array(fr.convert("RGBA")))
            durs.append(int(fr.info.get("duration", 100)))
    bg = out[0][0, 0].copy()
    if bg[3] == 255 and all(tuple(f[0, 0]) == tuple(bg) for f in out):
        for f in out:
            f[(f == bg).all(-1)] = 0
    k = L.detect_scale(out[0])
    return [L.downscale(f, k) for f in out], durs, list(range(1, len(out) + 1))


@mcp.tool()
async def analyze_animation(filename: str, layer_name: str = "", frames: str = "all"):
    """Review an animation's timing and motion, from an .aseprite file or a GIF.

    Use it on your own animation, or on a reference GIF to learn its timing.
    Reports the drawing sequence (re-used frames, ping-pong like 0-1-2-1,
    A-B-A-C), durations, holds, loop length, per-transition motion (whole
    pixel shift plus "residual" = pixels redrawn beyond that movement) and
    issues: two-frame loops, duplicate neighbouring frames, odd frames,
    mechanical (unchanging) spacing and timing, pixel-count drift, colours
    that exist in one frame only, durations that GIF export will round.
    Line boil (a still pose redrawn every frame) shows up as high residual
    on frames that should not move - it is not flagged automatically.

    Args:
        filename: .aseprite file or animated image (gif, webp, png)
        layer_name: Analyse one layer only (Aseprite files)
        frames: "all" (default), "2-6" or "1,3,5" (Aseprite files)

    Returns:
        JSON report and a contact strip of the frames with their durations.
    """
    try:
        imgs, durs, idx = _load_frames(filename, layer_name, frames)
        report = A.analyze(imgs, durs)
    except ValueError as e:
        return f"Failed to analyse animation: {e}"
    report["frame_indices"] = idx
    for i in report["issues"]:
        i["frames"] = [idx[f - 1] for f in i["frames"]]
    report.pop("pixel_counts", None)
    return [json.dumps(report), _png(_strip(imgs, durs))]


@mcp.tool()
async def set_frame_timing(
    filename: str,
    frames: str = "all",
    durations: list | None = None,
    preset: str = "",
    units: list | None = None,
    base: str = "game60",
    ease: str = "",
    mid_ms: int = 80,
    key_ms: int = 160,
) -> str:
    """Set frame durations from a preset, explicit values, frame units or easing.

    Pick ONE source:
    - durations: explicit ms per frame, e.g. [160, 80, 80, 160]
    - preset: typical timing for an action (applied to the first frames of
      the range): idle_breathing, idle_pingpong, walk_small, walk,
      walk_smooth, run, jump, attack, attack_fast, hit_react, smear,
      impact_fx, ui_loop, ui_pop, blink
    - units: holds in base frames, e.g. [2, 2, 1, 1, 2] "on twos/ones" with
      base game60 (16.7 ms), game30 (33 ms) or film24 (41.7 ms)
    - ease: "in_out", "in" or "out" - easing by timing alone: frames near
      the keys last key_ms, mid-motion frames mid_ms (e.g. 160,120,80,80,120,160)

    Args:
        filename: Aseprite file to modify
        frames: Frame range: "all", "3-8" or "1,4,6"
        durations, preset, units, ease: the timing source (see above)
        base: Base frame rate for units
        mid_ms, key_ms: Durations for ease
    """
    try:
        idx = pixels.parse_frames(filename, frames)
        if durations:
            vals = [int(v) for v in durations]
        elif preset:
            if preset not in A.TIMING_PRESETS:
                return f"Unknown preset '{preset}'. Available: {', '.join(A.TIMING_PRESETS)}"
            vals = A.TIMING_PRESETS[preset]["durations"]
        elif units:
            vals = A.units_to_ms([int(u) for u in units], base)
        elif ease:
            vals = A.ease_durations(len(idx), mid_ms, key_ms, ease)
        else:
            return "Give one of: durations, preset, units or ease"
        if len(vals) > len(idx):
            return f"{len(vals)} durations for {len(idx)} frames; add frames or narrow the values"
        applied = dict(zip(idx, vals))
        pixels.set_durations(filename, applied)
    except ValueError as e:
        return f"Failed to set timing: {e}"
    out = {"durations": {str(k): v for k, v in applied.items()}}
    if preset:
        out["preset"] = A.TIMING_PRESETS[preset]
    return json.dumps(out)


@mcp.tool()
async def animate_layer_path(
    filename: str,
    layer_name: str,
    start_frame: int,
    end_frame: int,
    dx: float,
    dy: float,
    breakdown_dx: float | None = None,
    breakdown_dy: float | None = None,
    easing: str = "ease_in_out",
    overshoot_px: float = 0.0,
    subpixel: bool = False,
) -> str:
    """Move a layer's drawing along a path over a frame range, with easing.

    The drawing on start_frame is the source; every frame up to end_frame
    gets it moved by (dx, dy) * progress. A breakdown point (offset at the
    middle of the move) bends the motion into an arc, as organic motion
    should be. easing: linear, ease_in, ease_out, ease_in_out (default) or
    back_out, which passes the target by overshoot_px and settles back - even
    1 px of overshoot adds snap. subpixel=true renders fractional positions
    with in-between tones instead of rounding (slow, subtle moves).
    Frames are added when missing.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer to move
        start_frame, end_frame: Frame range (start holds the source drawing)
        dx, dy: Total movement in pixels
        breakdown_dx, breakdown_dy: Offset at mid-move for an arc (optional)
        easing: linear, ease_in, ease_out, ease_in_out, back_out
        overshoot_px: Overshoot distance for back_out (default 0 -> 10% of move)
        subpixel: Use sub-pixel rendering for fractional positions
    """
    if end_frame <= start_frame:
        return "end_frame must be after start_frame"
    try:
        pixels.ensure_frames(filename, end_frame)
        src = pixels.read_image(filename, start_frame, layer_name)
        n = end_frame - start_frame + 1
        bd = None if breakdown_dx is None or breakdown_dy is None else (breakdown_dx, breakdown_dy)
        pos = A.path_positions((0, 0), (dx, dy), n, bd, easing, overshoot_px)
        for i, (px, py) in enumerate(pos[1:], start=1):
            if subpixel:
                img = A.subpixel_shift(src, px, py)
            else:
                img = A.shift(src, int(round(px)), int(round(py)))
            pixels.write_image(filename, layer_name, start_frame + i, img)
    except ValueError as e:
        return f"Failed to animate: {e}"
    return json.dumps({"positions": [[round(x, 2), round(y, 2)] for x, y in pos],
                       "frames": list(range(start_frame, end_frame + 1))})


@mcp.tool()
async def squash_stretch(
    filename: str,
    layer_name: str,
    frame_index: int,
    amount: float,
    axis: str = "y",
    anchor: str = "bottom",
    target_frame: int = 0,
) -> str:
    """Squash (amount > 0) or stretch (amount < 0) a layer's drawing, keeping its area.

    Squashing 0.2 along y makes it 20% shorter and ~25% wider, so the volume
    reads as constant. Anchor at the contact point (bottom for landing,
    top for hanging). Typical: 0.15-0.3 on impact frames, -0.1 to -0.2
    stretch just before and after. Nearest-neighbour: do a clean-up pass
    (cleanup_sprite with steps="corners,jaggies") afterwards.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer to deform
        frame_index: Source frame
        amount: -0.6..0.6; positive squashes along axis, negative stretches
        axis: "y" (default) or "x"
        anchor: bottom, top, center, left or right
        target_frame: Write here instead of overwriting (0 = same frame)
    """
    try:
        img = pixels.read_image(filename, frame_index, layer_name)
        out = A.squash_stretch(img, amount, axis, anchor)
        target = target_frame or frame_index
        pixels.ensure_frames(filename, target)
        pixels.write_image(filename, layer_name, target, out)
    except ValueError as e:
        return f"Failed to squash/stretch: {e}"
    return json.dumps({"pixels_before": int((img[..., 3] > 0).sum()), "pixels_after": int((out[..., 3] > 0).sum()),
                       "frame": target})


@mcp.tool()
async def subpixel_move(
    filename: str,
    layer_name: str,
    source_frame: int,
    target_frame: int,
    dx: float,
    dy: float,
    protect_colors: list | None = None,
) -> str:
    """Write a copy of a drawing moved by a fraction of a pixel (e.g. 0.5).

    Edges slide by changing tone instead of jumping a whole pixel, using
    only colours the drawing already has - the classic way to make idles,
    breathing and slow moves smooth on small sprites. Silhouette edges
    against transparency still snap (there is no in-between tone there).
    protect_colors keeps pixels of those colours (eyes, face details) moving
    by whole pixels only, so faces stay crisp. Use 0.5 px steps for
    in-betweens; amplitudes of 0.5-1 px for idles.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer to move
        source_frame: Frame with the drawing
        target_frame: Frame to write (added when missing)
        dx, dy: Movement in pixels, may be fractional
        protect_colors: Hex colours that only move by whole pixels
    """
    try:
        img = pixels.read_image(filename, source_frame, layer_name)
        protect = None
        if protect_colors:
            keys = L._keys(img)
            wanted = {hex_to_rgb(c) for c in protect_colors}
            protect = np.isin(keys, [k for k in np.unique(keys) if k != -1 and L._rgb(int(k)) in wanted])
        out = A.subpixel_shift(img, dx, dy, protect)
        pixels.ensure_frames(filename, target_frame)
        pixels.write_image(filename, layer_name, target_frame, out)
    except ValueError as e:
        return f"Failed to move: {e}"
    return json.dumps({"frame": target_frame, "moved": [dx, dy]})


@mcp.tool()
async def apply_overlap(
    filename: str,
    leader_layer: str,
    follower_layers: list,
    delay_frames: int = 1,
    frames: str = "all",
) -> str:
    """Make follower layers trail the leader's movement (overlap & follow-through).

    Hair, capes, tails, sleeves and loose parts should start moving after the
    body and keep going after it stops. This measures how the leader layer
    moves frame to frame and shifts each follower so it repeats that
    movement delay_frames later (1-2 is typical); followers therefore also
    settle after the leader has stopped. Followers keep their own drawings;
    only their positions lag.

    Args:
        filename: Aseprite file to modify
        leader_layer: Layer that leads the motion (body, head)
        follower_layers: Layers that should trail it
        delay_frames: Lag in frames (1-3)
        frames: Frame range to process ("all", "2-9", ...)
    """
    if not 1 <= delay_frames <= 3:
        return "delay_frames must be between 1 and 3"
    try:
        idx = pixels.parse_frames(filename, frames)
        track = [A.centroid(pixels.read_image(filename, f, leader_layer)) for f in idx]
        if any(t is None for t in track):
            return "The leader layer is empty on some frames"
        moved = {}
        for layer in follower_layers:
            shifts = []
            for i, f in enumerate(idx):
                j = max(0, i - delay_frames)
                ox = int(round(track[j][0] - track[i][0]))
                oy = int(round(track[j][1] - track[i][1]))
                if ox or oy:
                    img = pixels.read_image(filename, f, layer)
                    pixels.write_image(filename, layer, f, A.shift(img, ox, oy))
                shifts.append([ox, oy])
            moved[layer] = shifts
    except ValueError as e:
        return f"Failed to apply overlap: {e}"
    return json.dumps({"frames": idx, "follower_offsets": moved})


@mcp.tool()
async def build_loop(
    filename: str,
    keys: str,
    pattern: str = "pingpong",
    sequence: list | None = None,
    tag: str = "",
) -> str:
    """Arrange key drawings into a loop by re-using frames.

    Re-using drawings is how good pixel animation gets smooth loops from few
    frames. The key frames are replaced by copies in the chosen order (all
    layers, with their durations):
    - pingpong: 1-2-3-4-3-2 (there and back without repeating the ends)
    - abac: A-B-A-C, a 3-drawing breathing idle with variety
    - sequence: your own order as positions in keys, e.g. [0, 1, 2, 1]
    Then tune timing with set_frame_timing (hold the extremes longer).

    Args:
        filename: Aseprite file to modify
        keys: Key frames: "1-4" or "1,2,3"
        pattern: pingpong (default) or abac
        sequence: Custom order (0-based positions into keys)
        tag: Optional tag name over the resulting loop
    """
    try:
        key_list = pixels.parse_frames(filename, keys)
        k = len(key_list)
        if sequence:
            seq = [int(v) for v in sequence]
            if any(not 0 <= v < k for v in seq):
                return f"sequence values must be 0-{k - 1}"
        elif pattern == "pingpong":
            if k < 3:
                return "pingpong needs at least 3 key frames"
            seq = list(range(k)) + list(range(k - 2, 0, -1))
        elif pattern == "abac":
            if k != 3:
                return "abac needs exactly 3 key frames (A, B, C)"
            seq = [0, 1, 0, 2]
        else:
            return "pattern must be pingpong or abac (or pass sequence)"
        if len(set(seq)) < 3 and len(seq) > 1:
            note = "Loops with fewer than 3 distinct drawings read as flicker."
        else:
            note = ""
        pixels.resequence_frames(filename, key_list, seq, tag)
    except ValueError as e:
        return f"Failed to build loop: {e}"
    out = {"sequence": [key_list[i] for i in seq], "frames": len(seq)}
    if note:
        out["note"] = note
    return json.dumps(out)


@mcp.tool()
async def generate_smear(
    filename: str,
    layer_name: str,
    frame_a: int,
    frame_b: int,
    target_frame: int,
    mode: str = "elongated",
    copies: int = 2,
) -> str:
    """Draw a smear in-between from pose A to pose B (fake motion blur).

    elongated: the shape is stretched along the path from A to B with a
    crisp leading edge. multiples: after-images of A along the path, one
    palette step lighter, under B. Only the layer's own colours are used.
    Give the smear frame 40-50 ms - felt, not seen. Never dither a smear.

    Args:
        filename: Aseprite file to modify
        layer_name: Layer with the moving part
        frame_a: Pose before the smear
        frame_b: Pose after the smear
        target_frame: Frame to write the smear into (added when missing)
        mode: elongated (default) or multiples
        copies: Number of after-images for multiples (1-4)
    """
    if not 1 <= copies <= 4:
        return "copies must be between 1 and 4"
    try:
        a = pixels.read_image(filename, frame_a, layer_name)
        b = pixels.read_image(filename, frame_b, layer_name)
        out = A.smear(a, b, mode, copies)
        pixels.ensure_frames(filename, target_frame)
        pixels.write_image(filename, layer_name, target_frame, out)
    except ValueError as e:
        return f"Failed to draw smear: {e}"
    return json.dumps({"frame": target_frame, "mode": mode, "suggested_duration_ms": 40})


@mcp.tool()
async def draw_slash_arc(
    filename: str,
    layer_name: str,
    start_frame: int,
    center_x: int,
    center_y: int,
    radius: int,
    start_deg: float,
    end_deg: float,
    thickness: int = 5,
    colors: list | None = None,
    frames: int = 3,
) -> str:
    """Draw a weapon-trail slash over several frames.

    Frame 1 is a full crescent (bright core on the outside, body colour
    inside), then the trail disappears from where the swing started and
    thins, leaving a tail at the weapon's end. The crescent is thickest in
    the middle and tapers to points. Angles: 0 = right, 90 = down, sweeping
    from start_deg to end_deg. Pair with 30-60 ms frames on the strike.

    Args:
        filename: Aseprite file to modify
        layer_name: Effect layer (created when missing)
        start_frame: First frame of the slash (frames are added when missing)
        center_x, center_y: Pivot of the swing (shoulder or hand)
        radius: Outer radius of the arc
        start_deg, end_deg: Sweep angles
        thickness: Thickness at the middle of the crescent
        colors: [core, body, tail] hex colours (default white, light cyan, blue)
        frames: Number of frames (1-4)
    """
    if not 1 <= frames <= 4:
        return "frames must be between 1 and 4"
    try:
        cols = [hex_to_rgb(c) for c in (colors or ["#FFFFFF", "#9FE8FF", "#3E7BD9"])]
        pixels.ensure_frames(filename, start_frame + frames - 1)
        h, w = pixels.read_image(filename, start_frame).shape[:2]
        arcs = A.slash_arc((w, h), (center_x, center_y), radius, start_deg, end_deg, thickness, cols, frames)
        if layer_name not in pixels.list_layers(filename):
            pixels.add_layer(filename, layer_name)
        for i, arc in enumerate(arcs):
            f = start_frame + i
            base = pixels.read_image(filename, f, layer_name)
            m = arc[..., 3] > 0
            base[m] = arc[m]
            pixels.write_image(filename, layer_name, f, base)
    except ValueError as e:
        return f"Failed to draw slash: {e}"
    return json.dumps({"frames": list(range(start_frame, start_frame + frames)),
                       "pixels": [int((a[..., 3] > 0).sum()) for a in arcs]})
