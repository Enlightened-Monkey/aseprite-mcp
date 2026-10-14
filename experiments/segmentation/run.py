"""Element segmentation experiment.

usage: uv run --extra ai python experiments/segmentation/run.py IMAGE [OUT_DIR] [HINTS.json]

Without HINTS the elements are named by a local vision model and found by a
text detector; with HINTS (agent-provided names + points) SAM cuts them.

Writes, for IMAGE:
  <name>_elements.png           flat colour per element, with a legend strip
  <name>_elements_overlay.png   colours blended over the art
  <name>_elements_contours.png  element borders drawn on the art
  <name>_elements.aseprite      one layer per element (original pixels)
  <name>_classical.png          classical-only result (graph regions merged)
  <name>_sam_auto.png           SAM 2.1 automatic masks (no text)
  <name>_elements.json          element names, sources, pixel counts
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from aseprite_mcp.ai import segment as AI  # noqa: E402
from aseprite_mcp.core import pixels  # noqa: E402
from aseprite_mcp.pixelart import segment as SG  # noqa: E402

SCALE = 3


def big(arr):
    im = Image.fromarray(arr, "RGBA")
    return im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)


def with_legend(arr, names, colours):
    im = big(arr)
    rows = (len(names) + 2) // 3
    sheet = Image.new("RGBA", (im.width, im.height + rows * 18 + 8), (255, 255, 255, 255))
    sheet.paste(im, (0, 0))
    d = ImageDraw.Draw(sheet)
    col_w = im.width // 3
    for i, n in enumerate(names):
        x, y = (i % 3) * col_w + 6, im.height + 6 + (i // 3) * 18
        d.rectangle([x, y, x + 12, y + 12], fill=(*colours[i], 255), outline=(0, 0, 0))
        d.text((x + 18, y), n, fill=(0, 0, 0))
    return sheet


def save_elements(img, labels, elements, out_dir, base, tag):
    names = [e.name for e in elements]
    colours = SG.label_colours(len(elements))
    flat = SG.render_labels(labels, colours)
    with_legend(flat, names, colours).save(f"{out_dir}/{base}_{tag}.png")
    with_legend(SG.render_overlay(img, labels, 0.55, colours), names, colours).save(
        f"{out_dir}/{base}_{tag}_overlay.png")
    big(SG.render_contours(img, labels)).save(f"{out_dir}/{base}_{tag}_contours.png")
    ase = os.path.abspath(f"{out_dir}/{base}_{tag}.aseprite")
    pixels.create_sprite(ase, img, "original")
    for i, e in enumerate(elements):
        part = np.zeros_like(img)
        part[labels == i] = img[labels == i]
        pixels.add_layer(ase, e.name)
        pixels.write_image(ase, e.name, 1, part)
    from aseprite_mcp.core.commands import AsepriteCommand
    AsepriteCommand.execute_lua_script_checked(
        'local spr = app.activeSprite\nfor _, l in ipairs(spr.layers) do l.isVisible = (l.name ~= "original") end\n'
        'spr:saveAs(spr.filename)\nprint("OK")', ase)
    return [{"name": e.name, "source": e.source, "pixels": int((labels == i).sum()), "score": round(e.score, 3)}
            for i, e in enumerate(elements)]


def main(path, out_dir, hints=None):
    if hints:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        img = np.array(Image.open(path).convert("RGBA"))
        t = time.time()
        labels, elements = AI.segment_with_hints(img, json.load(open(hints)))
        rep = {"image": path, "mode": "agent hints", "elements": save_elements(img, labels, elements, out_dir,
                                                                              base, "agent"),
               "timing_s": round(time.time() - t, 1)}
        json.dump(rep, open(f"{out_dir}/{base}_agent.json", "w"), indent=2)
        print(json.dumps(rep, indent=2))
        return
    _main_auto(path, out_dir)


def _main_auto(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    img = np.array(Image.open(path).convert("RGBA"))
    t0 = time.time()

    # Classical only, for comparison.
    regions = SG.felzenszwalb(img, k=300, min_size=20)
    objects = SG.merge_regions(img, regions, border_delta=14, colour_delta=30, max_objects=24)
    big(SG.render_labels(objects)).save(f"{out_dir}/{base}_classical.png")
    t1 = time.time()

    # SAM automatic, no text.
    auto = AI.automatic_masks(img)
    big(SG.render_labels(auto)).save(f"{out_dir}/{base}_sam_auto.png")
    t2 = time.time()

    # Named elements (VLM -> detector -> SAM -> pixel snapping -> classical fill-in).
    labels, elements = AI.segment_elements(img)
    t3 = time.time()
    names = [e.name for e in elements]
    colours = SG.label_colours(len(elements))
    flat = SG.render_labels(labels, colours)
    with_legend(flat, names, colours).save(f"{out_dir}/{base}_elements.png")
    with_legend(SG.render_overlay(img, labels, 0.55, colours), names, colours).save(
        f"{out_dir}/{base}_elements_overlay.png")
    big(SG.render_contours(img, labels)).save(f"{out_dir}/{base}_elements_contours.png")

    ase = os.path.abspath(f"{out_dir}/{base}_elements.aseprite")
    pixels.create_sprite(ase, img, "original")
    for i, e in enumerate(elements):
        layer = e.name
        part = np.zeros_like(img)
        part[labels == i] = img[labels == i]
        pixels.add_layer(ase, layer)
        pixels.write_image(ase, layer, 1, part)
    # Show the element layers, hide only the copy of the original.
    from aseprite_mcp.core.commands import AsepriteCommand
    AsepriteCommand.execute_lua_script_checked(
        'local spr = app.activeSprite\nfor _, l in ipairs(spr.layers) do l.isVisible = (l.name ~= "original") end\n'
        'spr:saveAs(spr.filename)\nprint("OK")', ase)

    report = {"image": path, "elements": [
        {"name": e.name, "source": e.source, "pixels": int((labels == i).sum()), "score": round(e.score, 3)}
        for i, e in enumerate(elements)],
        "timing_s": {"classical": round(t1 - t0, 1), "sam_auto": round(t2 - t1, 1), "named": round(t3 - t2, 1)}}
    json.dump(report, open(f"{out_dir}/{base}_elements.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out", sys.argv[3] if len(sys.argv) > 3 else None)
