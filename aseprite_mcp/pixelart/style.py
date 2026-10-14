"""Style recognition: measure a sprite and file it under our pixel-art rules.

analyze_style() returns a profile of measurable traits (outline, shading,
anti-aliasing, dithering, palette, size, cleanliness), a style label, the
guide rules that apply to each trait, and lint settings that respect the
style (e.g. a deliberate black outline is not reported as "pure black").
"""
from collections import Counter

import numpy as np

from . import lint as L
from .color import hue_delta, lightness, palette_budget, rgb_to_hex, rgb_to_hsv

_N4 = [(0, -1), (-1, 0), (1, 0), (0, 1)]


def _edge_mask(keys: np.ndarray) -> np.ndarray:
    h, w = keys.shape
    pad = np.pad(keys, 1, constant_values=-1)
    edge = np.zeros((h, w), dtype=bool)
    for dx, dy in _N4:
        edge |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] == -1
    return edge & (keys != -1)


def _inner_neighbour(keys, edge, x, y, reach: int = 3):
    """First different colour found walking inward from an edge pixel.

    Walks up to `reach` px in each 4-direction that leads into the sprite.
    Returns None when the edge pixel's colour continues that far (it is part
    of a fill, not a line) or no direction leads inward.
    """
    h, w = keys.shape
    own = keys[y, x]
    best = None
    for dx, dy in _N4:
        for step in range(1, reach + 1):
            nx, ny = x + dx * step, y + dy * step
            if not (0 <= nx < w and 0 <= ny < h) or keys[ny, nx] == -1:
                break
            if keys[ny, nx] != own:
                if best is None or step < best[0]:
                    best = (step, int(keys[ny, nx]))
                break
    return best[1] if best else None


def _hue_families(colors: list[tuple[tuple[int, int, int], int]]) -> list[list]:
    """Group colours into ramps: greys together, others by hue within 30 deg."""
    fams = []
    for rgb, n in sorted(colors, key=lambda c: -c[1]):
        h, s, v = rgb_to_hsv(rgb)
        grey = s < 12 or v < 12
        for f in fams:
            if f["grey"] == grey and (grey or abs(hue_delta(f["h"], h)) <= 30):
                f["members"].append((rgb, n))
                break
        else:
            fams.append({"h": h, "grey": grey, "members": [(rgb, n)]})
    return fams


_DIRS = {"top-left": (-1, -1), "top-right": (1, -1), "bottom-left": (-1, 1), "bottom-right": (1, 1)}


def _outline(img, keys):
    """Classify the outline from how edge pixels compare with what is inside.

    Each silhouette-edge pixel is compared with the first different colour
    found walking inward. The light direction is the diagonal whose facing
    edges are least often darkened. Styles: none, black_contour,
    black_inline, dark_contour, colored, shaded (darker all round but
    lighter toward the light) and selective (dark on the shadow side,
    dropped or lightened toward the light).
    """
    edge = _edge_mask(keys)
    n_edge = int(edge.sum())
    if n_edge < 8:
        return {"style": "none", "colors": []}
    light = lightness(img[..., :3])
    ys, xs = np.nonzero(edge)
    cy, cx = ys.mean(), xs.mean()
    samples = []  # (x, y, dL, same_hue)
    for y, x in zip(ys, xs):
        inner = _inner_neighbour(keys, edge, x, y, reach=2)
        le = float(light[y, x])
        if inner is None:
            # The edge colour runs 3+ px inward: a fill, not a line.
            samples.append((x - cx, y - cy, 0.0, le, False))
            continue
        li = float(lightness(L._rgb(inner)))
        he, se, _ = rgb_to_hsv(img[y, x, :3])
        hi, si, _ = rgb_to_hsv(L._rgb(inner))
        same_hue = se >= 15 and si >= 15 and abs(hue_delta(he, hi)) <= 40
        samples.append((x - cx, y - cy, le - li, le, same_hue))
    if len(samples) < 8:
        return {"style": "none", "colors": []}
    arr = np.array([(dx, dy, d, le) for dx, dy, d, le, _ in samples])
    darker = arr[:, 2] < -0.08
    darker_share = float(darker.mean())

    best = None
    for name, (lx, ly) in _DIRS.items():
        facing = arr[:, 0] * lx + arr[:, 1] * ly
        lit, shade = facing > 0, facing < 0
        if lit.sum() < 4 or shade.sum() < 4:
            continue
        gap = float(darker[shade].mean() - darker[lit].mean())
        l_gap = float(arr[lit, 3].mean() - arr[shade, 3].mean())
        if best is None or gap + l_gap > best[1] + best[2]:
            best = (name, gap, l_gap, float(darker[lit].mean()), float(darker[shade].mean()))
    light_dir, gap, l_gap, darker_lit, darker_shade = best or ("unknown", 0, 0, darker_share, darker_share)

    counts = Counter(int(k) for k in keys[edge])
    top, top_n = counts.most_common(1)[0]
    top_rgb = L._rgb(top)
    main = [rgb_to_hex(L._rgb(k)) for k, n in counts.most_common() if n >= 0.05 * n_edge]
    out = {"colors": main, "darker_edge_share": round(darker_share, 2), "light_from_outline": light_dir,
           "pure_black": top_rgb == (0, 0, 0)}
    single = top_n >= 0.7 * n_edge and float(lightness(top_rgb)) < 0.3
    if not single and darker_share < 0.5 and darker_shade < 0.6:
        out["style"] = "none"
        return out
    if single:
        interior = int(((keys == top) & ~edge).sum())
        out["style"] = "black_inline" if interior >= 0.25 * top_n else "black_contour"
        _, top_s, _ = rgb_to_hsv(top_rgb)
        out["tinted"] = top_rgb != (0, 0, 0) and top_s >= 15
    elif darker_shade >= 0.6 and darker_lit < 0.4:
        out["style"] = "selective"
    elif l_gap > 0.08:
        out["style"] = "shaded"
    elif sum(1 for *_, h in samples if h) >= 0.5 * max(1, int(darker.sum())):
        out["style"] = "colored"
    else:
        out["style"] = "dark_contour"
    out["light_contrast"] = round(gap + l_gap, 2)
    return out


def _aa_share(img, keys):
    """How much of the high-contrast colour boundaries is softened by AA.

    Counts short runs (a pixel with at most one same-colour neighbour) that
    sit between two contrasting colours (lightness gap >= 0.2) and are
    lighter than one and darker than the other, relative to all pixels on
    such high-contrast boundaries.
    """
    h, w = keys.shape
    pad = np.pad(keys, 1, constant_values=-1)
    light = lightness(img[..., :3])
    lpad = np.pad(light, 1)
    contrast = aa = 0
    for y in range(h):
        for x in range(w):
            k = keys[y, x]
            if k == -1:
                continue
            nbs = [(pad[1 + y + dy, 1 + x + dx], lpad[1 + y + dy, 1 + x + dx]) for dx, dy in _N4]
            opaque = [(c, l) for c, l in nbs if c != -1]
            ls = [l for _, l in opaque] + [light[y, x]]
            if max(ls) - min(ls) < 0.2:
                continue
            contrast += 1
            others = [l for c, l in opaque if c != k]
            same = sum(c == k for c, _ in nbs)
            if same <= 1 and len(others) >= 2 and min(others) + 0.05 < light[y, x] < max(others) - 0.05:
                aa += 1
    return aa / contrast if contrast else 0.0


def _light_direction(img, keys) -> dict:
    """Where the light comes from, judged by where the light pixels sit.

    Lightness-weighted offset of every opaque pixel from the sprite centre;
    a clear lean toward one side means light from that side. Strength is
    the size of that lean relative to the sprite (below ~0.03: no clear
    direction, e.g. flat or pillow shading).
    """
    opaque = keys != -1
    if opaque.sum() < 16:
        return {"from": "unknown", "strength": 0.0}
    ys, xs = np.nonzero(opaque)
    lum = lightness(img[ys, xs, :3])
    wgt = lum - lum.mean()
    size = max(np.ptp(xs), np.ptp(ys), 1)
    vx = float((wgt * (xs - xs.mean())).sum() / (np.abs(wgt).sum() + 1e-9) / size)
    vy = float((wgt * (ys - ys.mean())).sum() / (np.abs(wgt).sum() + 1e-9) / size)
    strength = (vx * vx + vy * vy) ** 0.5
    if strength < 0.03:
        return {"from": "none (flat or all-round)", "strength": round(strength, 3)}
    angle = np.degrees(np.arctan2(vy, vx)) % 360
    names = ["right", "bottom-right", "bottom", "bottom-left", "left", "top-left", "top", "top-right"]
    return {"from": names[int(((angle + 22.5) % 360) // 45)], "strength": round(strength, 3)}


def analyze_style(img: np.ndarray) -> dict:
    scale = L.detect_scale(img)
    work = L.downscale(img, scale)
    alpha = work[..., 3]
    semi = float(((alpha > 0) & (alpha < 255)).sum() / max(1, (alpha > 0).sum()))
    keys = L._keys(work)
    opaque = keys != -1
    rows = np.where(opaque.any(axis=1))[0]
    cols = np.where(opaque.any(axis=0))[0]
    bbox_h = int(rows[-1] - rows[0] + 1) if rows.size else 0
    bbox_w = int(cols[-1] - cols[0] + 1) if cols.size else 0
    colors = [(L._rgb(int(k)), n) for k, n in Counter(int(k) for k in keys[opaque]).items()]
    n_colors = len(colors)
    budget = palette_budget(bbox_h)

    size_class = ("tiny" if max(bbox_h, bbox_w) <= 16 else "small" if max(bbox_h, bbox_w) <= 32
                  else "medium" if max(bbox_h, bbox_w) <= 64 else "large")
    palette_class = ("1-bit" if n_colors <= 2 else "4-shade" if n_colors <= 4 else
                     "limited" if n_colors <= 16 else "rich" if n_colors <= 64 else "unrestricted")

    outline = _outline(work, keys)
    # Shape-based line detection is the more robust witness: when it finds
    # an outline on most of the silhouette, trust it (the edge heuristic
    # above fails when parts of the edge are cut against other elements or
    # a lineless effect like smoke owns most of the silhouette).
    from .structure import analyze as _structure
    struct = _structure(work)["summary"]
    if outline["style"] == "none" and struct["has_outline"] and struct["line_colors"]:
        from .color import hex_to_rgb
        top = hex_to_rgb(struct["line_colors"][0]["color"])
        top_l = float(lightness(top))
        outline = {**outline, "style": "black_contour" if top_l < 0.15 else "dark_contour",
                   "colors": [c["color"] for c in struct["line_colors"][:4]],
                   "pure_black": top == (0, 0, 0), "from_structure": True}
    outline["coverage"] = struct["outline_coverage"]
    outline["inner_line_share"] = struct["inner_line_share"]
    fams = _hue_families(colors)
    tones = [len(f["members"]) for f in fams if sum(n for _, n in f["members"]) >= 0.03 * opaque.sum()]
    tones_per_ramp = round(float(np.mean(tones)), 1) if tones else 0.0
    pairs = L.shading_pairs(work)
    verdicts = Counter(p["verdict"] for p in pairs)
    hue_shift = "n/a" if not pairs else max(("ok", "flat", "reversed"), key=lambda v: verdicts[v])
    hue_shift = {"ok": "shifted", "flat": "flat", "reversed": "reversed", "n/a": "n/a"}[hue_shift]

    a, b = keys[:-1, :-1], keys[:-1, 1:]
    c, d = keys[1:, :-1], keys[1:, 1:]
    chk = (a == d) & (b == c) & (a != b) & (a != -1) & (b != -1)
    dither_share = float(chk.sum() / max(1, opaque.sum()))
    dithering = "none" if dither_share < 0.01 else "sparse" if dither_share < 0.08 else "heavy"

    lighting = _light_direction(work, keys)
    # Pillow only when the outline shows no clear light direction either.
    pillow = bool(L.check_pillow_shading(work)) and lighting["strength"] < 0.05
    if tones_per_ramp <= 1.3:
        shading = "flat"
    elif pillow:
        shading = "pillow"
    elif dithering == "heavy":
        shading = "dithered"
    elif tones_per_ramp >= 6 or n_colors > 48:
        shading = "soft"
    else:
        shading = "cel"

    aa = _aa_share(work, keys)
    # Soft-edge share: real AA and single mid-tone shading pixels both count,
    # so only the extremes are meaningful.
    antialiasing = "none" if aa < 0.05 else "light" if aa < 0.25 else "heavy"

    orphans = [i for i in L.check_orphans(work)]
    singles = sum(len(i["pixels"]) for i in orphans)
    noise = round(singles / max(1, opaque.sum()), 3)
    filtered = n_colors > 64 or semi > 0.05

    if filtered:
        label = "not clean pixel art (blurred, filtered or resized)"
    elif palette_class in ("1-bit", "4-shade"):
        label = "minimal / handheld 4-shade" if palette_class == "4-shade" else "1-bit"
    elif n_colors <= 8 and antialiasing == "none" and outline["style"] in ("none", "black_contour", "black_inline"):
        label = "8-bit retro"
    elif n_colors <= 32 and antialiasing != "heavy":
        label = "16-bit"
    else:
        label = "modern hi-bit"

    profile = {
        "label": label,
        "size": {"canvas": [int(work.shape[1]), int(work.shape[0])], "sprite": [bbox_w, bbox_h],
                 "class": size_class, "upscaled": scale},
        "palette": {"colors": n_colors, "class": palette_class, "budget": budget["max_colors"],
                    "ramps": len(tones), "tones_per_ramp": tones_per_ramp, "hue_shift": hue_shift},
        "outline": outline,
        "shading": shading,
        "lighting": lighting,
        "antialiasing": {"level": antialiasing, "share": round(aa, 3)},
        "dithering": {"level": dithering, "share": round(dither_share, 3)},
        "cleanliness": {"semi_transparent_share": round(semi, 3), "single_pixel_share": noise,
                        "filtered": filtered},
    }
    profile["rules"] = _rules(profile)
    profile["lint"] = _lint_settings(profile, outline)
    return profile


def _rules(p: dict) -> list[dict]:
    """Map each detected trait to the guide rules that govern it."""
    r = []

    def add(trait, topic, rule):
        r.append({"trait": trait, "topic": topic, "rule": rule})

    o = p["outline"]["style"]
    if p["cleanliness"]["filtered"]:
        add("filtered", "cleanup", "Not clean pixel art yet: recover the pixel grid, quantise the palette and "
            "remove soft edges before anything else (cleanup_sprite).")
    if o == "none":
        add("outline: none", "lineart", "Every colour boundary is the line: keep staircases even on region edges "
            "and separate the sprite from the background by value contrast.")
    elif o in ("black_contour", "dark_contour"):
        add(f"outline: {o}", "lineart", "Keep a single dark contour on the silhouette only; separate inner parts "
            "with colour and value, not dark lines. Keep 1 px width and even steps.")
    elif o == "black_inline":
        add("outline: black_inline", "lineart", "Retro inline style: dark lines inside and outside. Keep it consistent; "
            "avoid lines in small areas where they turn into blobs.")
    elif o == "colored":
        add("outline: colored", "lineart", "Each part is outlined with the darkest shade of its own ramp; new parts "
            "must follow the same rule.")
    elif o in ("shaded", "selective"):
        add(f"outline: {o}", "subpixel", "Outline lightens toward the light. Sel-out reads well on dark backgrounds "
            "only; check readability_preview on a light background.")
    if p["outline"].get("pure_black"):
        add("pure black outline", "color", "Pure black is part of this style; keep it consistent (a tinted near-black "
            "would be a style change).")
    pal = p["palette"]
    add(f"palette: {pal['class']} ({pal['colors']} colours)", "color",
        f"Reuse existing colours for edits; stay within ~{pal['budget']} colours and "
        f"{pal['tones_per_ramp']:g} tones per ramp, matching the current style.")
    if pal["hue_shift"] == "flat":
        add("hue shift: flat", "color", "Shading only changes brightness. If that is not deliberate, rebuild ramps "
            "with generate_color_ramp (shadows toward violet, highlights toward yellow).")
    elif pal["hue_shift"] == "reversed":
        add("hue shift: reversed", "color", "Shadows lean toward yellow/green. Unless that is a deliberate mood, "
            "shift shadows toward violet.")
    s = p["shading"]
    if s == "pillow":
        add("shading: pillow", "antialiasing", "Shading follows the outline from every side. Choose one light "
            "direction and reshade by form.")
    elif s == "flat":
        add("shading: flat", "color", "Flat colours: keep shapes clean and clusters deliberate; add shading only "
            "if the style calls for it.")
    elif s == "cel":
        add("shading: cel", "cleanup", "Cel shading: hard-edged light and shadow shapes with one light direction; "
            "keep clusters clean, no noise.")
    elif s == "soft":
        add("shading: soft", "antialiasing", "Many tones: watch for banding and near-duplicate colours.")
    d = p["dithering"]["level"]
    if d != "none":
        small = p["size"]["class"] in ("tiny", "small")
        add(f"dithering: {d}", "dithering", "Dither only between neighbouring ramp colours and keep the pattern "
            "aligned to the canvas." + (" On a sprite this small it mostly reads as noise." if small else ""))
    aa = p["antialiasing"]["level"]
    if aa == "none":
        add("anti-aliasing: none", "antialiasing", "Crisp, aliased style: do not add AA to new parts.")
    else:
        add(f"anti-aliasing: {aa}", "antialiasing", "AA about half a step long, 1-2 shades, inside the silhouette "
            "only; new curves need the same treatment.")
    return r


def _lint_settings(p: dict, outline: dict) -> dict:
    """Which lint checks fit this style, and the line colours to use."""
    checks = list(L.CHECKS)
    skip = {}
    if outline.get("pure_black"):
        skip["colors.pure_black"] = "pure black is part of the style"
    if p["dithering"]["level"] == "heavy" and p["size"]["class"] in ("medium", "large"):
        checks.remove("dither")
        skip["dither"] = "deliberate dithering on a large sprite"
    if outline["style"] == "selective":
        checks.remove("outer_aa")
        skip["outer_aa"] = "selective outline lightens edges on purpose"
    if outline["style"] == "none" and outline.get("inner_line_share", 0) < 0.03:
        for c in ("l_corners", "jaggies"):
            checks.remove(c)
        skip["l_corners/jaggies"] = "no line art: lineless shapes (smoke, painted backdrop) have no lines to judge"
    return {"checks": checks, "line_colors": outline.get("colors", [])[:2]
            if outline["style"] not in ("none",) else [], "skipped": skip}
