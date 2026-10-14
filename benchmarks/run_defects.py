"""Measure the linter and cleanup on injected defects.

usage: uv run python benchmarks/run_defects.py [OUT_DIR]

For every base sprite x defect type (fixed seeds):
  detect_warn  share of defect pixels within 1 px of a lint *warning*
  detect_any   same, warnings or info
  false_flags  lint warnings away from the defect (on the corrupted sprite)
  fix          share of defect pixels cleanup restored exactly (default mode)
  fix_aggr     same with keep_details=False
  resolved     share of defect pixels the expected check no longer flags after cleanup
  harm         pixels that were fine and cleanup changed (default mode)
Whole-image defects (soft alpha, upscale) score the share of pixels that
match the clean base after cleanup.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aseprite_mcp.pixelart import cleanup as CL  # noqa: E402
from aseprite_mcp.pixelart import lint as L  # noqa: E402
from aseprite_mcp.pixelart.color import oklab  # noqa: E402
from aseprite_mcp.pixelart.style import analyze_style  # noqa: E402
from benchmarks import defects as D  # noqa: E402

BG = (200, 203, 208)
TRAITS = {"pillow", "dither_small", "mixed_blacks"}  # judged as a whole, not pixel by pixel
try:
    FONT = ImageFont.truetype("/usr/share/fonts/google-noto/NotoSans-Regular.ttf", 16)
except OSError:
    FONT = ImageFont.load_default()


def same(a, b):
    """Per pixel: visually the same (both clear, or both opaque and close)."""
    ca, cb = a[..., 3] > 0, b[..., 3] > 0
    d = np.linalg.norm(oklab(a[..., :3]) - oklab(b[..., :3]), axis=-1) * 100
    return (~ca & ~cb) | (ca & cb & (d < 5) & (np.abs(a[..., 3].astype(int) - b[..., 3]) < 40))


def near(mask, pts, r=1):
    """Which mask pixels lie within r (Chebyshev) of any point."""
    hit = np.zeros_like(mask)
    h, w = mask.shape
    for x, y in pts:
        hit[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)] = True
    return mask & hit


def flags(issues, severity=None):
    return [tuple(p) for i in issues if severity in (None, i["severity"]) for p in i["pixels"]]


def flat(a):
    out = np.zeros((*a.shape[:2], 3), np.uint8)
    out[...] = BG
    al = a[..., 3:4] / 255.0
    return (a[..., :3] * al + out * (1 - al)).astype(np.uint8)


def panel(a, s, boxes=(), colour=(255, 0, 200)):
    im = Image.fromarray(flat(a)).resize((a.shape[1] * s, a.shape[0] * s), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for (x, y), c in boxes:
        d.rectangle([x * s, y * s, x * s + s - 1, y * s + s - 1], outline=c or colour, width=max(1, s // 4))
    return im


def sheet(panels, path):
    pad, lab = 12, 26
    W = sum(p.width for _, p in panels) + pad * (len(panels) + 1)
    H = max(p.height for _, p in panels) + lab + 2 * pad
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    x = pad
    for t, p in panels:
        d.text((x, pad), t, fill=(20, 20, 20), font=FONT)
        out.paste(p, (x, pad + lab))
        x += p.width + pad
    out.save(path)


def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for bname, bfun in D.BASES.items():
        clean = bfun()
        for dname, (inject, _) in D.DEFECTS.items():
            rng = np.random.default_rng(7)
            bad, mask = inject(clean, rng)
            st = analyze_style(bad)
            fixed, _ = CL.cleanup(bad, style=st)
            aggr, _ = CL.cleanup(bad, style=st, keep_details=False)
            row = {"base": bname, "defect": dname}
            if mask is None or bad.shape != clean.shape or dname == "soft_alpha":
                for key, res in (("fix", fixed), ("fix_aggr", aggr)):
                    row[key] = round(float(same(res, clean).mean()), 3) if res.shape == clean.shape else 0.0
                row["size_after"] = list(fixed.shape[:2][::-1])
                row["defect_px"] = int(mask.sum()) if mask is not None else None
                rows.append(row)
                continue
            iss = L.lint(bad)
            warn, anyf = flags(iss, "warning"), flags(iss)
            n = int(mask.sum())
            row["defect_px"] = n
            expected = D.DEFECTS[dname][1]
            if dname in TRAITS:
                # A whole-sprite trait: detected = the right check was raised.
                hit = [i for i in iss if i["check"] in expected]
                row["detect_warn"] = float(any(i["severity"] == "warning" for i in hit))
                row["detect_any"] = float(bool(hit))
            else:
                row["detect_warn"] = round(near(mask, warn).sum() / n, 2)
                row["detect_any"] = round(near(mask, anyf).sum() / n, 2)
            dil = near(np.ones_like(mask), [(x, y) for y, x in zip(*np.nonzero(mask))], 1)
            row["false_flags"] = sum(1 for x, y in warn if not dil[y, x])
            ok_before = same(bad, clean)
            row["fix"] = round(float((same(fixed, clean) & mask).sum() / n), 2)
            row["fix_aggr"] = round(float((same(aggr, clean) & mask).sum() / n), 2)
            # Fixed differently from the original still counts here: the
            # expected check no longer fires at the defect.
            left = [tuple(p) for i in L.lint(fixed) if i["check"] in expected for p in i["pixels"]]
            row["resolved"] = round(1 - near(mask, left).sum() / n, 2)
            row["harm"] = int((ok_before & ~same(fixed, clean) & ~mask).sum())
            row["harm_aggr"] = int((ok_before & ~same(aggr, clean) & ~mask).sum())
            rows.append(row)
            if bname in ("orb", "shapes"):
                s = 10
                dpts = [((x, y), None) for y, x in zip(*np.nonzero(mask))]
                lint_boxes = [(p, (255, 60, 60)) for p in warn] + [(p, (255, 170, 0)) for p in anyf if p not in warn]
                def boxes(res):
                    ok = same(res, clean)
                    return [((x, y), (0, 170, 0)) for y, x in zip(*np.nonzero(mask & ok))] + \
                           [((x, y), (230, 0, 0)) for y, x in zip(*np.nonzero(mask & ~ok))] + \
                           [((x, y), (255, 0, 200)) for y, x in zip(*np.nonzero(ok_before & ~ok & ~mask))]
                sheet([("czysty", panel(clean, s)), (f"błąd ({n} px)", panel(bad, s, dpts, (255, 0, 200))),
                       ("linter: czerw.=warning, pomar.=info", panel(bad, s, lint_boxes)),
                       ("clean-up domyślny", panel(fixed, s, boxes(fixed))),
                       ("clean-up dokładny", panel(aggr, s, boxes(aggr)))],
                      f"{out_dir}/{bname}_{dname}.png")
    json.dump(rows, open(f"{out_dir}/results.json", "w"), indent=1)
    return rows


def table(rows):
    cols = ["detect_warn", "detect_any", "false_flags", "fix", "fix_aggr", "resolved", "harm", "harm_aggr"]
    out = []
    for d in D.DEFECTS:
        rs = [r for r in rows if r["defect"] == d]
        agg = {}
        for c in cols:
            v = [r[c] for r in rs if c in r]
            if v:
                agg[c] = round(float(np.mean(v)), 2)
        out.append((d, agg))
    return out


if __name__ == "__main__":
    rows = run(sys.argv[1] if len(sys.argv) > 1 else "out/defects")
    for d, agg in table(rows):
        print(f"{d:14} " + "  ".join(f"{k}={v}" for k, v in agg.items()))
