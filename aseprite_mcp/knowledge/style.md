# Recognising and matching a style

## Key rules
- Before editing or extending someone else's sprite, run `analyze_style` and match what it finds. New pixels must use the same outline style, palette, shading and edge treatment.
- A trait is only a mistake if it is not deliberate. Pure black in a black-outline style, or dithering on a large retro background, is style, not error. Use `lint_pixel_art(respect_style=true)` to skip those findings.
- If the label says "not clean pixel art", the image was blurred, resized or filtered. Clean it first (`cleanup_sprite`), then analyse again.

## Style dimensions

| Dimension | Values | How it is recognised |
|---|---|---|
| Outline | none, black_contour, black_inline, dark_contour, colored, shaded, selective | Silhouette-edge pixels compared with the first different colour inward. One dark colour on most of the edge is a contour; the same colour also used inside means inline. Darker edges in the hue of the part they wrap mean colored. Edges lighter toward the light mean shaded; dropped there entirely means selective. |
| Shading | flat, cel, soft, pillow, dithered | Tones per hue family: about 1 is flat, 2-5 with clean shapes is cel, 6+ is soft. Lightness rising from every edge with no light direction is pillow. Checkerboard over a large area is dithered. |
| Light | direction + strength | Where the light pixels sit relative to the sprite's centre. Strength below ~0.03 means no clear direction (flat or all-round shading). |
| Edges | none / light / heavy soft-edge share | Single in-between pixels on high-contrast boundaries. Real AA and single mid-tone shading pixels both count, so treat it as a coarse signal. |
| Palette | 1-bit (2), 4-shade (<=4), limited (<=16), rich (<=64), unrestricted | Unique opaque colours. More than 64 almost always means filtering. |
| Hue shift | shifted, flat, reversed | Every pair of touching shades of one hue: the lighter step should turn toward yellow. |
| Size | tiny <=16, small <=32, medium <=64, large | Bounding box of opaque pixels. |

## Style labels
| Label | Typical traits |
|---|---|
| 1-bit / minimal 4-shade | 2-4 colours, shapes and value only, no AA. |
| 8-bit retro | <=8 colours, black or no outline, no AA, flat or simple cel shading. |
| 16-bit | <=32 colours, outlined (often coloured or shaded), cel shading, little AA. |
| modern hi-bit | Many colours, heavy AA, selective/shaded outlines, soft shading, sub-pixel animation. |
| not clean pixel art | Blur, semi-transparent edges, non-integer scaling or JPEG noise. Run `cleanup_sprite`. |

## Matching a style when adding to a sprite
- **Colours:** pick only from the sprite's existing colours (`analyze_palette`). A new material gets a ramp with the same tones per ramp and the same hue-shift habit.
- **Outline:** copy the outline rule exactly. For colored outlines, each new part is outlined in the darkest shade of its own ramp. For selective outlines, drop or lighten the outline on the lit side, on the detected light direction.
- **Light:** shade new parts from the detected light direction.
- **Edges:** heavy-AA styles need AA on new curves too (about half a step, 1-2 shades, inside the silhouette). No-AA styles stay crisp.
- **Detail density:** match the pixel size of details. Never add features smaller or finer than the existing ones (mixels in spirit).

## Clean-up of a user's sprite
`cleanup_sprite` runs the rule-based fixes in this order:
1. Soft alpha edges become fully opaque or fully clear.
2. The pixel grid is recovered, including non-integer upscales, blur and JPEG.
3. Near-identical colours merge. A far-too-large palette from blur or resampling is reduced to the size budget; a clean sprite keeps its palette.
4. Floating pixels are removed.
5. Low-contrast specks are recoloured. Distinct single pixels (eyes, shines) are kept for review.
6. Gaps in a single-colour outline are closed.
7. Pixel-perfect pass on lines.
8. Line staircases get an even rhythm, keeping their end points.
9. Bands that copy a staircase are shortened to about half a step.
10. Light pixels outside the outline are removed, unless the outline is selective.

Dithered areas are never touched by the pixel-level fixes. It recognises the style first so deliberate traits stay, then lists the remaining issues. Pillow shading, remaining jaggies on curves and readability problems need an artist's judgement: fix them by hand, guided by `lint_pixel_art` and `readability_preview`.

## Tools in this MCP
`analyze_structure` (line art and pixel roles), `analyze_style`, `cleanup_sprite`, `lint_pixel_art` (respect_style), `analyze_palette`, `readability_preview`, `suggest_outline_color`, `generate_color_ramp`, `pixel_art_guide`.
