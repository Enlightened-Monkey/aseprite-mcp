# Dithering

## Key rules
- **Don't dither by default.** Shade with clean, solid shapes first, and dither only if a specific need remains.
- Never dither small sprites (under about 48 px) or anything animated. Dither patterns shimmer as they move.
- Dither only between **adjacent, low-contrast ramp colours**. To cover a wide value range, add intermediate colours and dither each neighbouring pair.
- Use ordered patterns aligned to the **canvas** grid. Never let two minority pixels touch orthogonally ("wide pixels").
- Use less than you think. Dither should read as a blend, not as a pattern.

## When to use it
| Use | Avoid |
|---|---|
| Large static gradients: skies, space, big walls | Small sprites, icons, UI glyphs |
| Very large cel-shaded areas that look striped and flat | Anything that moves or animates |
| Hard palette limits (2-4 colours) | Tilesets that are mirrored or rotated |
| Texture on large surfaces (patches, not gradients) | High-contrast colour pairs |
| A small, static soft transition (one highlight melting into a shade) | Main shading of a character |

Low-contrast cel bands without dither already look soft. Banding in a sky is fine when the steps between bands are small.

## Ordered checker ladder
These are 4×4 cells, with `#` as the minority colour, going from colour A toward colour B:
```
1/16   2/16   4/16   6/16   8/16 (checker)
#...   #...   #.#.   #.#.   #.#.
....   ....   ....   .#..   .#.#
....   ..#.   #.#.   #.#.   #.#.
....   ....   ....   ...#   .#.#
```
Above 50%, swap the colours and run the ladder backwards (10/16 = 6/16 inverted, and so on).

| Gradient length (px) | Levels to use |
|---|---|
| <8 | none, or a single 50% band at the seam |
| 8-24 | 50% only, or 25/50/75 |
| 24-64 | 25/50/75, optionally 12.5/87.5 |
| >64 | the full ladder, or add more real ramp colours |
- Each level forms a **band**. Make bands about equal in width, in hand-made, banded fashion rather than per-pixel noise.
- Compute every pattern from absolute canvas coordinates, such as `(x+y) % 2` for the checker. Separate fills then share the same phase, and no doubled pixels appear where they meet.
- When a dither band follows a curve or diagonal, fix phase breaks by nudging the whole block 1 px, then patch the seam pixels.
- Mirroring an even-width dithered tile flips the checker phase on the axis. Make a separate tile instead.

## Contrast limit
- If two colours are more than about one ramp step apart (roughly ΔV > 20, or obviously different hues), the dither looks harsh and noisy. Insert a middle colour.
- With an unlimited palette, prefer adding a real in-between colour over dithering.

## Other patterns
| Pattern | Look | Use |
|---|---|---|
| Parallel lines (1 on, 1 off, horizontal or vertical) | 50% mix with a scanline feel | fake translucency, retro style, motion streaks in limited animation |
| Broken lines | lines that break into dashes toward the light | scanline-style gradients |
| Dents (one checker row on a colour boundary) | serrated edge | grass edges, rims; softening a single band seam on backgrounds |
| Interwoven | dither patches of different levels interpenetrating with wavy borders | clouds, foliage, planets |
| Stylised motifs (dashes, blades, dots, rings, scales) that shrink toward the other colour | organic texture | grass, water, rock, clouds; needs space |
| Random noise | grainy | generally avoid. If you need it, make it a repeating tile and clean it up by hand |

- **Texture is not a gradient.** Texture describes material (bricks, shingles, bark). A gradient describes light. Texture can be drawn with clean clusters and no dither at all.
- Dithering two different hues produces an optical mix. This matters only under hard palette limits.

## Checklist
1. Is the area static, larger than about 16 px along the gradient, and not part of a small sprite? If not, don't dither.
2. Are the two colours adjacent on a ramp? If not, add an intermediate colour.
3. Is the pattern from the ordered ladder and canvas-aligned?
4. Are there any 2×1 minority clumps or phase seams?
5. Check at 1x. If the pattern is visible as a pattern, reduce it.

## Automatic dithering
`dither_fill` mixes only neighbouring ramp colours, keeps the pattern aligned to the canvas so separate fills tile seamlessly, and can be restricted to existing pixels (`mask="opaque"`) or one colour (`only_color`). `levels=2` gives hard bands for comparison.

## Tools in this MCP
`dither_fill` (ramp dither with the clean density ladder, aligned to the canvas, optional shape mask), `apply_dither_gradient`, `apply_dither_pattern`, `generate_color_ramp`, `draw_pixels`, `lint_pixel_art` (dither on small sprites), `readability_preview` (blur view shows whether dither blends), `get_color_stats`.
