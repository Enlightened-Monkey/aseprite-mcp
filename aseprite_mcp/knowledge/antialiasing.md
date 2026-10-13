# Anti-aliasing (AA)

## Key rules
- AA means intermediate-colour pixels placed in the concave corners of a staircase between two contrasting colours. Each AA colour must sit between those two colours in value.
- Make each AA segment about **half the step length**, using **1-2 shades**. Too little AA is better than too much.
- On a transparent background, put AA **inside** the silhouette (between the fill and the outline), never outside it.
- Pure 45° lines usually get no AA. AA that copies the step pattern of a line causes banding.
- Judge AA at 1x or 2x zoom only. Effects that look fine zoomed in often look blurry at 1x.

## Placement
The AA pixel continues the higher step's row on the light side of the corner:
```
no AA           1 shade (a)        2 shades (a dark-mid, b light-mid)
##......        ##a.....           ########aabb........
..##....        ..##a...           ............########aabb
....##..        ....##a.
```
| Step length L | AA pixels per step | Shades |
|---|---|---|
| 1 (45°) | 0 (or a few sparse ones) | 1 |
| 2-3 | 1 | 1 |
| 4-6 | 2-3 | 1-2 |
| 7+ | ≈ L/2 | 2, darker next to the corner, lighter further away |

- AA darkness should decrease with distance from the corner, the way coverage falls off.
- Wrong: AA running the full step length. That forms a second line next to the first, which is banding.
- Wrong: AA on every corner of a 45° line on both sides. The line then reads 3 px wide and blurry.
- Pick AA colours from existing ramps. Reuse the same 1-2 AA colours across the whole sprite.

## Where to use AA
| Use AA | Skip or minimise AA |
|---|---|
| High-contrast edges (dark outline on a light fill) | Low-contrast colour pairs |
| Tight curves and round shapes | Tiny sprites (≤16 px); AA blurs them |
| Focal areas such as eyes and faces | Very small palettes (≤4-8 colours) |
| Large sprites (64 px and up need it) | Deliberately crisp or flat styles |
| Hand clean-up of downscaled art | The outer silhouette on unknown or varied backgrounds |
| Changing apparent line weight; sub-pixel animation | Pure 45° lines |

## Curves at 45°
On a 45° section that is part of a curve, the AA pattern changes the perceived curvature:
- **Convex** (bulging outward): use lighter AA in the middle of the section and darker AA at its ends.
- **Concave**: use darker AA in the middle and lighter AA at the ends.
- Put these AA pixels on the fill side of the outline.

## Irregular edges: gap filling
For a line whose runs are irregular, imagine the ideal straight line. Fill the **largest** gaps between the pixels and that line with a mid colour. That often suffices. Optionally fill smaller gaps with a lighter colour: the smaller the gap, the closer the colour to the background.

## Line weight through colour
At 1 px, value controls the apparent thickness. On a light background, darker means thicker and lighter means thinner. On a dark background this reverses.

| Effect | How |
|---|---|
| Thinner line | Draw it in a mid colour instead of the darkest |
| Bolder line | Add darker-mid AA in the step corners |
| Tapered stroke | Make the last 1-2 pixels at each end lighter |
| About 1.5 px line | One dark row plus a parallel half-tone row on part of its length |

Elaborate 3-shade AA looks almost identical to 1-2 shades at 1x. Don't spend colours on it.

## Banding
**Banding** is two or more runs of different colours lying exactly alongside each other, with the same length and aligned step positions. It makes curves blocky and lines thick, and it blurs outlines.
```
banding (bad)      fixed
###aa.....         ###a......
...###aa..         ...###....
......###aa        ......###a
```
Fixes:
- Shorten the band by 1-2 px at alternating ends.
- Add 1-2 px to the band so its steps no longer align with the line.
- Use a darker AA shade so the band reads as part of the line.
- Offset the step positions of parallel shading bands, as in hair or cloth stripes.

## Pillow shading
**Pillow shading** is shading arranged as concentric rings that follow the outline: dark at the edge and light in the centre, ignoring the light source. It is banding all around the silhouette and looks flat and amateurish.
Fix: choose a light direction. Put the shadow shape on the side away from the light and the highlight toward it. Let the shadow edge follow the **form** (where the surface turns away from the light), not the outline.

## Automatic AA
`antialias_lines` applies these rules to every step of a line colour: AA in the concave corners, about half the neighbouring step long, darker next to the corner with 2 shades, never on transparency, 45-degree parts left crisp, colours snapped to the palette. Check the result at 1x with `readability_preview` and run `lint_pixel_art` for banding.

## Tools in this MCP
`antialias_lines` (corner AA about half a step long, 1-2 shades, inside only), `draw_pixels`, `get_pixels_rect`, `get_composite_rect`, `shade_color`, `analyze_palette`, `lint_pixel_art` (banding, pillow shading, AA outside the outline), `readability_preview` (1x and blur views reveal banding), `export_frame`.
