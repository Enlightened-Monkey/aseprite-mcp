# Sub-pixel techniques

## Key rules
- Sub-pixel work imitates detail or motion smaller than 1 px by recolouring edge pixels. In a still image it equals AA and line weight. In animation it makes shapes appear to move by fractions of a pixel.
- Conserve the "colour mass". Shifting an edge by half a pixel turns the boundary pixel into a tone halfway between the shape and the background.
- Use **1-2 intermediate tones** per colour pair, taken from the existing palette. Don't invent new colours for every frame.
- Keep faces crisp, never let a new row or column pop in as a single pixel, and don't overdo it.
- Use it where 1 px moves are too coarse: idles and breathing, easing near keys, sprites of 32 px or smaller, and wind, shiver or shake.

## Colour-mass shifting
Think of each pixel's contrast against the background as water in a cup. Moving the shape pours part of it into the next cup, and the total stays about one pixel's worth.
```
frame   pixels   mass split
0       [D][.]   100 / 0
1       [d][l]    75 / 25   d = dark-mid, l = light-mid
2       [M][M]    50 / 50
3       [l][d]    25 / 75
4       [.][D]     0 / 100
```
- The split needn't be 50/50. A 25/75 split gives a quarter-pixel shift.
- In a real sprite, only the **edge** row or column changes. The interior stays the same.
```
edge moving left by 1/2 px (B = body, C = background, m = existing mid tone)
f0: B B C C      f1: B B m C
```
- Any existing colour that is roughly between the two works. Exact mixes aren't needed.
- Hue carries over too. The intermediate tone should sit on the ramp between the two colours.

## Line weight and split pixels
- A thinner line looks lighter on a light background. A width-0.5 line is a 1 px line in a light-mid tone. A width-1.5 line is a dark row plus a half-tone row.
- On a dark background the logic reverses: thin lines are closer to the background value.
- **Split pixel**: where a 1 px line jogs by one column, draw the transition row as two pixels of about 50% value instead of a hard step. The line then seems to pass between the columns. Use this for gentle curves in thin lines (mouths, brows, collars, veins).
```
hard jog   split
D .        D .
D D        M M
. D        . D
```
- Where you put AA changes the perceived shape. AA in the outer corners of a diagonal makes it bulge, AA in the inner corners pinches it, and AA on one side of a rectangle skews it.

## Selective outlines and background dependence
- A selective outline is dark on the shadow side and lighter, coloured or broken on the lit side. It is really AA against whatever is behind the sprite.
- It looks good on **dark** backgrounds and jagged on light ones. For mixed backgrounds, make it fuller and less broken, or use a solid shaded outline.
- Never put light AA on the outer outline of a moving sprite with a transparent background. The outer edge must stay outline-dark or become background.

## Animating with sub-pixels
| Rule | Detail |
|---|---|
| Row pops | When a curve moves, a new extreme row or column must appear (or vanish) with **at least 2-3 px**, never 1 |
| Direction | The sub-pixel motion runs **along the edge angle**. A near-horizontal edge moving up shows its steps and AA sliding sideways |
| Shift amount | For a vertical shift f on an edge of slope s, slide the step positions by about f / s horizontally |
| Amplitude | 0.5-1 px for idles. Use whole-pixel moves for anything larger |
| Faces | Keep eyes and mouth in 2 crisp states and snap between them while the body sub-pixels. Morphing eyes stand out badly |
| Offsets | Not everything moves: the shoulders lead, the head lags 1 frame, the feet stay fixed. Some parts can just slide by whole pixels |
| Favour keys | In-betweens should sit close to one key (25/75), not at 50/50 |
| Extremes | Define the start and stop keys first, then sub-pixel only the in-betweens |
| Timing | Hold extremes 1.5-2.5x longer than in-betweens, e.g. 160-200 ms at the keys and 80-120 ms between |
| Tones | 1-2 AA shades per colour pair. More shades only smooth the steps a little |
| No AA available | With very few colours, let a few whole pixels "swim" across a surface instead |

Example of a 6-frame breathing idle: the chest and shoulders shift 0.5 → 1 → 0.5 → 0 px, the head moves 0 or 1 px with a one-frame delay, and the feet stay fixed. Durations are 180, 100, 100, 180, 100, 100 ms.

- A whole circular loop can happen within a 3×2 px area. It is good for floating, bobbing and vibrating.
- The silhouette may barely change, which makes onion skin useless. Compare frames by pixel difference instead.
- Rounded outline corners can gain or lose a pixel between frames for extra life. Use this sparingly.

## Don't overdo it
- If more than about 25-30% of the opaque pixels change colour per frame in a subtle idle, the sprite looks like melting jelly.
- Sub-pixel shifting is about **movement**, not animated light.
- Shifting everything can create banding. Whole-pixel moves are often fine. Motion and readability come first.

## Tools in this MCP
`duplicate_frame_range`, `copy_frame`, `draw_pixels`, `get_pixels_rect`, `compare_frames` (diff instead of onion skin), `set_frame_duration`, `tween_cel_positions_eased`, `oscillate_cel_positions`, `offset_cel_positions`, `audit_animation`, `shade_color`, `get_palette`, `readability_preview` (light/mid/dark backgrounds for sel-out), `lint_pixel_art` (AA outside the outline).
