# Line art

## Key rules
- Use one line width across the whole sprite. 1 px is the norm. A thicker width is allowed only if it stays uniform.
- Straight lines are built from equal runs. Pick clean ratios (1:1, 2:1, 3:1, 4:1) and never mix run lengths inside one line.
- On curves, run lengths must change monotonically: shrinking toward the 45° part, then growing. A run that is shorter or longer than both of its neighbours is a jaggy.
- 1 px lines step diagonally. Delete L-corner pixels ("pixel-perfect").
- Every colour boundary is a line, including shadow edges and lineless shapes. They all follow the same staircase rules.

## Terms
- **Run (step)**: consecutive pixels along the main direction of a line, between two diagonal moves.
- **Jaggy**: a place where the run rhythm breaks, which makes the line look bumpy.
- **Double / L-corner**: an extra pixel where a 1 px line goes horizontal and then vertical instead of stepping diagonally. It locally thickens the line.

```
L-corner (bad)    clean
##..              ##..
.##.              ..#.
..##              ...#
```
A pixel P is redundant when it has one line neighbour horizontally and one vertically, and those two neighbours touch each other diagonally.

## Straight lines
| Ratio | Run pattern | Approx. angle |
|---|---|---|
| 1:1 | 1,1,1,… | 45° |
| 2:1 | 2,2,2,… | 26.6° (isometric) |
| 3:1 | 3,3,3,… | 18.4° |
| 4:1 | 4,4,4,… | 14° |
| 1:0 | one run | 0° / 90° |

- Snap lines to these ratios whenever the design allows it. Move an endpoint by a pixel rather than accept irregular runs.
- If you need a slope between two ratios, use a strictly **periodic** pattern (2,1,2,1 or 3,2,3,2). A random mix such as 2,2,1,2,2,2,1 reads as jagged.

```
2:1 clean          mixed (bad)
##......           ##.......
..##....           ..#......   <- lone 1-px run
....##..           ...##....
......##           .....##..
```

## Curves
- Split a curve at its extremes (top, bottom, left, right) and at inflection points. Within each arc the run sequence must be **unimodal**. Example: `5,3,2,1,1,1,2,3,5` is good. `4,2,3,1,1,2,1,3` is bad.
- Keep the 45° section of round shapes short. A long 1,1,1 stretch makes a circle look octagonal.
- Similar sections of a symmetric shape should use mirrored run sequences.
- Diagnostic: imagine the smooth vector curve over the pixels. Every pixel that strays more than about half a pixel from it is a candidate fix.
- Draw fast, then **chisel**. Add or remove single pixels, or shift whole segments with a selection, rather than redrawing the line.

```
bump (bad)        monotone (good)
.......####       .......####
......#....       ......#....
....##.....       .....#.....
...#.......       ....#......
.##........       ...#.......
#..........       .##........
```

## Line width
- Check for 2×2 blocks of line colour. They mean the line thickens there (often an L-corner or a bad join).
- Interior detail lines may be thinner or lighter than the outer contour, but only as a deliberate, consistent choice.
- Small areas need thin lines. To make a line look thinner without removing pixels, recolour it lighter (see the `antialiasing` topic, line weight).
- On very small sprites, shapes often work better than line art. Lines eat the space that features need.

## Outline styles
| Style | Description | Use when |
|---|---|---|
| None | Flat colour shapes, edges defined by value contrast | Large, painterly or minimalist art; backgrounds |
| Black inline | Dark line around the outside **and** between all inner parts | Deliberate retro look. Otherwise it reads muddy and wastes pixels |
| Dark contour | Dark only on the outer silhouette; inner parts separated by colour | Small or moving game sprites that must pop on any background |
| Coloured | Each part outlined in the darkest shade of its own ramp | Softer, modern look; good for mid-size sprites |
| Shaded / selective | Outline colour follows the light: darker on the shadow side, lighter or absent on the lit side | Modern default for sprites on dark backgrounds. Test it on light backgrounds, where it can look broken |

- Use the same style across every asset in a scene.
- The outline "black" should be a tinted near-black (see the `color` topic), not #000000.
- Thicker or softer outlines need more AA to stay smooth.

## Mixels and scaling
- **Mixels** are pixels of different sizes in one image, such as a 2x-scaled element next to 1x art, or a rotated sprite whose grid is skewed. Avoid them.
- Scale only by integers (2x, 3x, 4x) with nearest-neighbour. Never use 1.5x or 0.5x and call it finished.
- Rotation by anything other than 90°, 180° or 270° breaks the grid. Rotate once, then clean every line by hand. Exception: free rotation required by gameplay (aiming arms, spinning props).

## Draw it right the first time
Describe straight edges with `draw_clean_line` (use `mode="snap"` for man-made objects so every step is identical) and curves with `draw_smooth_curve` control points instead of placing pixels by hand. Outline a finished shape with `outline_sprite` in the chosen style.

## Finding and editing existing line art
`analyze_structure` finds the line art in any sprite by shape: 1-2 px strips darker than both sides, forming strokes. It separates the outline (on the silhouette) from inner lines and reports how much of the silhouette the outline covers, the line width, the line colours, and the parts the lines separate. Dark shading bands are not counted as lines.
- `extract_lineart` puts the line art on its own layer and fills its place with the neighbouring colours (flats and shading underneath).
- `recolor_lineart` turns black lines into colored line art (the darkest shade of each part), into a tinted near-black, or into one chosen colour. It can work on the outline only or on the inner lines only.
- `lint_pixel_art` and `cleanup_sprite` apply the line rules (L-corners, staircase rhythm) to the detected line art only.

## Tools in this MCP
`analyze_structure`, `extract_lineart`, `recolor_lineart`, `draw_clean_line` (exact or snapped to a clean slope), `draw_smooth_curve` (Bezier control points to a pixel-perfect curve with gradual steps), `outline_sprite` (black, colored, shaded, selective), `draw_line`, `draw_path`, `draw_polygon`, `draw_circle`, `draw_ellipse_at`, `draw_pixels`, `erase_region`, `move_region`, `outline_cel`, `outline_native`, `suggest_outline_color`, `rotate_layer`, `flip_layer`, `get_pixels_rect`, `lint_pixel_art` (jaggy runs, L-corners, doubles), `readability_preview`.
