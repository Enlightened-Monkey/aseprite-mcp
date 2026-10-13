# Perspective and projections

## Key rules
- Pick one projection for gameplay, then draw the world, the characters and the props in it. Never mix camera angles in one scene.
- Game projections are orthographic: no vanishing points, and objects don't shrink with distance. Depth comes from parallax layers, overlap and value.
- Receding edges use clean, constant ratios: **2:1** for isometric and **1:1** for 45° oblique or dimetric. Vertical edges stay perfectly vertical.
- Work on a grid. Build objects from primitives (box, cylinder, pyramid, cone, sphere) before adding detail.
- Cheat for readability. Characters and upright objects show their front even in top-down views.

## Projection catalogue
| Projection | Receding lines | Faces shown | Typical use |
|---|---|---|---|
| Side view | none (only horizontal and vertical) | 1 (front), optionally a sliver of top | platformers, fighting games |
| Top-down (3/4) | none; top and front faces stacked vertically | top + front | RPGs, adventure games |
| Top (90°) | none | top only | maps, shooters, vehicles |
| Isometric | 2:1 (≈26.6°) | top + 2 sides | tactics, city builders |
| 45° dimetric (planometric) | 1:1 | top (an unsquashed square rotated 45°) + 2 tall sides | maps with tall structures |
| Oblique | 1:1 (sometimes 2:1) for depth only | flat front + top + one side | towns, interiors, furniture |
| True perspective | converge to vanishing points | any | illustrations, not grids |

## Side view
- Only 1-2 planes are visible. Build depth from parallax layers: nearer layers move faster and have more contrast, farther layers move slower and are lighter or less saturated.
- A slight top tilt that shows the walkable ground surface is common. Keep it consistent across all tiles.

## Top-down (3/4)
- Use a square tile grid. The camera height is set by the ratio of a block's visible **front** face to its **top** face:

| Front : Top | Camera | Feel |
|---|---|---|
| >1 | low | brawlers, lots of character front |
| ≈1 | medium | most RPGs |
| =1 exactly | medium | easiest tiles (one top tile plus one front tile, same size) |
| <1 | high | terrain and altitude read well |
- Every block in a scene must use the same ratio.
- Draw characters at about 45° tilt (face plus top of head visible), whatever the room geometry.
- Walls drawn converging toward the centre of a room clash with 45° sprites near the bottom wall. Omit the near wall.

## Isometric
- Lines are 2 px horizontal per 1 px vertical, with constant runs. Never mix in 1 px or 3 px runs.
```
##
  ##
    ##
      ##
```
- **Tiles**: a diamond W wide × W/2 tall. Use widths that are multiples of 4 (16×8, 32×16, 64×32).
- **Lineless diamond** (best for tiling floors): the top and bottom tips are 2 px wide, the left and right tips are 2 px tall, and each row grows by 2 px per side. For a 16×8 pitch the rows are 2, 6, 10, 14, 14, 10, 6, 2, centred. Neighbouring diamonds in an alternate colour fill the 1 px leftovers.
```
.......##.......
.....######.....
...##########...
.##############.
.##############.
...##########...
.....######.....
.......##.......
```
- Outlined diamonds on a strict grid create chunky 4 px blobs where the lines of neighbouring tiles cross. Prefer lineless tiles separated by colour, or accept free-form outlines that aren't grid-exact.
- **Odd/even**: an even width gives symmetric 2 px tips. An odd width forces a 1 px or 3 px tip and an asymmetric shape. Keep all iso dimensions even.
- **Screen position** of tile (c, r) at height h: `sx = ox + (c - r) * W/2`, `sy = oy + (c + r) * W/4 - h`. Draw back to front, by ascending c + r, then by height.
- **Cube**: draw the top diamond, then drop vertical edges from its left, right and front corners by the cube height, then close the bottom with 2:1 lines parallel to the top edges. For a cube, the height is commonly W/2 px (a mathematically true iso cube is about 0.58 × W tall). Use one convention for the whole scene. Shade with three tones: top lightest, the side facing the light mid, the other side darkest.
- **Circle on the floor**: an ellipse at 2:1 (horizontal axis ≈ 0.71 × diamond width, vertical axis half that), touching the midpoints of the diamond's edges.
- **Circle on a wall**: draw a normal circle at the face's size, shear it vertically by 0.5 px per px in the direction of the face, then clean the jaggies by hand.
- **Pyramid or cone apex**: raise the centre of the base by the height and connect it to the base corners, or to the outer points of the ellipse for a cone.
- **Flat side art to iso**: scale the width to about 87%, shear 1 px up per 2 px across, then re-pose the parts and clean up. Shear in the direction the object faces.
- Terrain is mostly flat, with height as stacked cubes. Slopes run only along the grid axes.

## 45° dimetric and oblique
- **Dimetric**: the top face is a true square rotated 45°. Edges are 1:1 single-pixel diagonals. Sides look taller than in iso.
- **Oblique**: the front face stays undistorted on a 90° grid. Depth edges are parallel 1:1 lines (or 2:1, or chunky 2×2 or 3×3 steps) and never converge. A top-down map sheared 1:1 vertically becomes oblique.
- Use the same depth direction and slope for every object.

## Organic objects and consistency
- Block organic forms with a few ovals on a sketch layer, trace their union as the silhouette, then add masses, details and light.
- Put objects side by side to compare top/front ratio, slope and light direction. If an object seen from one view can be misread, change its silhouette.

## Tools in this MCP
`draw_line`, `draw_line_at`, `draw_polygon`, `draw_path`, `draw_ellipse_at`, `fill_area`, `create_tilemap_layer`, `set_tiles`, `draw_on_tile`, `get_tilemap_info`, `add_layer` (grid and sketch layers), `set_layer_opacity`, `offset_cel_positions` (parallax), `lint_pixel_art` (jaggy runs), `readability_preview`.
