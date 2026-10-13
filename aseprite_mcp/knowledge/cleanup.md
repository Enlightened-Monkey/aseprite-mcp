# Clean-up

## Key rules
- Clean-up is a pass you repeat, not a final touch. Rough work (blobs or loose lines) can change completely before it is finished.
- Think in **clusters**: every same-colour group of pixels is a deliberate shape that describes form, material or light. Remove noise and orphan pixels.
- Never ship raw transforms. Downscaled, rotated or upscaled art is only an underlay that you redraw by hand.
- Read sharpness from light and dark masses and contrast, not from internal lines or filters.
- Check every pass against a rubric, and make variants when unsure.

## Clusters and shape language
- Treat each cluster like a mosaic tile or a flat vector shape, with clean edges and an intentional outline that follows the line-art staircase rules.
- A texture with about 5 colours and clean clusters beats a noisy texture with more colours.
- Avoid uniform "lazy" marks (scattered dots, even speckle) that describe nothing.
- Start shading as blocked shapes of light and shadow. Detailed art still sits on that block structure.
- AA can refine a cluster's edge. It cannot rescue a badly shaped cluster.

## Noise and orphan pixels
- An **orphan** is a single pixel whose colour differs strongly from all 8 of its neighbours.
- Remove orphans, **except** intentional single pixels: eye pupils, specular glints, and AA pixels whose value lies between two neighbouring colours.
- If more than about 20-30% of the clusters in a region are 1-2 px in size, the region is noisy. Merge those pixels into the majority colour around them.
- Dither or speckle near facial features reads as dirt. Keep faces clean.

## Downscale, then redraw
1. Resize the big sketch or image to the target size with nearest-neighbour or majority-colour sampling, so no blended colours appear. Quantise to the palette.
2. Lock the result as an underlay (about 30-50% opacity), with an empty layer above it.
3. Redraw the eyes, outline, hands and key edges first, eyedropping existing colours.
4. Drop the details that don't survive the smaller size, and favour readability.
5. Lint for orphans, doubles and colour count.
Going big to small works well. Going small to big is essentially a full redraw.

## Rotation clean-up
- Rotate only by 90°, 180° or 270° for free. Any other angle produces broken lines and doubled or missing pixels.
- After rotating, rebuild each line with clean runs, restore the outline continuity, remove orphans, and restore single-pixel features such as eyes.
- When animating a rotation, clean every frame. Unedited rotated parts look robotic.

## Selection moves
- Fix proportions, perspective and off-model parts by moving regions (head size, limb length, how much of a lid shows) instead of redrawing.
- After a move, check the 1 px seam around the moved area for doubles, gaps and broken outlines.

## Sharpness without filters
1. Replace internal lines on small forms with clean light and shadow shapes. Small internal lines merge into blotches.
2. Widen the value range: add brighter highlights and darker accents in deep crevices.
3. Separate overlapping forms (a cheek against the head, an arm over the torso) with a darker edge or a cast shadow.
4. Raise contrast across the whole sprite by recolouring the palette ramps (value, saturation) rather than moving pixels.

## Tangents
Avoid shapes that just touch at a corner, and shapes lined up end to end with a 1 px seam. They merge into one ambiguous form.
```
tangent (bad)   fixed
.DWD.           .DWD.
.DWD.           .DWD.
.DDD.  <-seam   .DDD.
.DWD.           .DDD.
.DWD.           .DDW.
```
Shorten, offset or merge one shape so the two clearly overlap or clearly separate. A brief tangent in a fast animation frame is acceptable.

## Light consistency
- Choose the light direction at the start. Shading, highlights, outline shading and cast shadows all follow it.
- Upper parts cast shadows onto lower ones (a hat brim onto the face, an arm onto the torso).
- A shaded outline is lighter on the lit side and darker on the shadow side.
- Rounded edges plus consistent light make an object read as 3D.

## Variants
When a feature won't read (a smile, a pose, a held object), make 2-4 variants on hidden layers or extra frames and compare them side by side at 1x. Pick one, or combine the best parts. If none works after several tries, redraw that area from scratch.

## Review rubric
| Axis | Question |
|---|---|
| Silhouette | Does it read as a flat fill? Do the important parts stick out? |
| Design | Is it one main colour plus one accent? Does the key feature read? |
| Clusters | Are the shapes intentional? Any orphans or noise? |
| Lines | Uniform width? Clean runs? No L-corners? |
| Light | One direction? Cast shadows? No pillow shading? |
| AA | Sparse, inside the silhouette, no banding? |
| Colour | Within budget? Hue-shifted ramps? No near-duplicates or pure black? |
| Readability | Does it pass the 1x, blur, value and blind-read checks? |

## Automatic pass
Run `cleanup_sprite` first on any rough, resized or noisy sprite. It recovers the pixel grid, merges near-duplicate colours, removes floating pixels and low-contrast specks, closes outline gaps, applies pixel-perfect lines and removes AA outside the outline, while keeping style traits it recognises. Everything it cannot judge (jaggies, banding, pillow shading, readability) is listed for a manual pass. See the `style` topic.

## Tools in this MCP
`cleanup_sprite`, `analyze_style`, `lint_pixel_art` (orphans, doubles, jaggies, banding, pillow shading, colour count), `readability_preview`, `move_region`, `copy_region`, `erase_region`, `erase_color`, `draw_pixels`, `rotate_layer`, `flip_layer`, `import_image_as_layer`, `quantize_to_palette`, `duplicate_layer`, `set_layer_visibility`, `set_layer_opacity`, `adjust_hsl`, `get_color_stats`, `analyze_palette`.
