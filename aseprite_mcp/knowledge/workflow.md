# Workflow

## Key rules
- Decide the size, palette budget and light direction **before** you draw.
- Work from big to small: silhouette, line art, flats, shading, AA, details. Don't polish one area while the silhouette is still wrong.
- Keep the rough sketch on its own low-opacity layer as a safety net. Merge drawing layers when they multiply.
- After every stage, look at the result at 1x and run the linter. Fix problems by chiselling (adding or removing single pixels) instead of redrawing.
- Plan at least one review-and-iterate pass. First passes are never final.

## 1. Sizing decisions

1. List the features that must read, such as eyes with expression, a held item, fingers or an animated mouth.
2. Give the smallest of them ≥1-2 px of its own plus 1 px of clearance, then build the sprite size around it.
3. Choose proportions by function: about 2 heads tall for tiny overworld sprites, 3-4 for battle sprites, 6-7 for realistic ones.
4. Bigger sprites need cleaner lines and more frames to animate smoothly. Pick the smallest size that does the job.

| Sprite height | Features that fit | Colours (incl. outline) | Steps per ramp |
|---|---|---|---|
| ≤16 px | silhouette, 1-2 identity features, 1-2 px eyes | 4-8 | 2-3 |
| 16-32 px | eyes with pupils, simple mouth, mitten hands | 8-12 | 3-4 |
| 32-64 px | expressions, thumb and finger by value, AA | 12-16 | 4-5 |
| >64 px | full detail, careful AA, sub-pixel animation | 16-32 | 5-7 |

**Canvas presets**: these are handy for full scenes or for judging a sprite against the screen.

| Preset | Size |
|---|---|
| Tile | 16×16 or 32×32 |
| Handheld 8-bit | 160×144 |
| Handheld 32-bit | 240×160 (a good maximum for a first scene) |
| Home 8/16-bit | 256×224 / 256×240 |
| Modern low-res | 320×180 or 384×216 (16:9, integer-scales to 1080p) |

A game character usually covers about 0.5-4% of the screen area. 2-3% feels comfortable. Export only at integer scales.

## 2. Single sprite pipeline

| # | Stage | What to do | Check |
|---|---|---|---|
| 1 | Setup | `create_canvas`, then build the palette from 2-3 main hues plus one accent (`generate_color_ramp`, `set_palette`). Choose the light direction. | Colour count fits the budget |
| 2 | Sketch | Use a `sketch` layer at about 30% opacity. Block the shape with simple ovals and boxes. | Proportions, pose |
| 3 | Silhouette | Fill the whole shape in one flat colour on a new layer. Make limbs, weapons and hair stick out from the body. | `readability_preview` silhouette view reads at 1x |
| 4 | Line art | Use 1 px lines (`draw_path`, `draw_line`), or skip lines and work with shapes. Clean the runs and L-corners. | `lint_pixel_art` jaggies and doubles |
| 5 | Flats | Fill each part with its mid ramp colour (`fill_area`). Touching parts need different colours or values. | No two adjacent parts look the same |
| 6 | Shading | Block light and shadow as clean shapes that follow the form and the single light source. Add highlights last. | No pillow shading, no banding |
| 7 | Outline | Recolour the outline in your chosen style (black contour, coloured, shaded or selective) with `outline_cel` and `suggest_outline_color`. | No pure black unless that is the intended style |
| 8 | AA | Add selective AA on high-contrast curves and at the face. 1-2 shades, inside the silhouette. | AA not outside on a transparent background |
| 9 | Lint | Run `lint_pixel_art` and fix the reported pixels with `draw_pixels`. | Orphans, doubles, banding, colour count |
| 10 | Readability | Run `readability_preview` (1x/2x, blur, silhouette, value, light/mid/dark background). Then describe what the sprite reads as without assuming the intent. | Does the blind read match the intent? |
| 11 | Iterate | Fix the worst problem and repeat from 9. When unsure, make 2-3 variants on hidden layers and compare them. | |

**Value-first alternative**: draw stages 5-6 in 3-5 greys, check the value structure, then map each grey to a ramp step with `replace_color`.

**From a large or reference image**: downscale to the target size with nearest-neighbour or quantised sampling (`import_image_as_layer`, `quantize_to_palette`). Keep the result as a locked underlay and redraw the eyes, outline and key features by hand on a layer above it. Never ship a raw downscale.

## 3. Animation pipeline

1. **Design for motion**: make sure the still sprite is clean first. Animating multiplies every flaw.
2. **Keys**: draw the extremes and contact poses as rough silhouettes on a blocking layer. Keys must differ clearly in silhouette.
3. **Timing pass**: set durations (`set_frame_duration`) and play the keys back. Fix timing and arcs before adding detail.
4. **Breakdowns and in-betweens**: start each in-between as a near-copy of the key it favours (`copy_frame`, `move_region`). Follow arcs.
5. **Secondary motion**: make hair, cloth and tails lag 1-2 frames behind the main motion (`offset_cel_positions`, `tween_cel_positions_eased` on part layers).
6. **Clean-up per frame**: redraw any part that was moved but not edited. Keep the palette identical across frames.
7. **Check**: run `audit_animation` and `compare_frames` to look for unintended flicker, and preview with `render_onion_skin` or `export_tag`. Tag loops with `set_tag`.

Default frame counts and timings are in the `animation` topic.

## Working on an existing sprite
1. `analyze_style` to learn its outline, palette, light direction and edge treatment.
2. If it is rough, resized or noisy, run `cleanup_sprite`.
3. Edit using only its palette and rules (`style` topic), then `lint_pixel_art(respect_style=true)`.

## Tools in this MCP
`analyze_style`, `cleanup_sprite`, `create_canvas`, `add_layer`, `set_layer_opacity`, `set_layer_visibility`, `generate_color_ramp`, `analyze_palette`, `set_palette`, `draw_path`, `draw_line`, `draw_pixels`, `fill_area`, `outline_cel`, `suggest_outline_color`, `shade_color`, `replace_color`, `quantize_to_palette`, `lint_pixel_art`, `readability_preview`, `export_frame`, `copy_frame`, `move_region`, `set_frame_duration`, `set_tag`, `audit_animation`, `compare_frames`, `render_onion_skin`, `animation_workflow_guide`, `pixel_art_guide`.
