# Readability

## Key rules
- **Readability** is how quickly a viewer who wasn't told the intent correctly identifies what the sprite is and what it is doing. It beats detail, every time.
- Get the silhouette right first. It must show the role and the action as a solid fill.
- Size the sprite from the smallest feature that must read. Drop the details that don't fit.
- Give every feature space: 1 px gaps between features, bands at least 2 px wide, no tangents.
- Review at 1x, blurred, as a silhouette and in greyscale, then do a blind read. Iterate.

## Silhouette
- Make the head, limbs, weapon, hair and hat protrude from the body mass. Show negative space, such as a gap between arm and torso.
- Avoid overlapping limbs in key poses. Where overlap can't be avoided, separate the parts with colour or value contrast.
- A compact silhouette (blob monsters, crouched poses) has to carry its readability through internal colour contrast.
- One diagnostic detail is often enough to name an object. Example: adding a lid line and a 2 px lock turns a brown box into a chest. Keep the shapes simple and add that one detail.
- Stray highlight pixels can accidentally read as eyes. Check that spots of high contrast mean something.

## Features by size
**Eyes** are the first thing a viewer looks for, so spend polish there.
| Head width | Eyes |
|---|---|
| <5 px | omit them; suggest them with a shadow band under the brow or hair |
| 5-7 px | 1×1 or 1×2 dark pixels, no whites |
| 8-12 px | dark pupil plus 1 light pixel (white or light skin) |
| 13+ px | lid, iris colour, shine pixel, light AA |
- Pupils of 2×2 read well on heads of about 12-32 px. Keep them off the outline so they don't merge with it.
- A single shine pixel makes eyes look alive. Moving the lid, pupil or an AA pixel by 1 px changes the expression.
- Tiny expressions: stretch the eyes 2 px tall for surprise, slant them inward or diagonal for anger, use a flat 2×1 line for closed or happy eyes.
- For glasses, draw either the frames without eyes or only the top rim.

**Mouth**: needs ≥1 px of clear space above and below. Skip it on ≤16 px sprites unless it is the key feature.

**Hands**:
| Hand size | Approach |
|---|---|
| 3-4 px | 2-3 colour blob, at most a thumb notch |
| 5-6 px | mitten plus thumb, finger gaps by value |
| 7-9 px | individual 1 px fingers in alternating skin values |
- Block hands as flat colour shapes, not line art. Start from a mitten.
- A thumb plus an index finger is enough to grip, point or pinch. Three fingers plus a thumb is fine.
- Separate fingers by value, never by outline.

**Small NPC variety**: keep one body template and vary only the hair silhouette, the eye pixels and the clothing hue.

## Spacing
| Rule | Value |
|---|---|
| Gap between face features | ≥1 px |
| Minimum readable band (hair strip, belt, stripe) | 2 px |
| Pupils on mid-size heads | 2×2 px |
| Detail kept away from the outline | ≥1 px |
- Remove dark inner lines between parts. They cost pixels and make the sprite muddy. Separate parts with colour, or with the darkest shade of the adjacent fill.
- If two features collide, move the less important one (an ear, a strap) rather than the identity-defining one (the eyes).
- Avoid **tangents**: two contours that just touch at a point, or two shapes lined up end to end with a 1 px seam between them. Make them overlap clearly or separate clearly.
- Light and dark accents can replace outlines. Use the darkest tones for crevices and separations, and the brightest for edges and key details.

## Foreground and background separation
Combine all three methods:
1. **Outline** the foreground sprites and objects.
2. **Soften the background**: lower its saturation and contrast and shift its value (usually lighter, or darker for night scenes). Foreground mean saturation should be clearly above the background's.
3. **Focus**: keep the foreground crisp. Make the background flatter, with fewer hard edges and larger shapes.
Objects that touch need different colours or values along their shared border. Otherwise they merge into one shape.

## Dither and AA at small sizes
- Dithering on small sprites reads as noise. Use flat shading instead.
- Moderate AA can help a curve read on a mid-size sprite. On tiny sprites, prefer aliased pixels.

## Review method
1. **1x view**: keep it open at all times. Judging zoomed in is misleading.
2. **Blur** (about 1 px radius): shows blobs, banding and shapes that merge.
3. **Silhouette**: flat fill on white. Can you identify the pose and role?
4. **Value view** (greyscale): are the lightest and darkest values where the important details are? Is the value range wide enough?
5. **Backgrounds**: check the sprite on light, mid and dark backgrounds, especially with selective outlines.
6. **Blind read**: describe what the sprite shows without using the intended label, then compare that with the intent. If they differ, fix the biggest ambiguity and repeat.
7. **Variants**: when unsure, make 2-3 versions and compare them side by side.

## Tools in this MCP
`readability_preview` (1x/2x, blur, silhouette, value, light/mid/dark backgrounds), `lint_pixel_art`, `export_frame`, `get_composite_rect`, `outline_cel`, `adjust_hsl` (soften backgrounds), `move_region`, `duplicate_layer`, `set_layer_visibility`, `validate_scene`, `start_preview_server`.
