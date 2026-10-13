# Pixel art: core rules

1. Place every pixel deliberately. Don't use blur, soft brushes, smooth gradients or non-integer scaling.
2. Size the sprite from the smallest feature that must read. Fewer, bigger features beat many tiny ones.
3. Work silhouette first. It must read as a solid fill at 1x.
4. Keep one line width, usually 1 px. Straight lines need equal runs (1:1, 2:1, 3:1). Curve runs change monotonically. Remove L-corner doubles.
5. Keep palettes small: 4-8 colours at 16 px, ≤16 for a typical character.
6. Build HSV ramps with hue shifting. Shadows move toward violet (~270°) and highlights toward yellow (~60°), 10-30° per step, by the shortest arc.
7. Avoid #000000. Use a tinted near-black (V 8-25, S 20-60).
8. Use one light source. Shade by form, never as rings that follow the outline (pillow shading). Avoid banding.
9. Use AA sparingly: 1-2 shades, about half a step long, and inside the silhouette only.
10. Separate parts with colour or value, not black inner lines. Leave 1 px gaps between features. Make bands at least 2 px wide.
11. Never dither small or animated sprites. Dither only between adjacent ramp colours.
12. Animation: start from distinct keys. Use arcs, easing, overlap and ≥3 frames per loop. Frames usually last 80-120 ms.
13. Review at 1x, blurred, as a silhouette and in greyscale. Lint, fix, repeat.
14. Editing someone else's sprite: run `analyze_style` first and match its outline, palette, light and edges. Rough or resized input: `cleanup_sprite` first.

For details, call `pixel_art_guide(topic)`. Topics: workflow, style, lineart, antialiasing, color, readability, dithering, perspective, cleanup, subpixel, animation.
