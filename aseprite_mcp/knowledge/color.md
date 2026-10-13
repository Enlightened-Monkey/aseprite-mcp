# Colour

## Key rules
- Think in **HSV** (H 0-360°, S and V 0-100). A ramp is an ordered dark-to-light run of related colours.
- Shift hue along the ramp. Darker steps rotate toward **violet/blue (~260-280°)**, lighter steps toward **yellow (~55-65°)**. Always take the **shortest arc** on the hue wheel.
- Never shade by value alone. Each shadow step needs a change in hue and saturation as well.
- Use a tinted near-black instead of #000000. Share dark and light colours between ramps.
- Spend colours according to sprite size. Keep 2-3 main hues plus one accent per character.

## Ramp construction
| Parameter | Typical value |
|---|---|
| Steps per ramp | 3-6 for characters, up to 7-9 for large art or effects |
| Hue shift per step | 10-30° (15-20° is a good default) |
| Total hue span | 40-130° over 4-6 steps |
| Value change per step | 10-25 points, roughly linear |
| Lightest step | V 89-100, S 15-50 (desaturated, tinted near-white, not pure white) |
| Middle steps | highest saturation, S 70-100 |
| Darkest step | V 15-40 (lower for long ramps), S about 0.75-0.9 × the mid S |

Direction examples, shortest arc toward each pole:
| Base | Shadows go to | Highlights go to |
|---|---|---|
| Red ~0° | crimson/magenta (340-320°) | orange (15-30°) |
| Orange/brown ~30-40° | red/magenta (lower H) | yellow (higher H) |
| Green ~110° | teal/blue (higher H) | yellow-green (lower H) |
| Blue ~220° | violet (higher H) | cyan (lower H) |
| Purple ~280° | barely moves; use value and saturation | warm pink/magenta, toward 330-360° |

Shadows that rotate toward yellow (olive greens, yellowish blues) look sickly. Use that only on purpose.

**One-step shadow from a base colour**: rotate H 20-30° toward violet, multiply S by 0.5-0.8, lower V by 25-40.
**One-step highlight**: rotate H 10-20° toward yellow, raise V by 10-20, and lower S by 10-25 near the top.

**Saturation curves** direct attention:
- **Arc** (the default): mids are most saturated.
- **Saturated shadows with dull highlights**: works for metal and cold materials.
- **Vivid highlights with dull shadows**: works for warm, soft materials.
- **Single saturated step**: makes that step act as an accent.

## Near-blacks for outlines and the deepest shade
| Sprite palette | Near-black hue | S | V |
|---|---|---|---|
| warm or neutral | plum/purple, 290-340° | 20-50 | 9-25 |
| warm or organic | dark brown, 20-35° | 40-60 | 10-20 |
| cool | dark teal, 170-200° | 20-50 | 9-20 |
| foliage | dark green, 120-150° | 20-40 | 10-18 |
Pure black is valid only as a deliberate high-contrast style. Very dark tints look similar on many screens, so the choice is subtle but harmonises the sprite.

## Palette budget
| Sprite height | Total colours | Steps per ramp |
|---|---|---|
| ≤16 px | 4-8 | 2-3 |
| 16-32 px | 8-12 | 3-4 |
| 32-64 px | 12-16 | 4-5 |
| >64 px / scenes | 16-32 | 5-7 |
- About 16 is the usual cap for one character. Quality drops visibly below about 10 colours on a 60 px sprite.
- On small sprites, two similar shades can't be told apart. Merge them.
- Share colours: a common dark shade at the bottom of several ramps and a common near-white at the top. In scenes, unrelated objects can reuse colours, but objects that **touch** must not share their edge colours.

## Colour design
- Use **one main colour** for most of the body or its key features, plus **one accent** that contrasts strongly with it (≥60° hue apart or ≥30 V apart).
- Keep 2-3 major hue groups plus neutrals. Merge stray minor hues.
- Colour carries meaning. An orange point on a face reads as a beak. Choose accents that match what the part is.
- Define small details with contrasting colour or value instead of dark inner lines.

## Reducing colours
Cut in this order and re-check readability after each step:
1. Barely visible extremes, such as an extra darkest step or a one-off AA colour.
2. Merge similar ramps (skin with hair, green with teal).
3. Let ramps share one highlight.
4. Share or fuse the darkest shades.
5. Replace neutral ramps (greys, whites) with the light end of a warm ramp.
Never merge the colours that separate the silhouette from the main fill.

## Value-first and gradient-map workflow
1. Shade in 3-5 greys only, and check the structure in a value view.
2. Order the palette by perceived luminance, not by hue.
3. Map each grey to the step with matching luminance (`replace_color`). A palette sorted this way gives correct values and free hue shifting.
4. Fix any spots where the mapping looks wrong.

## Greys, ambient light, mood
- A mix of two complementary colours gives a near-grey. Greys work well for AA and blends between complementary areas.
- Under strong coloured light (night, fire, toxic glow), lower the saturation of the local colours by 40-70% and rotate them toward the light's hue. They still read as their "real" colours.
- Cool hues (180-280°) read as cold or sad. Warm hues (0-60°) read as warm or happy. Low saturation (S < 20) reads as neutral or bleak.
- Several hues inside one ramp feel livelier, as long as neighbouring steps stay close in hue and value.
- Revisit colours late in the process, and compare them against the other assets in the same scene.

## Animation
Use one fixed palette across all frames. A colour that appears in only one frame flickers.

## Tools in this MCP
`generate_color_ramp`, `shade_color`, `suggest_outline_color`, `analyze_palette`, `get_palette`, `set_palette`, `extract_palette`, `get_color_stats`, `replace_color`, `adjust_hsl`, `adjust_hsl_native`, `quantize_to_palette`, `remap_colors_in_cel_range`, `list_palette_presets`, `apply_palette_preset`, `set_color_mode`, `lint_pixel_art` (colour count, near-duplicates, pure black), `readability_preview` (value view).
