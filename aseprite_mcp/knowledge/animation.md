# Animation

## Key rules
- Start with **key poses** whose silhouettes differ clearly. More in-betweens can't rescue keys that are too similar.
- Ease by **spacing** (in-betweens bunched near the keys) or by **timing** (longer frames near the keys). Even spacing and even timing look mechanical and jittery.
- Moves follow **arcs**. Parts **overlap**: the leader moves first and followers lag 1-2 frames, then overshoot and settle.
- Loops need **at least 3 frames**. Two frames only flicker. Reuse frames (ping-pong) to save work.
- Keep the palette identical across frames, and copy-shift parts instead of redrawing them. Redrawing causes line boil.

## Timing units
| Base | on ones | on twos | on threes | on fours |
|---|---|---|---|---|
| 60 fps game | ≈17 ms | ≈33 ms | 50 ms | ≈67 ms |
| 30 fps game | ≈33 ms | ≈67 ms | 100 ms | ≈133 ms |
| 24 fps film | ≈42 ms | ≈83 ms | 125 ms | ≈167 ms |
GIF stores durations in 10 ms units, so 17, 33 and 67 ms export as 20, 30 and 70 ms. Merge identical consecutive frames into one longer frame.

## Timing table
| Action | Frames | ms per frame | Notes |
|---|---|---|---|
| Idle / breathing | 3-8 (up to 12) | 100-250 | Ping-pong 0-1-2-1 or A-B-A-C. Hold the extremes 150-250 ms, in-betweens 80-120 ms. Torso moves 1 px (or 0.5 px), feet fixed |
| Blink | 2-3 inside an idle | 60-100, then hold 2-4 s | Insert it every few loops |
| Walk | 4 (tiny), 6, 8 | 80-120 | Cycle 0.5-0.9 s. Second half mirrors the first |
| Run | 6-8 (minimum 3) | 60-80 | Cycle about 0.4-0.5 s |
| Jump | anticipation 1, rise 1-2, apex 1-2, fall 1-2, land 1-2 | 60-120; apex 120-200 | Squash on take-off and landing |
| Attack | anticipation 1-2, strike/smear 1-2, contact 1, recovery 2-4 | 100-200 / 20-50 / 100-200 / 60-100 | End hold 300-400 ms, or return to idle |
| Hit / flinch | 2-3 | 60-100 | Snap away, ease back |
| Impact / smoke FX | 6-12 | 40-60 | Expand, break into fragments, erode pixel by pixel |
| Slash arc FX | 3-4 | 40-100 | Thin line, then a bright wide crescent, then the body colour, then a thinning tail |
| UI shine / glint | 4-6 | 60-100 | Diagonal highlight band sweeping across a static shape |
| UI pop-in | 3, then hold | 40 | Small, then oversized (110-120%), then final size |
| Ambient loop (fire, water, flag) | 4-8 | 80-120 | |

Defaults when unsure: 100 ms for normal frames, 40-50 ms for smears, 150-400 ms for holds.

## Principles
- **Anticipation**: a short preparatory move opposite to the action (crouch before a jump, wind-up before a throw). One frame is often enough. For player-input actions, keep it at ≤1 frame or ≤100 ms so the controls feel responsive. Cosmetic actions can take longer.
- **Breakdown**: the mid-key that defines *how* A reaches B, for example an arm swinging on an arc instead of a straight line. With few frames, spend one on the breakdown.
- **Easing**: an in-between near a key should look like that key, shifted slightly. Timing-based easing works well in pixel art. Example for one cycle with keys at both ends: 160, 120, 80, 80, 120, 160 ms. Holds at both ends plus quick in-betweens give slow-in/slow-out.
- **Overshoot**: the moving part passes its destination by 1-2 px, then snaps or settles back over 1-2 frames. Even 1 px is felt. This works for whole bodies, limbs, UI elements and scale.
- **Overlap and follow-through**: animate the leading action first. Hair, capes, tails, sleeves and belts lag 1-2 frames, stretch while being dragged, keep moving 1-3 frames after the leader stops, overshoot once, then settle. On tiny areas, use **delayed pixels**: neighbouring pixels catch up one frame later.
- **Smears**: an elongated in-between, or multiple after-images along the path. Show it for about 40-50 ms so it is felt rather than seen. Don't use dithered motion blur.
- **Squash and stretch**: preserve area. If height scales by (1 − a), width scales by about 1/(1 − a). Anchor the change at the contact point (feet, ground). Use a ≈ 0.15-0.3 for a bouncy squash, and less for heavy or stiff objects.
- **Holds**: a dead hold freezes the sprite. A **moving hold** keeps it alive with 1 px drifts of secondary parts, decreasing over a few frames.

## Loop construction
- Use ping-pong (0-1-2-3-2-1) for breathing and swaying, and A-B-A-C for a breathing loop with a variant. Combine both for walks with a mirrored half.
- With 3 frames, move parts around a small **circle** (1→2→3) instead of back and forth, so the motion has a path.
- Check the seam: the last frame must lead naturally into the first. No pose may repeat across the seam unless it is meant as a hold.
- In walks, animate the character travelling, and keep the planted foot fixed relative to the ground. Then re-centre the frames. Arms counter-swing the legs. The body is lowest just after contact and highest at the passing position.

## Line boil and jitter
- **Boil**: still parts of the sprite change pixels every frame because they were redrawn. Fix it by copying the static region from the key frame.
- **Jitter**: one odd frame, in-betweens favouring the wrong key, motion off the arc, or even timing on eased spacing. A single bad frame can spoil the whole loop.
- **Flicker**: a pixel toggling A→B→A over 3 frames, or a colour that appears in only one frame.
- Use onion skin for blocking far-apart keys. For close frames, compare by pixel difference, and favour a key rather than drawing "between the lines".

## Production and reference
- **Methods**: block the motion as flat silhouettes first. Recycle frames by moving parts, always followed by clean-up. Downscale high-resolution or 3D renders and clean them by hand. Or draw line art and shading for every frame.
- **Parallax**: nearer layers scroll faster. For a seamless loop, each layer's speed × frame count must be a multiple of that layer's width.
- **Reference**: step through it frame by frame, mark the extremes (the poses just before a change of direction), use them as keys, draw your own in-betweens, and exaggerate.

## Tools in this MCP
`add_frames`, `copy_frame`, `duplicate_frame_range`, `set_frame_duration`, `set_tag`, `tween_cel_positions_eased`, `oscillate_cel_positions`, `offset_cel_positions`, `tween_cel_scale_eased`, `move_region`, `render_onion_skin`, `compare_frames`, `audit_animation`, `export_tag`, `animation_workflow_guide`, `lint_pixel_art`.
