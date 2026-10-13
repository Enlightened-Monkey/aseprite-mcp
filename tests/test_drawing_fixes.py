"""B3 (cel-normalization, no silent clip) + B4 (per-pixel alpha) fixes."""
from conftest import ok, run

from aseprite_mcp.tools import canvas, drawing, pixel_read


def _rgba(pixel_result: str) -> tuple[int, int, int, int]:
    s = str(pixel_result)
    return (int(s.split("r=")[1].split(",")[0]),
            int(s.split("g=")[1].split(",")[0]),
            int(s.split("b=")[1].split(",")[0]),
            int(s.split("a=")[1].split(")")[0]))


def test_polygon_offcanvas_vertices_no_clip(sprite):
    # Two vertices are outside the 32x32 canvas; pset() must bounds-guard
    # (no crash) and the in-canvas fill must still land at sprite-global (16,16).
    ok(run(canvas.add_layer(sprite, "poly")))
    pts = [{"x": 16, "y": -5}, {"x": -5, "y": 30}, {"x": 30, "y": 30}]
    ok(run(drawing.draw_polygon(sprite, "poly", 1, pts, "#00FF00", True, True)))
    r, g, b, _ = _rgba(run(pixel_read.get_pixel_color(sprite, 16, 16, "poly", 1)))
    assert (r, g, b) == (0, 255, 0), (r, g, b)


def test_gradient_lands_at_sprite_global(sprite):
    ok(run(canvas.add_layer(sprite, "grad")))
    ok(run(drawing.apply_gradient_rect(sprite, "grad", 1, 4, 4, 16, 1,
                                       "#000000", "#FFFFFF", True, True)))
    left = _rgba(run(pixel_read.get_pixel_color(sprite, 4, 4, "grad", 1)))
    right = _rgba(run(pixel_read.get_pixel_color(sprite, 19, 4, "grad", 1)))
    assert left[0] < right[0], (left, right)  # dark -> light, left to right


def test_draw_accepts_rgba_alpha(sprite):
    # "#RRGGBBAA" was rejected before B4 (len != 6) and left the layer empty.
    ok(run(canvas.add_layer(sprite, "alpha")))
    ok(run(drawing.draw_pixels_at(sprite, "alpha", 1,
                                  [{"x": 6, "y": 6, "color": "#FF000080"}], True)))
    r, g, b, a = _rgba(run(pixel_read.get_pixel_color(sprite, 6, 6, "alpha", 1)))
    assert (r, g, b) == (255, 0, 0)
    assert a == 0x80  # semi-transparency preserved


def test_draw_accepts_short_hex(sprite):
    ok(run(canvas.add_layer(sprite, "short")))
    ok(run(drawing.draw_pixels_at(sprite, "short", 1,
                                  [{"x": 5, "y": 5, "color": "#0F0"}], True)))
    assert _rgba(run(pixel_read.get_pixel_color(sprite, 5, 5, "short", 1))) == (0, 255, 0, 255)


def test_draw_rejects_bad_hex(sprite):
    ok(run(canvas.add_layer(sprite, "bad")))
    res = run(drawing.draw_pixels_at(sprite, "bad", 1,
                                     [{"x": 0, "y": 0, "color": "#GG0000"}], True))
    assert res.startswith(("Invalid", "Failed"))



def test_pixels_outside_trimmed_cel_are_kept(base_dir):
    """Drawing outside an existing (trimmed) cel must grow it, not drop pixels."""
    path = f"{base_dir}/trimmed_cel.aseprite"
    ok(run(canvas.create_canvas(32, 32, path)))
    ok(run(canvas.add_layer(path, "body")))
    ok(run(drawing.draw_rectangle_at(path, "body", 1, 8, 8, 4, 4, "#D04648", True)))
    ok(run(drawing.draw_pixels_at(path, "body", 1, [{"x": 1, "y": 1, "color": "#00FF00"}])))
    ok(run(drawing.draw_line_at(path, "body", 1, 20, 30, 30, 30, "#0000FF")))
    assert run(pixel_read.get_pixel_color(path, 1, 1, "body")).startswith("#00ff00")
    assert run(pixel_read.get_pixel_color(path, 25, 30, "body")).startswith("#0000ff")
    assert run(pixel_read.get_pixel_color(path, 8, 8, "body")).startswith("#d04648")
    # Active-cel tools draw on whichever layer is active; check the composite.
    ok(run(drawing.draw_pixels(path, [{"x": 31, "y": 0, "color": "#FFFF00"}])))
    ok(run(drawing.draw_line(path, 0, 31, 3, 31, "#FF00FF")))
    assert run(pixel_read.get_composite_pixel(path, 31, 0)).startswith("#ffff00")
    assert run(pixel_read.get_composite_pixel(path, 2, 31)).startswith("#ff00ff")
