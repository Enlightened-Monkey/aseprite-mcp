import pytest

from aseprite_mcp.pixelart.color import (
    delta_e, generate_ramp, hex_to_rgb, hue_delta, rotate_toward, shade, suggest_outline,
)


def test_rotate_toward_takes_shortest_arc_and_stops_at_pole():
    assert rotate_toward(30, 270, 20) == pytest.approx(10)    # orange -> red, not -> green
    assert rotate_toward(110, 270, 20) == pytest.approx(130)  # green -> teal
    assert rotate_toward(265, 270, 20) == pytest.approx(270)  # no overshoot


def _hues(ramp):
    return [c["h"] for c in ramp]


@pytest.mark.parametrize("base", ["#E08020", "#D04648", "#CCAA66"])
def test_warm_shadows_turn_toward_magenta_not_yellow(base):
    ramp = generate_ramp(hex_to_rgb(base), 5)
    base_h = ramp[2]["h"]
    for c in ramp[:2]:
        assert hue_delta(base_h, c["h"]) < 0, ramp  # hue decreases (wrapping) toward red/violet
    for c in ramp[3:]:
        assert hue_delta(base_h, c["h"]) > 0, ramp  # highlights toward yellow


def test_green_and_blue_shadows_turn_cool():
    green = generate_ramp(hex_to_rgb("#43F00F"), 5)
    assert 120 <= green[0]["h"] <= 190
    blue = generate_ramp(hex_to_rgb("#3060D0"), 5)
    assert blue[0]["h"] > blue[2]["h"] and blue[4]["h"] < blue[2]["h"]


def test_ramp_shape():
    ramp = generate_ramp(hex_to_rgb("#3060D0"), 6, base_index=2)
    assert ramp[2]["hex"] == "#3060D0"
    values = [c["v"] for c in ramp]
    assert values == sorted(values)
    sats = [c["s"] for c in ramp]
    assert sats[-1] < max(sats) * 0.7      # washed-out top
    assert ramp[-1]["v"] >= 90


def test_ramp_validation():
    with pytest.raises(ValueError):
        generate_ramp((1, 2, 3), 1)
    with pytest.raises(ValueError):
        generate_ramp((1, 2, 3), 4, base_index=4)


def test_shade_and_outline():
    red = hex_to_rgb("#D04648")
    dark = shade(red, "shadow")
    assert delta_e(dark, red) > 10
    light = shade(red, "highlight")
    assert sum(light) > sum(red)
    outline = suggest_outline([(red, 100), ((90, 30, 40), 40)])
    assert max(outline) < 60 and outline != (0, 0, 0)
    assert suggest_outline([]) != (0, 0, 0)
