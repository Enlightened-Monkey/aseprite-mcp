"""Integration tests for the numpy <-> Aseprite pixel bridge."""
import numpy as np
import pytest

from aseprite_mcp.core import pixels
from aseprite_mcp.tools import animation, canvas
from conftest import ok, run


def test_read_composite_and_layer(sprite):
    comp = pixels.read_image(sprite)
    assert comp.shape == (32, 32, 4)
    # conftest paints a 16x16 #D04648 square at (8,8) on "body"
    assert tuple(comp[8, 8]) == (0xD0, 0x46, 0x48, 255)
    assert comp[0, 0, 3] == 0
    body = pixels.read_image(sprite, layer="body")
    assert (body == comp).all()


def test_write_round_trip(sprite):
    ok(run(canvas.add_layer(sprite, "fx")))
    arr = np.zeros((32, 32, 4), np.uint8)
    arr[3, 5] = (1, 2, 3, 255)
    arr[31, 31] = (200, 100, 50, 128)
    pixels.write_image(sprite, "fx", 1, arr)
    back = pixels.read_image(sprite, layer="fx")
    assert (back == arr).all()


def test_hidden_layer_excluded_from_composite(sprite):
    ok(run(animation.set_layer_visibility(sprite, "fx", False)))
    comp = pixels.read_image(sprite)
    assert comp[3, 5, 3] == 0


def test_errors(sprite):
    with pytest.raises(ValueError, match="Layer not found"):
        pixels.read_image(sprite, layer="nope")
    with pytest.raises(ValueError, match="out of range"):
        pixels.read_image(sprite, frame=9)
    with pytest.raises(ValueError, match="shape"):
        pixels.write_image(sprite, "body", 1, np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(ValueError, match="does not match"):
        pixels.write_image(sprite, "body", 1, np.zeros((8, 8, 4), np.uint8))
