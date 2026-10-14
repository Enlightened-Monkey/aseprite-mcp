import numpy as np

from aseprite_mcp.pixelart import segment as SG


def _scene():
    """Sky with flakes, a two-tone ball and ground."""
    a = np.zeros((40, 40, 4), np.uint8)
    a[...] = (90, 120, 150, 255)                     # sky
    a[30:] = (220, 230, 235, 255)                    # snowy ground
    yy, xx = np.mgrid[0:40, 0:40]
    ball = (xx - 20) ** 2 + (yy - 18) ** 2 <= 64
    a[ball] = (180, 60, 60, 255)
    a[ball & (xx > 20)] = (120, 40, 50, 255)         # shaded half
    for x, y in [(3, 3), (10, 6), (33, 4), (5, 20), (36, 15), (30, 26), (8, 12)]:
        a[y, x] = (240, 245, 250, 255)               # snow flakes
    return a, ball


def test_edges_follow_colour_borders():
    a, ball = _scene()
    e = SG.edge_map(a)
    assert e[18, 12] or e[18, 11]                   # left rim of the ball
    assert not e[25, 5]                             # inside the plain sky


def test_graph_segmentation_and_merge():
    a, ball = _scene()
    regions = SG.felzenszwalb(a, k=60, min_size=4)
    assert regions[2, 20] == regions[25, 2]         # sky is one region
    assert regions[18, 20] != regions[2, 20]        # the ball is separate
    objects = SG.merge_regions(a, regions, border_delta=12, colour_delta=60, max_objects=3)
    assert objects.max() + 1 <= 3
    assert objects[18, 16] == objects[18, 24]       # both halves of the ball join


def test_particles_and_background():
    a, ball = _scene()
    p = SG.particle_mask(a)
    assert p[3, 3] and p[6, 10] and not p[18, 20]
    regions = SG.felzenszwalb(a, k=60, min_size=4)
    bg = SG.background_mask(a, regions)
    assert bg[2, 20] and not bg[18, 18]


def test_mask_helpers():
    a, ball = _scene()
    hr = np.repeat(np.repeat(ball, 4, axis=0), 4, axis=1)
    assert (SG.downsample_mask(hr, 4) == ball).all()
    big = np.zeros((40, 40), bool); big[5:35, 5:35] = True
    labels = SG.masks_to_labels([big, ball], (40, 40))
    assert labels[18, 20] == 1 and labels[6, 6] == 0     # smaller mask on top
    regions = SG.felzenszwalb(a, k=60, min_size=4)
    holey = ball.copy(); holey[16:20, 16:20] = False
    lab = SG.resolve_by_regions([holey], regions)
    assert (lab[16:20, 16:20] == 0).all()                # region owner fills the hole
    flat = SG.render_labels(lab)
    assert flat.shape == (40, 40, 4) and SG.render_contours(a, lab).shape == a.shape
