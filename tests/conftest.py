import numpy as np
import pytest


def synthetic_landmarks(cx=300.0, cy=300.0, w=200.0):
    """A rough 68-point face layout, good enough to check which way effects move things."""
    k = w / 200.0
    p = np.zeros((68, 2))
    for i in range(17):  # jaw: a U shape from ear to ear
        t = i / 16
        p[i] = (cx - w / 2 + w * t, cy + k * (40 + 110 * (1 - abs(2 * t - 1) ** 2)))
    p[8] = (cx, cy + 150 * k)
    for i, x in enumerate(np.linspace(-80, -25, 5)):
        p[17 + i] = (cx + k * x, cy - 55 * k)
    for i, x in enumerate(np.linspace(25, 80, 5)):
        p[22 + i] = (cx + k * x, cy - 55 * k)
    left_eye = [(-80, 0), (-65, -6), (-45, -6), (-30, 0), (-45, 6), (-65, 6)]
    right_eye = [(30, 0), (45, -6), (65, -6), (80, 0), (65, 6), (45, 6)]
    for i, (dx, dy) in enumerate(left_eye):
        p[36 + i] = (cx + k * dx, cy + k * (-30 + dy))
    for i, (dx, dy) in enumerate(right_eye):
        p[42 + i] = (cx + k * dx, cy + k * (-30 + dy))
    nose = {27: (0, -40), 28: (0, -20), 29: (0, -5), 30: (0, 10),
            31: (-25, 20), 32: (-12, 25), 33: (0, 27), 34: (12, 25), 35: (25, 20)}
    mouth = {48: (-45, 70), 49: (-25, 62), 50: (-12, 60), 51: (0, 59), 52: (12, 60), 53: (25, 62),
             54: (45, 70), 55: (25, 80), 56: (12, 84), 57: (0, 85), 58: (-12, 84), 59: (-25, 80)}
    for table in (nose, mouth):
        for i, (dx, dy) in table.items():
            p[i] = (cx + k * dx, cy + k * dy)
    for i in range(8):
        p[60 + i] = (p[48 + i] + p[[48, 50, 51, 52, 54, 56, 57, 58][i]]) / 2
    return p


@pytest.fixture
def pts():
    return synthetic_landmarks()


def blob_image(centers, size=(600, 600), radius=8):
    """Black image with a white dot at each centre, so we can track where content moves."""
    img = np.zeros((size[0], size[1], 3), np.uint8)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    for x, y in centers:
        img[(xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2] = 255
    return img


def centroid(img, box):
    x0, y0, x1, y1 = box
    g = img[y0:y1, x0:x1, 0].astype(float)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return (xs * g).sum() / g.sum(), (ys * g).sum() / g.sum()


def around(c, r=40):
    """A box of half-size r around a point, for centroid()."""
    return int(c[0]) - r, int(c[1]) - r, int(c[0]) + r, int(c[1]) + r
