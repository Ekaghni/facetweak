"""Local image warps with a smooth Gaussian falloff.

Every function returns a new image and only touches a small window around
the centre point, so one warp costs well under a millisecond on a face-sized
region instead of a full-image remap.
"""

from typing import Callable, Optional, Tuple

import cv2
import numpy as np

Field = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]


def _remap_window(image: np.ndarray, cx: float, cy: float, half: float, field: Field) -> np.ndarray:
    """Remap a square window around (cx, cy).

    ``field`` receives the pixel offsets from the centre (dx, dy) and returns the
    source offsets to sample from, plus an optional blend weight in [0, 1].
    """
    h, w = image.shape[:2]
    x0 = max(int(np.floor(cx - half)), 0)
    y0 = max(int(np.floor(cy - half)), 0)
    x1 = min(int(np.ceil(cx + half)) + 1, w)
    y1 = min(int(np.ceil(cy + half)) + 1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return image

    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xs - np.float32(cx)
    dy = ys - np.float32(cy)
    src_dx, src_dy, alpha = field(dx, dy)

    map_x = (np.float32(cx) + src_dx - x0).astype(np.float32)
    map_y = (np.float32(cy) + src_dy - y0).astype(np.float32)
    window = np.ascontiguousarray(image[y0:y1, x0:x1])
    warped = cv2.remap(window, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if alpha is not None:
        a = alpha[..., None] if window.ndim == 3 else alpha
        warped = (warped * a + window * (1.0 - a)).astype(image.dtype)

    out = image.copy()
    out[y0:y1, x0:x1] = warped
    return out


def _gauss(dist: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)


def magnify(image: np.ndarray, cx: float, cy: float, rx: float, ry: Optional[float] = None,
            strength: float = 0.0, sigma: float = 0.4) -> np.ndarray:
    """Bulge (strength > 0) or pinch (strength < 0) around a point.

    ``rx`` and ``ry`` are the radii of an axis-aligned ellipse. Leave ``ry`` out for a circle.
    """
    ry = rx if ry is None else ry
    if strength == 0 or rx < 1 or ry < 1:
        return image

    def field(dx, dy):
        dist = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
        fall = _gauss(dist, sigma)
        scale = np.maximum(1.0 + strength * fall * (1.0 - dist / 2.0), 0.5)
        return dx / scale, dy / scale, fall

    return _remap_window(image, cx, cy, 4.0 * sigma * max(rx, ry) + 2, field)


def shift(image: np.ndarray, cx: float, cy: float, radius: float,
          move_x: float, move_y: float) -> np.ndarray:
    """Move the content around (cx, cy) by (move_x, move_y) pixels, fading out with distance."""
    if (move_x == 0 and move_y == 0) or radius < 1:
        return image

    def field(dx, dy):
        fall = _gauss(np.sqrt(dx * dx + dy * dy) / radius, 0.5)
        return dx - move_x * fall, dy - move_y * fall, None

    pad = max(abs(move_x), abs(move_y))
    return _remap_window(image, cx, cy, 2.0 * radius + pad + 2, field)


def stretch(image: np.ndarray, cx: float, cy: float, radius: float,
            sx: float, sy: float) -> np.ndarray:
    """Scale the content around a point per axis.

    Positive values enlarge along that axis, negative values squeeze it.
    """
    if (sx == 0 and sy == 0) or radius < 1:
        return image

    def field(dx, dy):
        fall = _gauss(np.sqrt(dx * dx + dy * dy) / radius, 0.4)
        return dx * (1.0 - sx * fall), dy * (1.0 - sy * fall), None

    return _remap_window(image, cx, cy, 4.0 * 0.4 * radius + 2, field)


def rotate(image: np.ndarray, cx: float, cy: float, radius: float, degrees: float) -> np.ndarray:
    """Twist the content around a point. Positive degrees turn it clockwise on screen."""
    if degrees == 0 or radius < 1:
        return image

    def field(dx, dy):
        fall = _gauss(np.sqrt(dx * dx + dy * dy) / radius, 0.5)
        ang = np.radians(degrees * fall)
        c, s = np.cos(ang), np.sin(ang)
        # sampling with the inverse rotation turns the content the requested way
        return dx * c + dy * s, -dx * s + dy * c, None

    return _remap_window(image, cx, cy, 2.0 * radius + 2, field)
