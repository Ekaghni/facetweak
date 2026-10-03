import numpy as np
import pytest

from facetweak import PARAMS, PRESETS, Adjustments, effects, warp

from conftest import around, blob_image, centroid, synthetic_landmarks


def noise(size=600, seed=0, channels=3):
    shape = (size, size, channels) if channels else (size, size)
    return np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)


def test_every_param_is_a_field():
    assert set(PARAMS) == set(Adjustments().__dict__)


def test_zero_adjustments_return_same_pixels(pts):
    img = noise()
    assert np.array_equal(effects.apply(img, pts, Adjustments()), img)


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_each_param_changes_pixels_near_the_face_only(name, sign, pts):
    img = noise(seed=1)
    out = effects.apply(img, pts, Adjustments(**{name: sign}))
    assert out.shape == img.shape and out.dtype == img.dtype
    assert not np.array_equal(out, img)
    for corner in (np.s_[:10, :10], np.s_[:10, -10:], np.s_[-10:, :10], np.s_[-10:, -10:]):
        assert np.array_equal(out[corner], img[corner])


def test_values_outside_range_are_clipped(pts):
    img = noise(seed=3)
    a = effects.apply(img, pts, Adjustments(face_slim=5.0))
    b = effects.apply(img, pts, Adjustments(face_slim=1.0))
    assert np.array_equal(a, b)


def test_eye_distance_positive_moves_eyes_apart(pts):
    left, right = pts[36:42].mean(0), pts[42:48].mean(0)
    out = effects.apply(blob_image([left, right], radius=6), pts, Adjustments(eye_distance=1.0))
    lx, _ = centroid(out, around(left))
    rx, _ = centroid(out, around(right))
    assert lx < left[0] - 1 and rx > right[0] + 1


@pytest.mark.parametrize("name,idx,sign", [
    ("eye_height", range(36, 42), 1),     # positive = down
    ("brow_lift", range(17, 22), -1),     # positive = up
    ("nose_height", [30], -1),
    ("mouth_height", range(48, 60), -1),
    ("chin_length", [8], 1),
])
def test_vertical_moves_go_the_documented_way(name, idx, sign, pts):
    c = pts[list(idx)].mean(0)
    out = effects.apply(blob_image([c], radius=6), pts, Adjustments(**{name: 1.0}))
    _, y = centroid(out, around(c))
    assert (y - c[1]) * sign > 2


def test_brow_asymmetry_raises_left_and_lowers_right(pts):
    l, r = pts[17:22].mean(0), pts[22:27].mean(0)
    out = effects.apply(blob_image([l, r], radius=6), pts, Adjustments(brow_asymmetry=1.0))
    _, ly = centroid(out, around(l))
    _, ry = centroid(out, around(r))
    assert ly < l[1] - 1 and ry > r[1] + 1


def test_lip_gap_separates_lips(pts):
    up, lo = pts[[50, 51, 52]].mean(0), pts[[56, 57, 58]].mean(0)
    out = effects.apply(blob_image([up, lo], radius=4), pts, Adjustments(lip_gap=1.0))
    _, uy = centroid(out, around(up, 12))
    _, ly = centroid(out, around(lo, 12))
    assert uy < up[1] - 0.5 and ly > lo[1] + 0.5


def test_face_slim_moves_jaw_inward(pts):
    left, right = pts[1:6].mean(0), pts[11:16].mean(0)
    out = effects.apply(blob_image([left, right], radius=6), pts, Adjustments(face_slim=1.0))
    lx, _ = centroid(out, around(left, 30))
    rx, _ = centroid(out, around(right, 30))
    assert lx > left[0] + 1 and rx < right[0] - 1


def white_band(y0, y1):
    img = np.zeros((600, 600, 3), np.uint8)
    img[y0:y1, 150:450] = 255
    return img


def band_height(img, x):
    return int((img[:, x, 0] > 127).sum())


def test_brow_thickness_changes_band_height_the_right_way(pts):
    c = pts[17:22].mean(0)
    y, x = int(c[1]), int(c[0])
    img = white_band(y - 5, y + 5)
    thick = effects.apply(img, pts, Adjustments(brow_thickness=1.0))
    thin = effects.apply(img, pts, Adjustments(brow_thickness=-1.0))
    assert band_height(thick, x) > band_height(img, x) > band_height(thin, x)


def test_eye_openness_positive_makes_eyes_taller(pts):
    c = pts[36:42].mean(0)
    y, x = int(c[1]), int(c[0])
    img = white_band(y - 4, y + 4)
    assert band_height(effects.apply(img, pts, Adjustments(eye_openness=1.0)), x) > 8
    assert band_height(effects.apply(img, pts, Adjustments(eye_openness=-1.0)), x) < 8


def test_eye_size_positive_enlarges_the_eye(pts):
    c = pts[36:42].mean(0)
    img = np.zeros((600, 600, 3), np.uint8)
    yy, xx = np.mgrid[0:600, 0:600]
    img[((xx - c[0]) / 12) ** 2 + ((yy - c[1]) / 6) ** 2 <= 1] = 255
    bigger = effects.apply(img, pts, Adjustments(eye_size=1.0))
    smaller = effects.apply(img, pts, Adjustments(eye_size=-1.0))
    assert (bigger[..., 0] > 127).sum() > (img[..., 0] > 127).sum() > (smaller[..., 0] > 127).sum()


def test_brow_slant_positive_raises_outer_ends(pts):
    outer_left = pts[17]
    out = effects.apply(blob_image([outer_left], radius=4), pts, Adjustments(brow_slant=1.0))
    _, y = centroid(out, around(outer_left, 30))
    assert y < outer_left[1] - 0.5


def test_effect_size_follows_face_width():
    shifts = []
    for width, size in ((200, 600), (400, 1200)):
        p = synthetic_landmarks(size / 2, size / 2, width)
        c = p[36:42].mean(0)
        img = blob_image([c], (size, size), radius=int(6 * width / 200))
        out = effects.apply(img, p, Adjustments(eye_height=1.0))
        _, y = centroid(out, around(c, int(40 * width / 200)))
        shifts.append(y - c[1])
    assert 1.6 < shifts[1] / shifts[0] < 2.4


def test_grayscale_image_is_supported(pts):
    img = noise(seed=4, channels=0)
    out = effects.apply(img, pts, Adjustments(eye_size=1.0, face_slim=1.0))
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_warps_near_image_edges_do_not_crash():
    img = noise(50, seed=5)
    for fn in (lambda i: warp.magnify(i, 0, 0, 30, strength=0.5),
               lambda i: warp.shift(i, 49, 49, 30, 5, 5),
               lambda i: warp.stretch(i, 25, 0, 40, 0.3, 0.3),
               lambda i: warp.rotate(i, -10, 25, 30, 20),
               lambda i: warp.shift(i, 500, 500, 20, 5, 5)):
        assert fn(img).shape == img.shape


def test_input_image_is_never_modified(pts):
    img = noise(seed=6)
    keep = img.copy()
    effects.apply(img, pts, PRESETS["doll"])
    assert np.array_equal(img, keep)


def test_presets_are_valid():
    for name, adj in PRESETS.items():
        assert adj.active(), name
        assert all(-1 <= v <= 1 for v in adj.active().values())


def test_from_dict_rejects_unknown_names():
    with pytest.raises(KeyError, match="eye_sze"):
        Adjustments.from_dict({"eye_sze": 0.3})


def test_scaled_halves_values():
    assert Adjustments(eye_size=0.8).scaled(0.5).eye_size == pytest.approx(0.4)
