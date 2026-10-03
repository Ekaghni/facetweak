"""The adjustments, and the code that turns them into warps.

Every adjustment is a float in [-1, 1]. Zero means "leave it alone". The sign is
described in ``PARAMS`` and in ``facetweak params``. Distances scale with the face
width, so the same value looks similar on a 300 px selfie and a 4000 px portrait.
"""

from dataclasses import dataclass, fields
from typing import Dict, Mapping, Tuple

import numpy as np

from . import warp

# Pixel distances in the effects below were tuned on a face about 200 px wide.
REFERENCE_FACE_WIDTH = 200.0

# name -> (group, what a positive value does)
PARAMS: Dict[str, Tuple[str, str]] = {
    "eye_size": ("eyes", "bigger eyes (negative: smaller)"),
    "eye_distance": ("eyes", "eyes further apart (negative: closer together)"),
    "eye_height": ("eyes", "eyes lower on the face (negative: higher)"),
    "eye_openness": ("eyes", "eyes taller / more open (negative: narrower)"),
    "eye_width": ("eyes", "eyes wider (negative: shorter)"),
    "brow_lift": ("eyebrows", "both eyebrows higher (negative: lower)"),
    "brow_asymmetry": ("eyebrows", "left eyebrow up and right down (negative: the other way)"),
    "brow_length": ("eyebrows", "eyebrows longer (negative: shorter)"),
    "brow_slant": ("eyebrows", "outer ends up, inner ends down (negative: the reverse)"),
    "brow_thickness": ("eyebrows", "thicker eyebrows (negative: thinner)"),
    "nose_size": ("nose", "bigger nose (negative: smaller)"),
    "nose_height": ("nose", "nose higher on the face (negative: lower)"),
    "nose_bridge": ("nose", "wider bridge (negative: narrower)"),
    "nostril_width": ("nose", "wider nostrils (negative: narrower)"),
    "nostril_height": ("nose", "nostrils higher (negative: lower)"),
    "lip_size": ("lips", "fuller lips (negative: thinner)"),
    "lip_gap": ("lips", "lips further apart (negative: pressed together)"),
    "lip_width": ("lips", "wider mouth (negative: narrower)"),
    "mouth_height": ("lips", "mouth higher on the face (negative: lower)"),
    "face_slim": ("face", "narrower face (negative: wider)"),
    "chin_fullness": ("face", "fuller chin and jaw (negative: leaner)"),
    "chin_length": ("face", "longer chin (negative: shorter)"),
    "forehead_size": ("face", "taller forehead (negative: smaller)"),
}


@dataclass
class Adjustments:
    """All adjustments in one place. Every field defaults to 0 (no change)."""

    eye_size: float = 0.0
    eye_distance: float = 0.0
    eye_height: float = 0.0
    eye_openness: float = 0.0
    eye_width: float = 0.0
    brow_lift: float = 0.0
    brow_asymmetry: float = 0.0
    brow_length: float = 0.0
    brow_slant: float = 0.0
    brow_thickness: float = 0.0
    nose_size: float = 0.0
    nose_height: float = 0.0
    nose_bridge: float = 0.0
    nostril_width: float = 0.0
    nostril_height: float = 0.0
    lip_size: float = 0.0
    lip_gap: float = 0.0
    lip_width: float = 0.0
    mouth_height: float = 0.0
    face_slim: float = 0.0
    chin_fullness: float = 0.0
    chin_length: float = 0.0
    forehead_size: float = 0.0

    @classmethod
    def from_dict(cls, values: Mapping[str, float]) -> "Adjustments":
        """Build from a dict. Unknown names raise ``KeyError`` instead of being ignored."""
        unknown = set(values) - set(PARAMS)
        if unknown:
            raise KeyError("unknown adjustment(s): " + ", ".join(sorted(unknown)))
        return cls(**{k: float(v) for k, v in values.items()})

    def clipped(self) -> "Adjustments":
        """Copy with every value forced into [-1, 1]."""
        return Adjustments(**{f.name: float(np.clip(getattr(self, f.name), -1.0, 1.0)) for f in fields(self)})

    def active(self) -> Dict[str, float]:
        """Only the adjustments that are not zero."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != 0}

    def scaled(self, factor: float) -> "Adjustments":
        """Every value multiplied by ``factor`` (handy for a strength slider)."""
        return Adjustments(**{f.name: getattr(self, f.name) * factor for f in fields(self)}).clipped()


PRESETS: Dict[str, Adjustments] = {
    "subtle": Adjustments(eye_size=0.25, face_slim=0.2, lip_size=0.15, chin_fullness=-0.1),
    "slim": Adjustments(face_slim=0.6, chin_fullness=-0.4, nose_bridge=-0.2),
    "doll": Adjustments(eye_size=0.7, eye_openness=0.2, face_slim=0.4, nose_size=-0.3, lip_size=0.3),
    "sculpted": Adjustments(face_slim=0.35, chin_fullness=-0.35, brow_lift=0.2, nose_bridge=-0.3, lip_size=0.2),
}


def _mid(p: np.ndarray, idx) -> np.ndarray:
    return p[list(idx)].mean(axis=0)


def apply(image: np.ndarray, p: np.ndarray, adj: Adjustments) -> np.ndarray:
    """Apply ``adj`` to ``image`` using the 68 landmarks ``p`` (pixel coordinates, shape (68, 2))."""
    adj = adj.clipped()
    if not adj.active():
        return image

    face_w = max(abs(float(p[16, 0] - p[0, 0])), 1.0)
    u = face_w / REFERENCE_FACE_WIDTH  # pixels-per-reference-pixel
    out = image
    for step in (_eyes, _eyebrows, _nose, _lips, _face):
        out = step(out, p, adj, face_w, u)
    return out


def _eyes(img, p, a, face_w, u):
    lc, rc = _mid(p, range(36, 42)), _mid(p, range(42, 48))
    lw = abs(float(p[39, 0] - p[36, 0]))
    rw = abs(float(p[45, 0] - p[42, 0]))
    eyes = ((lc, lw), (rc, rw))

    if a.eye_size:
        r = max((lw + rw) / 2 * 1.8, 20 * u)
        for c, _ in eyes:
            img = warp.magnify(img, c[0], c[1], r, strength=a.eye_size * 0.4)

    if a.eye_distance:
        amount = abs(float(rc[0] - lc[0])) * a.eye_distance * 0.3 * 0.5
        img = warp.shift(img, lc[0], lc[1], lw * 2.0, -amount, 0)
        img = warp.shift(img, rc[0], rc[1], rw * 2.0, amount, 0)

    for c, w in eyes:
        if a.eye_height:
            img = warp.shift(img, c[0], c[1], w * 2.0, 0, a.eye_height * 12 * u)
        if a.eye_openness:
            img = warp.stretch(img, c[0], c[1], w * 1.5, 0, a.eye_openness * 0.4)
        if a.eye_width:
            img = warp.stretch(img, c[0], c[1], w * 1.5, a.eye_width * 0.3, 0)
    return img


def _eyebrows(img, p, a, face_w, u):
    left, right = _mid(p, range(17, 22)), _mid(p, range(22, 27))
    lw = abs(float(p[21, 0] - p[17, 0]))
    rw = abs(float(p[26, 0] - p[22, 0]))
    brows = ((left, lw), (right, rw))

    for c, w in brows:
        if a.brow_lift:
            img = warp.shift(img, c[0], c[1], w * 0.8, 0, -a.brow_lift * 10 * u)
        if a.brow_length:
            img = warp.stretch(img, c[0], c[1], w * 0.6, a.brow_length * 0.3, 0)

    if a.brow_asymmetry:
        d = a.brow_asymmetry * 8 * u
        img = warp.shift(img, left[0], left[1], lw * 0.8, 0, -d)
        img = warp.shift(img, right[0], right[1], rw * 0.8, 0, d)

    if a.brow_slant:
        # positive lifts the outer end of each brow: counter-clockwise on the left, clockwise on the right
        deg = a.brow_slant * 15
        img = warp.rotate(img, left[0], left[1], lw * 0.6, deg)
        img = warp.rotate(img, right[0], right[1], rw * 0.6, -deg)

    if a.brow_thickness:
        for c, w in brows:
            # only the vertical axis changes: the brow gets taller or flatter, not longer
            img = warp.stretch(img, c[0], c[1], w * 0.5, 0, a.brow_thickness * 0.25)
    return img


def _nose(img, p, a, face_w, u):
    nose_w = abs(float(p[35, 0] - p[31, 0]))
    nose_h = abs(float(p[33, 1] - p[27, 1]))
    tip = p[30]

    if a.nose_size:
        r = max((nose_w + nose_h) * 0.4, 18 * u)
        img = warp.magnify(img, tip[0], tip[1], r, strength=a.nose_size * 0.6)
    if a.nose_height:
        r = max((nose_w + nose_h) * 0.5, 25 * u)
        img = warp.shift(img, tip[0], tip[1], r, 0, -a.nose_height * 15 * u)
    if a.nose_bridge:
        cy = (p[27, 1] + p[28, 1]) / 2
        img = warp.stretch(img, p[27, 0], cy, nose_w * 0.8, a.nose_bridge * 0.3, 0)

    nostrils = (p[31] + p[35]) / 2
    if a.nostril_width:
        img = warp.stretch(img, nostrils[0], nostrils[1], nose_w * 1.2, a.nostril_width * 0.3, 0)
    if a.nostril_height:
        img = warp.shift(img, nostrils[0], nostrils[1], nose_w * 0.8, 0, -a.nostril_height * 8 * u)
    return img


def _lips(img, p, a, face_w, u):
    mouth_w = abs(float(p[54, 0] - p[48, 0]))
    mouth_h = abs(float(p[57, 1] - p[51, 1]))

    if a.lip_size:
        cx = (p[48, 0] + p[54, 0]) / 2
        cy = (p[51, 1] + p[57, 1]) / 2
        img = warp.magnify(img, cx, cy, mouth_w * 0.8, max(mouth_h * 1.5, 3.0), a.lip_size * 0.5)

    if a.lip_gap:
        upper = _mid(p, (50, 51, 52))
        lower = _mid(p, (56, 57, 58))
        d = a.lip_gap * 0.35 * 10 * u
        img = warp.shift(img, upper[0], upper[1], mouth_w * 0.5, 0, -d)
        img = warp.shift(img, lower[0], lower[1], mouth_w * 0.5, 0, d)

    if a.lip_width:
        d = a.lip_width * 0.3 * 12 * u
        img = warp.shift(img, p[48, 0], p[48, 1], mouth_w * 0.4, -d, 0)
        img = warp.shift(img, p[54, 0], p[54, 1], mouth_w * 0.4, d, 0)

    if a.mouth_height:
        c = _mid(p, range(48, 60))
        r = max(mouth_w, mouth_h) * 0.8
        img = warp.shift(img, c[0], c[1], r, 0, -a.mouth_height * 15 * u)
    return img


def _face(img, p, a, face_w, u):
    if a.face_slim:
        s = a.face_slim * 0.6
        left = _mid(p, range(1, 6))
        right = _mid(p, range(11, 16))
        img = warp.shift(img, left[0], left[1], face_w * 0.2, s * 15 * u, 0)
        img = warp.shift(img, right[0], right[1], face_w * 0.2, -s * 15 * u, 0)
        img = warp.shift(img, p[2, 0], p[2, 1], face_w * 0.15, s * 8 * u, 0)
        img = warp.shift(img, p[14, 0], p[14, 1], face_w * 0.15, -s * 8 * u, 0)

    if a.chin_fullness:
        c = _mid(p, range(3, 14))
        cy = c[1] + 0.05 * face_w
        v = a.chin_fullness
        if v > 0:
            r = face_w * 0.4
            img = warp.magnify(img, c[0], cy, r, strength=v * 0.35)
            img = warp.magnify(img, c[0], cy + 0.075 * face_w, r * 0.7, strength=v * 0.35 * 0.6)
        else:
            img = warp.stretch(img, c[0], cy, face_w * 0.35, 0, v * 0.4)
            jr = face_w * 0.15
            img = warp.stretch(img, p[4, 0], p[4, 1], jr, 0, v * 0.2)
            img = warp.stretch(img, p[12, 0], p[12, 1], jr, 0, v * 0.2)

    if a.chin_length:
        img = warp.shift(img, p[8, 0], p[8, 1], face_w * 0.3, 0, a.chin_length * 12 * u)

    if a.forehead_size:
        brow_y = (p[19, 1] + p[24, 1]) / 2
        fx = (p[19, 0] + p[24, 0]) / 2
        fy = brow_y - 0.25 * face_w
        r = abs(float(p[26, 0] - p[17, 0])) * 0.8
        img = warp.stretch(img, fx, fy, r, 0, a.forehead_size * 0.2)
    return img
