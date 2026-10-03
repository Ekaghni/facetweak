"""Regenerate the README comparison images in assets/. Run from the repo root:
    python benchmarks/make_examples.py
Each image is cropped around the face so the change is visible, original on the left.
"""

from pathlib import Path

import cv2
import numpy as np

from facetweak import Adjustments, FaceTweak, PRESETS, read_image, write_image

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    ("portrait.jpg", "before_after.jpg",
     Adjustments(eye_size=0.9, face_slim=0.9, nose_bridge=-0.6, lip_size=0.5, chin_fullness=-0.6)),
    ("portrait2.jpg", "compare_2.jpg", PRESETS["doll"].scaled(1.0)),
    ("portrait3.jpg", "compare_3.jpg",
     Adjustments(eye_size=0.8, face_slim=0.8, brow_lift=0.5, nose_size=-0.6, lip_size=0.6, chin_fullness=-0.5)),
    ("portrait4.jpg", "compare_4.jpg",
     Adjustments(eye_size=0.9, brow_lift=0.6, lip_size=0.7, nose_size=-0.7, face_slim=0.6, eye_openness=0.5)),
]


def crop_box(face, shape, margin=0.9):
    cx = (face.box[0] + face.box[2]) / 2
    cy = (face.box[1] + face.box[3]) / 2
    half = face.width * (0.5 + margin)
    h, w = shape[:2]
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
    return x0, y0, x1, y1


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    tool = FaceTweak()
    for src, dst, adj in CASES:
        img = read_image(ROOT / "examples" / src)
        faces = tool.detect(img)
        out = tool.apply(img, faces, adj)
        x0, y0, x1, y1 = crop_box(max(faces, key=lambda f: f.width), img.shape)
        pair = np.hstack([label(img[y0:y1, x0:x1], "original"),
                          np.full((y1 - y0, 4, 3), 255, np.uint8),
                          label(out[y0:y1, x0:x1], "facetweak")])
        write_image(ROOT / "assets" / dst, pair, 90)
        print(dst, {k: round(v, 2) for k, v in adj.active().items()})

    img = read_image(ROOT / "examples" / "portrait.jpg")
    faces = tool.detect(img)
    x0, y0, x1, y1 = crop_box(faces[0], img.shape)
    panels = []
    for s in (0.0, 0.5, 1.0):
        out = img if s == 0 else tool.apply(img, faces, PRESETS["doll"].scaled(s * 1.0))
        panels.append(label(out[y0:y1, x0:x1], "doll x %.1f" % s))
    write_image(ROOT / "assets" / "strength.jpg", np.hstack(panels), 90)


if __name__ == "__main__":
    main()
