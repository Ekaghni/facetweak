"""Time the original face.py warps against facetweak on the same image.

Needs dlib (only because legacy/face.py imports it). Run from the repo root:
    python benchmarks/compare_legacy.py
The two stacks are similar, not identical: the old code warps the whole frame per effect.
"""
import sys, time, statistics
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "legacy"))
import face as L
from facetweak import FaceTweak, Adjustments, read_image
img = read_image(__import__("pathlib").Path(__file__).resolve().parent.parent / "examples" / "portrait.jpg")
f = FaceTweak("mediapipe"); fc = f.detect(img)[0]
h, w = img.shape[:2]
norm = [L.NormalizedLandmark(x/w, y/h, 0) for x, y in fc.points]
def old(): 
    r = img
    for fn, v in ((L.FaceParts.eyes_magnified, 70), (L.FaceParts.nose_magnified, 35), (L.FaceParts.lips_magnified, 60), (L.FaceParts.face_slim, 40), (L.FaceParts.eyes_vertical_stretch, 30)):
        r = fn(norm, r, v)
    return r
adj = Adjustments(eye_size=0.7, nose_size=-0.3, lip_size=0.6, face_slim=0.4, eye_openness=0.4)
def new(): return f.apply(img, [fc], adj)
for n, fn in (("legacy", old), ("facetweak", new)):
    t = []
    for _ in range(10):
        s = time.perf_counter(); fn(); t.append((time.perf_counter()-s)*1000)
    print(n, round(statistics.median(t), 1), "ms", img.shape)
