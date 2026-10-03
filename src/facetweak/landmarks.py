"""Face landmark detection.

Both backends return the classic 68-point layout (jaw 0-16, brows 17-26, nose 27-35,
eyes 36-47, mouth 48-67) in pixel coordinates, so the effects don't care which one ran.
"""

import bz2
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# 68-point index -> MediaPipe face mesh index
MESH_68 = [
    234, 93, 132, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 361, 323, 454,  # jaw
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,  # brows
    168, 6, 197, 1, 129, 98, 2, 327, 358,  # nose
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  # eyes
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,  # outer lips
    78, 81, 13, 311, 308, 402, 14, 178,  # inner lips
]


class FaceTweakError(Exception):
    """Base class for errors the CLI shows without a traceback."""


class BackendUnavailable(FaceTweakError):
    pass


class NoFaceFound(FaceTweakError):
    pass


@dataclass
class Face:
    """One detected face."""

    points: np.ndarray  # (68, 2) float pixel coordinates
    box: Tuple[int, int, int, int]  # left, top, right, bottom

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])


def cache_dir() -> Path:
    """Where downloaded models live. Override with FACETWEAK_CACHE."""
    root = os.environ.get("FACETWEAK_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "facetweak"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path) -> None:
    print("Downloading %s ..." % dest.name, flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except OSError as exc:
        if tmp.exists():
            tmp.unlink()
        raise FaceTweakError(
            "could not download %s (%s). Download it by hand and put it in %s" % (url, exc, dest.parent)
        ) from exc
    tmp.replace(dest)


def _mediapipe_model() -> Path:
    path = cache_dir() / "face_landmarker.task"
    if not path.exists():
        _download(MEDIAPIPE_MODEL_URL, path)
    return path


def _dlib_model() -> Path:
    path = cache_dir() / "shape_predictor_68_face_landmarks.dat"
    if path.exists():
        return path
    packed = path.with_suffix(".dat.bz2")
    _download(DLIB_MODEL_URL, packed)
    with bz2.BZ2File(packed, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    packed.unlink()
    return path


class MediaPipeBackend:
    name = "mediapipe"

    def __init__(self, max_faces: int = 5, model_path: Optional[str] = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise BackendUnavailable(
                "mediapipe is not installed. Run: pip install mediapipe"
            ) from exc
        self._mp = mp
        data = Path(model_path).read_bytes() if model_path else _mediapipe_model().read_bytes()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=data),
            num_faces=max_faces,
            min_face_detection_confidence=0.4,
        )
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except OSError as exc:
            hint = ""
            if "libEGL" in str(exc) or "libGL" in str(exc):
                hint = " On Debian/Ubuntu run: sudo apt install libegl1 libgles2 libgl1"
            raise BackendUnavailable("mediapipe could not load its native library (%s).%s" % (exc, hint)) from exc

    def detect(self, image_bgr: np.ndarray) -> List[Face]:
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        h, w = image_bgr.shape[:2]
        faces = []
        for mesh in result.face_landmarks:
            xy = np.array([[m.x * w, m.y * h] for m in mesh], dtype=np.float64)
            pts = xy[MESH_68]
            # the jaw points sit on the face outline, so they give a usable box
            box = (int(pts[0, 0]), int(pts[17:27, 1].min()), int(pts[16, 0]), int(pts[8, 1]))
            faces.append(Face(pts, box))
        return faces


class DlibBackend:
    name = "dlib"

    def __init__(self, max_faces: int = 5, model_path: Optional[str] = None):
        try:
            import dlib
        except ImportError as exc:
            raise BackendUnavailable(
                'dlib is not installed. Run: pip install "facetweak[dlib]" (it needs a C++ compiler and CMake)'
            ) from exc
        self._dlib = dlib
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(model_path or _dlib_model()))

    def detect(self, image_bgr: np.ndarray) -> List[Face]:
        import cv2

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = []
        for rect in self._detector(gray):
            shape = self._predictor(gray, rect)
            pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float64)
            faces.append(Face(pts, (rect.left(), rect.top(), rect.right(), rect.bottom())))
        return faces


def make_backend(name: str = "auto", **kwargs):
    """``auto`` tries MediaPipe first and falls back to dlib."""
    if name == "mediapipe":
        return MediaPipeBackend(**kwargs)
    if name == "dlib":
        return DlibBackend(**kwargs)
    if name != "auto":
        raise ValueError("backend must be 'auto', 'mediapipe' or 'dlib', got %r" % name)
    try:
        return MediaPipeBackend(**kwargs)
    except BackendUnavailable as first:
        try:
            return DlibBackend()
        except BackendUnavailable:
            raise first
