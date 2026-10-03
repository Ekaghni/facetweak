"""Public API: detect faces and apply adjustments."""

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from . import effects
from .effects import Adjustments
from .landmarks import Face, FaceTweakError, NoFaceFound, make_backend

ImageLike = Union[str, Path, np.ndarray]


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image as BGR. Works with non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise FaceTweakError("could not read an image from %s" % path)
    return img


def write_image(path: Union[str, Path], image: np.ndarray, quality: int = 95) -> None:
    ext = Path(path).suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise FaceTweakError("could not encode %s as %s" % (path, ext))
    buf.tofile(str(path))


class FaceTweak:
    """Holds a landmark backend so repeated calls don't reload the model.

    >>> ft = FaceTweak()
    >>> out = ft.retouch("photo.jpg", eye_size=0.3, face_slim=0.4)
    """

    def __init__(self, backend: str = "auto", max_faces: int = 5):
        self.backend = make_backend(backend, max_faces=max_faces)

    def detect(self, image: ImageLike) -> List[Face]:
        img = read_image(image) if not isinstance(image, np.ndarray) else image
        return sorted(self.backend.detect(img), key=lambda f: f.box[0])

    def retouch(
        self,
        image: ImageLike,
        adjustments: Union[Adjustments, Mapping[str, float], None] = None,
        faces: Union[str, Sequence[int]] = "all",
        **kwargs: float
    ) -> np.ndarray:
        """Return a retouched copy of ``image``.

        ``adjustments`` can be an :class:`Adjustments`, a dict, or you can pass the
        values as keyword arguments. ``faces`` is ``"all"``, ``"largest"`` or a list of
        indexes (left to right). Raises :class:`NoFaceFound` when nothing is detected.
        """
        img = read_image(image) if not isinstance(image, np.ndarray) else image
        if adjustments is None:
            adj = Adjustments.from_dict(kwargs)
        elif kwargs:
            raise TypeError("pass either `adjustments` or keyword values, not both")
        elif isinstance(adjustments, Adjustments):
            adj = adjustments
        else:
            adj = Adjustments.from_dict(adjustments)

        return self.apply(img, self.detect(img), adj, faces)

    def apply(self, image: np.ndarray, found: List[Face], adj: Adjustments, faces="all") -> np.ndarray:
        """Apply ``adj`` to faces you already detected with :meth:`detect`."""
        if not found:
            raise NoFaceFound("no face found in the image")
        out = image
        for face in _choose(found, faces):
            out = effects.apply(out, face.points, adj)
        return out if out is not image else image.copy()


def _choose(found: List[Face], faces: Union[str, Sequence[int]]) -> List[Face]:
    if faces == "all":
        return found
    if faces == "largest":
        return [max(found, key=lambda f: f.width)]
    if isinstance(faces, str):
        raise ValueError("faces must be 'all', 'largest' or a list of indexes")
    try:
        return [found[i] for i in faces]
    except IndexError:
        raise FaceTweakError("face index out of range: found %d face(s)" % len(found)) from None


def retouch(image: ImageLike, adjustments=None, backend: str = "auto", faces="all", **kwargs: float) -> np.ndarray:
    """One-shot helper. Loads the model every call, so use :class:`FaceTweak` in a loop."""
    return FaceTweak(backend).retouch(image, adjustments, faces=faces, **kwargs)


def side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Put two images next to each other with a thin divider."""
    divider = np.full((before.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.hstack([before, divider, after])


def draw_landmarks(image: np.ndarray, faces: Sequence[Face]) -> np.ndarray:
    """Copy of the image with the 68 points drawn on it, for debugging."""
    out = image.copy()
    radius = max(1, int(min(out.shape[:2]) / 250))
    for face in faces:
        for i, (x, y) in enumerate(face.points):
            cv2.circle(out, (int(x), int(y)), radius, (0, 255, 0), -1, cv2.LINE_AA)
    return out
