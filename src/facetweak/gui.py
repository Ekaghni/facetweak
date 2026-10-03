"""Interactive OpenCV window with sliders. Imported only by `facetweak gui`."""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2

from .core import FaceTweak, read_image, write_image
from .effects import PARAMS, Adjustments
from .landmarks import FaceTweakError

SECTIONS = ["eyes", "eyebrows", "nose", "lips", "face"]
WINDOW = "facetweak"
CONTROLS = "controls"


class _Session:
    def __init__(self, tool: FaceTweak, image, found):
        self.tool, self.image, self.found = tool, image, found
        self.values = Adjustments()
        self.section = 0
        self.output = None

    def names(self) -> List[str]:
        return [n for n, (grp, _) in PARAMS.items() if grp == SECTIONS[self.section]]

    def render(self) -> None:
        out = self.tool.apply(self.image, self.found, self.values)
        self.output = out
        h, w = out.shape[:2]
        scale = min(1.0, 900 / w, 700 / h)
        shown = cv2.resize(out, (int(w * scale), int(h * scale))) if scale < 1 else out
        cv2.imshow(WINDOW, shown)

    def build_controls(self) -> None:
        cv2.destroyWindow(CONTROLS)
        cv2.namedWindow(CONTROLS, cv2.WINDOW_AUTOSIZE)
        for name in self.names():
            start = int(round(getattr(self.values, name) * 100)) + 100
            cv2.createTrackbar(name, CONTROLS, start, 200, self._handler(name))
        cv2.setWindowTitle(CONTROLS, "%s  (keys 1-5 sections, s save, r reset, q quit)" % SECTIONS[self.section])

    def _handler(self, name: str):
        def on_change(pos: int) -> None:
            setattr(self.values, name, (pos - 100) / 100.0)
            self.render()
        return on_change


def launch(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="facetweak gui", description="Adjust a photo with sliders.")
    parser.add_argument("image")
    parser.add_argument("--backend", choices=["auto", "mediapipe", "dlib"], default="auto")
    parser.add_argument("-o", "--output", help="where 's' saves (default: <name>_tweaked.<ext>)")
    args = parser.parse_args(argv)

    src = Path(args.image)
    image = read_image(src)
    tool = FaceTweak(args.backend)
    found = tool.detect(image)
    if not found:
        raise FaceTweakError("no face found in the image")
    dest = Path(args.output) if args.output else src.with_name(src.stem + "_tweaked" + src.suffix)

    session = _Session(tool, image, found)
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        session.build_controls()
        session.render()
    except cv2.error as exc:
        raise FaceTweakError(
            "OpenCV could not open a window (%s). If you installed opencv-python-headless, "
            "install opencv-python instead." % exc.msg.strip().splitlines()[0]
        ) from exc

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("s") and session.output is not None:
            write_image(dest, session.output)
            print("saved", dest)
        elif key == ord("r"):
            session.values = Adjustments()
            session.build_controls()
            session.render()
        elif ord("1") <= key <= ord("5"):
            session.section = key - ord("1")
            session.build_controls()
    cv2.destroyAllWindows()
    return 0
