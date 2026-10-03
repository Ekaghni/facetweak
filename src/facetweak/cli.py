"""Command line interface."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import __version__
from .core import FaceTweak, draw_landmarks, read_image, side_by_side, write_image
from .effects import PARAMS, PRESETS, Adjustments
from .landmarks import FaceTweakError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="facetweak",
        description="Retouch faces with landmark-guided warps. Every adjustment is a number "
        "from -1 to 1; 0 changes nothing. Other commands: `facetweak params`, `facetweak gui IMAGE`.",
    )
    p.add_argument("image", help="image file, or a folder of images (then -o must be a folder)")
    p.add_argument("-o", "--output", help="output file or folder (default: <name>_tweaked.<ext>)")
    p.add_argument("--preset", choices=sorted(PRESETS), help="start from a preset, then apply any flags on top")
    p.add_argument("--strength", type=float, default=1.0, help="scale every adjustment, 0 to 1 (default 1)")
    p.add_argument("--face", default="all", help="'all' (default), 'largest', or indexes like 0,2 (left to right)")
    p.add_argument("--backend", choices=["auto", "mediapipe", "dlib"], default="auto")
    p.add_argument("--compare", action="store_true", help="save before and after side by side")
    p.add_argument("--landmarks", action="store_true", help="save a debug image with the detected points and exit")
    p.add_argument("--json", action="store_true", help="print a machine-readable summary")
    p.add_argument("--version", action="version", version="facetweak " + __version__)
    group = p.add_argument_group("adjustments (-1 to 1, see `facetweak params`)")
    for name in PARAMS:
        group.add_argument(_flag(name), type=float, metavar="V", dest=name)
    return p


def _parse_faces(text: str):
    if text in ("all", "largest"):
        return text
    try:
        return [int(x) for x in text.split(",")]
    except ValueError:
        raise FaceTweakError("--face must be 'all', 'largest' or a comma list of numbers, got %r" % text)


def _list_params() -> int:
    group = None
    for name, (grp, doc) in PARAMS.items():
        if grp != group:
            print("\n[%s]" % grp)
            group = grp
        print("  %-16s positive = %s" % (_flag(name), doc))
    print("\npresets: " + ", ".join(sorted(PRESETS)))
    return 0


def _images_in(path: Path) -> List[Path]:
    return sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)


def _out_path(src: Path, output: Optional[str], batch: bool) -> Path:
    if batch:
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / src.name
    if output:
        return Path(output)
    return src.with_name(src.stem + "_tweaked" + src.suffix)


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    src = Path(args.image)
    if not src.exists():
        raise FaceTweakError("no such file or folder: %s" % src)
    batch = src.is_dir()
    if batch and not args.output:
        raise FaceTweakError("give an output folder with -o when the input is a folder")
    if not 0 <= args.strength <= 1:
        raise FaceTweakError("--strength must be between 0 and 1")

    values = {n: getattr(args, n) for n in PARAMS if getattr(args, n) is not None}
    adj = PRESETS[args.preset] if args.preset else Adjustments()
    merged = {**adj.active(), **values}
    adj = Adjustments.from_dict(merged).scaled(args.strength)
    if not adj.active() and not args.landmarks:
        raise FaceTweakError("nothing to do: pass an adjustment such as --eye-size 0.3, or --preset subtle")
    faces = _parse_faces(args.face)

    tool = FaceTweak(args.backend)
    files = _images_in(src) if batch else [src]
    summary = []
    for f in files:
        start = time.time()
        img = read_image(f)
        entry = {"input": str(f), "backend": tool.backend.name}
        try:
            if args.landmarks:
                found = tool.detect(img)
                dest = _out_path(f, args.output, batch)
                if not batch and not args.output:
                    dest = f.with_name(f.stem + "_landmarks.png")
                write_image(dest, draw_landmarks(img, found))
                entry.update(output=str(dest), faces=len(found))
            else:
                found = tool.detect(img)
                out = tool.apply(img, found, adj, faces)
                dest = _out_path(f, args.output, batch)
                write_image(dest, side_by_side(img, out) if args.compare else out)
                entry.update(output=str(dest), faces=len(found), applied=adj.active())
        except FaceTweakError as exc:
            if not batch:
                raise
            entry["error"] = str(exc)
        entry["seconds"] = round(time.time() - start, 3)
        summary.append(entry)
        if not args.json:
            print("%s -> %s" % (f.name, entry.get("output", "SKIPPED: " + entry.get("error", ""))))

    if args.json:
        print(json.dumps(summary if batch else summary[0], indent=2))
    return 1 if any("error" in e for e in summary) else 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv[:1] == ["params"]:
            return _list_params()
        if argv[:1] == ["gui"]:
            from .gui import launch

            return launch(argv[1:])
        return run(argv)
    except FaceTweakError as exc:
        print("facetweak: " + str(exc), file=sys.stderr)
        return 1
