"""Measure how long facetweak takes. Run: python benchmarks/latency.py

Uses examples/portrait.jpg resized to a few widths. Prints a markdown table.
Numbers depend on your CPU, so run it yourself before trusting mine.
"""

import platform
import statistics
import sys
import time
from pathlib import Path

import cv2

from facetweak import PRESETS, FaceTweak, read_image

ROOT = Path(__file__).resolve().parent.parent
WIDTHS = [480, 960, 1920, 3840]
RUNS = 20


def timed(fn, runs=RUNS):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "mediapipe"
    base = read_image(ROOT / "examples" / "portrait.jpg")

    t = time.perf_counter()
    tool = FaceTweak(backend)
    tool.detect(base)
    cold = (time.perf_counter() - t) * 1000

    print("cpu: %s, python %s, backend %s" % (platform.processor() or platform.machine(), platform.python_version(), backend))
    print("cold start (load model + first detection): %.0f ms\n" % cold)
    print("| width | detect (ms) | apply `doll` preset (ms) | total (ms) |")
    print("|---:|---:|---:|---:|")
    for w in WIDTHS:
        img = cv2.resize(base, (w, int(base.shape[0] * w / base.shape[1])), interpolation=cv2.INTER_CUBIC)
        found = tool.detect(img)
        d = timed(lambda: tool.detect(img))
        a = timed(lambda: tool.apply(img, found, PRESETS["doll"]))
        print("| %d | %.1f | %.1f | %.1f |" % (w, d, a, d + a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
