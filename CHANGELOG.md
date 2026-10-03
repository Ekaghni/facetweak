# Changelog

## 0.2.0

- Stronger maximum effect for eye size, eye openness, face slimming, lip size and nose size, so a value of 1.0 is clearly visible. Values you used before will look stronger.
- README comparison images are cropped to the face and use stronger settings. `benchmarks/make_examples.py` regenerates them.

## 0.1.3

- README: more before/after examples, a strength comparison and a table of what has and hasn't been tested. Three more public domain sample photos in `examples/`.

## 0.1.2

- The Linux hint now lists all three libraries (`libegl1 libgles2 libgl1`). 0.1.1 missed `libgles2`.

## 0.1.1

- Missing system libraries on Linux (`libEGL.so.1`, `libGLESv2.so.2`) now give a message with the apt command instead of a raw OSError.
- README: Linux note. CI installs the libraries and runs the model tests on Ubuntu.

## 0.1.0

First release. Rewrite of the original single-file script as a package.

- 23 adjustments for eyes, eyebrows, nose, lips and face, each from -1 to 1.
- MediaPipe landmarks by default, dlib as an optional extra.
- CLI with presets, folder mode, `--compare`, `--landmarks`, `--json`, and an OpenCV slider window.
- Python API: `FaceTweak`, `Adjustments`.
- Warps only touch a window around each feature, about 10x faster than the old full-frame version on a similar effect stack.
- Fixed adjustments whose direction contradicted their label in the old code.
