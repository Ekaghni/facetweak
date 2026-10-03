# Changelog

## 0.1.0

First release. Rewrite of the original single-file script as a package.

- 23 adjustments for eyes, eyebrows, nose, lips and face, each from -1 to 1.
- MediaPipe landmarks by default, dlib as an optional extra.
- CLI with presets, folder mode, `--compare`, `--landmarks`, `--json`, and an OpenCV slider window.
- Python API: `FaceTweak`, `Adjustments`.
- Warps only touch a window around each feature, about 10x faster than the old full-frame version on a similar effect stack.
- Fixed adjustments whose direction contradicted their label in the old code.
