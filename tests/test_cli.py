import json
from pathlib import Path

import numpy as np
import pytest

import facetweak
from facetweak import cli, core, landmarks

ROOT = Path(__file__).resolve().parent.parent
PORTRAIT = ROOT / "examples" / "portrait.jpg"


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert facetweak.__version__ in capsys.readouterr().out


def test_params_lists_every_adjustment(capsys):
    assert cli.main(["params"]) == 0
    out = capsys.readouterr().out
    for name in facetweak.PARAMS:
        assert "--" + name.replace("_", "-") in out
    assert "presets:" in out


def test_missing_file_gives_clean_error(capsys, tmp_path):
    assert cli.main([str(tmp_path / "nope.jpg"), "--eye-size", "0.3"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_nothing_to_do_is_an_error(capsys):
    assert cli.main([str(PORTRAIT)]) == 1
    assert "nothing to do" in capsys.readouterr().err


def test_folder_input_needs_output_folder(capsys, tmp_path):
    assert cli.main([str(tmp_path), "--eye-size", "0.3"]) == 1
    assert "output folder" in capsys.readouterr().err


def test_bad_strength(capsys):
    assert cli.main([str(PORTRAIT), "--eye-size", "0.3", "--strength", "2"]) == 1
    assert "--strength" in capsys.readouterr().err


def test_unknown_flag_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.main([str(PORTRAIT), "--eye-sze", "0.3"])
    assert exc.value.code == 2


def test_bad_face_selector(capsys):
    assert cli.main([str(PORTRAIT), "--eye-size", "0.3", "--face", "left"]) == 1
    assert "--face" in capsys.readouterr().err


def test_read_image_rejects_non_images(tmp_path):
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(facetweak.FaceTweakError):
        core.read_image(bad)


def test_image_roundtrip_with_unicode_path(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    path = tmp_path / "foto é中.png"
    core.write_image(path, img)
    assert np.array_equal(core.read_image(path), img)


def test_side_by_side_shape():
    a = np.zeros((10, 20, 3), np.uint8)
    assert core.side_by_side(a, a).shape == (10, 44, 3)


def test_mesh_mapping_has_68_distinct_points():
    assert len(landmarks.MESH_68) == 68
    assert len(set(landmarks.MESH_68)) == 68


def test_missing_mediapipe_gives_install_hint(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("mediapipe"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(landmarks.BackendUnavailable, match="pip install mediapipe"):
        landmarks.MediaPipeBackend()


def test_missing_dlib_gives_install_hint(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "dlib":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(landmarks.BackendUnavailable, match=r"facetweak\[dlib\]"):
        landmarks.DlibBackend()


def test_bad_backend_name():
    with pytest.raises(ValueError):
        landmarks.make_backend("opencv")


# ---- tests below load the real model (downloaded once, about 4 MB) ----

@pytest.fixture(scope="module")
def tool():
    return facetweak.FaceTweak("mediapipe")


@pytest.mark.slow
def test_detects_one_face_with_sane_geometry(tool):
    faces = tool.detect(PORTRAIT)
    assert len(faces) == 1
    p = faces[0].points
    assert p.shape == (68, 2)
    assert p[36:42, 0].mean() < p[42:48, 0].mean()  # image-left eye first
    assert p[17:27, 1].mean() < p[36:48, 1].mean() < p[48:68, 1].mean() < p[8, 1]


@pytest.mark.slow
def test_retouch_changes_the_face_and_not_the_background(tool):
    before = core.read_image(PORTRAIT)
    after = tool.retouch(PORTRAIT, eye_size=0.6, face_slim=0.6)
    assert after.shape == before.shape
    assert np.abs(after.astype(int) - before.astype(int)).sum() > 0
    assert np.array_equal(after[:20, :20], before[:20, :20])  # backdrop corner


@pytest.mark.slow
def test_no_face_raises(tool):
    blank = np.full((300, 300, 3), 127, np.uint8)
    with pytest.raises(facetweak.NoFaceFound):
        tool.retouch(blank, eye_size=0.3)


@pytest.mark.slow
def test_face_selection_errors(tool):
    with pytest.raises(facetweak.FaceTweakError, match="out of range"):
        tool.retouch(PORTRAIT, faces=[3], eye_size=0.3)
    assert tool.retouch(PORTRAIT, faces="largest", eye_size=0.3).shape[2] == 3


@pytest.mark.slow
def test_cli_end_to_end_json(tmp_path, capsys):
    out = tmp_path / "out.jpg"
    code = cli.main([str(PORTRAIT), "-o", str(out), "--preset", "subtle", "--eye-size", "0.5", "--json"])
    assert code == 0
    info = json.loads(capsys.readouterr().out)
    assert info["faces"] == 1 and info["applied"]["eye_size"] == pytest.approx(0.5)
    assert out.exists()


@pytest.mark.slow
def test_cli_batch_skips_images_without_faces(tmp_path, capsys):
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.jpg").write_bytes(PORTRAIT.read_bytes())
    core.write_image(src / "blank.png", np.full((200, 200, 3), 90, np.uint8))
    code = cli.main([str(src), "-o", str(tmp_path / "out"), "--face-slim", "0.4"])
    assert code == 1  # one image was skipped
    assert (tmp_path / "out" / "a.jpg").exists()
    assert not (tmp_path / "out" / "blank.png").exists()
    assert "SKIPPED" in capsys.readouterr().out


@pytest.mark.slow
def test_cli_compare_is_double_width(tmp_path):
    out = tmp_path / "cmp.png"
    assert cli.main([str(PORTRAIT), "-o", str(out), "--eye-size", "0.3", "--compare"]) == 0
    w = core.read_image(PORTRAIT).shape[1]
    assert core.read_image(out).shape[1] == 2 * w + 4


@pytest.mark.slow
def test_landmarks_debug_image(tmp_path):
    out = tmp_path / "lm.png"
    assert cli.main([str(PORTRAIT), "-o", str(out), "--landmarks"]) == 0
    assert out.exists()
