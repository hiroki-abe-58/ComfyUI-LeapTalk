"""Runtime config validation and the job inputs written by the node (pure CPU)."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from leaptalk_comfy import client
from leaptalk_comfy.config import ConfigError, load_runtimes, parse_runtime

REPO = Path(__file__).resolve().parents[1]
ABS = "C:\\rt" if sys.platform == "win32" else "/rt"
BASE = {
    "kind": "native",
    "python": f"{ABS}/venv/python",
    "upstream_dir": f"{ABS}/LeapTalk",
    "models": {"soulx_dir": f"{ABS}/m/soulx", "wav2vec_dir": f"{ABS}/m/w2v", "leaptalk_dir": f"{ABS}/m/lt"},
    "ffmpeg": f"{ABS}/ffmpeg",
}


def test_example_config_parses():
    data = json.loads((REPO / "examples" / "leaptalk.runtimes.example.json").read_text(encoding="utf-8"))
    for rid, rt in data["runtimes"].items():
        if sys.platform == "win32" and rid.startswith("linux"):
            continue
        if sys.platform != "win32" and rid.startswith("windows"):
            continue
        r = parse_runtime(rid, rt)
        assert r.kind == "native" and set(r.models) == {"soulx_dir", "wav2vec_dir", "leaptalk_dir"}


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"kind": "wsl"}, "kind"),
        ({"python": "venv/python"}, "absolute"),
        ({"upstream_dir": f"{ABS}/../x"}, "'..'"),
        ({"models": {"soulx_dir": f"{ABS}/a"}}, "models"),
        ({"models": {**BASE["models"], "extra": f"{ABS}/e"}}, "models"),
        ({"ffmpeg": ""}, "ffmpeg"),
        ({"env": {"HF_TOKEN": "x"}}, "env"),
        ({"env": {"CUDA_VISIBLE_DEVICES": 0}}, "env"),
        ({"timeout_minutes": 0}, "timeout"),
        ({"max_audio_seconds": 5000}, "max_audio_seconds"),
        ({"command": "rm -rf /"}, "unknown keys"),
    ],
)
def test_rejects_bad_runtime(patch, message):
    with pytest.raises(ConfigError, match=message):
        parse_runtime("rt", {**BASE, **patch})


def test_rejects_bad_ids_and_files(tmp_path, runtimes_file):
    with pytest.raises(ConfigError):
        parse_runtime("bad id; rm", BASE)
    assert load_runtimes(tmp_path / "none.json") == {}
    p = tmp_path / "broken.json"
    p.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_runtimes(p)
    p.write_text(json.dumps({"runtimes": {}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="schema_version"):
        load_runtimes(p)
    runtimes_file({"a": BASE})
    assert list(load_runtimes()) == ["a"]


def _read_float_wav(path: Path):
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    pos, fmt, samples = 12, None, None
    while pos < len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            samples = np.frombuffer(body, dtype="<f4")
        pos += 8 + size + (size & 1)
    tag, ch, sr, _rate, _block, bits = fmt
    assert tag == 3 and bits == 32
    return samples.reshape(-1, ch).T, sr


def test_wav_is_written_unchanged(tmp_path):
    rng = np.random.default_rng(0)
    for ch, sr in ((1, 16000), (2, 44100), (2, 48000)):
        wav = (rng.standard_normal((ch, sr // 2)) * 0.1).astype(np.float32)
        info = client.save_wav_float32(wav, sr, tmp_path / f"a{ch}{sr}.wav", 60)
        back, sr2 = _read_float_wav(tmp_path / f"a{ch}{sr}.wav")
        assert sr2 == sr and back.shape == wav.shape and np.array_equal(back, wav)
        assert info["channels"] == ch and info["sample_rate"] == sr and info["samples"] == sr // 2


@pytest.mark.parametrize(
    "wav, sr, message",
    [
        (np.zeros((1, 0), np.float32), 16000, "empty"),
        (np.zeros((1, 1000), np.float32), 16000, "at least"),
        (np.zeros((1, 16000 * 61), np.float32), 16000, "up to 60 s"),
        (np.full((1, 16000), np.nan, np.float32), 16000, "NaN"),
        (np.zeros((9, 16000), np.float32), 16000, "channels"),
        (np.zeros(16000, np.float32), 16000, "channels x samples"),
        (np.zeros((1, 16000), np.float32), 4000, "sample rate"),
    ],
)
def test_wav_rejects_bad_audio(tmp_path, wav, sr, message):
    with pytest.raises(ValueError, match=message):
        client.save_wav_float32(wav, sr, tmp_path / "x.wav", 60)


def test_silence_is_accepted_and_flagged(tmp_path):
    info = client.save_wav_float32(np.zeros((1, 16000), np.float32), 16000, tmp_path / "s.wav", 60)
    assert info["silent"] is True


def test_image_writer(tmp_path):
    from PIL import Image

    rgba = np.zeros((100, 80, 4), np.uint8)
    rgba[..., 0] = 200
    info = client.save_image(rgba, tmp_path / "i.png")
    im = Image.open(tmp_path / "i.png")
    assert im.mode == "RGB" and im.size == (80, 100) and info["alpha_dropped"]
    assert np.asarray(im)[0, 0].tolist() == [200, 0, 0]
    for bad in (np.zeros((32, 32, 3), np.uint8), np.zeros((100, 100), np.uint8), np.zeros((100, 100, 3), np.float32)):
        with pytest.raises(ValueError):
            client.save_image(bad, tmp_path / "x.png")


def test_child_env_has_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-value")
    monkeypatch.setenv("HF_TOKEN", "secret-value")
    monkeypatch.setenv("REGISTRY_ACCESS_TOKEN", "secret-value")
    env = client.child_env(tmp_path, sys.executable)
    assert "secret-value" not in env.values()
    assert not [k for k in env if "TOKEN" in k.upper()]
    assert env["TEMP"] == str(tmp_path / "tmp") and env["HF_HUB_OFFLINE"] == "1"
