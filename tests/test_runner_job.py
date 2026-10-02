"""The runtime runner's own validation and the official chunk/padding arithmetic (pure CPU, no torch)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from leaptalk_comfy import client
from leaptalk_comfy.config import parse_runtime

REPO = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("leaptalk_job_under_test", REPO / "runtime" / "leaptalk_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _job_dir(tmp_path, rt):
    job_id = "job-20260101-000000-abcdef012345"
    d = tmp_path / job_id
    d.mkdir()
    (d / "input.png").write_bytes(b"png")
    (d / "input.wav").write_bytes(b"wav")
    job = client.make_job(rt, job_id, decoder="lite_tae", audio_guidance=1.0, preview=True)
    return d, job


def _rt(tmp_path):
    a = str(tmp_path / "abs")
    return parse_runtime(
        "r",
        {
            "python": sys.executable,
            "upstream_dir": a + "/up",
            "models": {k: a + "/" + k for k in ("soulx_dir", "wav2vec_dir", "leaptalk_dir")},
            "ffmpeg": a + "/ffmpeg",
            "env": {"CUDA_VISIBLE_DEVICES": "0"},
        },
    )


def test_job_round_trip(tmp_path):
    lj = _runner()
    d, job = _job_dir(tmp_path, _rt(tmp_path))
    (d / "job.json").write_text(json.dumps(job), encoding="utf-8")
    assert lj.load_job(d / "job.json")["decoder"] == "lite_tae"


@pytest.mark.parametrize(
    "patch",
    [
        {"job_id": "../x"},
        {"decoder": "ltx_vae"},
        {"audio_guidance": 9.0},
        {"audio_guidance": True},
        {"max_audio_seconds": 10**6},
        {"video_crf": "18"},
        {"preview": "yes"},
        {"env": {"LD_PRELOAD": "/x.so"}},
        {"env": {"HF_TOKEN": "x"}},
        {"upstream_dir": "relative/up"},
        {"ffmpeg": "/x/../ffmpeg"},
        {"models": {"soulx_dir": "/a"}},
        {"command": "id"},
    ],
)
def test_runner_rejects_tampered_jobs(tmp_path, patch):
    lj = _runner()
    d, job = _job_dir(tmp_path, _rt(tmp_path))
    job.update(patch)
    (d / "job.json").write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(lj.JobError):
        lj.load_job(d / "job.json")


def test_runner_needs_inputs_in_its_own_directory(tmp_path):
    lj = _runner()
    d, job = _job_dir(tmp_path, _rt(tmp_path))
    (d / "input.wav").unlink()
    (d / "job.json").write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(lj.JobError, match="input.wav"):
        lj.load_job(d / "job.json")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "job.json").write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(lj.JobError, match="own job directory"):
        lj.load_job(other / "job.json")


@pytest.mark.parametrize(
    "seconds, samples_16k, chunks, out_frames",
    [
        (9.35, 149600, 9, 234),  # measured: official run wrote 9 chunks x 28 = 252 frames
        (33.975, 543600, 31, 850),  # measured: 31 chunks / 868 frames, 850 kept
        (0.5, 8000, 2, 13),  # short input: padded to one window, then to 2 slices
        (1.12, 17920, 2, 28),
        (2.24, 35840, 2, 56),
        (60.0, 960000, 54, 1500),
    ],
)
def test_chunk_plan_matches_the_official_padding(seconds, samples_16k, chunks, out_frames):
    lj = _runner()
    plan = lj.chunk_plan(samples_16k, seconds)
    assert plan["chunks"] == chunks and plan["out_frames"] == out_frames
    assert plan["slice_len"] == 28 and plan["motion_frames"] == 5
    assert plan["padded_samples"] % 17920 == 0 and plan["padded_samples"] >= max(samples_16k, 21120)
    assert plan["chunks"] * 28 >= plan["out_frames"] == math.ceil(seconds * 25 - 1e-9)


def test_upstream_hashes_ignore_line_endings(tmp_path):
    lj = _runner()
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert lj._sha256_text(lf) == lj._sha256_text(crlf) == hashlib.sha256(b"a = 1\nb = 2\n").hexdigest()
    assert len(lj.UPSTREAM_FILES) == 14 and all(len(v) == 64 for v in lj.UPSTREAM_FILES.values())
    with pytest.raises(lj.JobError, match="does not match"):
        lj.check_upstream(tmp_path)


def test_model_file_check_and_decoder_specific_files(tmp_path):
    lj = _runner()
    models = {k: str(tmp_path / k) for k in lj.MODEL_FILES}
    probs = lj.check_model_files(models, "lite_tae")
    assert any("taew2_1.pth" in p for p in probs) and not any("Wan2.1_VAE" in p for p in probs)
    probs = lj.check_model_files(models, "wan_vae")
    assert any("Wan2.1_VAE" in p for p in probs) and not any("taew2_1" in p for p in probs)


def test_secret_variables_are_removed(tmp_path, monkeypatch):
    lj = _runner()
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("MY_API_KEY", "x")
    lj.apply_env({"CUDA_VISIBLE_DEVICES": "0"}, tmp_path)
    import os

    assert "HF_TOKEN" not in os.environ and "MY_API_KEY" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0" and os.environ["HF_HUB_OFFLINE"] == "1"
