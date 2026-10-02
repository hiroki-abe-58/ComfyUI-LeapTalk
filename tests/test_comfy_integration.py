"""Against a real ComfyUI checkout on CPU: registration from an alias folder, validation, execution with the
fake runner, VIDEO/AUDIO types, interrupts."""

from __future__ import annotations

import asyncio
import json
import sys

import numpy as np
import pytest
from conftest import FAKE_JOB, make_runtime, speech_like

pytestmark = pytest.mark.comfy
NODE_IDS = ("LeapTalkRuntime", "LeapTalkGenerate", "LeapTalkDoctor")


def _mod(comfyui):
    return sys.modules[comfyui["nodes"].NODE_CLASS_MAPPINGS["LeapTalkGenerate"].__module__]


def _use_fake(mod, rt, monkeypatch):
    # ComfyUI imported its own copy of the package (from the alias folder); point that copy at the fake runner
    monkeypatch.setitem(mod.client.RUNTIME_SCRIPTS, mod.client.JOB_SCRIPT, FAKE_JOB)
    monkeypatch.setattr(mod, "_resolve", lambda rid: rt)


def test_registered_from_alias_folder_without_heavy_imports(comfyui):
    mappings = comfyui["nodes"].NODE_CLASS_MAPPINGS
    for nid in NODE_IDS:
        assert nid in mappings
    mod = _mod(comfyui)
    assert "LeapTalk alias" in mod.__file__ or "LeapTalk alias" in str(comfyui["alias"])
    for heavy in ("flash_head", "inference", "peft", "diffusers", "xfuser", "librosa", "vibt"):
        assert heavy not in sys.modules, heavy


def test_no_runtime_configured(comfyui, tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_LEAPTALK_CONFIG", str(tmp_path / "missing.json"))
    mod = _mod(comfyui)
    assert mod.LeapTalkRuntime.INPUT_TYPES()["required"]["runtime_id"][0] == [mod.NO_RUNTIME]
    with pytest.raises(ValueError, match="not configured"):
        mod.LeapTalkRuntime().select(mod.NO_RUNTIME)
    out = mod.LeapTalkDoctor().check(mod.NO_RUNTIME, False)
    assert json.loads(out["result"][0])["overall"] == "fail"


def _prompt(runtime_id="fake", decoder="lite_tae", guidance=1.0):
    return {
        "1": {"class_type": "LeapTalkRuntime", "inputs": {"runtime_id": runtime_id}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "portrait 'test' ü.png"}},
        "3": {"class_type": "LoadAudio", "inputs": {"audio": "speech test ü.wav"}},
        "4": {
            "class_type": "LeapTalkGenerate",
            "inputs": {"runtime": ["1", 0], "image": ["2", 0], "audio": ["3", 0], "decoder": decoder, "audio_guidance": guidance, "preview": True},
        },
        "5": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0], "filename_prefix": "leaptalk/test", "format": "auto", "codec": "auto"}},
    }


def test_validate_prompt(comfyui, runtimes_file, tmp_path):
    import execution
    import folder_paths
    from PIL import Image

    inp = folder_paths.get_input_directory()
    Image.new("RGB", (128, 128), (100, 120, 140)).save(f"{inp}/portrait 'test' ü.png")
    import struct

    pcm = (np.zeros(16000, np.int16)).tobytes()
    hdr = (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
    )
    with open(f"{inp}/speech test ü.wav", "wb") as f:
        f.write(hdr + pcm)
    a = str(tmp_path / "abs")
    runtimes_file(
        {
            "fake": {
                "python": sys.executable,
                "upstream_dir": a + "/u",
                "models": {k: a + "/" + k for k in ("soulx_dir", "wav2vec_dir", "leaptalk_dir")},
                "ffmpeg": a + "/ffmpeg",
            }
        }
    )
    ok = asyncio.run(execution.validate_prompt("p1", _prompt(), None))
    assert ok[0] is True and not ok[3], ok
    for bad in (_prompt(decoder="ltx"), _prompt(runtime_id="not-registered"), _prompt(guidance=9.0)):
        res = asyncio.run(execution.validate_prompt("p2", bad, None))
        assert res[0] is False or res[3]


def _audio(seconds=1.5, sr=24000, ch=1, batch=1):
    import torch

    w = torch.from_numpy(np.stack([speech_like(seconds, sr, ch)] * batch))
    return {"waveform": w, "sample_rate": sr}


def test_generate_returns_video_with_audio(comfyui, tmp_path, monkeypatch):
    import torch

    mod = _mod(comfyui)
    rt = make_runtime(tmp_path, "ok")
    _use_fake(mod, rt, monkeypatch)
    image = torch.rand(1, 200, 160, 3)
    video, report = mod.LeapTalkGenerate().generate({"runtime_id": "fake"}, image, _audio(1.5, 48000, 2), "lite_tae", 1.0, True)
    from comfy_api.latest import InputImpl

    assert isinstance(video, InputImpl.VideoFromFile)
    assert video.get_dimensions() == (512, 512)
    comps = video.get_components()
    assert comps.images.shape[0] == 38 and float(comps.frame_rate) == 25.0  # ceil(1.5 * 25)
    assert comps.audio is not None and comps.audio["waveform"].shape[1] == 2  # original stereo muxed
    rep = json.loads(report)
    assert rep["output_frames"] == 38 and rep["host"]["audio"]["sample_rate"] == 48000 and rep["host"]["audio"]["channels"] == 2
    job_dir = mod.client.jobs_root(rt) / rep["job"]
    from PIL import Image

    assert Image.open(job_dir / "input.png").size == (160, 200)


def test_rejects_batches_and_bad_inputs(comfyui, tmp_path, monkeypatch):
    import torch

    mod = _mod(comfyui)
    _use_fake(mod, make_runtime(tmp_path, "ok"), monkeypatch)
    gen = mod.LeapTalkGenerate()
    with pytest.raises(ValueError, match="exactly one reference image"):
        gen.generate({"runtime_id": "fake"}, torch.rand(2, 64, 64, 3), _audio(), "lite_tae", 1.0, False)
    with pytest.raises(ValueError, match="one audio clip"):
        gen.generate({"runtime_id": "fake"}, torch.rand(1, 64, 64, 3), _audio(batch=2), "lite_tae", 1.0, False)
    with pytest.raises(ValueError, match="NaN"):
        bad = _audio()
        bad["waveform"][0, 0, 10] = float("nan")
        gen.generate({"runtime_id": "fake"}, torch.rand(1, 64, 64, 3), bad, "lite_tae", 1.0, False)
    with pytest.raises(ValueError, match="connect a LeapTalk Runtime"):
        gen.generate({}, torch.rand(1, 64, 64, 3), _audio(), "lite_tae", 1.0, False)


def test_interrupt_maps_to_comfy_interrupt(comfyui, tmp_path, monkeypatch):
    import comfy.model_management as mm
    import torch

    mod = _mod(comfyui)
    rt = make_runtime(tmp_path, "hang")
    _use_fake(mod, rt, monkeypatch)
    calls = {"n": 0}

    def interrupted():
        calls["n"] += 1
        return calls["n"] > 10

    monkeypatch.setattr(mm, "processing_interrupted", interrupted)
    with pytest.raises(mm.InterruptProcessingException):
        mod.LeapTalkGenerate().generate({"runtime_id": "fake"}, torch.rand(1, 64, 64, 3), _audio(), "lite_tae", 1.0, False)
    job_dir = max(mod.client.jobs_root(rt).iterdir(), key=lambda p: p.stat().st_mtime)
    assert json.loads((job_dir / "stop_report.json").read_text(encoding="utf-8"))["active_after"] == 0
