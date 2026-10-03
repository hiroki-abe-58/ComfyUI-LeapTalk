"""Against a real ComfyUI checkout on CPU: registration from an alias folder, validation, execution with the
fake runner, VIDEO/AUDIO types, interrupts."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import numpy as np
import pytest
from conftest import FAKE_JOB, REPO_ROOT, fake_worker_runtime_dir, make_runtime, speech_like

pytestmark = pytest.mark.comfy
NODE_IDS = ("LeapTalkRuntime", "LeapTalkGenerate", "LeapTalkDoctor", "LeapTalkWorker")


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


def _write_inputs(image_name: str, audio_name: str) -> None:
    import struct

    import folder_paths
    from PIL import Image

    inp = folder_paths.get_input_directory()
    Image.new("RGB", (128, 128), (100, 120, 140)).save(f"{inp}/{image_name}")
    pcm = (np.zeros(16000, np.int16)).tobytes()
    hdr = (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
    )
    with open(f"{inp}/{audio_name}", "wb") as f:
        f.write(hdr + pcm)


def _v01_runtime(tmp_path) -> dict:
    """A runtime entry in the v0.1 config format (no backend / worker / memory_guard keys)."""
    a = str(tmp_path / "abs")
    return {
        "python": sys.executable,
        "upstream_dir": a + "/u",
        "models": {k: a + "/" + k for k in ("soulx_dir", "wav2vec_dir", "leaptalk_dir")},
        "ffmpeg": a + "/ffmpeg",
    }


def test_validate_prompt(comfyui, runtimes_file, tmp_path):
    import execution

    _write_inputs("portrait 'test' ü.png", "speech test ü.wav")
    runtimes_file({"fake": _v01_runtime(tmp_path)})
    ok = asyncio.run(execution.validate_prompt("p1", _prompt(), None))
    assert ok[0] is True and not ok[3], ok
    for bad in (_prompt(decoder="ltx"), _prompt(runtime_id="not-registered"), _prompt(guidance=9.0)):
        res = asyncio.run(execution.validate_prompt("p2", bad, None))
        assert res[0] is False or res[3]
    persistent = _prompt()
    persistent["1"]["inputs"]["backend"] = "persistent"
    assert asyncio.run(execution.validate_prompt("p3", persistent, None))[0] is True
    persistent["1"]["inputs"]["backend"] = "gpu-server"
    res = asyncio.run(execution.validate_prompt("p4", persistent, None))
    assert res[0] is False or res[3]


def test_shipped_api_workflows_validate_unchanged(comfyui, runtimes_file, tmp_path):
    """The v0.1 workflow (no backend input) and the new ones validate as shipped, against a runtime
    config without the new keys."""
    import execution

    _write_inputs("leaptalk_example_portrait.png", "leaptalk_example_speech.wav")
    runtimes_file({"windows-native": _v01_runtime(tmp_path)})
    names = sorted(p.stem for p in (REPO_ROOT / "workflows" / "api").glob("*.json") if p.name != "layout.json")
    expected = {
        "leaptalk_portrait_speech",
        "leaptalk_doctor",
        "leaptalk_persistent",
        "leaptalk_persistent_lower_memory",
        "leaptalk_worker_status",
        "leaptalk_worker_unload",
    }
    assert expected <= set(names)
    for name in names:
        api = json.loads((REPO_ROOT / "workflows" / "api" / f"{name}.json").read_text(encoding="utf-8"))
        res = asyncio.run(execution.validate_prompt(f"p-{name}", api, None))
        assert res[0] is True and not res[3], (name, res)
    old = json.loads((REPO_ROOT / "workflows" / "api" / "leaptalk_portrait_speech.json").read_text(encoding="utf-8"))
    assert "backend" not in old["1"]["inputs"]  # the v0.1 example stays as it was: one-shot by default
    # the Lite TAE examples stay Lite TAE; only the explicitly named lower-memory workflow uses wan_vae
    lite = json.loads((REPO_ROOT / "workflows" / "api" / "leaptalk_persistent.json").read_text(encoding="utf-8"))
    low = json.loads((REPO_ROOT / "workflows" / "api" / "leaptalk_persistent_lower_memory.json").read_text(encoding="utf-8"))
    assert old["4"]["inputs"]["decoder"] == lite["4"]["inputs"]["decoder"] == "lite_tae"
    assert low["4"]["inputs"]["decoder"] == "wan_vae" and low["1"]["inputs"]["backend"] == "persistent"
    assert low["7"]["class_type"] == "LeapTalkWorker" and low["7"]["inputs"]["action"] == "status"


def test_failure_causes_stay_distinguishable(comfyui, tmp_path, monkeypatch):
    """Memory guard refusal/stop, user cancel, timeout and runtime errors reach ComfyUI as different errors."""
    import comfy.model_management as mm
    import torch

    mod = _mod(comfyui)
    _use_fake(mod, make_runtime(tmp_path, "ok"), monkeypatch)
    cases = [
        (
            mod.worker.MemoryGuardRefused("LeapTalk memory guard: did not start a worker: x", kind="admission"),
            RuntimeError,
            "LeapTalk memory guard: did not start",
        ),
        (
            mod.worker.MemoryGuardRefused("LeapTalk memory guard: stopped the job and the worker - y", kind="stopped"),
            RuntimeError,
            "LeapTalk memory guard: stopped",
        ),
        (mod.client.JobTimeout("LeapTalk job exceeded 60 s"), RuntimeError, "LeapTalk timeout: "),
        (mod.client.RuntimeJobError("runtime job failed (exit 3): boom"), RuntimeError, "LeapTalk runtime error: "),
        (mod.worker.WorkerError("the worker exited unexpectedly"), RuntimeError, "LeapTalk runtime error: "),
        (mod.client.JobCancelled("cancelled by the user"), mm.InterruptProcessingException, None),
    ]
    for exc, expect, prefix in cases:

        def boom(*a, exc=exc, **k):
            raise exc

        monkeypatch.setattr(mod.client, "generate", boom)
        with pytest.raises(expect) as e:
            mod.LeapTalkGenerate().generate({"runtime_id": "fake"}, torch.rand(1, 64, 64, 3), _audio(), "lite_tae", 1.0, False)
        if prefix:
            assert str(e.value).startswith(prefix), str(e.value)


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
    # a v0.1 Runtime output (no "backend" key) and a v0.1 runtime config: one-shot, as before
    assert rep["backend"] == {"requested": "runtime default", "runtime_default": "one-shot", "effective": "one-shot"}
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
    job_dir = max((p for p in mod.client.jobs_root(rt).iterdir() if p.name.startswith("job-")), key=lambda p: p.stat().st_mtime)
    assert json.loads((job_dir / "stop_report.json").read_text(encoding="utf-8"))["active_after"] == 0


@pytest.fixture
def comfy_worker(comfyui, tmp_path, monkeypatch):
    """ComfyUI's own copy of the package (alias folder) with the fake-Engine worker runtime."""
    mod = _mod(comfyui)
    monkeypatch.setattr(mod.client, "RUNTIME_DIR", fake_worker_runtime_dir(tmp_path))
    mod.worker.MANAGER.unload("test setup", wait_s=30)
    yield mod
    mod.worker.MANAGER.unload("test teardown", wait_s=30)


def test_persistent_backend_through_the_nodes(comfy_worker, tmp_path, monkeypatch):
    import torch

    mod = comfy_worker
    rt = make_runtime(tmp_path, "ok")
    monkeypatch.setattr(mod, "_resolve", lambda rid: rt)
    node = mod.LeapTalkWorker()
    first = node.run("status")
    assert json.loads(first["result"][0])["worker"] is None and "no worker loaded" in first["ui"]["text"][0]
    assert mod.worker.MANAGER._worker is None  # status started nothing
    a = mod.LeapTalkWorker.IS_CHANGED(action="status")
    time.sleep(0.02)
    assert mod.LeapTalkWorker.IS_CHANGED(action="status") != a  # never served from ComfyUI's cache
    assert mod.LeapTalkRuntime().select("fake")[0]["backend"] is None  # v0.1 graph: runtime default
    (runtime,) = mod.LeapTalkRuntime().select("fake", "persistent")
    reps = []
    for v in (0.2, 0.8, 0.2):
        _video, report = mod.LeapTalkGenerate().generate(runtime, torch.full((1, 96, 128, 3), v), _audio(1.0), "lite_tae", 1.0, True)
        reps.append(json.loads(report))
    for r in reps:
        assert r["backend"] == {"requested": "persistent", "runtime_default": "one-shot", "effective": "persistent"}
        assert r["engine"]["counters"]["init"] == 1 and r["engine"]["counters"]["lora_merge"] == 1
    assert len({r["host"]["process"]["worker"]["worker_instance_id"] for r in reps}) == 1
    assert [r["host"]["process"]["started_new_worker"] for r in reps] == [True, False, False]
    assert reps[0]["frames_sha256"] == reps[2]["frames_sha256"] != reps[1]["frames_sha256"]
    st = node.run("status")
    assert "is idle with the model loaded" in st["ui"]["text"][0] and "3 jobs" in st["ui"]["text"][0]
    un = node.run("unload")
    out = json.loads(un["result"][0])
    assert out["status"] == "unloaded" and out["active_after"] == 0 and out["now"]["worker"] is None
    assert "no worker loaded" in un["ui"]["text"][0]
