"""Memory guard: units, estimates, the admission formula, the run-time rule, and that one-shot jobs and
worker start-up are guarded too (CPU, fake runtimes; no real memory pressure is created)."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from conftest import make_runtime, speech_like

from leaptalk_comfy import memory

GIB = memory.GIB


def snap(commit, limit=93.3, avail=20.0, enforced=True):
    return {
        "measurable": True,
        "commit_enforced": enforced,
        "commit_bytes": int(commit * GIB),
        "commit_limit_bytes": int(limit * GIB),
        "available_bytes": int(avail * GIB),
        "commit_pct": round(100 * commit / limit, 2),
        "t": time.monotonic(),
    }


def _img():
    return np.full((96, 128, 3), 120, np.uint8)


def test_snapshot_is_in_bytes_with_display_values():
    s = memory.snapshot()
    assert s["measurable"], "Windows (GetPerformanceInfo) and Linux (/proc/meminfo) are measurable"
    for k in ("commit", "commit_limit", "available", "total"):
        assert isinstance(s[f"{k}_bytes"], int) and s[f"{k}_bytes"] > 0
        assert s[f"{k}_gib"] == round(s[f"{k}_bytes"] / GIB, 2)
    assert s["commit_pct"] == round(100 * s["commit_bytes"] / s["commit_limit_bytes"], 2)


def test_unknown_memory_is_never_unlimited():
    cfg = memory.settings({})
    for kind in ("cold", "warm"):
        rep = memory.admit({"measurable": False, "t": 0}, cfg, {"bytes": GIB}, kind)
        assert not rep["admitted"] and "cannot measure" in rep["reason"]
    p = memory.SustainedPressure(cfg)
    assert not p.update({"measurable": False, "t": 0.0}) and p.update({"measurable": False, "t": 5.0})
    assert memory._b({}, "commit") is None and memory._b({"commit_gib": 2.0}, "commit") == 2 * GIB


def test_estimates_per_decoder_profile_and_high_water():
    cfg = memory.settings({})
    e = memory.PeakEstimates()
    lite, wan, lite_cfg = ("lite_tae", "single"), ("wan_vae", "single"), ("lite_tae", "cfg")
    assert memory.profile_of({"decoder": "wan_vae", "audio_guidance": 1.0}) == wan
    assert memory.profile_of({"decoder": "lite_tae", "audio_guidance": 2.0}) == lite_cfg
    assert e.estimate(lite, "cold", cfg)["bytes"] == int(16.6 * GIB) and e.estimate(lite, "warm", cfg)["bytes"] == int(8.1 * GIB)
    assert e.estimate(wan, "cold", cfg)["bytes"] == int(13.1 * GIB) and e.estimate(wan, "warm", cfg)["bytes"] == int(4.1 * GIB)
    # never measured (audio guidance > 1): the job part counts twice
    assert e.estimate(lite_cfg, "warm", cfg)["bytes"] == int((8.1 + 8.1) * GIB)
    assert e.estimate(lite_cfg, "cold", cfg)["bytes"] == int((16.6 + 8.1) * GIB)
    assert "never measured" in e.estimate(lite_cfg, "warm", cfg)["source"]
    # a measurement below the fallback does not make the estimate less conservative
    e.observe(wan, "warm", 3 * GIB, "commit")
    assert e.estimate(wan, "warm", cfg)["bytes"] == int(4.1 * GIB)
    # above it, the high-water is used, and a later small job (a 0.5 s clip) does not lower it
    e.observe(wan, "warm", 6 * GIB, "cuda_reserved")
    e.observe(wan, "warm", 1 * GIB, "commit")
    est = e.estimate(wan, "warm", cfg)
    assert est["bytes"] == 6 * GIB and est["samples"] == 3 and est["high_water_bytes"] == 6 * GIB and "measured" in est["source"]
    assert e.estimate(lite, "warm", cfg)["bytes"] == int(8.1 * GIB) and e.estimate(wan, "cold", cfg)["bytes"] == int(13.1 * GIB)  # no mixing
    e.observe(wan, "warm", 0, "commit")  # nothing measured (e.g. no sample): ignored
    assert e.estimate(wan, "warm", cfg)["samples"] == 3
    over = memory.settings({"expected_job_gib": 10.0})  # administrator fallback, still never below a measurement
    assert e.estimate(lite, "warm", over)["bytes"] == 10 * GIB and "runtime config" in e.estimate(lite, "warm", over)["source"]
    assert e.estimate(wan, "warm", memory.settings({"expected_job_gib": 2.0}))["bytes"] == 6 * GIB


def test_admission_formula_boundaries_and_report():
    cfg = memory.settings({})
    est = {"bytes": 8 * GIB, "source": "test", "profile": "lite_tae, audio guidance 1.0"}
    limit = 100.0  # GiB, so GiB and % read the same
    ok = memory.admit(snap(84.9, limit), cfg, est, "warm")  # 84.9 + 8 + 2 = 94.9 < 95
    assert ok["admitted"] and ok["safety_margin_bytes"] == 2 * GIB and ok["safety_margin_pct"] == 25.0
    assert ok["projected_commit_bytes"] == int(84.9 * GIB) + 10 * GIB
    at = memory.admit(snap(85.0, limit), cfg, est, "warm")  # exactly at the stop level: refused
    assert not at["admitted"]
    bad = memory.admit(snap(85.5, limit), cfg, est, "warm")
    assert not bad["admitted"] and bad["shortfall_bytes"] == int(85.5 * GIB) + 10 * GIB - int(0.95 * int(100 * GIB))
    assert "limit is 95 % (the stop level)" in bad["reason"] and "does not guarantee" in bad["reason"]
    # projected between the stop level (95) and max_projected (97): refused - a job predicted to be stopped is not started
    mid = memory.admit(snap(86.0, limit), cfg, est, "warm")
    assert not mid["admitted"] and mid["projected_commit_pct"] == 96.0 and mid["projected_limit_pct"] == 95.0
    # a stricter configured projection limit applies
    strict = memory.admit(snap(80.5, limit), memory.settings({"max_projected_commit_pct": 90.0}), est, "warm")
    assert not strict["admitted"] and "max_projected_commit_pct" in strict["reason"]
    # cold = load + first job from the commit before loading; warm = the job on top of the resident model
    cold = memory.admit(snap(70.0, limit), cfg, {"bytes": 17 * GIB}, "cold")
    assert cold["admitted"] and cold["projected_commit_bytes"] == int(70 * GIB) + 17 * GIB + int(round(17 * GIB * 0.25))
    warm = memory.admit(snap(79.0, limit), cfg, {"bytes": int(8.5 * GIB)}, "warm")  # 79 already contains the worker
    assert warm["projected_commit_bytes"] == int(79 * GIB) + int(8.5 * GIB) + int(round(8.5 * GIB * 0.25))
    assert "no model is loaded at or above 90 %" in memory.admit(snap(90.0, limit), cfg, {"bytes": GIB}, "cold")["reason"]
    assert "physical memory available" in memory.admit(snap(50.0, limit, avail=7.9), cfg, {"bytes": GIB}, "cold")["reason"]
    assert "at or above the stop level" in memory.admit(snap(95.0, limit), cfg, {"bytes": GIB}, "warm")["reason"]
    for k in (
        "commit_now_bytes",
        "commit_limit_bytes",
        "physical_available_bytes",
        "estimated_additional_peak_bytes",
        "safety_margin_bytes",
        "projected_commit_bytes",
        "start_limit_pct",
        "stop_limit_pct",
        "admitted",
        "shortfall_bytes",
    ):
        assert k in bad
    assert memory.admit(snap(99.0, limit), memory.settings({"enabled": False}), est, "warm")["admitted"]
    # Linux without strict overcommit: only available memory counts
    lx = memory.admit(snap(150.0, limit, avail=3.0, enforced=False), cfg, {"bytes": GIB}, "cold")
    assert not lx["admitted"] and "memory available" in lx["reason"]


def test_one_shot_is_refused_before_starting(tmp_path, fake_runner, monkeypatch):
    client = fake_runner
    from leaptalk_comfy import worker

    monkeypatch.setattr(worker.MANAGER, "estimates", memory.PeakEstimates())
    rt = make_runtime(tmp_path, "ok", memory_guard={})
    monkeypatch.setattr(memory, "snapshot", lambda: snap(91.0))
    with pytest.raises(memory.MemoryGuardRefused, match="LeapTalk memory guard: did not start the one-shot job") as e:
        client.generate(rt, _img(), speech_like(1.0), 24000, backend="one-shot")
    assert e.value.kind == "admission"
    jobs = list(client.jobs_root(rt).glob("job-*"))
    assert jobs and not any((j / "events.jsonl").exists() for j in jobs)  # the runtime never started
    assert ":\\" not in json.dumps(e.value.report) and "/home/" not in json.dumps(e.value.report)


def _loaded_seen(client, rt):
    return any(b'"loaded"' in p.read_bytes() for p in client.jobs_root(rt).glob("job-*/events.jsonl"))


def test_one_shot_is_checked_again_after_loading(tmp_path, fake_runner, monkeypatch):
    client = fake_runner
    from leaptalk_comfy import worker

    monkeypatch.setattr(worker.MANAGER, "estimates", memory.PeakEstimates())
    rt = make_runtime(tmp_path, "hang", memory_guard={})
    # before loading: 60 GiB (load allowed); once the runner reports "loaded": 88 GiB, too much for the job
    monkeypatch.setattr(memory, "snapshot", lambda: snap(88.0) if _loaded_seen(client, rt) else snap(60.0))
    with pytest.raises(memory.MemoryGuardRefused, match="loaded the model but did not start the one-shot job") as e:
        client.generate(rt, _img(), speech_like(1.0), 24000, backend="one-shot")
    assert e.value.kind == "admission" and e.value.report["kind"] == "warm"
    assert e.value.report["stop"]["active_after"] == 0  # the runtime tree (incl. a grandchild) was ended


def test_one_shot_is_stopped_by_sustained_pressure(tmp_path, fake_runner, monkeypatch):
    client = fake_runner
    from leaptalk_comfy import worker

    monkeypatch.setattr(worker.MANAGER, "estimates", memory.PeakEstimates())
    rt = make_runtime(tmp_path, "hang", memory_guard={"stop_sustain_seconds": 0.5})
    calls = {"after_load": 0}

    def fake_snapshot():
        if not _loaded_seen(client, rt):
            return snap(60.0)
        calls["after_load"] += 1
        return snap(60.0) if calls["after_load"] == 1 else snap(96.0)  # the job is admitted, then commit stays at 96 %

    monkeypatch.setattr(memory, "snapshot", fake_snapshot)
    t = time.monotonic()
    with pytest.raises(memory.MemoryGuardRefused, match="stopped the one-shot job while generating") as e:
        client.generate(rt, _img(), speech_like(1.0), 24000, backend="one-shot")
    assert e.value.kind == "stopped" and time.monotonic() - t < 30
    assert e.value.report["stop"]["active_after"] == 0


def test_worker_start_up_is_watched(tmp_path, worker_env, monkeypatch):
    """Commit stays high while the worker imports/loads (it does not answer then): ComfyUI ends the tree."""
    client, worker = worker_env
    rt = make_runtime(tmp_path, "slow_init", memory_guard={"stop_sustain_seconds": 0.5})
    calls = {"n": 0}

    def fake_snapshot():
        calls["n"] += 1
        return snap(60.0) if calls["n"] <= 2 else snap(96.0)  # admission and the tracker start, then pressure

    monkeypatch.setattr(memory, "snapshot", fake_snapshot)
    with pytest.raises(memory.MemoryGuardRefused, match="stopped the worker while it was loading") as e:
        client.generate(rt, _img(), speech_like(1.0), 24000, backend="persistent")
    assert e.value.kind == "stopped" and worker.MANAGER._worker is None
    stopped = [h for h in worker.MANAGER.history if h["event"] == "stopped"]
    assert stopped and stopped[-1]["active_after"] == 0 and "memory guard" in stopped[-1]["reason"]


def test_persistent_refusal_never_falls_back(tmp_path, worker_env, monkeypatch):
    client, worker = worker_env
    rt = make_runtime(tmp_path, "ok", memory_guard={})
    monkeypatch.setattr(memory, "snapshot", lambda: snap(91.0))
    with pytest.raises(memory.MemoryGuardRefused, match="did not start a worker"):
        client.generate(rt, _img(), speech_like(1.0), 24000, backend="persistent", decoder="lite_tae")
    jobs = list(client.jobs_root(rt).glob("job-*"))
    assert jobs and not any((j / "events.jsonl").exists() for j in jobs)  # no one-shot run, no other decoder
    assert [h for h in worker.MANAGER.history if h["event"] == "refused"][-1]["backend"] == "persistent"
    assert not any(h["event"] == "starting" for h in worker.MANAGER.history)


def test_refusal_hint_depends_on_the_decoder():
    from leaptalk_comfy.worker import _refusal

    lite = memory.admit(snap(91.0), memory.settings({}), {"bytes": GIB, "profile": "lite_tae, audio guidance 1.0"}, "cold")
    wan = memory.admit(snap(91.0), memory.settings({}), {"bytes": GIB, "profile": "wan_vae, audio guidance 1.0"}, "cold")
    assert "lower-memory workflow" in _refusal("did not start", lite)
    assert "lower-memory workflow" not in _refusal("did not start", wan) and "close other applications" in _refusal("did not start", wan)


def test_messages_and_reports_have_no_private_paths():
    from leaptalk_comfy.worker import redact

    win = "C:" + "\\Users\\someone\\models\\x.safetensors"  # built at run time: no private-path pattern in the source
    unix = "/home" + "/someone/leaptalk/x.pth"
    assert "<path>" in redact(f"failed in {win}") and "someone" not in redact(win)
    assert "someone" not in redact(unix)
    rep = memory.admit(snap(91.0), memory.settings({}), {"bytes": int(16.6 * GIB), "source": "built-in"}, "cold")
    text = json.dumps(rep) + rep["reason"]
    assert ":\\" not in text and "/home/" not in text and "Users" not in text
