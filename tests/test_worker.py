"""Persistent worker with the real leaptalk_worker.py and a fake Engine (CPU, no weights)."""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO_ROOT, make_runtime, speech_like


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def _wait_gone(pids, seconds=20.0) -> bool:
    t = time.time()
    while time.time() - t < seconds:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


def _img(v=120):
    return np.full((96, 128, 3), v, np.uint8)


def _gen(client, rt, v=120, seconds=1.0, **kw):
    return client.generate(rt, _img(v), speech_like(seconds), 24000, backend="persistent", **kw)


def _w(out):
    return out.result["host"]["process"]["worker"]


def test_reuse_across_jobs_and_portraits(tmp_path, worker_env):
    client, worker = worker_env
    rt = make_runtime(tmp_path, "ok")
    a1 = _gen(client, rt, 50)
    b = _gen(client, rt, 200)
    a2 = _gen(client, rt, 50)
    ws = [_w(o) for o in (a1, b, a2)]
    assert len({x["worker_instance_id"] for x in ws}) == 1 and len({x["engine_instance_id"] for x in ws}) == 1
    assert len({x["interpreter_pid"] for x in ws}) == 1
    assert a1.result["host"]["process"]["started_new_worker"] is True and b.result["host"]["process"]["started_new_worker"] is False
    assert ws[-1]["jobs_done"] == 3 and ws[-1]["engine_counters_at_start"]["init"] == 1 and ws[-1]["engine_counters_at_start"]["lora_merge"] == 1
    live = a2.result["engine"]["counters"]  # the Engine's own counters after the third job
    assert (live["init"], live["lora_merge"], live["audio_proj_load"], live["jobs_started"], live["jobs_ok"]) == (1, 1, 1, 3, 3)
    # the portrait of the previous job does not leak: A -> B -> A gives A's frames again
    assert a1.result["frames_sha256"] == a2.result["frames_sha256"] != b.result["frames_sha256"]
    assert a1.result["engine"]["engine_instance_id"] == a2.result["engine"]["engine_instance_id"]
    # every job has its own folder, events and result
    assert len({o.job_dir for o in (a1, b, a2)}) == 3
    for o in (a1, b, a2):
        evs = [json.loads(line) for line in (o.job_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert evs[0]["event"] == "started" and evs[0]["worker_instance_id"] == ws[0]["worker_instance_id"]
        assert evs[-1]["event"] == "done"


def test_identity_change_restarts_and_status_does_not_start(tmp_path, worker_env):
    client, worker = worker_env
    assert worker.MANAGER.status()["worker"] is None  # status never starts a worker
    assert worker.MANAGER.status()["worker"] is None
    rt = make_runtime(tmp_path, "ok")
    first = _gen(client, rt)
    st1 = worker.MANAGER.status()["worker"]
    time.sleep(1.2)
    st2 = worker.MANAGER.status()["worker"]
    assert st2["seconds_since_last_activity"] >= st1["seconds_since_last_activity"]  # status does not refresh the idle timer
    second = client.generate(rt, _img(), speech_like(1.0), 24000, backend="persistent", decoder="wan_vae")
    p = second.result["host"]["process"]
    assert p["started_new_worker"] is True and "decoder" in p["restart_reason"]
    assert _w(second)["worker_instance_id"] != _w(first)["worker_instance_id"]
    assert _wait_gone([_w(first)["interpreter_pid"], _w(first)["launcher_pid"]])


def test_unload_and_idle_timeout(tmp_path, worker_env):
    client, worker = worker_env
    rt = make_runtime(tmp_path, "ok")
    out = _gen(client, rt)
    pids = [_w(out)["interpreter_pid"], _w(out)["launcher_pid"]]
    rep = worker.MANAGER.unload("test")
    assert rep["status"] == "unloaded" and rep["active_after"] == 0
    assert _wait_gone(pids) and worker.MANAGER.status()["worker"] is None
    assert worker.MANAGER.unload("again")["status"] == "no_worker"
    out = _gen(client, rt)
    w = worker.MANAGER._worker
    w.last_activity -= 10_000  # pretend the idle time has passed
    t = time.time()
    while worker.MANAGER._worker is not None and time.time() - t < 10:
        time.sleep(0.1)
    assert worker.MANAGER._worker is None
    assert any(h["event"] == "stopped" and "idle timeout" in h["reason"] for h in worker.MANAGER.history)
    assert _wait_gone([_w(out)["interpreter_pid"]])


def test_idle_timer_and_unload_never_cut_a_running_job(tmp_path, worker_env):
    client, worker = worker_env
    rt = make_runtime(tmp_path, "slow")
    _gen(client, rt)  # start the worker
    res = {}

    def job():
        res["out"] = _gen(client, rt, seconds=8.0)  # 8 chunks x 0.6 s

    th = threading.Thread(target=job)
    th.start()
    t = time.time()
    while worker.MANAGER._worker.state != "busy" and time.time() - t < 30:
        time.sleep(0.05)
    worker.MANAGER._worker.last_activity -= 10_000  # idle deadline long gone, but the worker is busy
    time.sleep(1.5)  # the idle timer ticks while the job runs
    busy = worker.MANAGER.unload("test while busy", wait_s=0.3)
    assert busy["status"] == "busy"
    th.join(60)
    assert res["out"].result["status"] == "ok"
    assert all("idle timeout" not in h.get("reason", "") for h in worker.MANAGER.history if h["event"] == "stopped")


def test_concurrent_requests_start_one_worker(tmp_path, worker_env):
    client, worker = worker_env
    rt = make_runtime(tmp_path, "ok")
    outs, errs = [], []

    def job(v):
        try:
            outs.append(_gen(client, rt, v))
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    ths = [threading.Thread(target=job, args=(v,)) for v in (10, 90, 170)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(120)
    assert not errs and len(outs) == 3
    assert len({_w(o)["worker_instance_id"] for o in outs}) == 1
    assert sum(1 for h in worker.MANAGER.history if h["event"] == "starting") == 1


def test_cancel_and_timeout_then_next_job_succeeds(tmp_path, worker_env):
    client, worker = worker_env
    rt = make_runtime(tmp_path, "hang")
    calls = {"n": 0}

    def interrupted():  # press "cancel" a little after the job is running on the worker
        w = worker.MANAGER._worker
        if w is not None and w.state == "busy":
            calls["n"] += 1
        return calls["n"] > 5

    with pytest.raises(client.JobCancelled):
        _gen(client, rt, interrupted=interrupted)
    assert any(h["event"] == "job_cancelled" for h in worker.MANAGER.history)
    with pytest.raises(client.JobTimeout):
        _gen(client, rt, timeout_s=2)
    ok = _gen(client, make_runtime(tmp_path, "ok"))
    assert ok.result["status"] == "ok"


def test_worker_crash_during_job_and_recovery(tmp_path, worker_env):
    client, worker = worker_env
    rt = make_runtime(tmp_path, "ok")
    first = _gen(client, rt)
    up = Path(rt.upstream_dir)
    up.mkdir(parents=True, exist_ok=True)
    (up / "CRASH_NEXT").write_text("x", encoding="utf-8")
    with pytest.raises(worker.WorkerError, match="exited unexpectedly"):
        _gen(client, rt)
    assert worker.MANAGER._worker is None
    assert _wait_gone([_w(first)["interpreter_pid"], _w(first)["launcher_pid"]])
    again = _gen(client, rt)  # a fresh worker, not a retry of the failed job
    assert _w(again)["worker_instance_id"] != _w(first)["worker_instance_id"]


def test_unexpected_error_in_comfyui_during_a_job_ends_the_worker(tmp_path, worker_env):
    """A failure on the ComfyUI side while the worker generates (here: the progress callback) leaves the
    job's state unknown; the worker is not reused and its late result cannot reach the next job."""
    client, worker = worker_env
    rt = make_runtime(tmp_path, "slow")
    first = _gen(client, rt)

    def boom(done, total, preview):
        raise ValueError("progress display failed")

    with pytest.raises(ValueError, match="progress display failed"):
        client.generate(rt, _img(), speech_like(3.0), 24000, backend="persistent", on_progress=boom)
    assert worker.MANAGER._worker is None
    assert any(h["event"] == "stopped" and "not reused" in h["reason"] for h in worker.MANAGER.history)
    assert _wait_gone([_w(first)["interpreter_pid"], _w(first)["launcher_pid"]])
    again = _gen(client, rt)
    assert _w(again)["worker_instance_id"] != _w(first)["worker_instance_id"] and again.result["status"] == "ok"


def test_init_failure_leaves_nothing(tmp_path, worker_env):
    client, worker = worker_env
    with pytest.raises(worker.WorkerError, match="simulated load failure"):
        _gen(client, make_runtime(tmp_path, "init_fail"))
    assert worker.MANAGER._worker is None
    stopped = [h for h in worker.MANAGER.history if h["event"] == "stopped"]
    assert stopped and stopped[-1]["active_after"] == 0
    assert _gen(client, make_runtime(tmp_path, "ok")).result["status"] == "ok"


def test_protocol_garbage_is_rejected(tmp_path, worker_env, monkeypatch):
    client, worker = worker_env
    bad = client.RUNTIME_DIR / "leaptalk_worker.py"
    bad.write_text("import sys, os\nsys.stdout.write('this is not json\\n'); sys.stdout.flush()\nimport time; time.sleep(60)\n", encoding="utf-8")
    with pytest.raises(worker.WorkerError, match="protocol error"):
        _gen(client, make_runtime(tmp_path, "ok"))
    stopped = [h for h in worker.MANAGER.history if h["event"] == "stopped"]
    assert stopped and stopped[-1]["active_after"] == 0


def test_worker_env_has_no_secrets(tmp_path, worker_env, monkeypatch):
    client, worker = worker_env
    for k in ("GITHUB_TOKEN", "HF_TOKEN", "REGISTRY_ACCESS_TOKEN", "MY_API_KEY"):
        monkeypatch.setenv(k, "secret-value")
    out = _gen(client, make_runtime(tmp_path, "ok"))
    keys = out.result["env_keys"]
    assert not [k for k in keys if any(s in k.upper() for s in ("TOKEN", "SECRET", "API_KEY", "PASSWORD"))]


def test_memory_guard_blocks_start_and_warm_job(tmp_path, worker_env, monkeypatch):
    client, worker = worker_env
    from leaptalk_comfy import memory

    rt = make_runtime(tmp_path, "ok", memory_guard={"expected_worker_gib": 16, "expected_job_gib": 8})
    high = {
        "measurable": True,
        "commit_enforced": True,
        "total_gib": 64,
        "available_gib": 20,
        "commit_gib": 86,
        "commit_limit_gib": 93.3,
        "commit_pct": 92.2,
        "t": time.monotonic(),
    }
    monkeypatch.setattr(memory, "snapshot", lambda: dict(high, t=time.monotonic()))
    with pytest.raises(worker.MemoryGuardRefused, match="commit charge is 92.2"):
        _gen(client, rt)
    assert worker.MANAGER._worker is None  # nothing was started
    unknown = {"measurable": False, "t": time.monotonic()}
    monkeypatch.setattr(memory, "snapshot", lambda: dict(unknown, t=time.monotonic()))
    with pytest.raises(worker.MemoryGuardRefused, match="cannot measure"):
        _gen(client, rt)
    ok_snap = dict(high, commit_gib=60, commit_pct=64.3)
    monkeypatch.setattr(memory, "snapshot", lambda: dict(ok_snap, t=time.monotonic()))
    first = _gen(client, rt)  # 60 + 16 GiB -> 81 %: allowed
    assert first.result["host"]["process"]["memory_check"]["expected_add_gib"] == 16
    # the warm job only adds its measured extra peak over the resident model (fake: 2 GiB x 1.25),
    # not the whole worker again and not the 8 GiB default
    warm = _gen(client, rt)
    assert warm.result["host"]["process"]["memory_check"]["expected_add_gib"] == 2.5
    # 92 % with the model resident: too high for a new load (90 %), fine for a warm job (92 % + 2.5 GiB < 97 %)
    resident = dict(high, commit_gib=85.8, commit_pct=92.0)
    monkeypatch.setattr(memory, "snapshot", lambda: dict(resident, t=time.monotonic()))
    assert _gen(client, rt).result["host"]["process"]["started_new_worker"] is False
    for snap, msg in (
        (dict(high, commit_gib=88.3, commit_pct=94.6), "commit would reach about 97.3"),  # 88.3 + 2.5 GiB of 93.3
        (dict(high, commit_gib=89.0, commit_pct=95.4), "at or above the stop level"),
        (dict(high, commit_gib=80.0, commit_pct=85.7, available_gib=1.5), "only 1.5 GiB of physical memory"),
    ):
        monkeypatch.setattr(memory, "snapshot", lambda snap=snap: dict(snap, t=time.monotonic()))
        with pytest.raises(worker.MemoryGuardRefused, match=f"on the loaded worker: .*{msg}"):
            _gen(client, rt)
        assert worker.MANAGER._worker is not None and worker.MANAGER._worker.state == "idle"  # refused, not killed


def test_memory_pressure_hysteresis():
    from leaptalk_comfy import memory

    cfg = memory.settings({"stop_commit_pct": 95, "stop_sustain_seconds": 5})
    p = memory.SustainedPressure(cfg)

    def snap(pct, t):
        return {"measurable": True, "commit_enforced": True, "commit_pct": pct, "available_gib": 10, "t": t}

    assert not p.update(snap(96, 0)) and not p.update(snap(96, 4.9))
    assert p.update(snap(95.0, 5.0))  # 5 s without a sample below the stop level
    p2 = memory.SustainedPressure(cfg)
    assert not p2.update(snap(96, 0)) and not p2.update(snap(94.9, 4)) and not p2.update(snap(96, 4.5)) and not p2.update(snap(96, 9.4))
    assert p2.update(snap(96, 9.5))  # the sample below 95 % at t=4 restarted the timer at t=4.5
    # the pattern of a normal job on the measured machine: 93-95 % with momentary samples above 95 %
    p3 = memory.SustainedPressure(cfg)
    pattern = [93.5, 95.2, 94.1, 93.8, 95.1, 94.3, 95.4, 94.0, 93.6, 95.0, 94.2, 93.9, 95.3, 94.4, 93.7, 94.8]
    assert not any(p3.update(snap(pct, i)) for i, pct in enumerate(pattern))
    unknown = memory.SustainedPressure(cfg)  # unmeasurable counts as pressure, never as "unlimited"
    assert not unknown.update({"measurable": False, "t": 0}) and unknown.update({"measurable": False, "t": 5})
    off = memory.SustainedPressure(memory.settings({"enabled": False}))
    assert not off.update(snap(99, 0)) and not off.update(snap(99, 100))
    ok, why = memory.check_start({"measurable": True, "commit_enforced": False, "available_gib": 3.0}, memory.settings({}), 16)
    assert not ok and "available" in why


def test_one_shot_job_unloads_the_idle_worker(tmp_path, worker_env, fake_runner):
    client, worker = worker_env
    rt = make_runtime(tmp_path, "ok")
    first = _gen(client, rt)
    out = client.generate(rt, _img(), speech_like(1.0), 24000, backend="one-shot")
    assert out.result["host"]["process"]["backend"] == "one-shot"
    assert out.result["host"]["process"]["unloaded_persistent_worker"]["worker_instance_id"] == _w(first)["worker_instance_id"]
    assert worker.MANAGER._worker is None


def test_worker_rejects_bad_requests_directly(tmp_path, worker_env):
    """Talk to the real worker script without the manager: duplicate request ids and foreign job ids."""
    client, worker = worker_env
    from leaptalk_comfy.process import ProcessTree

    rt = make_runtime(tmp_path, "ok")
    root = client.jobs_root(rt)
    session = root / "sessions" / "wtestdirect0001"
    session.mkdir(parents=True)
    cfg = {
        "schema_version": 1,
        "protocol": 1,
        "worker_instance_id": "wtestdirect0001",
        "jobs_root": str(root),
        "upstream_dir": rt.upstream_dir,
        "models": dict(rt.models),
        "ffmpeg": rt.ffmpeg,
        "decoder": "lite_tae",
        "env": {},
    }
    (session / "worker.json").write_text(json.dumps(cfg), encoding="utf-8")
    tree = ProcessTree(
        [sys.executable, str(client.RUNTIME_DIR / "leaptalk_worker.py"), str(session)],
        cwd=session,
        env=client.child_env(session, sys.executable),
        stdout=None,
        stderr=session / "w.log",
    )

    def send(o):
        tree.proc.stdin.write((json.dumps({"v": 1, **o}) + "\n").encode())
        tree.proc.stdin.flush()

    def recv(types):
        while True:
            m = json.loads(tree.proc.stdout.readline())
            if m["type"] in types:
                return m

    try:
        assert recv({"hello"})["pid"] > 0
        send({"type": "init", "worker_instance_id": "wtestdirect0001"})
        assert recv({"ready", "fatal"})["type"] == "ready"
        send({"type": "generate", "request_id": "rq-000000001", "job_id": "../outside"})
        assert recv({"result"})["status"] == "invalid"
        send({"type": "generate", "request_id": "rq-000000002", "job_id": "job-does-not-exist-123"})
        assert recv({"result"})["status"] == "invalid"
        send({"type": "generate", "request_id": "rq-000000002", "job_id": "job-does-not-exist-123"})
        r = recv({"result"})
        assert r["status"] == "invalid" and "duplicate" in r["error"]
        send({"type": "status", "request_id": "rq-000000003"})
        assert recv({"status"})["state"] == "idle"
        send({"type": "shutdown", "request_id": "rq-000000004"})
        assert recv({"bye"})["type"] == "bye"
        assert tree.proc.wait(20) == 0
    finally:
        tree.terminate()
        tree.close()


PARENT = r"""
import sys, time, json, pathlib, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tests")
from leaptalk_comfy import client, worker
from conftest import make_runtime, speech_like
client.RUNTIME_DIR = pathlib.Path(sys.argv[2])
rt = make_runtime(pathlib.Path(sys.argv[3]), "ok")
out = client.generate(rt, np.full((96, 128, 3), 100, np.uint8), speech_like(1.0), 24000, backend="persistent")
w = out.result["host"]["process"]["worker"]
pathlib.Path(sys.argv[3], "pids.json").write_text(json.dumps([w["launcher_pid"], w["interpreter_pid"]]))
time.sleep(600)
"""


def test_parent_death_ends_an_idle_worker(tmp_path, worker_env):
    """Kill the 'ComfyUI' process hard while its worker idles: the worker tree must go away
    (Windows: job object closed by the OS; POSIX: stdin EOF -> the worker exits)."""
    client, worker = worker_env
    parent = subprocess.Popen([sys.executable, "-c", PARENT, str(REPO_ROOT), str(client.RUNTIME_DIR), str(tmp_path)])  # noqa: S603
    try:
        t = time.time()
        while not (tmp_path / "pids.json").exists() and time.time() - t < 120:
            time.sleep(0.2)
        pids = json.loads((tmp_path / "pids.json").read_text())
        assert all(_alive(p) for p in pids)
        parent.kill()
        parent.wait(10)
        assert _wait_gone(pids, 30), "worker survived its parent"
    finally:
        if parent.poll() is None:
            parent.kill()


def test_package_keeps_worker_files():
    patterns = [ln.strip() for ln in (REPO_ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    needed = [
        "runtime/leaptalk_worker.py",
        "runtime/leaptalk_job.py",
        "runtime/leaptalk_doctor.py",
        "leaptalk_comfy/worker.py",
        "leaptalk_comfy/memory.py",
        "leaptalk_comfy/process.py",
        "workflows/leaptalk_portrait_speech.json",
        "workflows/leaptalk_persistent.json",
    ]
    for path in needed:
        assert (REPO_ROOT / path).is_file(), path
        for pat in patterns:
            p = pat.rstrip("/")
            hit = fnmatch.fnmatch(path, p) or fnmatch.fnmatch(os.path.basename(path), p) or (pat.endswith("/") and path.startswith(p + "/"))
            assert not hit, f"{path} is excluded by .comfyignore pattern {pat!r}"
