"""Real subprocesses with the fake runner: success, progress/preview, failure, cancel, timeout, parent death."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import FAKE_JOB, make_runtime, speech_like

from leaptalk_comfy import client
from leaptalk_comfy.process import IS_WINDOWS, ProcessTree


def _alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def _wait_gone(pids, seconds=15):
    t = time.time()
    while time.time() - t < seconds:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


def _image():
    return np.full((96, 128, 3), 120, np.uint8)


def test_generate_success_with_progress_and_preview(tmp_path, fake_runner):
    rt = make_runtime(tmp_path, "ok")
    seen = []
    previews = []

    def on_progress(done, total, prev):
        seen.append((done, total))
        if prev is not None:
            previews.append(client.read_preview(prev) is not None)

    out = client.generate(rt, _image(), speech_like(2.3, 24000, 2), 24000, on_progress=on_progress)
    assert out.video.is_file() and out.video.parent == out.job_dir
    assert out.result["video"]["frames"] == 58 and out.result["chunks"] == 3  # ceil(2.3*25), padded to 3 slices
    assert seen and seen[-1] == (3, 3) and any(previews)
    host = out.result["host"]
    assert host["audio"]["channels"] == 2 and host["audio"]["sample_rate"] == 24000 and host["image"]["width"] == 128
    import av

    with av.open(str(out.video)) as c:
        assert len(c.streams.audio) == 1 and len(c.streams.video) == 1
        assert sum(1 for _ in c.decode(video=0)) == 58


def test_runtime_failure_is_reported_and_nothing_lingers(tmp_path, fake_runner):
    rt = make_runtime(tmp_path, "fail")
    with pytest.raises(client.RuntimeJobError, match="simulated runtime failure"):
        client.generate(rt, _image(), speech_like(1.0), 24000)


def test_audio_limit_is_enforced_before_launch(tmp_path, fake_runner):
    rt = make_runtime(tmp_path, "ok", max_audio_seconds=2)
    with pytest.raises(ValueError, match="up to 2 s"):
        client.generate(rt, _image(), speech_like(2.5), 24000)


def test_cancel_stops_the_whole_tree(tmp_path, fake_runner):
    rt = make_runtime(tmp_path, "hang")
    t0 = time.time()
    calls = {"n": 0}

    def interrupted():
        calls["n"] += 1
        return calls["n"] > 15

    with pytest.raises(client.JobCancelled):
        client.generate(rt, _image(), speech_like(1.0), 24000, interrupted=interrupted)
    job_dir = max(client.jobs_root(rt).iterdir(), key=lambda p: p.stat().st_mtime)
    stop = json.loads((job_dir / "stop_report.json").read_text(encoding="utf-8"))
    assert stop["active_after"] == 0 and stop["graceful"] is True
    gc = int((job_dir / "grandchild.pid").read_text())
    if IS_WINDOWS:
        assert _wait_gone([gc]), "grandchild survived the job object"
    else:
        # graceful stop: the fake runner exits on CANCEL; its grandchild is in the same process group,
        # which terminate() signals afterwards
        assert _wait_gone([gc])
    assert time.time() - t0 < 60


def test_timeout_stops_the_tree(tmp_path, fake_runner):
    rt = make_runtime(tmp_path, "hang")
    with pytest.raises(client.JobTimeout):
        client.generate(rt, _image(), speech_like(1.0), 24000, timeout_s=3)
    job_dir = max(client.jobs_root(rt).iterdir(), key=lambda p: p.stat().st_mtime)
    assert json.loads((job_dir / "stop_report.json").read_text(encoding="utf-8"))["active_after"] == 0
    assert _wait_gone([int((job_dir / "grandchild.pid").read_text())])


def test_process_tree_terminate_kills_grandchildren(tmp_path):
    code = (
        "import subprocess,sys,time,pathlib;"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)']);"
        "pathlib.Path('gc.pid').write_text(str(p.pid));time.sleep(600)"
    )
    tree = ProcessTree([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ), stdout=tmp_path / "o.log", stderr=tmp_path / "e.log")
    try:
        for _ in range(100):
            if (tmp_path / "gc.pid").exists():
                break
            time.sleep(0.1)
        gc = int((tmp_path / "gc.pid").read_text())
        rep = tree.terminate()
        assert rep["active_after"] == 0
        assert _wait_gone([tree.pid, gc])
    finally:
        tree.close()


PARENT = r"""
import sys, time, pathlib
sys.path.insert(0, sys.argv[1])
from leaptalk_comfy.process import ProcessTree
work = pathlib.Path(sys.argv[2])
code = "import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)']);pathlib.Path('gc.pid').write_text(str(p.pid))\n" \
       "import threading\n" \
       "def watch():\n" \
       "    while sys.stdin.buffer.read(1): pass\n" \
       "    import os; os._exit(0)\n" \
       "threading.Thread(target=watch, daemon=True).start()\n" \
       "time.sleep(600)"
tree = ProcessTree([sys.executable, "-c", code], cwd=work, env=dict(__import__("os").environ), stdout=work / "o.log", stderr=work / "e.log")
(work / "child.pid").write_text(str(tree.pid))
time.sleep(600)
"""


def test_parent_death_ends_the_tree(tmp_path):
    """Kill the 'ComfyUI' process hard: Windows closes the job handle (KILL_ON_JOB_CLOSE); on POSIX the
    child sees stdin EOF (the real runner does this with --watch-stdin) and its group is gone."""
    repo = Path(__file__).resolve().parents[1]
    parent = subprocess.Popen([sys.executable, "-c", PARENT, str(repo), str(tmp_path)])  # noqa: S603
    try:
        for _ in range(150):
            if (tmp_path / "gc.pid").exists() and (tmp_path / "child.pid").exists():
                break
            time.sleep(0.1)
        child, gc = int((tmp_path / "child.pid").read_text()), int((tmp_path / "gc.pid").read_text())
        assert _alive(child)
        parent.kill()
        parent.wait(10)
        if IS_WINDOWS:
            assert _wait_gone([child, gc]), "job object did not end the tree"
        else:
            assert _wait_gone([child]), "child did not stop on stdin EOF"
            try:
                os.kill(gc, 9)  # this minimal child does not clean up its own group (the real runner's stop path does)
            except OSError:
                pass
    finally:
        if parent.poll() is None:
            parent.kill()


def test_fake_runner_path_is_the_one_used(fake_runner):
    assert client.RUNTIME_SCRIPTS[client.JOB_SCRIPT] == FAKE_JOB
