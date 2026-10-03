"""Test setup.

All tests here run on CPU without model weights. Tests marked ``comfy`` import a real ComfyUI checkout
given by ``COMFYUI_PATH`` and fail (not skip) when it is missing, so CI cannot pass by silently skipping
them; deselect them explicitly with ``-m "not comfy"``. Real-GPU checks live in ``scripts/gpu_e2e.py``
(see docs/TESTING.md).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):  # the package, and this folder (helpers via "from conftest import")
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
FAKE_RUNTIME = Path(__file__).resolve().parent / "fake_runtime"
FAKE_JOB = FAKE_RUNTIME / "fake_leaptalk_job.py"

_COMFY_STATE: dict = {}


def _boot_comfy(tmp_root: Path):
    if _COMFY_STATE:
        return _COMFY_STATE
    path = os.environ.get("COMFYUI_PATH")
    if not path or not (Path(path) / "comfy").is_dir():
        pytest.fail("COMFYUI_PATH must point to a ComfyUI checkout for tests marked 'comfy'", pytrace=False)
    if path not in sys.path:
        sys.path.insert(0, path)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    comfy.cli_args.args.disable_all_custom_nodes = False
    import folder_paths
    import nodes  # noqa: E402  (ComfyUI's nodes.py)

    for name in ("temp", "input", "output", "user"):
        (tmp_root / name).mkdir(parents=True, exist_ok=True)
    folder_paths.set_temp_directory(str(tmp_root / "temp"))
    folder_paths.set_input_directory(str(tmp_root / "input"))
    folder_paths.set_output_directory(str(tmp_root / "output"))
    folder_paths.set_user_directory(str(tmp_root / "user"))
    asyncio.run(nodes.init_builtin_extra_nodes())
    # install under a folder name that differs from the repository and the registry id
    alias = tmp_root / "custom_nodes" / "LeapTalk alias ü (test)"
    alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT, alias, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "_comfyui"))
    ok = asyncio.run(nodes.load_custom_node(str(alias)))
    if not ok:
        pytest.fail("ComfyUI failed to load the custom node package", pytrace=False)
    _COMFY_STATE.update(nodes=nodes, path=Path(path), alias=alias)
    return _COMFY_STATE


@pytest.fixture(scope="session")
def comfyui(tmp_path_factory):
    return _boot_comfy(tmp_path_factory.mktemp("comfy"))


def make_runtime(tmp_path: Path, mode: str = "ok", **over):
    """A 'native' runtime that runs the fake runner with this interpreter."""
    from leaptalk_comfy.config import parse_runtime

    root = tmp_path / "rt root ü"
    data = {
        "kind": "native",
        "python": sys.executable,
        "upstream_dir": str(root / "upstream" / mode),
        "models": {k: str(root / "models" / k) for k in ("soulx_dir", "wav2vec_dir", "leaptalk_dir")},
        "ffmpeg": str(root / "ffmpeg.exe"),
        "jobs_dir": str(tmp_path / "jobs dir 'quoted' ü"),
        "timeout_minutes": 5,
        "max_audio_seconds": 60,
        # CI machines are small: the guard logic is tested separately with fixed snapshots
        "memory_guard": {
            "expected_worker_gib": 0.1,
            "expected_job_gib": 0.1,
            "min_available_gib": 0.2,
            "start_max_commit_pct": 99,
            "max_projected_commit_pct": 100,
        },
    }
    data.update(over)
    return parse_runtime(f"fake-{mode}", data)


@pytest.fixture
def fake_runner(monkeypatch):
    from leaptalk_comfy import client

    monkeypatch.setitem(client.RUNTIME_SCRIPTS, client.JOB_SCRIPT, FAKE_JOB)
    return client


def fake_worker_runtime_dir(tmp_path: Path) -> Path:
    """A runtime code folder with the real leaptalk_worker.py and the fake-Engine leaptalk_job.py."""
    rt_dir = tmp_path / "runtime code"
    rt_dir.mkdir()
    shutil.copyfile(REPO_ROOT / "runtime" / "leaptalk_worker.py", rt_dir / "leaptalk_worker.py")
    fake = (FAKE_RUNTIME / "fake_engine_job.py").read_text(encoding="utf-8").replace("__REAL_JOB_MODULE__", str(REPO_ROOT / "runtime" / "leaptalk_job.py"))
    (rt_dir / "leaptalk_job.py").write_text(fake, encoding="utf-8")
    return rt_dir


@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    """Persistent-worker test setup: a runtime folder with the real leaptalk_worker.py and a fake Engine.

    Yields (client, worker module); the manager is emptied afterwards (no process left behind)."""
    from leaptalk_comfy import client, worker

    monkeypatch.setattr(client, "RUNTIME_DIR", fake_worker_runtime_dir(tmp_path))
    worker.MANAGER.unload("test setup", wait_s=30)
    worker.MANAGER.history.clear()
    yield client, worker
    worker.MANAGER.unload("test teardown", wait_s=30)


@pytest.fixture
def runtimes_file(tmp_path, monkeypatch):
    """Write a runtimes config and point COMFYUI_LEAPTALK_CONFIG at it."""

    def write(runtimes: dict) -> Path:
        p = tmp_path / "leaptalk.runtimes.json"
        p.write_text(json.dumps({"schema_version": 1, "runtimes": runtimes}), encoding="utf-8")
        monkeypatch.setenv("COMFYUI_LEAPTALK_CONFIG", str(p))
        return p

    return write


def speech_like(seconds: float, sr: int = 24000, channels: int = 1):
    """A deterministic speech-band test signal (not speech; for plumbing tests only)."""
    import numpy as np

    t = np.arange(int(seconds * sr)) / sr
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    sig = (0.3 * env * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.stack([sig * (1 - 0.2 * c) for c in range(channels)])


def pytest_configure(config):
    config.addinivalue_line("markers", "comfy: needs a ComfyUI checkout (COMFYUI_PATH)")
