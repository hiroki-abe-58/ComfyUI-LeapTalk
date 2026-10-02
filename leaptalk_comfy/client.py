"""Create, run, cancel and collect LeapTalk runtime jobs from the ComfyUI process.

Process-control design adapted from ComfyUI-MonarchRT / ComfyUI-CausalForcing (same author, Apache-2.0).

Design rules (see docs/SECURITY.md):

- The command line is an argv list built only from the administrator's runtime config and this
  package's own runtime scripts; never ``shell=True``.
- The workflow contributes data only (an image, an audio clip and a few bounded options). They are
  written into a fresh job directory (``input.png``, ``input.wav``, a schema-checked ``job.json``).
- The runtime process gets an allowlisted environment: no tokens, nothing else from ComfyUI's
  environment; its temp directory is inside the job directory.
- stdout/stderr go to files in the job directory; progress is read from ``events.jsonl``.
- Cancel/timeout first ask the runner to stop between chunks (``CANCEL`` file), then end the whole
  process tree (``process.ProcessTree``). Only paths inside the job directory are read back.

This isolates *configuration and arguments*; it is not a sandbox.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Runtime
from .process import IS_WINDOWS, ProcessTree

PACKAGE_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PACKAGE_DIR / "runtime"
JOB_SCRIPT = "leaptalk_job.py"
DOCTOR_SCRIPT = "leaptalk_doctor.py"
RUNTIME_SCRIPTS = {name: RUNTIME_DIR / name for name in (JOB_SCRIPT, DOCTOR_SCRIPT)}
DECODERS = ("lite_tae", "wan_vae")
FPS = 25
IMAGE_NAME = "input.png"
AUDIO_NAME = "input.wav"
OUTPUT_NAME = "output.mp4"
PREVIEW_NAME = "preview.jpg"
MIN_AUDIO_SECONDS = 0.2
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_PREVIEW_BYTES = 2 * 1024 * 1024
_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,80}")


class RuntimeJobError(RuntimeError):
    """The runtime reported an error (invalid input, runtime failure, missing files...)."""


class JobCancelled(RuntimeError):
    pass


class JobTimeout(RuntimeError):
    pass


@dataclass
class JobOutcome:
    job_dir: Path
    result: dict
    video: Path


# --------------------------------------------------------------------------- inputs


def save_image(image, path: Path) -> dict:
    """Write one HxWx3 (or 4) uint8 image as an 8-bit RGB PNG (the runtime applies the official 512x512 resize/crop)."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4) or arr.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 array")
    if arr.shape[0] < 64 or arr.shape[1] < 64:
        raise ValueError("image must be at least 64x64 pixels")
    if arr.shape[0] * arr.shape[1] > 4096 * 4096:
        raise ValueError("image is larger than 16 megapixels")
    Image.fromarray(np.ascontiguousarray(arr[:, :, :3])).save(path, format="PNG")
    return {"width": int(arr.shape[1]), "height": int(arr.shape[0]), "alpha_dropped": bool(arr.shape[2] == 4)}


def save_wav_float32(waveform, sample_rate: int, path: Path, max_seconds: int) -> dict:
    """Write a channels x samples float array as an IEEE-float WAV, unchanged (no resampling, no mixdown).

    The runtime resamples to 16 kHz mono for the audio encoder exactly like the official script and
    muxes this original file into the output.
    """
    import numpy as np

    a = np.asarray(waveform, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("audio must be channels x samples")
    channels, samples = a.shape
    if not 1 <= channels <= 8:
        raise ValueError(f"audio has {channels} channels; 1 to 8 are supported")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
        raise ValueError("audio sample rate must be an integer between 8000 and 192000 Hz")
    if samples == 0:
        raise ValueError("the audio is empty")
    seconds = samples / sample_rate
    if seconds < MIN_AUDIO_SECONDS:
        raise ValueError(f"the audio is {seconds:.2f} s long; at least {MIN_AUDIO_SECONDS} s is needed")
    if seconds > max_seconds:
        raise ValueError(f"the audio is {seconds:.1f} s long; this runtime accepts up to {max_seconds} s (it is not cut silently)")
    if not np.isfinite(a).all():
        raise ValueError("the audio contains NaN or Inf samples")
    data = np.ascontiguousarray(a.T).astype("<f4").tobytes()
    block = channels * 4
    fmt = struct.pack("<HHIIHHH", 3, channels, sample_rate, sample_rate * block, block, 32, 0)  # WAVE_FORMAT_IEEE_FLOAT
    fact = struct.pack("<I", samples)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"fact" + struct.pack("<I", 4) + fact + b"data" + struct.pack("<I", len(data)) + data
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    peak = float(np.abs(a).max())
    return {"sample_rate": sample_rate, "channels": channels, "samples": samples, "seconds": round(seconds, 4), "peak": round(peak, 5), "silent": peak < 1e-4}


# --------------------------------------------------------------------------- jobs


def jobs_root(rt: Runtime) -> Path:
    if rt.jobs_dir:
        root = Path(rt.jobs_dir)
    else:
        import folder_paths

        root = Path(folder_paths.get_temp_directory()) / "leaptalk"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def new_job_dir(rt: Runtime, kind: str = "job") -> tuple[str, Path]:
    job_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:12]}"
    assert _JOB_ID_RE.fullmatch(job_id)
    root = jobs_root(rt)
    job_dir = (root / job_id).resolve()
    if job_dir.parent != root:
        raise RuntimeJobError("job directory escaped the jobs root")
    job_dir.mkdir(parents=False, exist_ok=False)
    return job_id, job_dir


def make_job(rt: Runtime, job_id: str, *, decoder: str, audio_guidance: float, preview: bool, video_crf: int = 18) -> dict:
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}")
    g = float(audio_guidance)
    if not math.isfinite(g) or not 1.0 <= g <= 4.0:
        raise ValueError("audio_guidance must be between 1.0 and 4.0")
    return {
        "schema_version": 1,
        "job_id": job_id,
        "upstream_dir": rt.upstream_dir,
        "models": dict(rt.models),
        "ffmpeg": rt.ffmpeg,
        "decoder": decoder,
        "audio_guidance": round(g, 3),
        "max_audio_seconds": rt.max_audio_seconds,
        "video_crf": int(video_crf),
        "preview": bool(preview),
        "env": dict(rt.env),
    }


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def child_env(job_dir: Path, python: str) -> dict[str, str]:
    """Allowlisted environment for the runtime process: nothing secret, temp files inside the job."""
    tmp = job_dir / "tmp"
    tmp.mkdir(exist_ok=True)
    if IS_WINDOWS:
        src = {k.upper(): v for k, v in os.environ.items()}
        env = {
            k: src[k] for k in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS") if k in src
        }
        sysroot = env.get("SYSTEMROOT", r"C:\Windows")
        env["PATH"] = os.pathsep.join([str(Path(python).parent), sysroot + r"\System32", sysroot])
    else:
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "USER", "LOGNAME") if k in os.environ}
        env.setdefault("PATH", "/usr/bin:/bin")
    env.update(
        TEMP=str(tmp),
        TMP=str(tmp),
        TMPDIR=str(tmp),
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    return env


def read_events(path: Path, offset: int) -> tuple[list[dict], int]:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    events = []
    for line in data[: end + 1].splitlines():
        try:
            ev = json.loads(line.decode("utf-8"))
            if isinstance(ev, dict):
                events.append(ev)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return events, offset + end + 1


def error_from_events(job_dir: Path) -> str:
    events, _ = read_events(job_dir / "events.jsonl", 0)
    for ev in reversed(events):
        if ev.get("event") in ("failed", "invalid_job"):
            return f"{ev.get('error_type', ev['event'])}: {ev.get('error', '')}"[:2000]
    try:
        tail = (job_dir / "stderr.log").read_bytes()[-1500:].decode("utf-8", "replace")
    except OSError:
        tail = ""
    return f"runtime exited without a result; stderr tail:\n{tail}"


def stop_tree(tree: ProcessTree, job_dir: Path, grace_s: float = 15.0) -> dict:
    """Ask the runner to stop between chunks, then end the whole tree."""
    t = time.monotonic()
    try:
        (job_dir / "CANCEL").write_text("cancel\n", encoding="utf-8")
    except OSError:
        pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and tree.poll() is None:
        time.sleep(0.1)
    graceful = tree.poll() is not None
    report = tree.terminate()
    report.update(graceful=graceful, seconds=round(time.monotonic() - t, 2))
    try:
        _write_json(job_dir / "stop_report.json", report)
    except OSError:
        pass
    return report


def run_process(
    rt: Runtime,
    script: str,
    job_dir: Path,
    request_name: str,
    *,
    timeout_s: float,
    interrupted: Callable[[], bool] = lambda: False,
    on_event: Callable[[dict], None] | None = None,
    poll_s: float = 0.2,
) -> tuple[int, dict]:
    """Run one runtime script on ``job_dir/request_name``. Returns (exit code, process info)."""
    argv = [rt.python, str(RUNTIME_SCRIPTS[script])]
    if not IS_WINDOWS:
        argv.append("--watch-stdin")  # Windows: the job object ends the tree when ComfyUI goes away
    argv.append(str(job_dir / request_name))
    t0 = time.monotonic()
    tree = ProcessTree(argv, cwd=job_dir, env=child_env(job_dir, rt.python), stdout=job_dir / "stdout.log", stderr=job_dir / "stderr.log")
    deadline = time.monotonic() + timeout_s
    offset = 0
    info: dict = {"pid": tree.pid}
    try:
        while True:
            events, offset = read_events(job_dir / "events.jsonl", offset)
            if on_event:
                for ev in events:
                    on_event(ev)
            code = tree.poll()
            if code is not None:
                events, offset = read_events(job_dir / "events.jsonl", offset)
                if on_event:
                    for ev in events:
                        on_event(ev)
                left = tree.active_processes()
                if left > 0:  # nothing of a finished job may linger (e.g. a stuck encoder)
                    info["leftover_terminated"] = tree.terminate()
                info["wall_seconds"] = round(time.monotonic() - t0, 3)
                return code, info
            if interrupted():
                info["stop"] = stop_tree(tree, job_dir)
                raise JobCancelled("cancelled by the user")
            if time.monotonic() > deadline:
                info["stop"] = stop_tree(tree, job_dir, grace_s=5.0)
                raise JobTimeout(f"runtime job exceeded {timeout_s:.0f} s")
            time.sleep(poll_s)
    except (JobCancelled, JobTimeout):
        raise
    except BaseException:
        stop_tree(tree, job_dir, grace_s=5.0)
        raise
    finally:
        tree.close()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_result(job_dir: Path, job: dict, audio_info: dict) -> JobOutcome:
    path = job_dir / "result.json"
    if not path.is_file() or path.stat().st_size > MAX_RESULT_BYTES:
        raise RuntimeJobError("runtime finished without a valid result.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("job_id") != job["job_id"] or result.get("status") != "ok":
        raise RuntimeJobError("result.json does not belong to this job")
    if result.get("output") != OUTPUT_NAME:
        raise RuntimeJobError(f"unexpected output name in result.json: {result.get('output')!r}")
    video = (job_dir / OUTPUT_NAME).resolve()
    if video.parent != job_dir.resolve() or not video.is_file():
        raise RuntimeJobError("output video missing")
    if _sha256(video) != result.get("sha256"):
        raise RuntimeJobError("output video does not match its recorded sha256")
    expected_frames = math.ceil(audio_info["seconds"] * FPS - 1e-6)
    v = result.get("video") or {}
    if abs(int(v.get("frames", -1)) - expected_frames) > 1 or (v.get("width"), v.get("height")) != (512, 512):
        raise RuntimeJobError(f"unexpected output video in result.json: {v}")
    if not (result.get("audio") or {}).get("muxed"):
        raise RuntimeJobError("result.json reports no muxed audio")
    return JobOutcome(job_dir=job_dir, result=result, video=video)


def generate(
    rt: Runtime,
    image,
    waveform,
    sample_rate: int,
    *,
    decoder: str = "lite_tae",
    audio_guidance: float = 1.0,
    preview: bool = True,
    interrupted: Callable[[], bool] = lambda: False,
    on_progress: Callable[[int, int, Path | None], None] | None = None,
    timeout_s: float | None = None,
) -> JobOutcome:
    """image: HxWx3 uint8; waveform: channels x samples float32 at ``sample_rate``."""
    job_id, job_dir = new_job_dir(rt, "job")
    image_info = save_image(image, job_dir / IMAGE_NAME)
    audio_info = save_wav_float32(waveform, sample_rate, job_dir / AUDIO_NAME, rt.max_audio_seconds)
    job = make_job(rt, job_id, decoder=decoder, audio_guidance=audio_guidance, preview=preview)
    _write_json(job_dir / "job.json", job)

    def on_event(ev: dict) -> None:
        if on_progress and ev.get("event") == "progress":
            prev = job_dir / PREVIEW_NAME if ev.get("preview") else None
            on_progress(int(ev.get("chunk", 0)), int(ev.get("chunks", 1)), prev)

    code, info = run_process(rt, JOB_SCRIPT, job_dir, "job.json", timeout_s=timeout_s or rt.timeout_minutes * 60, interrupted=interrupted, on_event=on_event)
    if code == 4:
        raise JobCancelled("the runtime job was cancelled")
    if code != 0:
        raise RuntimeJobError(f"runtime job failed (exit {code}): {error_from_events(job_dir)}")
    outcome = load_result(job_dir, job, audio_info)
    outcome.result["host"] = {"image": image_info, "audio": audio_info, "process": info, "python": sys.version.split()[0], "platform": sys.platform}
    return outcome


def read_preview(path: Path | None):
    """Load the runner's latest preview frame (fixed name inside the job directory) or None."""
    if path is None or path.name != PREVIEW_NAME:
        return None
    try:
        if not path.is_file() or path.stat().st_size > MAX_PREVIEW_BYTES:
            return None
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")
    except Exception:  # noqa: BLE001 - the runner may be replacing the file right now
        return None


def doctor(rt: Runtime, *, verify_sha256: bool, interrupted: Callable[[], bool] = lambda: False) -> dict:
    job_id, job_dir = new_job_dir(rt, "doctor")
    req = {
        "schema_version": 1,
        "job_id": job_id,
        "upstream_dir": rt.upstream_dir,
        "models": dict(rt.models),
        "ffmpeg": rt.ffmpeg,
        "env": dict(rt.env),
        "verify_sha256": bool(verify_sha256),
    }
    _write_json(job_dir / "request.json", req)
    code, _info = run_process(rt, DOCTOR_SCRIPT, job_dir, "request.json", timeout_s=3600, interrupted=interrupted)
    path = job_dir / "doctor.json"
    if path.is_file() and path.stat().st_size < MAX_RESULT_BYTES:
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        report = {"overall": "fail", "checks": [{"name": "launch", "status": "fail", "detail": error_from_events(job_dir)}]}
    report["runtime_id"] = rt.id
    report["exit_code"] = code
    report["host"] = {"python": sys.version.split()[0], "platform": sys.platform}
    return report
