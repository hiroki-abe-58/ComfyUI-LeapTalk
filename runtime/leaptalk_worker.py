"""LeapTalk persistent worker: one loaded Engine, many jobs, driven by ComfyUI over a local pipe.

Runs with the runtime's own Python, started (and owned) by ComfyUI-LeapTalk's WorkerManager:

    python leaptalk_worker.py <session_dir>

``session_dir/worker.json`` (written by the manager) fixes everything this worker may use: the pinned
upstream checkout, the model folders, ffmpeg, the decoder, the allowlisted environment and the jobs
folder. Workflows never reach this process directly; they only cause the manager to write a job folder
(``<jobs_root>/<job_id>/{job.json,input.png,input.wav}``) and send its id.

Protocol (version 1): one JSON object per line, UTF-8, at most 64 KiB per line, every message carries
``"v": 1`` and ``"type"``. Worker -> manager on a duplicate of the original stdout; fd 1 itself is
redirected to the log, so prints from upstream code can never reach the channel. Manager -> worker on
a duplicate of the original stdin (fd 0 is redirected to NUL), read by a thread with ``os.read`` (no
Python buffer lock is held there; on Windows only after PeekNamedPipe reports data, see ``Channel``);
this process always ends with ``os._exit`` so that thread can never block interpreter shutdown.

    worker  -> hello      {pid, worker_instance_id}            before any heavy import
    manager -> init       {worker_instance_id}                 after checking the process is in its job object
    worker  -> loading    {step}                               while the Engine loads
    worker  -> ready      {engine}  |  fatal {error_type, error}
    manager -> generate   {request_id, job_id}                 the job path is derived from job_id, never sent
    worker  -> accepted   {request_id, job_id}
    worker  -> result     {request_id, job_id, status: ok|cancelled|invalid|error, error?, fatal}
    manager -> cancel     {job_id}                             (also: a CANCEL file in the job folder)
    manager -> status     {request_id}   ->   status {request_id, state, ...}
    manager -> shutdown   {request_id}   ->   bye
    stdin EOF: the manager is gone -> exit at once.

Unexpected errors in a job (CUDA errors, out of memory, anything not a validation error) are reported
with ``fatal: true`` and the worker exits; it is never reused after them.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leaptalk_job as lj  # noqa: E402

PROTOCOL_VERSION = 1
MAX_LINE = 64 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
EXIT_SHUTDOWN, EXIT_PARENT_GONE, EXIT_FATAL, EXIT_PROTOCOL, EXIT_CHANGED = 0, 7, 3, 5, 6
WORKER_KEYS = {"schema_version", "protocol", "worker_instance_id", "jobs_root", "upstream_dir", "models", "ffmpeg", "decoder", "env"}
_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,80}")


def load_worker_json(session: Path) -> dict:
    p = session / "worker.json"
    if p.stat().st_size > 64 * 1024:
        raise lj.JobError("worker.json is too large")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != WORKER_KEYS:
        raise lj.JobError(f"worker.json keys must be {sorted(WORKER_KEYS)}")
    if cfg["schema_version"] != 1 or cfg["protocol"] != PROTOCOL_VERSION:
        raise lj.JobError("unsupported worker.json / protocol version")
    if not isinstance(cfg["worker_instance_id"], str) or not _ID_RE.fullmatch(cfg["worker_instance_id"]):
        raise lj.JobError("invalid worker_instance_id")
    for k in ("jobs_root", "upstream_dir", "ffmpeg"):
        lj._abs_path(cfg[k], k)
    if not isinstance(cfg["models"], dict) or set(cfg["models"]) != set(lj.MODEL_FILES):
        raise lj.JobError("invalid models")
    for k, v in cfg["models"].items():
        lj._abs_path(v, f"models.{k}")
    if cfg["decoder"] not in lj.DECODERS:
        raise lj.JobError("invalid decoder")
    env = cfg["env"]
    if not isinstance(env, dict) or any(k not in lj.ENV_KEYS or not isinstance(v, str) for k, v in env.items()):
        raise lj.JobError("invalid env")
    return cfg


class Channel:
    """Line protocol: a reader thread on a private copy of stdin, writes on a private copy of stdout.

    fd 0 and fd 1 are pointed elsewhere (NUL, the log) so nothing else in this process - upstream
    prints, C runtimes of DLLs that initialise the standard handles, child processes - touches the
    pipes. On Windows the reader never leaves a synchronous ReadFile pending on the pipe: while one
    is pending, other threads' calls on that pipe (e.g. GetFileType while a DLL such as numpy's
    loads) block until data arrives, which hung imports in the worker. It polls with PeekNamedPipe
    and only reads bytes that are already there.
    """

    def __init__(self):
        self._out = os.fdopen(os.dup(1), "wb", buffering=0)
        os.dup2(2, 1)  # everything else printed to stdout goes to the log
        sys.stdout = sys.stderr
        self._in = os.dup(0)  # private, not inherited by child processes
        nul = os.open(os.devnull, os.O_RDONLY)
        os.dup2(nul, 0)
        os.close(nul)
        self.inbox: queue.Queue = queue.Queue()
        self.cancelled_jobs: set = set()
        self._lock = threading.Lock()
        threading.Thread(target=self._read, name="leaptalk-stdin", daemon=True).start()

    def send(self, obj: dict) -> None:
        data = (json.dumps({"v": PROTOCOL_VERSION, **obj}, ensure_ascii=False) + "\n").encode("utf-8")
        if len(data) > MAX_LINE:
            data = (json.dumps({"v": PROTOCOL_VERSION, "type": obj.get("type", "error"), "error": "message too large", "truncated": True}) + "\n").encode()
        with self._lock:
            self._out.write(data)

    def _chunk(self) -> bytes:
        """Next bytes from the manager; b"" when the pipe is closed (the manager is gone)."""
        if os.name != "nt":
            return os.read(self._in, 65536)  # POSIX: a blocking read on a raw fd holds no Python-level lock
        import _winapi
        import msvcrt

        handle = msvcrt.get_osfhandle(self._in)
        while True:
            try:
                avail, _left = _winapi.PeekNamedPipe(handle, 0)
            except OSError:  # ERROR_BROKEN_PIPE: the write end is closed
                return b""
            if avail:
                return os.read(self._in, min(avail, 65536))  # returns at once: the bytes are there
            time.sleep(0.02)

    def _read(self) -> None:
        buf = b""
        while True:
            try:
                chunk = self._chunk()
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._handle_line(line)
            if len(buf) > MAX_LINE:
                self._fatal_protocol("request line longer than 64 KiB")
        os._exit(EXIT_PARENT_GONE)  # the manager (ComfyUI) is gone: stop now, whatever we were doing

    def _handle_line(self, line: bytes) -> None:
        if len(line) > MAX_LINE:
            self._fatal_protocol("request line longer than 64 KiB")
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._fatal_protocol("request is not JSON")
            return
        if not isinstance(msg, dict) or msg.get("v") != PROTOCOL_VERSION or not isinstance(msg.get("type"), str):
            self._fatal_protocol("request without protocol version or type")
            return
        if msg["type"] == "cancel" and isinstance(msg.get("job_id"), str):
            with self._lock:
                self.cancelled_jobs.add(msg["job_id"])
        self.inbox.put(msg)

    def _fatal_protocol(self, why: str) -> None:
        try:
            self.send({"type": "fatal", "error_type": "ProtocolError", "error": why})
        finally:
            os._exit(EXIT_PROTOCOL)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self.cancelled_jobs


class JobControl:
    """Cooperative stop for one job: a cancel message for this job_id or a CANCEL file in its folder."""

    def __init__(self, channel: Channel, job_id: str, job_dir: Path):
        self.channel, self.job_id, self.cancel_file = channel, job_id, job_dir / "CANCEL"

    def check(self) -> None:
        if self.channel.is_cancelled(self.job_id) or self.cancel_file.exists():
            raise lj.Cancelled("cancel requested")


def _rotate_log(session: Path, n: list) -> None:
    try:
        if os.fstat(2).st_size <= MAX_LOG_BYTES:
            return
    except OSError:
        return
    n[0] += 1
    new = session / f"worker.{n[0]}.log"
    fd = os.open(new, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(fd, 2)
    os.dup2(fd, 1)
    os.close(fd)
    old = session / (f"worker.{n[0] - 2}.log" if n[0] > 2 else "worker.log")
    if n[0] >= 2:
        try:
            old.unlink()
        except OSError:
            pass


def main() -> int:
    session = Path(sys.argv[1]).resolve()
    ch = Channel()
    try:
        cfg = load_worker_json(session)
    except Exception as exc:  # noqa: BLE001
        ch.send({"type": "fatal", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
        os._exit(EXIT_FATAL)
    wid = cfg["worker_instance_id"]
    ch.send({"type": "hello", "pid": os.getpid(), "worker_instance_id": wid})
    try:
        first = ch.inbox.get(timeout=120)
    except queue.Empty:
        os._exit(EXIT_PROTOCOL)
    if first.get("type") != "init" or first.get("worker_instance_id") != wid:
        ch.send({"type": "fatal", "error_type": "ProtocolError", "error": "expected init"})
        os._exit(EXIT_PROTOCOL)
    jobs_root = Path(cfg["jobs_root"]).resolve()
    lj.apply_env(cfg["env"], session)  # worker-level caches/temp live in the session folder, not in a job
    t0 = time.perf_counter()
    try:
        engine = lj.Engine(cfg, on_step=lambda step: ch.send({"type": "loading", "step": step}))
    except Exception as exc:  # noqa: BLE001
        ch.send({"type": "fatal", "error_type": type(exc).__name__, "error": str(exc)[:2000]})
        os._exit(EXIT_FATAL)
    ready_s = round(time.perf_counter() - t0, 3)
    ch.send({"type": "ready", "worker_instance_id": wid, "engine": engine.describe(), "engine_load_seconds": ready_s})
    seen: collections.deque = collections.deque(maxlen=1024)
    jobs_done = 0
    log_n = [0]
    while True:
        msg = ch.inbox.get()
        kind = msg["type"]
        rid = msg.get("request_id")
        if kind == "cancel":
            continue  # recorded by the reader; only meaningful while that job runs
        if kind == "shutdown":
            ch.send({"type": "bye", "request_id": rid, "jobs_done": jobs_done})
            os._exit(EXIT_SHUTDOWN)
        if kind == "status":
            ch.send(
                {
                    "type": "status",
                    "request_id": rid,
                    "state": "idle",
                    "worker_instance_id": wid,
                    "engine_instance_id": engine.engine_instance_id,
                    "initialization_id": engine.initialization_id,
                    "counters": dict(engine.counters),
                    "jobs_done": jobs_done,
                    "memory": engine._memory_now(),
                    "resident_after_init": engine.resident,
                }
            )
            continue
        if kind != "generate":
            ch.send({"type": "error", "request_id": rid, "error": f"unknown request type {kind!r}"})
            continue
        job_id = msg.get("job_id")
        if not isinstance(rid, str) or not _ID_RE.fullmatch(rid) or not isinstance(job_id, str) or not _ID_RE.fullmatch(job_id):
            ch.send({"type": "result", "request_id": rid, "job_id": job_id, "status": "invalid", "error": "invalid request_id or job_id", "fatal": False})
            continue
        if rid in seen:
            ch.send({"type": "result", "request_id": rid, "job_id": job_id, "status": "invalid", "error": "duplicate request_id", "fatal": False})
            continue
        seen.append(rid)
        job_dir = jobs_root / job_id
        try:
            if job_dir.resolve().parent != jobs_root or lj._is_link(job_dir):
                raise lj.JobError("job folder is outside the jobs root")
            job = lj.load_job(job_dir / "job.json")
            for key in ("upstream_dir", "models", "ffmpeg", "decoder", "env"):
                if job[key] != cfg[key]:
                    raise lj.JobError(f"job {key} does not match this worker; the manager must restart the worker")
        except (lj.JobError, OSError, ValueError) as exc:
            ch.send({"type": "result", "request_id": rid, "job_id": job_id, "status": "invalid", "error": str(exc)[:1000], "fatal": False})
            continue
        ch.send({"type": "accepted", "request_id": rid, "job_id": job_id})
        events = lj.Events(job_dir / "events.jsonl")
        t_job = time.time()
        events.emit("started", pid=os.getpid(), worker_instance_id=wid, engine_instance_id=engine.engine_instance_id, backend="persistent")
        fatal_exit = None
        try:
            verify = engine.verify_unchanged()
            result = engine.generate(job, job_dir, events, JobControl(ch, job_id, job_dir))
            lj.finish_result(result, job, engine)
            jobs_done += 1
            result.update(
                backend="persistent",
                worker={"worker_instance_id": wid, "pid": os.getpid(), "jobs_done": jobs_done, "engine_verify": verify},
                process_seconds=round(time.time() - t_job, 3),
            )
            lj._write_json(job_dir / "result.json", result)
            events.emit("done", seconds=round(time.time() - t_job, 3))
            ch.send({"type": "result", "request_id": rid, "job_id": job_id, "status": "ok", "fatal": False})
        except lj.Cancelled as exc:
            events.emit("cancelled", reason=str(exc))
            ch.send({"type": "result", "request_id": rid, "job_id": job_id, "status": "cancelled", "fatal": False})
        except lj.EngineChanged as exc:
            events.emit("failed", error_type="EngineChanged", error=str(exc)[:2000])
            ch.send({"type": "result", "request_id": rid, "job_id": job_id, "status": "error", "error": str(exc)[:2000], "fatal": True})
            fatal_exit = EXIT_CHANGED
        except lj.JobError as exc:
            events.emit("failed", error_type="JobError", error=str(exc)[:2000])
            ch.send({"type": "result", "request_id": rid, "job_id": job_id, "status": "invalid", "error": str(exc)[:2000], "fatal": False})
        except BaseException as exc:  # noqa: BLE001 - CUDA errors, OOM, encoder failures: do not reuse this process
            events.emit("failed", error_type=type(exc).__name__, error=str(exc)[:2000])
            ch.send(
                {
                    "type": "result",
                    "request_id": rid,
                    "job_id": job_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "fatal": True,
                }
            )
            fatal_exit = EXIT_FATAL
        finally:
            events.f.close()
        if fatal_exit is not None:
            os._exit(fatal_exit)
        _rotate_log(session, log_n)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - log it; the process must still end through os._exit
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
    finally:
        os._exit(EXIT_FATAL)
