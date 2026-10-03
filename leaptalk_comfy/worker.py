"""Persistent LeapTalk worker owned by this ComfyUI process.

``MANAGER`` keeps at most one runtime worker (one loaded LeapTalk Engine on the GPU) for this ComfyUI
process. It is started lazily by the first persistent Generate, reused by later queue jobs while its
identity (runtime settings, decoder, code, protocol, upstream, model files) is unchanged, and stopped
by: an explicit Unload, the idle timeout (measured from the end of the last job), memory pressure while
idle, an identity change, a fatal error, or ComfyUI exiting. It is a local child process connected by
pipes - not a server, nothing listens on a port.

Ownership: the worker tree (venv launcher, interpreter, ffmpeg) lives in a Windows job object created
with KILL_ON_JOB_CLOSE (process group + stdin EOF on Linux/macOS), see ``process.py``. Normal job ends
keep the tree; unload/idle/fatal/exit end it. One lock serialises worker start, jobs, unload, the idle
timer and one-shot jobs, so two models never run on the GPU at once through this plugin. Other custom
nodes or other ComfyUI processes are not coordinated.
"""

from __future__ import annotations

import atexit
import collections
import hashlib
import json
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from . import client, memory
from .config import Runtime
from .process import IS_WINDOWS, ProcessTree

PROTOCOL_VERSION = 1
MAX_LINE = 64 * 1024
WORKER_SCRIPT = "leaptalk_worker.py"
CODE_FILES = ("leaptalk_worker.py", "leaptalk_job.py")
HELLO_TIMEOUT_S = 60.0
ACCEPT_TIMEOUT_S = 60.0
CANCEL_GRACE_S = 20.0
QUEUE_WAIT_TIMEOUT_S = 3600.0
RESPONSE_TYPES = {"hello", "loading", "ready", "fatal", "accepted", "result", "status", "bye", "error"}
_UPSTREAM_COMMIT_RE = re.compile(r'^UPSTREAM_COMMIT = "([0-9a-f]{40})"', re.M)


class WorkerError(RuntimeError):
    """The persistent worker failed or answered something unexpected; it is not reused."""


class MemoryGuardRefused(RuntimeError):
    """The memory guard refused to load a model or start a job."""


def redact(text: str) -> str:
    """Drop absolute paths from messages that are shown in ComfyUI or written to reports."""
    text = re.sub(r"[A-Za-z]:[\\/][^\s'\"<>|]*", "<path>", text)
    return re.sub(r"(?<![\w.])/(?:home|Users|mnt|opt|tmp|var|root)/[^\s'\"<>|]*", "<path>", text)


def code_digest() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode())
        h.update((client.RUNTIME_DIR / name).read_bytes())
    return h.hexdigest()


def _model_stamps(rt: Runtime, decoder: str) -> dict:
    text = (client.RUNTIME_DIR / "leaptalk_job.py").read_text(encoding="utf-8")
    # the same table the runtime checks; parsed instead of imported (the runtime file is not imported in ComfyUI)
    files = re.findall(r'\("([^"]+)", (\d+), "[0-9a-f]{64}"\)', text)
    out = {}
    for rel, _size in files:
        if decoder == "lite_tae" and rel == "VAE_Wan/Wan2.1_VAE.pth" or decoder == "wan_vae" and rel == "taew2_1.pth":
            continue
        for key, folder in rt.models.items():
            p = Path(folder) / rel
            if p.is_file():
                st = p.stat()
                out[f"{key}/{rel}"] = [st.st_size, st.st_mtime_ns]
    return out


def identity_parts(rt: Runtime, decoder: str) -> dict:
    text = (client.RUNTIME_DIR / "leaptalk_job.py").read_text(encoding="utf-8")
    m = _UPSTREAM_COMMIT_RE.search(text)
    return {
        "protocol": PROTOCOL_VERSION,
        "runtime_id": rt.id,
        "python": rt.python,
        "upstream_dir": rt.upstream_dir,
        "upstream_commit": m.group(1) if m else None,
        "models": dict(sorted(rt.models.items())),
        "ffmpeg": rt.ffmpeg,
        "env": dict(sorted(rt.env.items())),
        "decoder": decoder,
        "device": "cuda",
        "dtype": "bf16",
        "code": code_digest(),
        "model_files": _model_stamps(rt, decoder),
    }


def identity_of(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class _Worker:
    def __init__(self, rt: Runtime, decoder: str, parts: dict, reason: str):
        self.rt = rt
        self.decoder = decoder
        self.parts = parts
        self.identity = identity_of(parts)
        self.worker_instance_id = f"w{uuid.uuid4().hex[:20]}"
        self.start_reason = reason
        self.state = "starting"
        self.tree: ProcessTree | None = None
        self.inbox: queue.Queue = queue.Queue()
        self.dead_reason: str | None = None
        self.interpreter_pid: int | None = None
        self.engine: dict = {}
        self.jobs_done = 0
        self.last_activity = time.monotonic()
        self.started_at = time.monotonic()
        self.startup_seconds: float | None = None
        self.expected_job_gib: float | None = None
        self.loading_step: str | None = None
        self.session: Path | None = None

    # -- channel
    def _reader(self) -> None:
        stream = self.tree.proc.stdout
        while True:
            try:
                line = stream.readline(MAX_LINE + 1)
            except (OSError, ValueError):
                line = b""
            if not line:
                self.inbox.put(None)
                return
            if len(line) > MAX_LINE or not line.endswith(b"\n"):
                self.inbox.put({"type": "__protocol_error__", "error": "response line too long or truncated"})
                return
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.inbox.put({"type": "__protocol_error__", "error": "response is not JSON"})
                return
            if not isinstance(msg, dict) or msg.get("v") != PROTOCOL_VERSION or msg.get("type") not in RESPONSE_TYPES:
                self.inbox.put({"type": "__protocol_error__", "error": f"unexpected response {str(msg)[:200]}"})
                return
            self.inbox.put(msg)

    def send(self, obj: dict) -> None:
        data = (json.dumps({"v": PROTOCOL_VERSION, **obj}) + "\n").encode("utf-8")
        if len(data) > MAX_LINE:
            raise WorkerError("request too large")
        try:
            self.tree.proc.stdin.write(data)
            self.tree.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise WorkerError(f"the worker pipe is closed ({type(exc).__name__})") from exc

    def recv(self, timeout: float) -> dict | None:
        """Next message, None on timeout. Raises WorkerError on EOF / protocol errors / fatal."""
        try:
            msg = self.inbox.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if msg is None:
            code = self.tree.poll() if self.tree else None
            raise WorkerError(f"the worker exited unexpectedly (exit code {code})")
        if msg["type"] == "__protocol_error__":
            raise WorkerError(f"protocol error: {msg['error']}")
        if msg["type"] == "fatal":
            raise WorkerError(redact(f"worker failed: {msg.get('error_type')}: {msg.get('error')}"))
        return msg

    def alive(self) -> bool:
        return self.tree is not None and self.tree.poll() is None and self.state != "dead"

    def summary(self) -> dict:
        now = time.monotonic()
        return {
            "state": self.state,
            "worker_instance_id": self.worker_instance_id,
            "launcher_pid": self.tree.pid if self.tree else None,
            "interpreter_pid": self.interpreter_pid,
            "runtime_id": self.rt.id,
            "decoder": self.decoder,
            "identity": self.identity[:16],
            "engine_instance_id": self.engine.get("engine_instance_id"),
            "initialization_id": self.engine.get("initialization_id"),
            "engine_counters_at_start": self.engine.get("counters"),  # live counters: status "runtime" and each job's result
            "engine_init_timings": self.engine.get("init_timings"),  # paid once, when this worker started
            "jobs_done": self.jobs_done,
            "start_reason": self.start_reason,
            "startup_seconds": self.startup_seconds,
            "seconds_since_last_activity": round(now - self.last_activity, 1),
            "idle_timeout_seconds": self.rt.worker_idle_seconds,
        }


class WorkerManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._worker: _Worker | None = None
        self._idle_thread: threading.Thread | None = None
        self.history: collections.deque = collections.deque(maxlen=50)
        self._pressure: memory.SustainedPressure | None = None
        self._clock = time.monotonic
        atexit.register(self.shutdown_all)

    # ------------------------------------------------------------------ public API

    def status(self) -> dict:
        """Never starts a worker and never extends its idle time."""
        w = self._worker
        out = {"worker": w.summary() if w is not None else None, "host_memory": memory.snapshot(), "history": list(self.history)[-10:]}
        if w is not None and w.state == "idle":
            left = w.rt.worker_idle_seconds - (self._clock() - w.last_activity)
            out["worker"]["idle_seconds_left"] = round(max(0.0, left), 1)
            if self._lock.acquire(blocking=False):  # ask the idle worker for its own memory; skip if a job holds the lock
                try:
                    if self._worker is w and w.state == "idle" and w.alive():
                        rid = f"r{uuid.uuid4().hex[:20]}"
                        w.send({"type": "status", "request_id": rid})
                        deadline = self._clock() + 5.0
                        while self._clock() < deadline:
                            msg = w.recv(0.2)
                            if msg is not None and msg.get("type") == "status" and msg.get("request_id") == rid:
                                out["worker"]["runtime"] = {k: msg.get(k) for k in ("memory", "resident_after_init", "counters", "jobs_done")}
                                break
                except WorkerError as exc:
                    out["worker"]["runtime_error"] = redact(str(exc))
                finally:
                    self._lock.release()
        return out

    def unload(self, reason: str = "unload requested", wait_s: float = 10.0) -> dict:
        """Stop this plugin's worker. Waits up to ``wait_s`` for a running job; never cuts it off."""
        if not self._lock.acquire(timeout=wait_s):
            return {
                "status": "busy",
                "detail": "a LeapTalk job is running; unload waits for jobs to finish - try again afterwards or cancel the job in ComfyUI",
            }
        try:
            w = self._worker
            if w is None:
                return {"status": "no_worker"}
            rep = self._stop_worker(w, reason)
            return {"status": "unloaded", **rep}
        finally:
            self._lock.release()

    def shutdown_all(self) -> None:
        w = self._worker
        if w is not None:
            try:
                self._stop_worker(w, "ComfyUI is exiting", graceful_s=3.0)
            except Exception:  # noqa: BLE001
                pass

    def note_job_memory(self, result: dict) -> None:
        """Use the measured extra CUDA memory of a finished job (peak minus the resident baseline) as the
        expected addition for the next job on this worker, with a 25 % margin."""
        w = self._worker
        jm = (result or {}).get("job_memory") or {}
        try:
            base = jm["resident_before_job"]["cuda_reserved_bytes"]
            peak = max(p["peak_reserved_bytes"] for p in jm["phases"].values())
        except (KeyError, TypeError, ValueError):
            return
        if w is not None and w.engine.get("engine_instance_id") == (result.get("engine") or {}).get("engine_instance_id"):
            w.expected_job_gib = max(1.0, (peak - base) / memory.GIB * 1.25)

    def run_one_shot(self, fn):
        """Run a one-shot job under the same lock; unload an idle worker first (one model at a time)."""
        with self._lock:
            w = self._worker
            unloaded = None
            if w is not None:
                unloaded = self._stop_worker(w, "a one-shot job was requested (one model on the GPU at a time)")
            return fn(), unloaded

    def run_job(self, rt: Runtime, job: dict, job_dir: Path, *, interrupted, on_event, timeout_s: float | None = None) -> dict:
        """Run one prepared job on the persistent worker. Returns info for the report; raises on failure."""
        t_queue = self._clock()
        while not self._lock.acquire(timeout=0.5):
            if interrupted():
                raise client.JobCancelled("cancelled while waiting for the LeapTalk worker")
            if self._clock() - t_queue > QUEUE_WAIT_TIMEOUT_S:
                raise client.JobTimeout("waited too long for the LeapTalk worker")
        try:
            info: dict = {"queue_wait_s": round(self._clock() - t_queue, 3), "backend": "persistent"}
            decoder = job["decoder"]
            cfg = memory.settings(rt.memory_guard)
            parts = identity_parts(rt, decoder)
            ident = identity_of(parts)
            w = self._worker
            restart_reason = None
            if w is not None and not w.alive():
                restart_reason = f"previous worker was not running ({w.dead_reason or 'exited'})"
                self._stop_worker(w, restart_reason)
                w = None
            if w is not None and w.identity != ident:
                changed = sorted(k for k in parts if parts[k] != w.parts.get(k))
                restart_reason = f"worker identity changed: {', '.join(changed)}"
                self._stop_worker(w, restart_reason)
                w = None
            if w is not None:
                w.rt = rt  # same identity: take the current timeouts / idle time / memory settings
            started_new = w is None
            if started_new:
                snap = memory.snapshot()
                ok, why = memory.check_start(snap, cfg, cfg["expected_worker_gib"], load=True)
                info["memory_check"] = {"snapshot": snap, "decision": why, "expected_add_gib": cfg["expected_worker_gib"]}
                if not ok:
                    raise MemoryGuardRefused(f"LeapTalk did not start a worker: {why}")
                w = self._start_worker(rt, decoder, parts, restart_reason or "first persistent job", interrupted)
            else:
                snap = memory.snapshot()
                expect = w.expected_job_gib if w.expected_job_gib is not None else cfg["expected_job_gib"]
                ok, why = memory.check_start(snap, cfg, expect, load=False)
                info["memory_check"] = {"snapshot": snap, "decision": why, "expected_add_gib": round(expect, 2)}
                if not ok:
                    raise MemoryGuardRefused(f"LeapTalk did not start the job on the loaded worker: {why}")
            info["started_new_worker"] = started_new
            info["restart_reason"] = restart_reason
            info["startup_s"] = w.startup_seconds if started_new else 0.0
            self._run_on_worker(w, rt, job, job_dir, interrupted, on_event, timeout_s or rt.timeout_minutes * 60.0, cfg, info)
            info["worker"] = w.summary()
            return info
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ internals

    def _record(self, event: str, **kw) -> None:
        self.history.append({"event": event, "t": round(time.time(), 1), **kw})

    def _ensure_idle_thread(self) -> None:
        if self._idle_thread is None or not self._idle_thread.is_alive():
            self._idle_thread = threading.Thread(target=self._idle_loop, name="leaptalk-worker-idle", daemon=True)
            self._idle_thread.start()

    def _idle_loop(self) -> None:
        # one daemon thread for the life of the process (started with the first worker); it never exits
        # by itself, so a worker started right after an unload can never be left without an idle timer
        while True:
            time.sleep(1.0)
            if self._worker is None:
                continue
            if not self._lock.acquire(blocking=False):
                continue  # a job, a start or an unload is in progress: never interrupt it
            try:
                w = self._worker
                if w is None:
                    continue
                if not w.alive():
                    self._stop_worker(w, f"worker exited while idle ({w.dead_reason or 'exit code ' + str(w.tree.poll() if w.tree else None)})")
                    continue
                if w.state != "idle":
                    continue
                if self._clock() - w.last_activity >= w.rt.worker_idle_seconds:
                    self._stop_worker(w, f"idle timeout ({w.rt.worker_idle_seconds} s without a job)")
                    continue
                cfg = memory.settings(w.rt.memory_guard)
                if self._pressure is None or self._pressure.cfg != cfg:
                    self._pressure = memory.SustainedPressure(cfg)
                if self._pressure.update(memory.snapshot()):
                    self._stop_worker(w, f"memory pressure while idle (commit {self._pressure.last.get('commit_pct')} %)")
            finally:
                self._lock.release()

    def _start_worker(self, rt: Runtime, decoder: str, parts: dict, reason: str, interrupted) -> _Worker:
        w = _Worker(rt, decoder, parts, reason)
        t0 = self._clock()
        root = client.jobs_root(rt)
        session = root / "sessions" / w.worker_instance_id
        (session / "code").mkdir(parents=True)
        for name in CODE_FILES:
            shutil.copyfile(client.RUNTIME_DIR / name, session / "code" / name)  # frozen copy: edits do not affect a running worker
        worker_json = {
            "schema_version": 1,
            "protocol": PROTOCOL_VERSION,
            "worker_instance_id": w.worker_instance_id,
            "jobs_root": str(root),
            "upstream_dir": rt.upstream_dir,
            "models": dict(rt.models),
            "ffmpeg": rt.ffmpeg,
            "decoder": decoder,
            "env": dict(rt.env),
        }
        client._write_json(session / "worker.json", worker_json)
        w.session = session
        argv = [rt.python, str(session / "code" / WORKER_SCRIPT), str(session)]
        self._worker = w
        self._record("starting", worker_instance_id=w.worker_instance_id, reason=reason, decoder=decoder)
        try:
            w.tree = ProcessTree(argv, cwd=session, env=client.child_env(session, rt.python), stdout=None, stderr=session / "worker.log")
            threading.Thread(target=w._reader, name="leaptalk-worker-reader", daemon=True).start()
            hello = self._wait_for(w, {"hello"}, HELLO_TIMEOUT_S, interrupted)
            if hello.get("worker_instance_id") != w.worker_instance_id or not isinstance(hello.get("pid"), int):
                raise WorkerError("worker hello does not match")
            w.interpreter_pid = hello["pid"]
            member = w.tree.contains(w.interpreter_pid)
            if IS_WINDOWS and member is not True:
                raise WorkerError("the worker interpreter is not inside this plugin's job object; refusing to use it")
            w.send({"type": "init", "worker_instance_id": w.worker_instance_id})
            deadline = self._clock() + rt.worker_startup_timeout_seconds
            while True:
                if interrupted():
                    raise client.JobCancelled("cancelled while the LeapTalk worker was starting")
                left = deadline - self._clock()
                if left <= 0:
                    raise client.JobTimeout(f"the LeapTalk worker did not finish loading within {rt.worker_startup_timeout_seconds} s")
                msg = w.recv(min(0.5, left))
                if msg is None:
                    continue
                if msg["type"] == "loading":
                    w.loading_step = str(msg.get("step"))
                    continue
                if msg["type"] == "ready":
                    w.engine = msg.get("engine") or {}
                    break
                raise WorkerError(f"unexpected message while loading: {msg['type']}")
            counters = w.engine.get("counters") or {}
            if (counters.get("init"), counters.get("lora_merge"), counters.get("audio_proj_load")) != (1, 1, 1):
                raise WorkerError(f"the worker Engine reports unexpected load counters {counters}")
            w.startup_seconds = round(self._clock() - t0, 3)
            w.state = "idle"
            w.last_activity = self._clock()
            self._record("ready", worker_instance_id=w.worker_instance_id, startup_s=w.startup_seconds, engine_instance_id=w.engine.get("engine_instance_id"))
            self._ensure_idle_thread()
            return w
        except BaseException as exc:
            self._stop_worker(w, f"start failed: {type(exc).__name__}")
            if isinstance(exc, (client.JobCancelled, client.JobTimeout, WorkerError)):
                raise
            raise WorkerError(redact(f"could not start the LeapTalk worker: {exc}")) from exc

    def _wait_for(self, w: _Worker, types: set, timeout: float, interrupted) -> dict:
        deadline = self._clock() + timeout
        while True:
            if interrupted():
                raise client.JobCancelled("cancelled while the LeapTalk worker was starting")
            left = deadline - self._clock()
            if left <= 0:
                raise WorkerError(f"no {sorted(types)} from the worker within {timeout:.0f} s")
            msg = w.recv(min(0.5, left))
            if msg is None:
                continue
            if msg["type"] in types:
                return msg
            raise WorkerError(f"unexpected message {msg['type']!r} (expected {sorted(types)})")

    def _run_on_worker(self, w: _Worker, rt: Runtime, job: dict, job_dir: Path, interrupted, on_event, timeout_s: float, cfg: dict, info: dict) -> None:
        rid = f"r{uuid.uuid4().hex[:20]}"
        job_id = job["job_id"]
        w.state = "busy"
        w.last_activity = self._clock()
        t0 = self._clock()
        try:
            w.send({"type": "generate", "request_id": rid, "job_id": job_id})
            msg = None
            deadline = self._clock() + ACCEPT_TIMEOUT_S
            while msg is None:
                if self._clock() > deadline:
                    raise WorkerError("the worker did not accept the job")
                msg = w.recv(0.5)
                if msg is not None and msg["type"] == "status":
                    msg = None  # a late answer to an earlier status query: harmless, skip it
            if msg["type"] == "result" and msg.get("request_id") == rid:
                self._finish_result(w, msg, job_id)  # rejected before acceptance (invalid job)
                return
            if msg["type"] != "accepted" or msg.get("request_id") != rid or msg.get("job_id") != job_id:
                raise WorkerError(f"unexpected answer to generate: {msg.get('type')}")
            info["accepted_after_s"] = round(self._clock() - t0, 3)
            job_deadline = self._clock() + timeout_s
            pressure = memory.SustainedPressure(cfg)
            offset = 0
            while True:
                events, offset = client.read_events(job_dir / "events.jsonl", offset)
                for ev in events:
                    on_event(ev)
                msg = w.recv(0.2)
                if msg is not None and msg["type"] == "status":
                    msg = None  # late status answer
                if msg is not None:
                    if msg["type"] == "result" and msg.get("request_id") == rid and msg.get("job_id") == job_id:
                        events, offset = client.read_events(job_dir / "events.jsonl", offset)
                        for ev in events:
                            on_event(ev)
                        info["job_wall_s"] = round(self._clock() - t0, 3)
                        self._finish_result(w, msg, job_id)
                        return
                    raise WorkerError(f"unexpected message during the job: {msg.get('type')}")
                if interrupted():
                    self._cancel(w, rid, job_id, job_dir, "cancelled by the user")
                    raise client.JobCancelled("cancelled by the user")
                if self._clock() > job_deadline:
                    self._cancel(w, rid, job_id, job_dir, "job timeout")
                    raise client.JobTimeout(f"LeapTalk job exceeded {timeout_s:.0f} s")
                if pressure.update(memory.snapshot()):
                    self._cancel(w, rid, job_id, job_dir, "memory pressure")
                    self._stop_worker(w, f"memory pressure during a job (commit {pressure.last.get('commit_pct')} %)")
                    raise MemoryGuardRefused(
                        f"LeapTalk stopped the job: system commit stayed at or above {cfg['stop_commit_pct']:.0f} % for {cfg['stop_sustain_seconds']:.0f} s"
                    )
        except WorkerError as exc:
            w.dead_reason = str(exc)
            self._stop_worker(w, redact(f"worker error: {exc}"))
            raise
        except (client.JobCancelled, client.JobTimeout, MemoryGuardRefused, client.RuntimeJobError):
            raise  # handled above: the job was stopped cleanly (worker kept) or the worker was ended
        except BaseException as exc:
            # anything else (e.g. a failing progress callback) leaves the job's state unknown: do not reuse
            if w.state == "busy":
                self._stop_worker(w, f"job aborted in ComfyUI ({type(exc).__name__}); the worker is not reused")
            raise
        finally:
            if w.state == "busy":
                w.state = "idle"
            w.last_activity = self._clock()

    def _finish_result(self, w: _Worker, msg: dict, job_id: str) -> None:
        status = msg.get("status")
        if msg.get("fatal"):
            w.dead_reason = f"fatal job error ({msg.get('error_type') or status})"
            self._stop_worker(w, w.dead_reason)
        if status == "ok":
            w.jobs_done += 1
            return
        if status == "cancelled":
            raise client.JobCancelled("the LeapTalk job was cancelled")
        raise client.RuntimeJobError(redact(f"LeapTalk job failed ({status}): {msg.get('error_type') or ''} {msg.get('error') or ''}".strip()))

    def _cancel(self, w: _Worker, rid: str, job_id: str, job_dir: Path, why: str) -> None:
        """Ask the job to stop at the next chunk boundary; if it does not, end the worker."""
        try:
            (job_dir / "CANCEL").write_text(why + "\n", encoding="utf-8")
        except OSError:
            pass
        try:
            w.send({"type": "cancel", "job_id": job_id})
        except WorkerError:
            pass
        deadline = self._clock() + CANCEL_GRACE_S
        while self._clock() < deadline:
            try:
                msg = w.recv(0.2)
            except WorkerError:
                break
            if msg is not None and msg.get("type") == "result" and msg.get("request_id") == rid:
                if msg.get("status") == "cancelled" and not msg.get("fatal"):
                    self._record("job_cancelled", worker_instance_id=w.worker_instance_id, job_id=job_id, why=why, worker_kept=True)
                    return
                break
        self._stop_worker(w, f"job did not stop cleanly after {why}")

    def _stop_worker(self, w: _Worker, reason: str, graceful_s: float = 10.0) -> dict:
        rep: dict = {"worker_instance_id": w.worker_instance_id, "reason": reason}
        if w.tree is not None:
            if w.tree.poll() is None and w.state == "idle":
                try:
                    w.send({"type": "shutdown", "request_id": f"r{uuid.uuid4().hex[:20]}"})
                    deadline = self._clock() + graceful_s
                    while self._clock() < deadline and w.tree.poll() is None:
                        time.sleep(0.05)
                except WorkerError:
                    pass
            rep["graceful_exit"] = w.tree.poll() is not None
            rep.update(w.tree.terminate())
            w.tree.close()
        w.state = "dead"
        if self._worker is w:
            self._worker = None
        rep["host_memory_after"] = memory.snapshot()
        self._record("stopped", **{k: v for k, v in rep.items() if k != "host_memory_after"})
        return rep


MANAGER = WorkerManager()
