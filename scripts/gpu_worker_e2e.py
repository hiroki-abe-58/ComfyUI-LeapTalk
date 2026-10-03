"""Maintainer GPU series for the persistent worker (v0.2), through ComfyUI's HTTP API and websocket.

Starts its own ComfyUI (127.0.0.1, free port, --cpu, --cache-none so every queued prompt really runs),
queues real workflows one after another and records evidence for every job: prompt id, job id, backend,
worker instance id and PIDs, engine instance id, init / LoRA-merge counters, generator calls, frame
hashes, timings, previews received per prompt, system commit / available RAM / GPU memory around the
job. Never touches another ComfyUI instance; only processes started by its own ComfyUI are inspected or
stopped (a worker interpreter or launcher in the crash steps, after checking it descends from it).

    python scripts/gpu_worker_e2e.py --comfyui <ComfyUI dir> --python <ComfyUI python> --config <runtimes.json>
        --runtime <runtime_id> --workdir <empty dir> --fixtures fixtures.json --steps oneshot,persistent,...
        [--legacy] [--label v0.2.0]

fixtures.json: {"images": {"A": path, "B": path}, "audio": {"a_short": path, "b_long": path, ...}}

Steps
  status0      Worker status before any job: no worker, no worker process
  oneshot      the v0.1 workflow (no backend input) on the benchmark fixtures, --oneshot-reps times each
  persistent   per benchmark fixture: unload, first persistent job, --warm-reps warm jobs (+ worker memory)
  sequence     >= 8 jobs on one worker: portrait A/B/A, speech changes, stereo 48 kHz, 0.5 s, silence, ~34 s
  long96       ~96 s speech on the loaded worker
  fresh        unload, then the A/short input on a fresh worker (compare with the warm result)
  decoder      lite_tae -> wan_vae -> lite_tae on the persistent worker (restart reasons, hashes)
  oneshot_wan  wan_vae one-shot (parity with the persistent wan_vae result)
  cancel       /interrupt during a long persistent job, then a normal job
  timeout      a persistent job on a runtime with timeout_minutes=1, then a normal job
  error        a persistent job on a runtime with a missing model folder, then a normal job
  crash        the worker interpreter is killed during a job, then a normal job
  launcher     the worker's venv launcher is killed while idle, then a normal job
  idle         a runtime with worker_idle_seconds=15: status polls until the worker is gone
  unload       loaded-idle and after-unload memory around an explicit Unload
  race         generate, generate, unload, generate queued at once (one worker at a time)
  kill_idle    ComfyUI killed hard while the worker idles (worker tree must go away)
  kill_busy    ComfyUI killed hard during a persistent job
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import re
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("leaptalk_memory", REPO / "leaptalk_comfy" / "memory.py")
MEM = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = MEM  # dataclasses look the module up while it is executed
_spec.loader.exec_module(MEM)

OFFICIAL_REFERENCE = {  # frames_sha256 of the v0.1.0 package result that matched the official script frame by frame
    ("A", "a_short", "lite_tae"): "e7a290c491b9b5318ab5aa61b2113c71a99f0c5883e4896b14d167a970e13163",
    ("B", "b_long", "lite_tae"): "7bcffb23687269d4b20d6f9a629d2ad27e0cebdd9c68e9a6185e58a53460acc4",
}
BENCH = (("A", "a_short"), ("B", "b_long"))
V020_WAN_REFERENCE = {  # frames_sha256 of v0.2.0 with decoder = wan_vae (one-shot and persistent gave the same)
    ("A", "a_short"): "c63bd9b0822349bf3d84baa023ae85f20b0d44d83d82c7e69301740247bfc550",
    ("B", "b_short"): "5ee53e4df089378aa6dd2d6c14877c8f15c5d0f90f247281f6482c9bda7f349f",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(base: str, path: str, data: dict | None = None, timeout: float = 30):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json"})  # noqa: S310 - 127.0.0.1 only
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - 127.0.0.1 only
        body = r.read()
    return json.loads(body) if body else {}


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, timeout=30)  # noqa: S603,S607
        return int(out.stdout.decode().split()[0])
    except Exception:  # noqa: BLE001
        return None


def gen_wf(runtime_id, image, audio, prefix, *, backend=None, decoder="lite_tae", guidance=1.0) -> dict:
    rt_inputs = {"runtime_id": runtime_id}
    if backend is not None:  # None: the v0.1 graph, no backend input (runtime default)
        rt_inputs["backend"] = backend
    return {
        "1": {"class_type": "LeapTalkRuntime", "inputs": rt_inputs},
        "2": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "4": {
            "class_type": "LeapTalkGenerate",
            "inputs": {"runtime": ["1", 0], "image": ["2", 0], "audio": ["3", 0], "decoder": decoder, "audio_guidance": guidance, "preview": True},
        },
        # mp4 + codec auto: the runtime's H.264/AAC streams are copied, not re-encoded
        "5": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0], "filename_prefix": prefix, "format": "mp4", "format.codec": "auto"}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["4", 1]}},
    }


def worker_wf(action: str) -> dict:
    return {"1": {"class_type": "LeapTalkWorker", "inputs": {"action": action}}, "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}}}


def decode(path: Path) -> dict:
    import av

    with av.open(str(path)) as c:
        v = [s for s in c.streams if s.type == "video"]
        a = [s for s in c.streams if s.type == "audio"]
        info = {"video_streams": len(v), "audio_streams": len(a)}
        if v:
            info.update(width=v[0].codec_context.width, height=v[0].codec_context.height, rate=float(v[0].average_rate), codec=v[0].codec_context.name)
        if a:
            info.update(audio_codec=a[0].codec_context.name, audio_rate=a[0].codec_context.sample_rate, audio_channels=a[0].codec_context.channels)
    with av.open(str(path)) as c:
        info["frames"] = sum(1 for _ in c.decode(video=0))
    if a:
        with av.open(str(path)) as c:
            info["audio_seconds"] = round(sum(fr.samples for fr in c.decode(audio=0)) / info["audio_rate"], 4)
    return info


class Watch(threading.Thread):
    """Memory/GPU samples (about every 0.5-1 s), worker processes of *this* ComfyUI, and a watchdog:
    commit continuously >= 95 % for 5 s -> POST /interrupt (recorded). Nothing outside our ComfyUI is touched."""

    def __init__(self, csv_path: Path, comfy: Comfy):
        super().__init__(daemon=True)
        self.comfy = comfy
        self.rows: list[dict] = []
        self.stop_ev = threading.Event()
        self.f = open(csv_path, "a", encoding="utf-8")  # noqa: SIM115
        self.f.write("t,commit_pct,commit_gib,available_gib,gpu_mib,worker_launchers,worker_procs\n")
        self.max_launchers = 0
        self.watchdog_events: list[dict] = []
        self._high_since = None

    def run(self) -> None:
        n = 0
        gpu = None
        procs = (0, 0)
        while not self.stop_ev.is_set():
            s = MEM.snapshot()
            if n % 2 == 0:
                gpu = gpu_used_mib()
                procs = self.comfy.worker_process_counts()
                self.max_launchers = max(self.max_launchers, procs[0])
            row = {
                "t": time.time(),
                "commit_pct": s.get("commit_pct"),
                "commit_gib": s.get("commit_gib"),
                "available_gib": s.get("available_gib"),
                "gpu_mib": gpu,
            }
            self.rows.append(row)
            self.f.write(f"{row['t']:.2f},{row['commit_pct']},{row['commit_gib']},{row['available_gib']},{gpu},{procs[0]},{procs[1]}\n")
            self.f.flush()
            # same rule as the plugin's guard: 5 s continuously at or above 95 % (any lower sample resets)
            pct = s.get("commit_pct") or 0
            if pct >= 95.0:
                self._high_since = self._high_since or time.time()
                if time.time() - self._high_since >= 5.0:
                    self.watchdog_events.append({"t": time.time(), "commit_pct": pct})
                    try:
                        http(self.comfy.base, "/interrupt", {})
                    except Exception:  # noqa: BLE001
                        pass
                    self._high_since = None
            else:
                self._high_since = None
            n += 1
            self.stop_ev.wait(0.5)
        self.f.close()

    def window(self, t0: float, t1: float) -> dict:
        rows = [r for r in self.rows if t0 <= r["t"] <= t1 and r["commit_pct"] is not None]
        if not rows:
            return {}
        g = [r["gpu_mib"] for r in rows if r["gpu_mib"] is not None]
        return {
            "samples": len(rows),
            "commit_pct_max": max(r["commit_pct"] for r in rows),
            "commit_gib_max": max(r["commit_gib"] for r in rows),
            "commit_gib_start": rows[0]["commit_gib"],
            "available_gib_min": min(r["available_gib"] for r in rows),
            "gpu_mib_max": max(g) if g else None,
            "gpu_mib_start": g[0] if g else None,
        }

    def now(self) -> dict:
        s = MEM.snapshot()
        return {"commit_pct": s.get("commit_pct"), "commit_gib": s.get("commit_gib"), "available_gib": s.get("available_gib"), "gpu_mib": gpu_used_mib()}


class WS(threading.Thread):
    """ComfyUI websocket: per prompt - status messages and binary preview frames (attributed to the
    prompt that is executing)."""

    def __init__(self, port: int, cid: str):
        super().__init__(daemon=True)
        import websocket

        self.conn = websocket.create_connection(f"ws://127.0.0.1:{port}/ws?clientId={cid}", timeout=2)
        self.current: str | None = None
        self.events: dict = collections.defaultdict(list)
        self.previews: dict = collections.defaultdict(list)
        self.stop_ev = threading.Event()

    def run(self) -> None:
        import websocket

        while not self.stop_ev.is_set():
            try:
                msg = self.conn.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:  # noqa: BLE001
                return
            now = time.time()
            if isinstance(msg, bytes):
                if len(msg) > 8 and struct.unpack(">I", msg[:4])[0] == 1 and self.current:
                    self.previews[self.current].append((now, msg[8:]))
                continue
            try:
                d = json.loads(msg)
            except ValueError:
                continue
            kind, data = d.get("type"), d.get("data") or {}
            pid = data.get("prompt_id")
            if kind == "execution_start" and pid:
                self.current = pid
            if pid:
                small = {k: data.get(k) for k in ("node", "value", "max", "exception_message") if k in data}
                self.events[pid].append((now, kind, small))
            if kind in ("execution_success", "execution_error", "execution_interrupted") and pid == self.current:
                self.current = None

    def end_time(self, pid: str) -> float | None:
        for t, kind, _ in self.events.get(pid, []):
            if kind in ("execution_success", "execution_error", "execution_interrupted"):
                return t
        return None

    def progress_count(self, pid: str, node: str = "4") -> int:
        return sum(1 for _, kind, d in self.events.get(pid, []) if kind == "progress" and str(d.get("node")) == node)

    def close(self) -> None:
        self.stop_ev.set()
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass


class Comfy:
    def __init__(self, comfyui: Path, python: str, config: Path, workdir: Path, tag: str, base_directory: str = ""):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.workdir = workdir
        env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY", "CREDENTIAL"))}
        tmp = workdir / "tmp"
        tmp.mkdir(exist_ok=True)
        env.update(COMFYUI_LEAPTALK_CONFIG=str(config), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", TEMP=str(tmp), TMP=str(tmp))
        for sub in ("output", "temp", "user", "input"):
            (workdir / sub).mkdir(exist_ok=True)
        self.log = open(workdir / f"comfyui-{tag}.log", "ab")  # noqa: SIM115
        argv = [python, str(comfyui / "main.py"), "--listen", "127.0.0.1", "--port", str(self.port), "--cpu", "--disable-auto-launch", "--cache-none"]
        if base_directory:  # custom_nodes (and the rest) from another folder; the ComfyUI checkout is only read
            argv += ["--base-directory", base_directory]
        argv += ["--output-directory", str(workdir / "output"), "--temp-directory", str(workdir / "temp")]
        argv += ["--user-directory", str(workdir / "user"), "--input-directory", str(workdir / "input")]
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(argv, cwd=str(comfyui), stdout=self.log, stderr=subprocess.STDOUT, env=env, creationflags=flags)  # noqa: S603
        for _ in range(300):
            try:
                http(self.base, "/system_stats", timeout=2)
                break
            except Exception:  # noqa: BLE001
                if self.proc.poll() is not None:
                    raise RuntimeError("ComfyUI exited during start-up; see its log") from None
                time.sleep(1)
        else:
            raise RuntimeError("ComfyUI did not start")
        self.cid = str(uuid.uuid4())
        self.ws = WS(self.port, self.cid)
        self.ws.start()

    # -- processes started by this ComfyUI (identity by descent, never by name alone)
    def _descendants(self):
        import psutil

        try:
            return psutil.Process(self.proc.pid).children(recursive=True)
        except Exception:  # noqa: BLE001
            return []

    def worker_processes(self) -> dict:
        """{'launchers': [...], 'interpreters': [...], 'others': [...]}: worker processes of this ComfyUI."""
        out = {"launchers": [], "interpreters": [], "others": []}
        procs = self._descendants()
        worker = {p.pid for p in procs if any("leaptalk_worker.py" in c for c in _safe(p.cmdline, []))}
        for p in procs:
            ppid = _safe(p.ppid, None)
            if ppid is None:
                continue
            if p.pid in worker:
                (out["interpreters"] if ppid in worker else out["launchers"]).append(p.pid)
            elif ppid in worker or any(a.pid in worker for a in _safe_parents(p)):
                out["others"].append(p.pid)
        return out

    def worker_process_counts(self) -> tuple[int, int]:
        w = self.worker_processes()
        return len(w["launchers"]), len(w["launchers"]) + len(w["interpreters"]) + len(w["others"])

    def all_descendant_pids(self) -> list[int]:
        return [p.pid for p in self._descendants()]

    def queue(self, prompt: dict) -> str:
        pid = str(uuid.uuid4())
        res = http(self.base, "/prompt", {"prompt": prompt, "client_id": self.cid, "prompt_id": pid})
        if res.get("node_errors"):
            raise RuntimeError(f"validation failed: {res['node_errors']}")
        return res["prompt_id"]

    def wait(self, pid: str, timeout: float) -> dict:
        t = time.time()
        while time.time() - t < timeout:
            h = http(self.base, f"/history/{pid}")
            if pid in h and h[pid].get("status", {}).get("completed") is not None:
                st = h[pid]["status"]
                if st.get("status_str") in ("success", "error") or st.get("completed"):
                    return h[pid]
            time.sleep(0.5)
        raise TimeoutError(f"prompt {pid} did not finish in {timeout} s")

    def stop(self) -> None:
        self.ws.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()

    def kill_hard(self) -> None:
        self.ws.close()
        self.proc.kill()
        self.proc.wait(30)
        self.log.close()


def outcome(rec: dict) -> str:
    """completed / admission_refused / memory_stopped / user_cancelled / timeout / runtime_failed
    (from ComfyUI's status and the node's error text; a user cancel has no error text)."""
    err = rec.get("error") or ""
    if rec.get("status") == "success":
        return "completed"
    if "LeapTalk memory guard: stopped" in err or "stopped the job: system commit" in err:
        return "memory_stopped"
    if "LeapTalk memory guard:" in err or "did not start" in err:
        return "admission_refused"
    if "timeout" in err.lower() or "exceeded" in err:
        return "timeout"
    if "execution_interrupted" in (rec.get("messages") or []) and not err:
        return "user_cancelled"
    return "runtime_failed"


def _safe(fn, default):
    """psutil call on a process that may exit at any moment."""
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _safe_parents(p):
    return _safe(p.parents, [])


def pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def wait_gone(pids: list[int], seconds: float = 30) -> list[int]:
    t = time.time()
    while time.time() - t < seconds:
        if not [p for p in pids if p and pid_alive(p)]:
            return []
        time.sleep(0.25)
    return [p for p in pids if p and pid_alive(p)]


class Runner:
    def __init__(self, a):
        self.a = a
        self.fx = json.loads(a.fixtures.read_text(encoding="utf-8"))
        self.workdir = a.workdir
        self.report: dict = {"label": a.label, "legacy_workflow": a.legacy, "steps": {}, "sessions": []}
        self.hashes: dict = collections.defaultdict(list)  # (img, aud, decoder, backend) -> [frames_sha256]
        self.comfy: Comfy | None = None
        self.watch: Watch | None = None
        self.n_session = 0
        self.fd = a.failure_decoder  # decoder for the failure / lifecycle steps (their subject is the worker, not the decoder)
        self._redact_roots = [str(p) for p in (a.workdir, a.comfyui, Path(a.python).parent, a.config.parent)] + [
            str(Path(p).parent) for p in self._fixture_paths()
        ]

    def _fixture_paths(self):
        return list(self.fx["images"].values()) + list(self.fx["audio"].values())

    # ------------------------------------------------------------------ setup
    def write_config(self) -> Path:
        cfg = json.loads(self.a.config.read_text(encoding="utf-8"))
        base = dict(cfg["runtimes"][self.a.runtime])
        base.pop("jobs_dir", None)  # jobs live in this ComfyUI's temp folder
        self._redact_roots += [str(Path(base["python"]).parent), base["upstream_dir"], str(Path(base["ffmpeg"]).parent), *base["models"].values()]
        rts = {
            "main": base,
            "e2e-timeout": {**base, "timeout_minutes": 1},
            "e2e-missing-model": {**base, "models": {**base["models"], "leaptalk_dir": str(self.workdir / "no-such-leaptalk-dir")}},
        }
        if not self.a.legacy:
            rts["e2e-idle"] = {**base, "worker_idle_seconds": 15}
        p = self.workdir / "leaptalk.runtimes.e2e.json"
        p.write_text(json.dumps({"schema_version": 1, "runtimes": rts}, indent=1), encoding="utf-8")
        return p

    def start(self) -> None:
        self.n_session += 1
        self.comfy = Comfy(self.a.comfyui, self.a.python, self.cfg_path, self.workdir, f"s{self.n_session}", self.a.base_directory)
        self.watch = Watch(self.workdir / "memwatch.csv", self.comfy)
        self.watch.start()
        info = http(self.comfy.base, "/object_info")
        rt_in = info.get("LeapTalkRuntime", {}).get("input", {})
        sess = {
            "session": self.n_session,
            "nodes": sorted(n for n in info if n.startswith("LeapTalk")),
            "runtime_has_backend_input": "backend" in (rt_in.get("optional") or {}),
            "fresh_memory": self.watch.now(),
            "worker_processes_at_start": self.comfy.worker_processes(),
        }
        self.report["sessions"].append(sess)
        self.log(f"session {self.n_session} on port {self.comfy.port}: {sess['nodes']}")

    def stop(self, hard: bool = False) -> None:
        if self.comfy is None:
            return
        if hard:
            self.comfy.kill_hard()
        else:
            self.comfy.stop()
        if self.watch:
            self.watch.stop_ev.set()
            self.watch.join(5)
            self.report.setdefault("max_worker_launchers_seen", 0)
            self.report["max_worker_launchers_seen"] = max(self.report["max_worker_launchers_seen"], self.watch.max_launchers)
            self.report.setdefault("watchdog_interrupts", []).extend(self.watch.watchdog_events)
        self.comfy = None

    def log(self, text: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        with open(self.workdir / "progress.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def redact(self, obj):
        text = json.dumps(obj, ensure_ascii=False)
        for root in sorted(set(self._redact_roots), key=len, reverse=True):
            for form in (root, root.replace("\\", "\\\\"), root.replace("\\", "/")):
                text = text.replace(form, "<local>")
        text = re.sub(r"[A-Za-z]:\\\\Users\\\\[^\\\\\"]+", "<user>", text)
        return json.loads(text)

    def save(self) -> None:
        (self.workdir / "report.json").write_text(json.dumps(self.redact(self.report), indent=1, ensure_ascii=False), encoding="utf-8")

    def ensure_inputs(self) -> None:
        for p in self._fixture_paths():
            src = Path(p)
            dst = self.workdir / "input" / src.name
            if not dst.exists():
                shutil.copyfile(src, dst)

    def img(self, key: str) -> str:
        return Path(self.fx["images"][key]).name

    def aud(self, key: str) -> str:
        return Path(self.fx["audio"][key]).name

    # ------------------------------------------------------------------ one prompt
    def submit(self, wf: dict) -> tuple[str, float]:
        return self.comfy.queue(wf), time.time()

    def collect(self, name: str, pid: str, t0: float, *, timeout: float = 3600, meta: dict | None = None) -> dict:
        h = self.comfy.wait(pid, timeout)
        t_end = self.comfy.ws.end_time(pid) or time.time()
        st = h["status"]["status_str"]
        rec = {
            "name": name,
            "prompt_id": pid,
            "status": st,
            "queued_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
            "wall_s": round(t_end - t0, 3),
        }
        if meta:
            rec.update(meta)
        msgs = h["status"].get("messages", [])
        rec["messages"] = [m[0] for m in msgs]
        err = next((m[1].get("exception_message") for m in msgs if m[0] == "execution_error"), None)
        if err:
            rec["error"] = err[:600]
        rec["memory_window"] = self.watch.window(t0, t_end + 0.5)
        rec["outcome"] = outcome(rec)
        prev = self.comfy.ws.previews.get(pid, [])
        rec["ws"] = {
            "progress_msgs": self.comfy.ws.progress_count(pid),
            "previews": len(prev),
            "first_preview_after_s": round(prev[0][0] - t0, 2) if prev else None,
        }
        outs = h.get("outputs", {})
        text = (outs.get("6") or outs.get("2") or {}).get("text", [None])[0]
        if text:
            try:
                rep = json.loads(text)
            except ValueError:
                rep = {"raw": text[:2000]}
            if "worker" in rep and "host_memory" in rep or "status" in rep and ("now" in rep or rep.get("status") in ("no_worker", "busy")):
                rec["worker_node"] = rep
            else:
                rec["job"] = self.evidence(rep)
                self.check_video(rec, outs, rep)
                if prev:
                    rec["ws"]["preview_vs_job"] = self.preview_binding(prev[-1][1], rep)
                after = (outs.get("8") or {}).get("text", [None])[0]  # shipped persistent workflow: Worker status after the job
                if after:
                    st = json.loads(after)
                    w = st.get("worker") or {}
                    rec["worker_status_after_job"] = {k: w.get(k) for k in ("state", "worker_instance_id", "jobs_done", "idle_seconds_left")} if w else None
        return rec

    def run(self, name: str, wf: dict, **meta) -> dict:
        pid, t0 = self.submit(wf)
        rec = self.collect(name, pid, t0, meta=meta)
        j = rec.get("job") or {}
        self.log(
            f"{name}: {rec['status']} ({rec.get('outcome')}) {rec['wall_s']:.1f}s"
            + (
                f" job={j.get('job_id')} worker={j.get('worker_instance_id')} new={j.get('started_new_worker')} frames={str(j.get('frames_sha256'))[:12]}"
                if j
                else ""
            )
            + (f" error={rec.get('error', '')[:160]}" if rec.get("error") else "")
        )
        return rec

    @staticmethod
    def evidence(rep: dict) -> dict:
        host = rep.get("host") or {}
        proc = host.get("process") or {}
        w = proc.get("worker") or {}
        eng = rep.get("engine") or {}
        timings = rep.get("timings") or {}
        return {
            "job_id": rep.get("job"),
            "backend": rep.get("backend"),
            "runtime_id": rep.get("runtime_id"),
            "decoder": rep.get("decoder"),
            "worker_instance_id": w.get("worker_instance_id"),
            "worker_interpreter_pid": w.get("interpreter_pid"),
            "worker_launcher_pid": w.get("launcher_pid"),
            "one_shot_runner_pid": proc.get("pid"),
            "engine_instance_id": eng.get("engine_instance_id"),
            "initialization_id": eng.get("initialization_id"),
            "engine_counters": eng.get("counters"),
            "weight_fingerprint": eng.get("weight_fingerprint"),
            "worker_jobs_done": (rep.get("worker") or {}).get("jobs_done"),
            "started_new_worker": proc.get("started_new_worker"),
            "restart_reason": proc.get("restart_reason"),
            "unloaded_persistent_worker": proc.get("unloaded_persistent_worker"),
            "memory_check": (proc.get("memory_check") or {}).get("decision"),
            "expected_add_gib": (proc.get("memory_check") or {}).get("expected_add_gib"),
            "startup_s": proc.get("startup_s"),
            "worker_engine_init_timings": w.get("engine_init_timings"),
            "one_shot_engine_init_timings": rep.get("engine_init_timings"),
            "queue_wait_s": proc.get("queue_wait_s"),
            "job_wall_s": proc.get("job_wall_s"),
            "host_wall_s": proc.get("host_wall_s"),
            "node_wall_seconds": rep.get("wall_seconds"),
            "process_seconds": rep.get("process_seconds"),
            "timings": timings,
            "forwards": rep.get("forwards"),
            "chunks": rep.get("chunks"),
            "output_frames": rep.get("output_frames"),
            "audio": rep.get("audio"),
            "frames_sha256": rep.get("frames_sha256"),
            "reference_latent_sha256": rep.get("reference_latent_sha256"),
            "mp4_sha256": rep.get("sha256"),
            "job_memory": rep.get("job_memory"),
            "memory": rep.get("memory"),
            "lora_applied_tensors": ((rep.get("weights") or {}).get("lora") or {}).get("applied_tensors"),
            "effective_config": rep.get("effective_config"),
        }

    def check_video(self, rec: dict, outs: dict, rep: dict) -> None:
        vids = []
        for node in outs.values():
            for items in node.values():
                for item in items if isinstance(items, list) else []:
                    if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4"):
                        vids.append(self.workdir / "output" / item.get("subfolder", "") / item["filename"])
        rec["saved"] = [str(p.relative_to(self.workdir)).replace("\\", "/") for p in vids]
        if not vids:
            return
        d = decode(vids[0])
        rec["decoded"] = d
        exp = rep.get("output_frames")
        secs = ((rep.get("audio") or {}).get("input_seconds")) or 0
        rec["checks"] = {
            "one_video": len(vids) == 1,
            "has_audio": d.get("audio_streams") == 1,
            "frames_match_report": d.get("frames") == exp,
            "512x512_25fps": (d.get("width"), d.get("height"), d.get("rate")) == (512, 512, 25.0),
            "audio_tail_s": round(d.get("audio_seconds", 0) - secs, 4) if secs else None,
            "lora_applied": rec["job"].get("lora_applied_tensors") == 480,
        }

    def preview_binding(self, last_ws_jpeg: bytes, rep: dict) -> dict:
        """Compare the last websocket preview of this prompt with this job's own preview.jpg and with the
        previous job's preview.jpg (mean absolute pixel difference, 0-255)."""
        import io

        import numpy as np
        from PIL import Image

        root = self.workdir / "temp" / "temp" / "leaptalk"
        own = root / (rep.get("job") or "") / "preview.jpg"
        try:
            ws_img = np.asarray(Image.open(io.BytesIO(last_ws_jpeg)).convert("RGB").resize((256, 256)), dtype=np.float32)
        except Exception:  # noqa: BLE001
            return {"error": "undecodable preview"}
        out = {}
        if own.is_file():
            out["own_job_mad"] = round(float(np.abs(ws_img - np.asarray(Image.open(own).convert("RGB").resize((256, 256)), dtype=np.float32)).mean()), 2)
        jobs = sorted((p for p in root.glob("job-*") if p.name != rep.get("job") and (p / "preview.jpg").is_file()), key=lambda p: p.stat().st_mtime)
        if jobs:
            other = jobs[-1] / "preview.jpg"
            out["previous_job_mad"] = round(float(np.abs(ws_img - np.asarray(Image.open(other).convert("RGB").resize((256, 256)), dtype=np.float32)).mean()), 2)
            out["previous_job"] = jobs[-1].name
        return out

    def worker_status(self, name: str) -> dict:
        rec = self.run(name, worker_wf("status"))
        node = rec.get("worker_node") or {}
        w = node.get("worker") or {}
        return {
            "prompt_id": rec["prompt_id"],
            "worker": {
                k: w.get(k)
                for k in (
                    "state",
                    "worker_instance_id",
                    "interpreter_pid",
                    "engine_instance_id",
                    "engine_counters_at_start",
                    "jobs_done",
                    "idle_seconds_left",
                    "runtime",
                )
            }
            if w
            else None,
            "host_memory": {k: (node.get("host_memory") or {}).get(k) for k in ("commit_pct", "commit_gib", "available_gib")},
            "history_tail": (node.get("history") or [])[-4:],
            "worker_processes": self.comfy.worker_processes(),
            "gpu_mib": gpu_used_mib(),
        }

    def record(self, step: str, data) -> None:
        self.report["steps"][step] = data
        self.save()

    def note_hash(self, img, aud, dec, backend, rec) -> None:
        h = (rec.get("job") or {}).get("frames_sha256")
        if h:
            self.hashes[f"{img}|{aud}|{dec}|{backend}"].append(h)

    def gen(self, name, img, aud, *, backend="persistent", runtime="main", decoder="lite_tae", guidance=1.0) -> dict:
        b = None if backend == "default" else backend
        rec = self.run(
            name,
            gen_wf(runtime, self.img(img), self.aud(aud), f"e2e/{name}", backend=b, decoder=decoder, guidance=guidance),
            input={"image": img, "audio": aud, "decoder": decoder, "guidance": guidance, "backend_input": b, "runtime": runtime},
        )
        if rec["status"] == "success":
            reported = (rec.get("job") or {}).get("backend")
            effective = reported.get("effective") if isinstance(reported, dict) else "one-shot"  # v0.1 reports have no backend
            self.note_hash(img, aud, decoder, effective, rec)
        return rec

    def unload(self, name: str) -> dict:
        rec = self.run(name, worker_wf("unload"))
        rec["worker_processes_after"] = self.comfy.worker_processes()
        return rec

    def wait_progress(self, pid: str, n: int = 2, timeout: float = 600) -> bool:
        t = time.time()
        while time.time() - t < timeout:
            if self.comfy.ws.progress_count(pid) >= n:
                return True
            if self.comfy.ws.end_time(pid) is not None:  # finished (or was refused) before reaching chunk n
                return False
            time.sleep(0.2)
        return False

    # ------------------------------------------------------------------ steps
    def step_status0(self):
        self.record("status0", {"status": self.worker_status("status0"), "worker_processes_after_status": self.comfy.worker_processes()})

    def bench(self):
        return [tuple(x.split(":")) for x in self.a.bench.split(",")] if self.a.bench else list(BENCH)

    def step_oneshot(self):
        out = {}
        for img, aud in self.bench():
            runs = []
            for i in range(self.a.oneshot_reps):
                rec = self.gen(f"oneshot_{img}_{aud}_{i + 1}", img, aud, backend="default")
                time.sleep(2)
                rec["worker_processes_after"] = self.comfy.worker_processes()
                rec["descendants_after"] = len(self.comfy.all_descendant_pids())
                runs.append(rec)
                self.record("oneshot", {**out, f"{img}_{aud}": runs})
            out[f"{img}_{aud}"] = runs
        self.record("oneshot", out)

    def step_persistent(self):
        out = {}
        for img, aud in self.bench():
            runs = {"unload_before": self.unload(f"unload_before_{img}_{aud}")}
            time.sleep(3)
            runs["before_first"] = self.watch.now()
            runs["first"] = self.gen(f"persistent_first_{img}_{aud}", img, aud)
            time.sleep(3)
            runs["loaded_idle"] = self.worker_status(f"status_after_first_{img}_{aud}")
            warm = []
            for i in range(self.a.warm_reps):
                rec = self.gen(f"persistent_warm_{img}_{aud}_{i + 1}", img, aud)
                time.sleep(3)
                rec["after_job_status"] = self.worker_status(f"status_after_warm_{img}_{aud}_{i + 1}")
                warm.append(rec)
                runs["warm"] = warm
                self.record("persistent", {**out, f"{img}_{aud}": runs})
            out[f"{img}_{aud}"] = runs
        self.record("persistent", out)

    def step_sequence(self):
        plan = [
            ("A", "a_short"),
            ("B", "b_short"),
            ("A", "a_short"),
            ("A", "a_long"),
            ("B", "stereo48k"),
            ("A", "clip05"),
            ("B", "silence3"),
            ("B", "b_long"),
            ("A", "a_short"),
            ("B", "b_short"),
        ]
        out = {"unload_before": self.unload("sequence_unload_before"), "jobs": []}
        for i, (img, aud) in enumerate(plan, 1):
            rec = self.gen(f"seq{i:02d}_{img}_{aud}", img, aud)
            rec["after_job_status"] = self.worker_status(f"seq{i:02d}_status")
            out["jobs"].append(rec)
            self.record("sequence", out)

    def step_long96(self):
        self.record("long96", {"job": self.gen("long96_A_aba97", "A", "aba97"), "after": self.worker_status("long96_status")})

    def step_fresh(self):
        out = {"unload": self.unload("fresh_unload")}
        time.sleep(3)
        out["fresh"] = self.gen("fresh_A_a_short", "A", "a_short")
        out["warm_again"] = self.gen("fresh_warm_A_a_short", "A", "a_short")
        self.record("fresh", out)

    def step_decoder(self):
        out = {"lite_1": self.gen("decoder_lite_1", "A", "a_short")}
        out["wan_vae"] = self.gen("decoder_wan_vae", "A", "a_short", decoder="wan_vae")
        out["wan_vae_warm"] = self.gen("decoder_wan_vae_warm", "A", "a_short", decoder="wan_vae")
        out["lite_2"] = self.gen("decoder_lite_2", "A", "a_short")
        out["status"] = self.worker_status("decoder_status")
        self.record("decoder", out)

    def step_oneshot_wan(self):
        self.record(
            "oneshot_wan",
            {"job": self.gen("oneshot_wan_vae", "A", "a_short", backend="default", decoder="wan_vae"), "after": self.worker_status("oneshot_wan_status")},
        )

    def step_cancel(self):
        out = {}
        pid, t0 = self.submit(gen_wf("main", self.img("A"), self.aud("aba97"), "e2e/cancel", backend="persistent", decoder=self.fd))
        out["progress_seen"] = self.wait_progress(pid, 3)
        out["worker_processes_during"] = self.comfy.worker_processes()
        http(self.comfy.base, "/interrupt", {})
        out["cancel"] = self.collect("cancel_A_aba97", pid, t0)
        time.sleep(2)
        out["status_after"] = self.worker_status("cancel_status")
        out["next"] = self.gen("cancel_next_A_a_short", "A", "a_short", decoder=self.fd)
        self.log(f"cancel: {out['cancel']['status']} -> next {out['next']['status']}")
        self.record("cancel", out)

    def step_timeout(self):
        # wan_vae: about 1.3 s per chunk, so the ~96 s clip (87 chunks) outlasts timeout_minutes=1 on a loaded worker
        out = {"timeout": self.gen("timeout_A_aba97_wan", "A", "aba97", runtime="e2e-timeout", decoder="wan_vae")}
        time.sleep(2)
        out["status_after"] = self.worker_status("timeout_status")
        out["next"] = self.gen("timeout_next_A_a_short", "A", "a_short", decoder=self.fd)
        self.record("timeout", out)

    def step_error(self):
        out = {"error": self.gen("error_missing_model", "A", "a_short", runtime="e2e-missing-model")}
        time.sleep(2)
        out["status_after"] = self.worker_status("error_status")
        out["next"] = self.gen("error_next_A_a_short", "A", "a_short")
        self.record("error", out)

    def step_crash(self):
        out = {}
        pid, t0 = self.submit(gen_wf("main", self.img("A"), self.aud("a_long"), "e2e/crash", backend="persistent", decoder=self.fd))
        out["progress_seen"] = self.wait_progress(pid, 3)
        wp = self.comfy.worker_processes()
        out["worker_processes_before"] = wp
        killed = None
        if out["progress_seen"] and len(wp["interpreters"]) == 1:  # only while the job is generating
            import psutil

            killed = wp["interpreters"][0]
            psutil.Process(killed).kill()  # our own worker interpreter (descends from our ComfyUI)
        out["killed_interpreter_pid"] = killed
        out["job"] = self.collect("crash_A_a_long", pid, t0)
        out["left_after_10s"] = wait_gone(wp["launchers"] + wp["interpreters"] + wp["others"], 10)
        out["status_after"] = self.worker_status("crash_status")
        out["next"] = self.gen("crash_next_A_a_short", "A", "a_short", decoder=self.fd)
        self.record("crash", out)

    def step_launcher(self):
        out = {"job": self.gen("launcher_job", "A", "a_short", decoder=self.fd)}
        time.sleep(2)
        wp = self.comfy.worker_processes()
        out["worker_processes_before"] = wp
        if len(wp["launchers"]) == 1:
            import psutil

            psutil.Process(wp["launchers"][0]).kill()  # our own worker's venv launcher
            out["killed_launcher_pid"] = wp["launchers"][0]
            time.sleep(0.5)
            out["interpreter_alive_0_5s_after"] = [p for p in wp["interpreters"] if pid_alive(p)]
        out["left_after_15s"] = wait_gone(wp["launchers"] + wp["interpreters"] + wp["others"], 15)
        out["status_after"] = self.worker_status("launcher_status")
        out["next"] = self.gen("launcher_next_A_a_short", "A", "a_short", decoder=self.fd)
        self.record("launcher", out)

    def step_idle(self):
        out = {"job": self.gen("idle_job", "A", "a_short", runtime="e2e-idle", decoder=self.fd)}
        t_end = time.time()
        polls = []
        while time.time() - t_end < 90:
            st = self.worker_status(f"idle_poll_{len(polls) + 1}")
            polls.append({"t_since_job_end": round(time.time() - t_end, 1), **st})
            if st["worker"] is None:
                break
            time.sleep(4)
        out["polls"] = polls
        out["gone_after_s"] = polls[-1]["t_since_job_end"] if polls and polls[-1]["worker"] is None else None
        time.sleep(3)
        out["memory_after"] = self.watch.now()
        out["next_main"] = self.gen("idle_next_main", "A", "a_short", decoder=self.fd)
        self.record("idle", out)

    def step_unload(self):
        out = {"job": self.gen("unload_job", "A", "a_short", decoder=self.fd)}
        time.sleep(5)
        out["loaded_idle"] = self.worker_status("unload_loaded_idle")
        out["unload"] = self.unload("unload_explicit")
        time.sleep(5)
        out["after_unload"] = {**self.watch.now(), "worker_processes": self.comfy.worker_processes()}
        out["unload_again"] = self.unload("unload_again")
        self.record("unload", out)

    def step_race(self):
        prompts = [
            ("race_gen_long", gen_wf("main", self.img("A"), self.aud("a_long"), "e2e/race1", backend="persistent", decoder=self.fd)),
            ("race_gen_short", gen_wf("main", self.img("B"), self.aud("b_short"), "e2e/race2", backend="persistent", decoder=self.fd)),
            ("race_unload", worker_wf("unload")),
            ("race_status", worker_wf("status")),
            ("race_gen_after_unload", gen_wf("main", self.img("A"), self.aud("a_short"), "e2e/race3", backend="persistent", decoder=self.fd)),
        ]
        before = self.watch.max_launchers
        self.watch.max_launchers = 0
        queued = [(name, *self.submit(wf)) for name, wf in prompts]
        recs = [self.collect(name, pid, t0) for name, pid, t0 in queued]
        out = {"prompts": recs, "max_worker_launchers_during": self.watch.max_launchers}
        self.watch.max_launchers = max(before, self.watch.max_launchers)
        for r in recs:
            j = r.get("job") or {}
            self.log(f"{r['name']}: {r['status']} worker={j.get('worker_instance_id')} new={j.get('started_new_worker')}")
        self.record("race", out)

    def step_shipped(self):
        """The shipped API workflows of the installed package, unchanged except the runtime id and the
        input file names: v0.1 one-shot, the persistent example three times, status, unload."""
        api = Path(self.a.shipped_dir) / "workflows" / "api"

        def load(name, img="A", aud="a_short"):
            wf = json.loads((api / f"{name}.json").read_text(encoding="utf-8"))
            for node in wf.values():
                ins = node.get("inputs", {})
                if node.get("class_type") == "LeapTalkRuntime":
                    ins["runtime_id"] = "main"
                if node.get("class_type") == "LoadImage":
                    ins["image"] = self.img(img)
                if node.get("class_type") == "LoadAudio":
                    ins["audio"] = self.aud(aud)
                if node.get("class_type") == "LeapTalkGenerate" and self.a.shipped_decoder:
                    ins["decoder"] = self.a.shipped_decoder
            return wf

        out = {"v01_one_shot": self.run("shipped_v01_portrait_speech", load("leaptalk_portrait_speech"))}
        out["persistent"] = [
            self.run(f"shipped_persistent_{i}", load("leaptalk_persistent", *fx))
            for i, fx in enumerate((("A", "a_short"), ("B", "b_short"), ("A", "a_short")), 1)
        ]
        out["status"] = self.run("shipped_worker_status", load("leaptalk_worker_status"))
        out["unload"] = self.run("shipped_worker_unload", load("leaptalk_worker_unload"))
        out["worker_processes_after_unload"] = self.comfy.worker_processes()
        self.record("shipped", out)

    def _shipped_wf(self, name: str, img: str, aud: str) -> dict:
        wf = json.loads((Path(self.a.shipped_dir) / "workflows" / "api" / f"{name}.json").read_text(encoding="utf-8"))
        for node in wf.values():
            ins = node.get("inputs", {})
            if node.get("class_type") == "LeapTalkRuntime":
                ins["runtime_id"] = "main"
            if node.get("class_type") == "LoadImage":
                ins["image"] = self.img(img)
            if node.get("class_type") == "LoadAudio":
                ins["audio"] = self.aud(aud)
        return wf

    def step_lowmem(self):
        """The shipped lower-memory workflow (persistent + wan_vae) of the installed package, unchanged except
        the runtime id and input names: cold A, warm A again, B with other speech, A again, ~34 s, Status,
        Unload, then a new worker. Every attempt is recorded with its outcome; nothing is retried."""
        wfname = "leaptalk_persistent_lower_memory"
        plan = [
            ("lm1_A_short_cold", "A", "a_short"),
            ("lm2_A_short_warm", "A", "a_short"),
            ("lm3_B_short", "B", "b_short"),
            ("lm4_A_short_again", "A", "a_short"),
            ("lm5_B_long", "B", "b_long"),
        ]
        out = {"status_before": self.worker_status("lm_status_before"), "jobs": []}
        for name, img, aud in plan:
            rec = self.run(name, self._shipped_wf(wfname, img, aud), input={"image": img, "audio": aud, "decoder": "wan_vae", "workflow": wfname})
            out["jobs"].append(rec)
            self.record("lowmem", out)
        out["status"] = self.worker_status("lm_status")
        out["unload"] = self.unload("lm_unload")
        time.sleep(3)
        out["after_unload"] = {**self.watch.now(), "worker_processes": self.comfy.worker_processes()}
        rec = self.run(
            "lm6_A_short_new_worker",
            self._shipped_wf(wfname, "A", "a_short"),
            input={"image": "A", "audio": "a_short", "decoder": "wan_vae", "workflow": wfname},
        )
        out["jobs"].append(rec)
        out["status_end"] = self.worker_status("lm_status_end")
        self.record("lowmem", out)

    def step_oneshot_wan_ref(self):
        """One-shot (the v0.1 workflow with decoder = wan_vae) on the benchmark fixtures: the same-decoder reference."""
        out = {}
        for img, aud in BENCH:
            out[f"{img}_{aud}"] = self.gen(f"oneshot_wan_{img}_{aud}", img, aud, backend="default", decoder="wan_vae")
            self.record("oneshot_wan_ref", out)

    def step_lite_check(self):
        """Lite TAE with the shipped persistent workflow, guard as shipped: whatever the guard decides is recorded."""
        out = {"unload_before": self.unload("lite_unload_before")}
        time.sleep(3)
        out["jobs"] = [
            self.run(
                f"lite_{i}_A_short", self._shipped_wf("leaptalk_persistent", "A", "a_short"), input={"image": "A", "audio": "a_short", "decoder": "lite_tae"}
            )
            for i in (1, 2)
        ]
        out["status"] = self.worker_status("lite_status")
        out["unload"] = self.unload("lite_unload_after")
        self.record("lite_check", out)

    def step_kill_idle(self):
        out = {"job": self.gen("kill_idle_job", "A", "a_short", decoder=self.fd)}
        time.sleep(2)
        wp = self.comfy.worker_processes()
        out["worker_processes_before"] = wp
        t = time.time()
        self.stop(hard=True)
        out["left_after"] = wait_gone(wp["launchers"] + wp["interpreters"] + wp["others"], 30)
        out["seconds_to_clean"] = round(time.time() - t, 2)
        time.sleep(3)
        out["gpu_mib_after"] = gpu_used_mib()
        out["memory_after"] = {k: MEM.snapshot().get(k) for k in ("commit_pct", "commit_gib", "available_gib")}
        self.record("kill_idle", out)
        self.start()

    def step_kill_busy(self):
        out = {}
        pid, _t0 = self.submit(gen_wf("main", self.img("A"), self.aud("aba97"), "e2e/kill_busy", backend="persistent", decoder=self.fd))
        out["progress_seen"] = self.wait_progress(pid, 3)
        wp = self.comfy.worker_processes()
        out["worker_processes_before"] = wp
        t = time.time()
        self.stop(hard=True)
        out["left_after"] = wait_gone(wp["launchers"] + wp["interpreters"] + wp["others"], 30)
        out["seconds_to_clean"] = round(time.time() - t, 2)
        time.sleep(3)
        out["gpu_mib_after"] = gpu_used_mib()
        out["memory_after"] = {k: MEM.snapshot().get(k) for k in ("commit_pct", "commit_gib", "available_gib")}
        self.record("kill_busy", out)
        self.start()

    # ------------------------------------------------------------------ main
    def main(self) -> int:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.report["gpu_idle_mib_before"] = gpu_used_mib()
        self.report["memory_before"] = {k: MEM.snapshot().get(k) for k in ("total_gib", "commit_limit_gib", "commit_pct", "commit_gib", "available_gib")}
        self.cfg_path = self.write_config()
        self.start()
        self.ensure_inputs()
        try:
            for step in self.a.steps.split(","):
                fn = getattr(self, f"step_{step}", None)
                if fn is None:
                    raise SystemExit(f"unknown step {step}")
                self.log(f"=== {step}")
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - record and continue with the next step
                    self.log(f"step {step} raised {type(exc).__name__}: {exc}")
                    self.report["steps"].setdefault(step, {})
                    if isinstance(self.report["steps"][step], dict):
                        self.report["steps"][step]["driver_error"] = f"{type(exc).__name__}: {exc}"[:600]
                    self.save()
                    if self.comfy is None or self.comfy.proc.poll() is not None:
                        self.stop(hard=True)
                        self.start()
        finally:
            wp = self.comfy.worker_processes() if self.comfy else None
            self.stop()
            if wp:
                self.report["worker_processes_at_end"] = wp
                self.report["left_after_comfy_stop"] = wait_gone(wp["launchers"] + wp["interpreters"] + wp["others"], 30)
            time.sleep(3)
            self.report["gpu_idle_mib_after"] = gpu_used_mib()
            self.report["memory_after"] = {k: MEM.snapshot().get(k) for k in ("commit_pct", "commit_gib", "available_gib")}
            self.report["hashes"] = {k: sorted(set(v)) for k, v in self.hashes.items()}
            self.report["official_reference"] = {"|".join(k): v for k, v in OFFICIAL_REFERENCE.items()}
            self.save()
        self.summary()
        return 0

    def summary(self) -> None:
        lines = []
        for key, vals in sorted(self.hashes.items()):
            lines.append(f"{key}: {len(vals)} runs, {len(set(vals))} distinct frame hash(es)")
        for (img, aud, dec), ref in OFFICIAL_REFERENCE.items():
            got = {k: v for k, v in self.hashes.items() if k.startswith(f"{img}|{aud}|{dec}|")}
            for k, v in got.items():
                lines.append(f"official reference {img}/{aud}/{dec} vs {k}: {'match' if set(v) == {ref} else 'DIFFERENT'}")
        st = self.report["steps"]
        for name in ("oneshot", "persistent"):
            if name not in st:
                continue
            for fx, runs in st[name].items():
                if name == "oneshot":
                    walls = [r["wall_s"] for r in runs if r.get("status") == "success"]
                    if walls:
                        lines.append(f"oneshot {fx}: n={len(walls)} median {statistics.median(walls):.2f}s range {min(walls):.2f}-{max(walls):.2f}")
                elif isinstance(runs, dict) and runs.get("warm"):
                    walls = [r["wall_s"] for r in runs["warm"] if r.get("status") == "success"]
                    first = runs["first"]["wall_s"]
                    if walls:
                        lines.append(
                            f"persistent {fx}: first {first:.2f}s, warm n={len(walls)} median {statistics.median(walls):.2f}s range {min(walls):.2f}-{max(walls):.2f}"
                        )
        # every generation attempt and its outcome (worker status / unload prompts are not counted)
        counts: dict = {}

        def walk(obj):
            if isinstance(obj, dict):
                if (
                    "outcome" in obj
                    and "job" in obj
                    or obj.get("outcome") in ("admission_refused", "memory_stopped", "timeout", "runtime_failed", "user_cancelled")
                ):
                    yield obj
                for v in obj.values():
                    yield from walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from walk(v)

        seen = set()
        for rec in walk(st):
            if rec["prompt_id"] in seen or rec.get("worker_node") is not None:
                continue
            seen.add(rec["prompt_id"])
            counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
        lines.append(f"attempts: {sum(counts.values())} {counts}")
        self.report["attempt_outcomes"] = counts
        for rec in (st.get("lowmem") or {}).get("jobs", []):
            j = rec.get("job") or {}
            inp = rec.get("input") or {}
            ref = V020_WAN_REFERENCE.get((inp.get("image"), inp.get("audio")))
            h = j.get("frames_sha256")
            cmp = "no v0.2.0 reference" if ref is None else ("= v0.2.0 wan_vae" if h == ref else "DIFFERENT from v0.2.0 wan_vae")
            lines.append(
                f"{rec['name']}: {rec['outcome']} {rec['wall_s']:.1f}s worker={j.get('worker_instance_id')} new={j.get('started_new_worker')} frames={str(h)[:12]} {cmp if h else ''}"
            )
        self.save()
        text = "\n".join(lines)
        (self.workdir / "summary.txt").write_text(text + "\n", encoding="utf-8")
        print(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", type=Path, required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--steps", default="status0,oneshot,persistent")
    ap.add_argument("--legacy", action="store_true", help="the installed node is v0.1 (no persistent backend, no Worker node)")
    ap.add_argument("--label", default="")
    ap.add_argument("--oneshot-reps", type=int, default=3)
    ap.add_argument("--warm-reps", type=int, default=5)
    ap.add_argument("--failure-decoder", default="lite_tae", choices=("lite_tae", "wan_vae"), help="decoder of the failure / lifecycle steps")
    ap.add_argument("--base-directory", default="", help="ComfyUI --base-directory (custom_nodes etc. outside the ComfyUI checkout)")
    ap.add_argument("--shipped-decoder", default="", choices=("", "lite_tae", "wan_vae"), help="'shipped' step: replace the workflows' decoder (recorded)")
    ap.add_argument("--bench", default="", help="benchmark fixtures as IMG:AUDIO,... (default A:a_short,B:b_long)")
    ap.add_argument("--shipped-dir", default=str(REPO), help="installed package folder whose workflows/api the 'shipped' step queues")
    a = ap.parse_args()
    return Runner(a).main()


if __name__ == "__main__":
    sys.exit(main())
