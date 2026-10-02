"""Maintainer end-to-end test through ComfyUI's HTTP API with a real LeapTalk runtime (GPU, weights).

Starts its own ComfyUI (127.0.0.1, free port, --cpu: ComfyUI itself does no GPU work), queues real
workflows and checks the results. Never touches an existing ComfyUI instance.

    python scripts/gpu_e2e.py --comfyui <ComfyUI dir> --python <ComfyUI python> --config <leaptalk.runtimes.json>
        --runtime <runtime_id> --workdir <empty dir> --image portrait.png --audio speech.wav
        [--audio2 long.wav] [--steps generate,long,doctor,cancel,timeout,error,crash]

Steps
  generate  Load Image + Load Audio -> LeapTalk Generate -> Save Video; MP4 decoded (frames, audio stream)
  long      the same with --audio2 (e.g. ~30 s)
  doctor    LeapTalk Doctor reports ok
  cancel    /interrupt during generation -> execution_interrupted, runner tree gone, GPU back to idle
  timeout   runtime with timeout_minutes=1 and a long clip -> error, nothing left
  error     runtime with a wrong model folder -> error with the runner's message, nothing left
  crash     ComfyUI killed during generation -> runner tree gone (Windows job object / stdin EOF)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path


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


def pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def workflow(runtime_id: str, image: str, audio: str, prefix: str, decoder: str = "lite_tae", guidance: float = 1.0) -> dict:
    return {
        "1": {"class_type": "LeapTalkRuntime", "inputs": {"runtime_id": runtime_id}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "4": {
            "class_type": "LeapTalkGenerate",
            "inputs": {"runtime": ["1", 0], "image": ["2", 0], "audio": ["3", 0], "decoder": decoder, "audio_guidance": guidance, "preview": True},
        },
        "5": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0], "filename_prefix": prefix, "format": "mp4", "format.codec": "h264"}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["4", 1]}},
    }


def doctor_workflow(runtime_id: str) -> dict:
    return {
        "1": {"class_type": "LeapTalkDoctor", "inputs": {"runtime_id": runtime_id, "verify_sha256": False}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
    }


class Comfy:
    def __init__(self, comfyui: Path, python: str, config: Path, workdir: Path):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.workdir = workdir
        env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY"))}
        env.update(COMFYUI_LEAPTALK_CONFIG=str(config), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        for sub in ("output", "temp", "user", "input"):
            (workdir / sub).mkdir(exist_ok=True)
        self.log = open(workdir / "comfyui.log", "ab")  # noqa: SIM115
        argv = [python, str(comfyui / "main.py"), "--listen", "127.0.0.1", "--port", str(self.port), "--cpu", "--disable-auto-launch"]
        argv += [
            "--output-directory",
            str(workdir / "output"),
            "--temp-directory",
            str(workdir / "temp"),
            "--user-directory",
            str(workdir / "user"),
            "--input-directory",
            str(workdir / "input"),
        ]
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(argv, cwd=str(comfyui), stdout=self.log, stderr=subprocess.STDOUT, env=env, creationflags=flags)  # noqa: S603
        for _ in range(240):
            try:
                http(self.base, "/system_stats", timeout=2)
                return
            except Exception:  # noqa: BLE001
                if self.proc.poll() is not None:
                    raise RuntimeError("ComfyUI exited during start-up; see comfyui.log") from None
                time.sleep(1)
        raise RuntimeError("ComfyUI did not start")

    def queue(self, prompt: dict) -> str:
        pid = str(uuid.uuid4())
        res = http(self.base, "/prompt", {"prompt": prompt, "client_id": "leaptalk-e2e", "prompt_id": pid})
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
            time.sleep(1)
        raise TimeoutError(f"prompt {pid} did not finish in {timeout} s")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()

    def kill_hard(self) -> None:
        self.proc.kill()
        self.proc.wait(30)
        self.log.close()


def latest_job_dir(workdir: Path) -> Path | None:
    # ComfyUI uses <--temp-directory>/temp as its temp folder
    jobs = sorted((workdir / "temp" / "temp" / "leaptalk").glob("job-*"), key=lambda p: p.stat().st_mtime)
    return jobs[-1] if jobs else None


def wait_new_job_with_progress(workdir: Path, before: set, timeout: float = 300) -> Path | None:
    """Wait for a job directory created after ``before`` whose runner has reported chunk progress."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        root = workdir / "temp" / "temp" / "leaptalk"
        for jd in sorted(root.glob("job-*")) if root.is_dir() else []:
            if jd.name not in before and (jd / "events.jsonl").is_file() and '"progress"' in (jd / "events.jsonl").read_text(encoding="utf-8"):
                return jd
        time.sleep(0.3)
    return None


def job_names(workdir: Path) -> set:
    root = workdir / "temp" / "temp" / "leaptalk"
    return {p.name for p in root.glob("job-*")} if root.is_dir() else set()


def runner_pid(job_dir: Path) -> int | None:
    try:
        for line in (job_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev.get("event") == "started":
                return int(ev["pid"])
    except (OSError, ValueError):
        return None
    return None


def decode(path: Path) -> dict:
    import av

    with av.open(str(path)) as c:
        v = [s for s in c.streams if s.type == "video"]
        a = [s for s in c.streams if s.type == "audio"]
        info = {"video_streams": len(v), "audio_streams": len(a)}
        if v:
            info.update(width=v[0].codec_context.width, height=v[0].codec_context.height, rate=float(v[0].average_rate), codec=v[0].codec_context.name)
        if a:
            info.update(audio_codec=a[0].codec_context.name, audio_rate=a[0].codec_context.sample_rate)
    with av.open(str(path)) as c:
        info["frames"] = sum(1 for _ in c.decode(video=0))
    if a:
        with av.open(str(path)) as c:
            info["audio_seconds"] = round(sum(fr.samples for fr in c.decode(audio=0)) / info["audio_rate"], 4)
    return info


def wait_gone(pids: list[int], seconds: float = 30) -> list[int]:
    t = time.time()
    while time.time() - t < seconds:
        alive = [p for p in pids if p and pid_alive(p)]
        if not alive:
            return []
        time.sleep(0.5)
    return [p for p in pids if p and pid_alive(p)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", type=Path, required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--audio2", type=Path)
    ap.add_argument("--image2", type=Path)
    ap.add_argument("--steps", default="generate,doctor,cancel,timeout,error,crash")
    ap.add_argument("--cases", type=Path, help="JSON list of extra generations: {name, image, audio, decoder?, guidance?} (step 'cases')")
    a = ap.parse_args()
    steps = a.steps.split(",")
    a.workdir.mkdir(parents=True, exist_ok=True)
    report: dict = {"steps": {}, "gpu_idle_mib_before": gpu_used_mib()}
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    base_rt = cfg["runtimes"][a.runtime]
    test_cfg = a.workdir / "leaptalk.runtimes.e2e.json"
    rts = {a.runtime: base_rt}
    rts["e2e-timeout"] = {**base_rt, "timeout_minutes": 1}
    rts["e2e-missing-model"] = {**base_rt, "models": {**base_rt["models"], "leaptalk_dir": str(a.workdir / "no-such-leaptalk-dir")}}
    test_cfg.write_text(json.dumps({"schema_version": 1, "runtimes": rts}, indent=1), encoding="utf-8")
    comfy = Comfy(a.comfyui, a.python, test_cfg, a.workdir)
    cases = json.loads(a.cases.read_text(encoding="utf-8")) if a.cases else []
    for src in [a.image, a.audio, a.audio2, a.image2] + [Path(c[k]) for c in cases for k in ("image", "audio")]:
        if src:
            shutil.copyfile(src, a.workdir / "input" / src.name)
    report["port"] = comfy.port
    try:
        info = http(comfy.base, "/object_info")
        report["nodes_registered"] = {n: n in info for n in ("LeapTalkRuntime", "LeapTalkGenerate", "LeapTalkDoctor")}
        report["runtime_choices"] = info["LeapTalkRuntime"]["input"]["required"]["runtime_id"][0]

        def run_generate(name: str, image: Path, audio: Path, prefix: str, **kw):
            t = time.time()
            pid = comfy.queue(workflow(a.runtime, image.name, audio.name, prefix, **kw))
            h = comfy.wait(pid, 3600)
            wall = round(time.time() - t, 2)
            st = h["status"]["status_str"]
            out = {"status": st, "wall_s": wall}
            if st == "success":
                vids = []
                for node in h["outputs"].values():
                    for items in node.values():
                        for item in items if isinstance(items, list) else []:
                            if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4"):
                                vids.append(a.workdir / "output" / item.get("subfolder", "") / item["filename"])
                out["saved"] = [str(p.relative_to(a.workdir)) for p in vids]
                out["decoded"] = [decode(p) for p in vids]
                text = (h["outputs"].get("6") or {}).get("text", [None])[0]
                rep = json.loads(text) if text else {}
                out["report_excerpt"] = {
                    k: rep.get(k)
                    for k in (
                        "decoder",
                        "chunks",
                        "output_frames",
                        "generated_frames",
                        "trimmed_tail_frames",
                        "forwards",
                        "audio",
                        "timings",
                        "memory",
                        "weights",
                        "wall_seconds",
                        "wall_seconds_per_audio_second",
                        "process_seconds",
                    )
                }
                exp = rep.get("output_frames")
                out["checks"] = {
                    "one_video": len(vids) == 1,
                    "has_audio": bool(vids) and out["decoded"][0]["audio_streams"] == 1,
                    "frames_match_report": bool(vids) and out["decoded"][0]["frames"] == exp,
                    "512x512_25fps": bool(vids) and (out["decoded"][0]["width"], out["decoded"][0]["height"], out["decoded"][0]["rate"]) == (512, 512, 25.0),
                    "lora_applied": (rep.get("weights") or {}).get("lora", {}).get("applied_tensors") == 480,
                }
            else:
                out["messages"] = [m[0] for m in h["status"].get("messages", [])]
            report["steps"][name] = out

        if "generate" in steps:
            run_generate("generate", a.image, a.audio, "leaptalk_e2e/short")
        if "long" in steps and a.audio2:
            run_generate("long", a.image2 or a.image, a.audio2, "leaptalk_e2e/long")
        if "cases" in steps:
            for c in cases:
                run_generate(
                    c["name"],
                    Path(c["image"]),
                    Path(c["audio"]),
                    f"leaptalk_e2e/{c['name']}",
                    decoder=c.get("decoder", "lite_tae"),
                    guidance=c.get("guidance", 1.0),
                )
        if "doctor" in steps:
            pid = comfy.queue(doctor_workflow(a.runtime))
            h = comfy.wait(pid, 1200)
            text = (h["outputs"].get("2") or {}).get("text", ["{}"])[0]
            rep = json.loads(text)
            report["steps"]["doctor"] = {
                "status": h["status"]["status_str"],
                "overall": rep.get("overall"),
                "checks": [[c["name"], c["status"]] for c in rep.get("checks", [])],
            }
        for name, rid, audio, expect in (
            ("cancel", a.runtime, a.audio2 or a.audio, "interrupted"),
            ("timeout", "e2e-timeout", a.audio2 or a.audio, "error"),
            ("error", "e2e-missing-model", a.audio, "error"),
        ):
            if name not in steps:
                continue
            t = time.time()
            before = job_names(a.workdir)
            pid = comfy.queue(workflow(rid, a.image.name, audio.name, f"leaptalk_e2e/{name}"))
            if name == "cancel":
                # interrupt while chunks of *this* job are being generated
                if wait_new_job_with_progress(a.workdir, before) is None:
                    raise RuntimeError("cancel step: the job never reported progress")
                http(comfy.base, "/interrupt", {})
            h = comfy.wait(pid, 600)
            jd = latest_job_dir(a.workdir)
            rpid = runner_pid(jd) if jd else None
            left = wait_gone([rpid] if rpid else [], 30)
            msgs = [m[0] for m in h["status"].get("messages", [])]
            err = next((m[1].get("exception_message") for m in h["status"].get("messages", []) if m[0] == "execution_error"), None)
            stop_rep = None
            if jd and (jd / "stop_report.json").is_file():
                stop_rep = json.loads((jd / "stop_report.json").read_text(encoding="utf-8"))
            time.sleep(3)
            g = gpu_used_mib()
            report["steps"][name] = {
                "status": h["status"]["status_str"],
                "messages": msgs,
                "error_excerpt": (err or "")[:300] or None,
                "wall_s": round(time.time() - t, 1),
                "job_dir": jd.name if jd else None,
                "stop_report": stop_rep,
                "runner_pid_alive_after": bool(left),
                "gpu_used_mib_after": g,
                "expected": expect,
            }
        if "crash" in steps:
            before = job_names(a.workdir)
            pid = comfy.queue(workflow(a.runtime, a.image.name, (a.audio2 or a.audio).name, "leaptalk_e2e/crash"))
            jd = wait_new_job_with_progress(a.workdir, before)
            if jd is None:
                raise RuntimeError("crash step: the job never reported progress")
            rpid = runner_pid(jd)
            import psutil

            tree = []
            if rpid and pid_alive(rpid):
                p = psutil.Process(rpid)
                tree = [rpid] + [c.pid for c in p.children(recursive=True)]
                par = p.parent()  # the venv launcher (python.exe trampoline) on Windows
                if par is not None and par.pid != comfy.proc.pid:
                    tree.append(par.pid)
            t = time.time()
            comfy.kill_hard()  # simulate a ComfyUI crash: no cleanup code runs in ComfyUI
            left = wait_gone(tree, 30)
            time.sleep(3)
            report["steps"]["crash"] = {
                "job_dir": jd.name,
                "runner_pid": rpid,
                "runner_tree_before": len(tree),
                "left_after": left,
                "seconds_to_clean": round(time.time() - t, 1),
                "gpu_used_mib_after": gpu_used_mib(),
            }
            comfy = None
    finally:
        if comfy is not None:
            comfy.stop()
        report["gpu_idle_mib_after"] = gpu_used_mib()
        (a.workdir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1)[:12000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
