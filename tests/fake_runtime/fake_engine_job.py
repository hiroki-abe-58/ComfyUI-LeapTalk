"""Stand-in for runtime/leaptalk_job.py inside a persistent-worker session (CPU tests only).

The test fixture copies this file as ``leaptalk_job.py`` next to the real ``leaptalk_worker.py``. It
re-exports the real job module (job validation, Events, chunk plan, result fields) and replaces only
``Engine`` with a fake that needs no GPU and no weights. Behaviour is chosen by the last component of
``upstream_dir``: ok | slow | hang | init_fail | slow_init. A file named CRASH_NEXT in the upstream
folder makes the next job kill the worker process (and is removed first).
"""

import hashlib
import importlib.util
import os
import time
import uuid
from pathlib import Path

REAL = os.environ.get("LEAPTALK_TEST_REAL_JOB_MODULE") or r"__REAL_JOB_MODULE__"
_spec = importlib.util.spec_from_file_location("_real_leaptalk_job", REAL)
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)
globals().update({k: getattr(_real, k) for k in dir(_real) if not k.startswith("__")})


class Engine:
    def __init__(self, settings, on_step=None, check=None):
        self.mode = Path(settings["upstream_dir"]).name
        self.upstream_dir = Path(settings["upstream_dir"])
        self.settings = settings
        if on_step:
            on_step("pipeline")
        if self.mode == "slow_init":
            time.sleep(3)
        if self.mode == "init_fail":
            raise RuntimeError("simulated load failure (CUDA error)")
        self.engine_instance_id = uuid.uuid4().hex
        self.initialization_id = f"init-{self.engine_instance_id[:12]}"
        self.counters = {"init": 1, "lora_merge": 1, "audio_proj_load": 1, "jobs_started": 0, "jobs_ok": 0, "jobs_failed": 0}
        self.attention_backend = "fake"
        self.upstream = {"commit": "fake"}
        self.weights = {"lora": {"applied_tensors": 480}}
        self.fingerprint = {"fake": "0" * 64}
        self.init_timings = {"import_s": 0.0}
        self.resident = {"cuda_reserved_bytes": 0}
        self.env_seen = sorted(os.environ)

    def describe(self):
        return {
            "engine_instance_id": self.engine_instance_id,
            "initialization_id": self.initialization_id,
            "counters": dict(self.counters),
            "init_timings": self.init_timings,
            "env_keys": self.env_seen,
            "pid": os.getpid(),
        }

    def _memory_now(self):
        return {"cuda_reserved_bytes": 0, "process_rss_bytes": None}

    def verify_unchanged(self):
        return {"fake": True}

    def generate(self, job, job_dir, events, ctl):
        import av
        import numpy as np
        from PIL import Image

        self.counters["jobs_started"] += 1
        crash = self.upstream_dir / "CRASH_NEXT"
        if crash.exists():
            crash.unlink()
            os._exit(9)
        img = np.asarray(Image.open(job_dir / "input.png").convert("RGB"))
        color = [int(c) for c in img.reshape(-1, 3).mean(axis=0)]
        with av.open(str(job_dir / "input.wav")) as src:
            s = src.streams.audio[0]
            samples = sum(fr.samples for fr in src.decode(audio=0))
            sr = s.codec_context.sample_rate
            channels = s.codec_context.channels
        seconds = samples / sr
        plan = chunk_plan(int(round(seconds * 16000)), seconds)  # noqa: F821 - from the real module
        frame_hashes = []
        for i in range(plan["chunks"]):
            ctl.check()
            if self.mode == "hang":
                while True:
                    ctl.check()
                    time.sleep(0.05)
            if self.mode == "slow":
                time.sleep(0.6)
            frame = np.zeros((512, 512, 3), np.uint8)
            frame[:] = color
            frame[0, 0] = [i % 256, 0, 0]
            frame_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            if job["preview"]:
                Image.fromarray(frame).save(job_dir / "preview.tmp.jpg", quality=80)
                (job_dir / "preview.tmp.jpg").replace(job_dir / "preview.jpg")
            events.emit("progress", chunk=i + 1, chunks=plan["chunks"], frames=min((i + 1) * 28, plan["out_frames"]), preview=job["preview"])
        out = job_dir / "output.mp4"
        layout = {1: "mono", 2: "stereo"}.get(channels, f"{channels}c")
        with av.open(str(job_dir / "input.wav")) as src, av.open(str(out), "w") as c:
            try:
                vs = c.add_stream("libx264", rate=25)
            except Exception:  # noqa: BLE001
                vs = c.add_stream("mpeg4", rate=25)
            vs.width, vs.height, vs.pix_fmt = 512, 512, "yuv420p"
            as_ = c.add_stream("aac", rate=sr, layout=layout)
            for _ in range(plan["out_frames"]):
                fr = np.zeros((512, 512, 3), np.uint8)
                fr[:] = color
                for p in vs.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                    c.mux(p)
            for p in vs.encode():
                c.mux(p)
            res = av.AudioResampler(format="fltp", layout=layout, rate=sr)
            for fr in src.decode(audio=0):
                for f2 in res.resample(fr):
                    for p in as_.encode(f2):
                        c.mux(p)
            for f2 in res.resample(None):
                for p in as_.encode(f2):
                    c.mux(p)
            for p in as_.encode():
                c.mux(p)
        self.counters["jobs_ok"] += 1
        self.last_job_memory = {"resident_before_job": {"cuda_reserved_bytes": 0}, "phases": {"chunk_loop": {"peak_reserved_bytes": 2 * 1024**3}}}
        return {
            "status": "ok",
            "output": "output.mp4",
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "video": {"frames": plan["out_frames"], "fps": 25.0, "width": 512, "height": 512, "codec": "h264"},
            "audio": {"input_sample_rate": sr, "input_channels": channels, "input_seconds": round(seconds, 4), "muxed": {"codec": "aac"}},
            "chunks": plan["chunks"],
            "output_frames": plan["out_frames"],
            "frames_sha256": hashlib.sha256("".join(frame_hashes).encode()).hexdigest(),
            "portrait_color": color,
            "env_keys": sorted(os.environ),
            "timings": {"job_total_s": 0.0},
        }
