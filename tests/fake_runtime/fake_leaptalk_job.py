"""Stand-in for runtime/leaptalk_job.py in CPU tests (no GPU, no weights, no upstream code).

It validates the job with the real runner's ``load_job`` and ``chunk_plan``, then, depending on the
last component of ``upstream_dir`` (``.../ok``, ``.../fail``, ``.../hang``, ``.../slow``):
- ok:   writes real progress events and previews, and a real MP4 (H.264 + AAC from input.wav)
        with ceil(seconds * 25) frames, plus result.json;
- fail: reports a runtime failure (exit 3);
- hang: starts a grandchild process, then waits forever (until cancelled or killed);
- slow: like ok, but slowly (for timeout tests).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("leaptalk_job_real", REPO / "runtime" / "leaptalk_job.py")
lj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lj)


def write_video(job_dir: Path, frames: int) -> dict:
    import av
    import numpy as np

    out = job_dir / "output.mp4"
    with av.open(str(job_dir / "input.wav")) as src:
        a_in = src.streams.audio[0]
        rate = a_in.codec_context.sample_rate
        channels = a_in.codec_context.channels
        layout = {1: "mono", 2: "stereo"}.get(channels, f"{channels}c")  # float WAV without a channel mask
        with av.open(str(out), "w") as c:
            try:
                vs = c.add_stream("libx264", rate=25)
            except Exception:  # noqa: BLE001 - wheels without x264
                vs = c.add_stream("mpeg4", rate=25)
            vs.width, vs.height, vs.pix_fmt = 512, 512, "yuv420p"
            as_ = c.add_stream("aac", rate=rate, layout=layout)
            for i in range(frames):
                arr = np.full((512, 512, 3), (i * 7) % 255, dtype=np.uint8)
                for p in vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    c.mux(p)
            for p in vs.encode():
                c.mux(p)
            res = av.AudioResampler(format="fltp", layout=layout, rate=rate)
            for fr in src.decode(audio=0):
                for f2 in res.resample(fr):
                    for p in as_.encode(f2):
                        c.mux(p)
            for f2 in res.resample(None):
                for p in as_.encode(f2):
                    c.mux(p)
            for p in as_.encode():
                c.mux(p)
    return {"codec": "aac", "sample_rate": rate, "channels": channels}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-stdin", action="store_true")
    ap.add_argument("job")
    a = ap.parse_args()
    job_path = Path(a.job).resolve()
    job_dir = job_path.parent
    ev = lj.Events(job_dir / "events.jsonl")
    ev.emit("started", pid=os.getpid())
    try:
        job = lj.load_job(job_path)
    except (lj.JobError, OSError, ValueError) as exc:
        ev.emit("invalid_job", error=str(exc))
        return lj.EXIT_INVALID
    ctl = lj.Control(job_dir, a.watch_stdin)
    mode = Path(job["upstream_dir"]).name
    try:
        if mode == "fail":
            raise RuntimeError("simulated runtime failure (CUDA out of memory)")
        if mode == "hang":
            gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])  # noqa: S603
            (job_dir / "grandchild.pid").write_text(str(gc.pid), encoding="utf-8")
            ev.emit("progress", chunk=1, chunks=99, frames=28, preview=False)
            while True:
                ctl.check()
                time.sleep(0.1)
        import av

        with av.open(str(job_dir / "input.wav")) as src:
            s = src.streams.audio[0]
            samples = sum(fr.samples for fr in src.decode(audio=0))
            sr = s.codec_context.sample_rate
            channels = s.codec_context.channels
        seconds = samples / sr
        if seconds > job["max_audio_seconds"]:
            raise lj.JobError(f"the audio is {seconds:.1f} s long; this runtime accepts up to {job['max_audio_seconds']} s")
        plan = lj.chunk_plan(int(round(seconds * 16000)), seconds)
        from PIL import Image

        for i in range(plan["chunks"]):
            ctl.check()
            if job["preview"]:
                Image.new("RGB", (512, 512), (i * 20 % 255, 80, 160)).save(job_dir / "preview.tmp.jpg", quality=80)
                (job_dir / "preview.tmp.jpg").replace(job_dir / "preview.jpg")
            ev.emit("progress", chunk=i + 1, chunks=plan["chunks"], frames=min((i + 1) * 28, plan["out_frames"]), preview=job["preview"])
            time.sleep(1.5 if mode == "slow" else 0.01)
        muxed = write_video(job_dir, plan["out_frames"])
        out = job_dir / "output.mp4"
        result = {
            "job_id": job["job_id"],
            "status": "ok",
            "output": "output.mp4",
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "video": {"frames": plan["out_frames"], "fps": 25.0, "width": 512, "height": 512, "codec": "h264"},
            "audio": {"input_sample_rate": sr, "input_channels": channels, "input_seconds": round(seconds, 4), "muxed": muxed},
            "chunks": plan["chunks"],
            "output_frames": plan["out_frames"],
            "generated_frames": plan["chunks"] * plan["slice_len"],
            "forwards": {"total": plan["chunks"], "conditional": plan["chunks"], "unconditional": 0},
            "decoder": job["decoder"],
            "audio_guidance": job["audio_guidance"],
            "weights": {"lora": {"applied_tensors": 480}},
            "fake_runtime": True,
        }
        lj._write_json(job_dir / "result.json", result)
        ev.emit("done")
        return lj.EXIT_OK
    except lj.Cancelled as exc:
        ev.emit("cancelled", reason=str(exc))
        return lj.EXIT_CANCELLED
    except lj.JobError as exc:
        ev.emit("failed", error_type="JobError", error=str(exc))
        return lj.EXIT_INVALID
    except Exception as exc:  # noqa: BLE001
        ev.emit("failed", error_type=type(exc).__name__, error=str(exc))
        return lj.EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
