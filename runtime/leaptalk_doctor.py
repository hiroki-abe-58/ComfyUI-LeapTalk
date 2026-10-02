"""LeapTalk runtime doctor: checks a runtime without generating a video.

Usage: python leaptalk_doctor.py [--watch-stdin] <job_dir>/request.json  -> writes doctor.json
Checks: pinned upstream files, model files (size; sha256 on request), Python packages, CUDA device,
a small scaled-dot-product-attention probe, the attention backend LeapTalk will pick, and ffmpeg
(libx264 and AAC encoders).
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leaptalk_job as lj  # noqa: E402

REQUEST_KEYS = {"schema_version", "job_id", "upstream_dir", "models", "ffmpeg", "env", "verify_sha256"}
PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "librosa",
    "soundfile",
    "einops",
    "loguru",
    "xfuser",
    "mediapipe",
    "pyloudnorm",
    "av",
    "numpy",
    "pillow",
)


def load_request(path: Path) -> dict:
    if path.name != "request.json" or path.stat().st_size > 64 * 1024:
        raise lj.JobError("expected a small request.json")
    req = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(req, dict) or set(req) != REQUEST_KEYS:
        raise lj.JobError(f"request keys must be {sorted(REQUEST_KEYS)}")
    if req["schema_version"] != lj.SCHEMA_VERSION or not isinstance(req["job_id"], str) or path.parent.name != req["job_id"]:
        raise lj.JobError("invalid request")
    lj._abs_path(req["upstream_dir"], "upstream_dir")
    lj._abs_path(req["ffmpeg"], "ffmpeg")
    if not isinstance(req["models"], dict) or set(req["models"]) != set(lj.MODEL_FILES):
        raise lj.JobError("invalid models")
    for k, v in req["models"].items():
        lj._abs_path(v, f"models.{k}")
    env = req["env"]
    if not isinstance(env, dict) or any(k not in lj.ENV_KEYS for k in env) or not all(isinstance(v, str) for v in env.values()):
        raise lj.JobError("invalid env")
    if not isinstance(req["verify_sha256"], bool):
        raise lj.JobError("verify_sha256 must be a boolean")
    return req


def run(req: dict, job_dir: Path) -> dict:
    checks = []

    def add(name, status, detail=""):
        checks.append({"name": name, "status": status, "detail": detail})

    up = Path(req["upstream_dir"])
    try:
        lj.check_upstream(up)
        add("upstream_files", "ok", f"{len(lj.UPSTREAM_FILES)} files match LeapTalk {lj.UPSTREAM_COMMIT[:7]}")
    except lj.JobError as exc:
        add("upstream_files", "fail", str(exc))

    for key, files in lj.MODEL_FILES.items():
        for rel, size, digest in files:
            p = Path(req["models"][key]) / rel
            label = f"model:{key}/{rel}"
            optional = rel in lj.WANVAE_ONLY
            if not p.is_file():
                add(label, "warn" if optional else "fail", "missing" + (" (only needed for decoder wan_vae)" if optional else ""))
                continue
            if p.stat().st_size != size:
                add(label, "fail", f"size {p.stat().st_size}, expected {size}")
                continue
            if req["verify_sha256"]:
                h = lj._sha256_file(p)
                add(label, "ok" if h == digest else "fail", "sha256 ok" if h == digest else f"sha256 {h[:12]}..., expected {digest[:12]}...")
            else:
                add(label, "ok", "size ok")

    vers = {}
    for name in PACKAGES:
        try:
            vers[name] = md.version(name)
        except md.PackageNotFoundError:
            vers[name] = None
    missing = [k for k, v in vers.items() if v is None]
    add("python_packages", "fail" if missing else "ok", ("missing: " + ", ".join(missing)) if missing else json.dumps(vers))

    try:
        import torch

        if not torch.cuda.is_available():
            add("cuda", "fail", f"torch {torch.__version__}: CUDA not available")
        else:
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]}{cap[1]}"
            built = torch.cuda.get_arch_list()
            ok = arch in built
            add(
                "cuda",
                "ok" if ok else "fail",
                f"{torch.cuda.get_device_name(0)} {arch}, torch {torch.__version__} (CUDA {torch.version.cuda}), built for {built[-3:]}",
            )
            q = torch.randn(1, 12, 256, 128, device="cuda", dtype=torch.bfloat16)
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            add("sdpa_probe", "ok" if bool(torch.isfinite(out).all()) else "fail", "bf16 scaled_dot_product_attention on the GPU")
    except Exception as exc:  # noqa: BLE001
        add("cuda", "fail", f"{type(exc).__name__}: {exc}")

    backend = "torch_sdpa"
    for mod, name in (("sageattention", "sageattention"), ("flash_attn_interface", "flash_attn_3"), ("flash_attn", "flash_attn_2")):
        try:
            __import__(mod)
            backend = name
            break
        except ImportError:
            continue
    add("attention_backend", "ok", f"{backend} (LeapTalk picks the first available of sageattention, flash_attn_3, flash_attn_2, torch SDPA)")

    ff = req["ffmpeg"]
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, timeout=60, creationflags=flags)  # noqa: S603 - fixed argv
        text = enc.stdout.decode("utf-8", "replace")
        ver = subprocess.run([ff, "-hide_banner", "-version"], capture_output=True, timeout=60, creationflags=flags)  # noqa: S603
        first = ver.stdout.decode("utf-8", "replace").splitlines()[:1]
        have = {e: bool(re.search(rf"\s{e}\s", text)) for e in ("libx264", "aac")}
        add("ffmpeg", "ok" if all(have.values()) else "fail", f"{first[0] if first else '?'}; encoders {have}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        add("ffmpeg", "fail", f"{type(exc).__name__}: {exc}")

    order = {"ok": 0, "warn": 1, "fail": 2}
    overall = max((c["status"] for c in checks), key=lambda s: order[s])
    return {"overall": overall, "checks": checks, "upstream_commit": lj.UPSTREAM_COMMIT, "python": sys.version.split()[0]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-stdin", action="store_true")
    ap.add_argument("request")
    a = ap.parse_args(argv)
    path = Path(a.request).resolve()
    events = lj.Events(path.parent / "events.jsonl")
    events.emit("started", pid=os.getpid())
    try:
        req = load_request(path)
    except (lj.JobError, OSError, ValueError) as exc:
        events.emit("invalid_job", error=str(exc)[:1000])
        return lj.EXIT_INVALID
    lj.apply_env(req["env"], path.parent)
    t = time.time()
    report = run(req, path.parent)
    report["seconds"] = round(time.time() - t, 2)
    lj._write_json(path.parent / "doctor.json", report)
    events.emit("done", overall=report["overall"])
    return lj.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
