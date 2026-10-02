"""LeapTalk runtime runner: one job = one talking-head video with the official LeapTalk recipe.

Runs with the runtime's own Python (a separate venv with the pinned upstream code and weights),
never inside the ComfyUI process. Usage:

    python leaptalk_job.py [--watch-stdin] <job_dir>/job.json

What it does, in the order of the official ``inference.py`` (LeapTalk 5d8fef8, stream mode):

1. imports the pinned upstream checkout (file hashes checked first) and builds ``FlashHeadPipeline``
   with the SoulX-FlashHead-1_3B ``Model_Pro`` base, the selected decoder (LeapTalk Lite TAE or
   WanVAE) and wav2vec2-base-960h;
2. loads the LeapTalk LoRA with PEFT (every checkpoint tensor must be present in the model and
   loaded with its exact value), merges it, and loads ``audio_proj_step_10400.pt`` with
   ``strict=True`` and ``weights_only=True``;
3. encodes the reference image (resize/center-crop to 512x512) into the static anchor latent,
   builds ``ViBTScheduler`` with the official 1-step timesteps (shift 5);
4. loads the speech with ``librosa.load(sr=16000, mono=True)``, pads it like the official script
   and runs the official stream loop chunk by chunk (8 s audio ring buffer, wav2vec2 embeddings,
   ``_bridge_sample_one_chunk``, decode, color correction, VAE round-trip history update, drop of
   the 5 overlap frames), using the upstream helper functions unchanged.

Differences from the official script (they do not change the generated frames):
- frames are converted to uint8 exactly as the official writer does (bf16 -> fp16 -> uint8 cast)
  but streamed to ffmpeg chunk by chunk instead of being kept in memory until the end;
- the video is trimmed to ceil(audio_seconds * 25) frames and muxed with the *original* input
  audio as AAC, without ``-shortest`` (the official script muxes MP3 with ``-shortest`` and
  ignores ffmpeg errors); the result is decoded back and checked;
- the generator calls are counted, the job can be cancelled between chunks, and progress,
  previews and a JSON report are written to the job directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA_VERSION = 1
UPSTREAM_REPO = "https://github.com/zhangrongxiang/LeapTalk"
UPSTREAM_COMMIT = "5d8fef8dac3d1c9d3b3d5d9cceb6c76acc81e2c1"
# sha256 with CRLF -> LF normalisation (Windows git with core.autocrlf=true checks out CRLF)
UPSTREAM_FILES = {
    "inference.py": "b96ba8dd4d5c8e1e343ce486e34baac628611cdfaffbd8f213114604df879f9d",
    "flash_head/src/pipeline/flash_head_pipeline.py": "aba6760eb945d33dbfccbbd45093cdceac167fc0564221b9922c606b3dcd59d4",
    "flash_head/src/modules/flash_head_model.py": "1f837e73007e367a6b91d028b023f0583f1bffb7ac991c69f1b5041087f9c3fb",
    "flash_head/audio_analysis/wav2vec2.py": "c30cb3548889e52542d63e91b0301599f3dead5de97b7b69e144aa6232fa9476",
    "flash_head/audio_analysis/torch_utils.py": "615c93cf0cda6433e8aad599e95958a0c204e390f482156ec2941fd98c21e602",
    "flash_head/wan/modules/__init__.py": "c18715a0fdb38948da78cf1e8c65737c6cd978c51631ab5474f9d8b247ec0c14",
    "flash_head/wan/modules/tae.py": "91b8c4cf952e78311da73f134f887a7af5ccaa257b8c649909bb5d00e2a94963",
    "flash_head/wan/modules/vae.py": "7071d15350f30eed21a363c624e36c646ae8f7ec81303b83575dcf63bfbc49eb",
    "flash_head/utils/utils.py": "1b385b6c71eef4ded0875cf90a1552f639ac42182bb71de7ee32592257e8e094",
    "flash_head/utils/facecrop.py": "dc439a58b908a9693ab0192c0018efa0515ce6ff54322ba6db18d129d2a57a66",
    "flash_head/utils/cpu_face_handler.py": "975885324af1df32c29ec1d659d2f26f2f4dd5de15c6a57ea31e585dc34b221b",
    "flash_head/src/distributed/usp_device.py": "8cc913ea900d31af9532b8b3ca79eeb14806dfbbb15993fa3f554a349a3f43ac",
    "flash_head/configs/infer_params.yaml": "9c5111fa5ba2ebf20da09d3b6806e88ff952955fb82c6c3f74cefcc150bd7b71",
    "vibt/scheduler.py": "098ca288e01061ba73299c9ba4c7b8352ce74119937af73077b8142b2771ae8f",
}
# Model files used by this runner: (relative path, size in bytes, sha256). Revisions in docs/SETUP.md.
MODEL_FILES = {
    "soulx_dir": [
        ("Model_Pro/config.json", 353, "94b480d3af70e73989371ed0c106656c2c887201286b1210e835cd09e71e32b1"),
        ("Model_Pro/diffusion_pytorch_model.safetensors", 6030864656, "e47e61b9023ea1aac60c0c0fff077289bc6cca443c71907e5a76a411480af250"),
        ("VAE_Wan/Wan2.1_VAE.pth", 507609880, "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"),
    ],
    "wav2vec_dir": [
        ("config.json", 1596, "d3ec255c063d9f95057b553b19c20135b259875834a4fe9deb218a6be25b4cf3"),
        ("preprocessor_config.json", 159, "b225d617c025463b9e157e06afea8b90dc7078fc70b013c533328423e0486b4a"),
        ("model.safetensors", 377607901, "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a"),
    ],
    "leaptalk_dir": [
        ("lora/adapter_config.json", 1069, "46fd70a44d56fc5e92ca9dda65cabe0dc1b29f8de0cc6400dc55a21467e90ba7"),
        ("lora/adapter_model.safetensors", 188803712, "30fdb8ffb312b3abaad8f068fe2ce30a38f5124e30bd6e9a310cd0a620501cde"),
        ("audio_proj_step_10400.pt", 173651301, "40b9b53cdeedcad0b1839d51cfe4e122aaaa6bffafdcfed520fa29704a81b0c1"),
        ("taew2_1.pth", 22679486, "f986092baa4a124035dcf44ef59cd30e0bef91b2aa4d6a6e8e152e55e82ca30c"),
    ],
}
WANVAE_ONLY = {"VAE_Wan/Wan2.1_VAE.pth"}
TAE_ONLY = {"taew2_1.pth"}
DECODERS = ("lite_tae", "wan_vae")
# Official stream parameters (inference.py defaults / flash_head/configs/infer_params.yaml)
FPS = 25
SAMPLE_RATE = 16000
SIZE = 512
FRAME_NUM = 33
MOTION_FRAMES_LATENT_NUM = 2
CACHED_AUDIO_DURATION = 8
SHIFT_GAMMA = 5.0
NOISE_SCALE = 1.0
NUM_INFERENCE_STEPS = 1
COLOR_CORRECTION_STRENGTH = 1.0
SEED = 42  # official default; with 1 solver step the ViBT noise term is exactly 0, so the seed has no effect
MAX_AUDIO_SECONDS = 1800
MAX_INPUT_BYTES = {"input.png": 64 * 1024 * 1024, "input.wav": 2 * 1024 * 1024 * 1024}
ENV_KEYS = {"CUDA_VISIBLE_DEVICES", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"}
SECRET_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|CREDENTIAL|WANDB_API)", re.I)
JOB_KEYS = {
    "schema_version",
    "job_id",
    "upstream_dir",
    "models",
    "ffmpeg",
    "decoder",
    "audio_guidance",
    "max_audio_seconds",
    "video_crf",
    "preview",
    "env",
}
_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,80}")
EXIT_OK, EXIT_INVALID, EXIT_FAILED, EXIT_CANCELLED = 0, 2, 3, 4


class JobError(Exception):
    """The job file or its inputs are invalid (exit code 2)."""


class Cancelled(Exception):
    """A stop was requested (exit code 4)."""


# --------------------------------------------------------------------------- job validation


def _abs_path(value, what: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024 or any(c in value for c in "\0\n\r"):
        raise JobError(f"{what} must be a non-empty path")
    p = Path(value)
    if not p.is_absolute() or ".." in p.parts:
        raise JobError(f"{what} must be an absolute path without '..'")
    return value


def load_job(path: Path) -> dict:
    """Parse and validate job.json (the runtime does not trust the caller's validation)."""
    path = Path(path)
    if path.name != "job.json" or path.stat().st_size > 64 * 1024:
        raise JobError("expected a small job.json")
    job = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(job, dict):
        raise JobError("job.json must be an object")
    extra = set(job) - JOB_KEYS
    if extra:
        raise JobError(f"unknown job keys: {sorted(extra)}")
    missing = JOB_KEYS - set(job)
    if missing:
        raise JobError(f"missing job keys: {sorted(missing)}")
    if job["schema_version"] != SCHEMA_VERSION:
        raise JobError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(job["job_id"], str) or not _JOB_ID_RE.fullmatch(job["job_id"]):
        raise JobError("invalid job_id")
    if path.parent.name != job["job_id"]:
        raise JobError("job.json must be inside its own job directory")
    _abs_path(job["upstream_dir"], "upstream_dir")
    _abs_path(job["ffmpeg"], "ffmpeg")
    models = job["models"]
    if not isinstance(models, dict) or set(models) != set(MODEL_FILES):
        raise JobError(f"models must have exactly {sorted(MODEL_FILES)}")
    for k, v in models.items():
        _abs_path(v, f"models.{k}")
    if job["decoder"] not in DECODERS:
        raise JobError(f"decoder must be one of {DECODERS}")
    g = job["audio_guidance"]
    if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g) or not 1.0 <= g <= 4.0:
        raise JobError("audio_guidance must be a number in [1.0, 4.0]")
    mx = job["max_audio_seconds"]
    if isinstance(mx, bool) or not isinstance(mx, int) or not 1 <= mx <= MAX_AUDIO_SECONDS:
        raise JobError(f"max_audio_seconds must be an integer in [1, {MAX_AUDIO_SECONDS}]")
    crf = job["video_crf"]
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 40:
        raise JobError("video_crf must be an integer in [0, 40]")
    if not isinstance(job["preview"], bool):
        raise JobError("preview must be true or false")
    env = job["env"]
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise JobError("env must map strings to strings")
    bad = [k for k in env if k not in ENV_KEYS or SECRET_RE.search(k)]
    if bad:
        raise JobError(f"env keys not allowed: {bad}")
    for name, limit in MAX_INPUT_BYTES.items():
        f = path.parent / name
        if not f.is_file():
            raise JobError(f"{name} missing from the job directory")
        if f.stat().st_size > limit:
            raise JobError(f"{name} is too large")
    return job


# --------------------------------------------------------------------------- small helpers


class Events:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def emit(self, event: str, **kw) -> None:
        self.f.write(json.dumps({"event": event, "t": round(time.time(), 3), **kw}, ensure_ascii=False) + "\n")
        self.f.flush()


class Control:
    """Cooperative stop: a CANCEL file in the job directory, or stdin EOF when --watch-stdin is used."""

    def __init__(self, job_dir: Path, watch_stdin: bool):
        self.cancel_file = job_dir / "CANCEL"
        self.stdin_closed = threading.Event()
        if watch_stdin:
            threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self) -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except (OSError, ValueError):
            pass
        self.stdin_closed.set()

    def check(self) -> None:
        if self.stdin_closed.is_set():
            raise Cancelled("the parent process went away")
        if self.cancel_file.exists():
            raise Cancelled("cancel requested")


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def chunk_plan(samples_16k: int, input_seconds: float) -> dict:
    """Padding and chunking of the official stream mode, and the number of output frames kept.

    slice = 28 new frames per chunk (33 - 5 overlap) = 17920 samples at 16 kHz; a chunk window is
    33 frames = 21120 samples. The official script pads to a multiple of the slice, then to at least
    one window, then again to a multiple of the slice. Every chunk yields 28 new frames; frames
    after the end of the speech are dropped (ceil(seconds * 25) frames are kept).
    """
    motion_frames = (MOTION_FRAMES_LATENT_NUM - 1) * 4 + 1
    slice_len = FRAME_NUM - motion_frames
    slice_samples = slice_len * SAMPLE_RATE // FPS
    frame_samples = FRAME_NUM * SAMPLE_RATE // FPS
    n = int(samples_16k)
    if n % slice_samples:
        n += slice_samples - n % slice_samples
    if n < frame_samples:
        n = frame_samples
    if n % slice_samples:
        n += slice_samples - n % slice_samples
    chunks = n // slice_samples
    out_frames = int(math.ceil(input_seconds * FPS - 1e-9))
    return {"padded_samples": n, "chunks": chunks, "slice_len": slice_len, "motion_frames": motion_frames, "out_frames": out_frames}


def check_upstream(upstream_dir: Path) -> dict:
    bad = {}
    for rel, digest in UPSTREAM_FILES.items():
        p = upstream_dir / rel
        if not p.is_file():
            bad[rel] = "missing"
        elif _sha256_text(p) != digest:
            bad[rel] = "modified"
    if bad:
        raise JobError(f"upstream checkout does not match LeapTalk {UPSTREAM_COMMIT[:7]}: {bad}")
    return {"commit": UPSTREAM_COMMIT, "files_checked": len(UPSTREAM_FILES)}


def check_model_files(models: dict, decoder: str) -> list[str]:
    problems = []
    for key, files in MODEL_FILES.items():
        for rel, size, _digest in files:
            if decoder == "lite_tae" and rel in WANVAE_ONLY or decoder == "wan_vae" and rel in TAE_ONLY:
                continue
            p = Path(models[key]) / rel
            if not p.is_file():
                problems.append(f"{key}/{rel} missing")
            elif p.stat().st_size != size:
                problems.append(f"{key}/{rel} has size {p.stat().st_size}, expected {size}")
    return problems


def apply_env(env: dict, job_dir: Path) -> None:
    for k in list(os.environ):
        if SECRET_RE.search(k):
            del os.environ[k]  # never let credentials reach the model code
    os.environ.update(env)
    cache = job_dir / "tmp" / "cache"
    for k, sub in (("HF_HOME", "hf"), ("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "xdg")):
        os.environ.setdefault(k, str(cache / sub))  # nothing is downloaded; keep stray cache writes inside the job
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ["WANDB_MODE"] = "disabled"


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- the engine


class Engine:
    """Builds the official LeapTalk pipeline once and generates one video per call."""

    def __init__(self, job: dict, events: Events, ctl: Control):
        self.events = events
        self.ctl = ctl
        self.timings: dict = {}
        t = time.perf_counter()
        up = Path(job["upstream_dir"]).resolve()
        self.upstream = check_upstream(up)
        problems = check_model_files(job["models"], job["decoder"])
        if problems:
            raise JobError("model files: " + "; ".join(problems))
        if str(up) not in sys.path:
            sys.path.insert(0, str(up))
        import torch

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this runtime")
        import flash_head.src.modules.flash_head_model as fh_model_mod
        import flash_head.src.pipeline.flash_head_pipeline as fh_pipe_mod
        import inference as lt  # the official script as a module (its __main__ block does not run)

        for mod in (lt, fh_pipe_mod, fh_model_mod):
            if not Path(mod.__file__).resolve().is_relative_to(up):
                raise JobError(f"{mod.__name__} was imported from outside the pinned checkout")
        self.lt = lt
        self.timings["import_s"] = round(time.perf_counter() - t, 3)
        self.attention_backend = (
            "sageattention"
            if fh_model_mod.SAGE_ATTN_AVAILABLE
            else "flash_attn_3"
            if fh_model_mod.FLASH_ATTN_3_AVAILABLE
            else "flash_attn_2"
            if fh_model_mod.FLASH_ATTN_2_AVAILABLE
            else "torch_sdpa"
        )
        self.ctl.check()

        models = job["models"]
        lora_dir = os.path.join(models["leaptalk_dir"], "lora")
        audio_proj_path = os.path.join(models["leaptalk_dir"], "audio_proj_step_10400.pt")
        tae_path = os.path.join(models["leaptalk_dir"], "taew2_1.pth")
        self.decoder = job["decoder"]
        lite = self.decoder == "lite_tae"
        if lt._lora_checkpoint_needs_compiled_base(lora_dir):
            raise JobError("this LoRA was saved from a torch.compile()d model; only the published LeapTalk LoRA (plain keys) is supported")
        # Official: COMPILE_MODEL is only forced for compiled-base LoRA keys (not the case) and --compile off;
        # COMPILE_VAE = not lite. This runner keeps both off (eager); see docs/BENCHMARKS.md for the WanVAE note.
        fh_pipe_mod.COMPILE_MODEL = False
        fh_pipe_mod.COMPILE_VAE = False
        self.dtype = torch.bfloat16
        self.device = "cuda"
        self.memory_phases: dict = {}
        torch.cuda.reset_peak_memory_stats()

        events.emit("loading", step="pipeline")
        t = time.perf_counter()
        pipeline = fh_pipe_mod.FlashHeadPipeline(
            checkpoint_dir=models["soulx_dir"],
            model_type="pro",
            wav2vec_dir=models["wav2vec_dir"],
            device=self.device,
            param_dtype=self.dtype,
            use_usp=False,
            use_tae=lite,
            tae_path=tae_path if lite else None,
            tae_model_type="wan21",
        )
        self.timings["pipeline_load_s"] = round(time.perf_counter() - t, 3)
        self.mark_memory("pipeline_load", release_cache=True)
        self.ctl.check()

        events.emit("loading", step="lora")
        t = time.perf_counter()
        self.weights = {"lora": self._load_lora(pipeline, lora_dir)}
        self.timings["lora_load_merge_s"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        state = torch.load(audio_proj_path, map_location="cpu", weights_only=True)
        inner = lt._get_inner_flashhead_model(pipeline.model)
        before = inner.audio_proj.proj1.weight.detach().float().cpu().clone()
        inner.audio_proj.load_state_dict(state, strict=True)
        after = inner.audio_proj.proj1.weight.detach().float().cpu()
        exact = all(
            torch.equal(inner.audio_proj.state_dict()[k].detach().float().cpu(), v.to(inner.audio_proj.state_dict()[k].dtype).float()) for k, v in state.items()
        )
        if not exact:
            raise RuntimeError("audio_proj weights were not applied")
        self.weights["audio_proj"] = {
            "file": "audio_proj_step_10400.pt",
            "tensors": len(state),
            "strict": True,
            "values_applied": True,
            "proj1_rel_change_vs_base": round(float((after - before).norm() / before.norm()), 4),
        }
        self.timings["audio_proj_s"] = round(time.perf_counter() - t, 3)
        self.mark_memory("lora_merge_and_audio_proj", release_cache=True)
        self.pipeline = pipeline
        self.vae_stride_t = int(pipeline.config.vae_stride[0])
        self.weights["base"] = {"model": "SoulX-FlashHead-1_3B Model_Pro", "class": type(inner).__name__, "dtype": str(self.dtype).replace("torch.", "")}
        self.weights["decoder"] = {"lite_tae": "LeapTalk Lite TAE (taew2_1.pth, TAEHV)", "wan_vae": "Wan2.1 VAE (VAE_Wan/Wan2.1_VAE.pth)"}[self.decoder]
        self.weights["audio_encoder"] = "wav2vec2-base-960h"
        events.emit("loaded", seconds=round(sum(v for k, v in self.timings.items() if k.endswith("_s")), 3))

    def mark_memory(self, phase: str, release_cache: bool) -> None:
        """Record the CUDA peak of the phase that just ended; optionally return unused cached blocks to the driver.

        On Windows (WDDM) CUDA allocations count against the system commit charge, so releasing the cache
        after one-off load phases lowers the commit this process holds. Numerics are unaffected.
        """
        torch = self.torch
        torch.cuda.synchronize()
        rec = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        if release_cache:
            torch.cuda.empty_cache()
            rec["reserved_after_release_bytes"] = int(torch.cuda.memory_reserved())
        self.memory_phases[phase] = rec
        torch.cuda.reset_peak_memory_stats()

    def _load_lora(self, pipeline, lora_dir: str) -> dict:
        torch = self.torch
        from peft import PeftModel
        from safetensors import safe_open

        model = pipeline.model
        before = model.blocks[0].self_attn.q.weight.detach().float().cpu().clone()
        pm = PeftModel.from_pretrained(model, lora_dir, is_trainable=False)
        params = dict(pm.named_parameters())
        loaded, missing, mismatch = 0, [], []
        with safe_open(os.path.join(lora_dir, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for k in keys:
                name = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
                p = params.get(name)
                if p is None:
                    missing.append(k)
                elif torch.equal(p.detach().float().cpu(), f.get_tensor(k).float()):
                    loaded += 1
                else:
                    mismatch.append(k)
        model_lora = [n for n in params if ".lora_" in n]
        if missing or mismatch or loaded != len(keys) or len(model_lora) != len(keys):
            raise RuntimeError(
                f"LoRA not fully applied: {loaded}/{len(keys)} tensors, missing={missing[:3]}, mismatch={mismatch[:3]}, model has {len(model_lora)}"
            )
        cfg = pm.peft_config["default"]
        merged = pm.merge_and_unload()
        merged.eval().requires_grad_(False)
        if any(".lora_" in n for n, _ in merged.named_parameters()):
            raise RuntimeError("LoRA layers remain after merge")
        after = merged.blocks[0].self_attn.q.weight.detach().float().cpu()
        pipeline.model = merged
        return {
            "checkpoint_tensors": len(keys),
            "applied_tensors": loaded,
            "r": cfg.r,
            "alpha": cfg.lora_alpha,
            "scale": cfg.lora_alpha / cfg.r,
            "target_modules": sorted(cfg.target_modules),
            "merged": True,
            "blocks.0.self_attn.q_rel_change": round(float((after - before).norm() / before.norm()), 4),
        }

    def generate(self, job: dict, job_dir: Path) -> dict:
        torch, lt, pipeline = self.torch, self.lt, self.pipeline
        import librosa
        import numpy as np

        ev, ctl = self.events, self.ctl
        timings = dict(self.timings)
        image_path = str(job_dir / "input.png")
        audio_path = str(job_dir / "input.wav")

        # ---- audio: inspect the original, then load exactly like the official script
        import soundfile as sf

        info = sf.info(audio_path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise JobError("the audio is empty")
        orig_seconds = info.frames / info.samplerate
        if orig_seconds > job["max_audio_seconds"]:
            raise JobError(f"the audio is {orig_seconds:.1f} s long; this runtime accepts up to {job['max_audio_seconds']} s (no silent truncation)")
        if orig_seconds < 0.2:
            raise JobError("the audio is shorter than 0.2 s")
        t = time.perf_counter()
        audio_all, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        timings["audio_load_s"] = round(time.perf_counter() - t, 3)
        if not np.isfinite(audio_all).all():
            raise JobError("the audio contains NaN or Inf samples")
        n16 = int(len(audio_all))
        peak = float(np.abs(audio_all).max()) if n16 else 0.0

        sp = lt.StreamParams(
            frame_num=FRAME_NUM,
            motion_frames_latent_num=MOTION_FRAMES_LATENT_NUM,
            tgt_fps=FPS,
            sample_rate=SAMPLE_RATE,
            cached_audio_duration=CACHED_AUDIO_DURATION,
        ).init_with_stride(self.vae_stride_t)
        frame_num, motion_frames_num, slice_len = sp.frame_num, sp.motion_frames_num, sp.slice_len
        slice_samples = slice_len * SAMPLE_RATE // FPS
        frame_samples = frame_num * SAMPLE_RATE // FPS
        # padding sequence of the official script (stream mode)
        remainder = len(audio_all) % slice_samples
        if remainder > 0:
            audio_all = np.concatenate([audio_all, np.zeros(slice_samples - remainder, dtype=audio_all.dtype)])
        if len(audio_all) < frame_samples:
            audio_all = np.concatenate([audio_all, np.zeros(frame_samples - len(audio_all), dtype=audio_all.dtype)])
        remainder = len(audio_all) % slice_samples
        if remainder != 0:
            audio_all = np.concatenate([audio_all, np.zeros(slice_samples - remainder, dtype=audio_all.dtype)])
        slices = audio_all.reshape(-1, slice_samples)
        num_chunks = int(slices.shape[0])
        plan = chunk_plan(n16, orig_seconds)
        if (plan["padded_samples"], plan["chunks"], plan["slice_len"], plan["motion_frames"]) != (len(audio_all), num_chunks, slice_len, motion_frames_num):
            raise RuntimeError(f"internal: chunk plan {plan} disagrees with the official padding ({len(audio_all)} samples, {num_chunks} chunks)")
        out_frames = plan["out_frames"]
        if out_frames > num_chunks * slice_len:
            raise RuntimeError("internal: fewer generated frames than the audio needs")
        ev.emit("audio", seconds=round(orig_seconds, 4), chunks=num_chunks, frames=out_frames)

        # ---- reference image -> static anchor latent (official prepare_params)
        ctl.check()
        t = time.perf_counter()
        pipeline.prepare_params(
            cond_image_path_or_dir=image_path,
            target_size=(SIZE, SIZE),
            frame_num=frame_num,
            motion_frames_num=0,
            sampling_steps=4,
            seed=SEED,
            shift=SHIFT_GAMMA,
            color_correction_strength=COLOR_CORRECTION_STRENGTH,
            use_face_crop=False,
        )
        x0 = pipeline.ref_img_latent.to(device=self.device, dtype=self.dtype)
        torch.cuda.synchronize()
        timings["reference_encode_s"] = round(time.perf_counter() - t, 3)
        self.mark_memory("reference_encode", release_cache=True)

        scheduler = lt.ViBTScheduler(num_train_timesteps=1000)
        scheduler.timesteps = lt._build_infer_timesteps(
            step_list=None, num_inference_steps=NUM_INFERENCE_STEPS, shift_gamma=SHIFT_GAMMA, device=self.device, num_timesteps=1000
        )
        scheduler.num_inference_steps = int(scheduler.timesteps.numel())
        scheduler.set_parameters(noise_scale=NOISE_SCALE, shift_gamma=SHIFT_GAMMA, seed=SEED)
        latent_motion_frames = x0[:, :1].unsqueeze(0).clone()
        clamp_latent_len = int(latent_motion_frames.shape[2])
        guidance = float(job["audio_guidance"])

        # ---- encoder: raw RGB frames on stdin, original audio as the second input, no -shortest
        out_path = job_dir / "output.mp4"
        enc_argv = [
            job["ffmpeg"],
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{SIZE}x{SIZE}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(job["video_crf"]),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-y",
            str(out_path),
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        enc_log = open(job_dir / "ffmpeg.log", "wb")  # noqa: SIM115
        enc = subprocess.Popen(enc_argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=enc_log, creationflags=flags)  # noqa: S603 - argv list, no shell

        from collections import deque

        from PIL import Image

        cached_len = SAMPLE_RATE * sp.cached_audio_duration
        audio_end_idx = sp.cached_audio_duration * FPS
        audio_start_idx = audio_end_idx - frame_num
        audio_dq = deque([0.0] * cached_len, maxlen=cached_len)
        forwards = {"conditional": 0, "unconditional": 0}
        per_chunk = []
        frame_hashes = []
        written = 0
        model = pipeline.model
        cuda_ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731

        class _CountingModel:
            """Counts generator calls. With audio CFG (guidance != 1) every solver step calls the model twice:
            conditional first, then with zeroed audio context (unconditional), as in _bridge_sample_one_chunk."""

            def __init__(self):
                self.calls_this_chunk = 0

            def __call__(self, *a, **k):
                if guidance == 1.0 or self.calls_this_chunk % 2 == 0:
                    forwards["conditional"] += 1
                else:
                    forwards["unconditional"] += 1
                self.calls_this_chunk += 1
                return model(*a, **k)

        counter = _CountingModel()
        t_loop = time.perf_counter()
        try:
            for chunk_idx in range(num_chunks):
                ctl.check()
                torch.cuda.synchronize()
                c0 = time.perf_counter()
                e = {k: (cuda_ev(), cuda_ev()) for k in ("audio", "denoise", "decode", "color", "hist")}
                e["audio"][0].record()
                audio_dq.extend(slices[chunk_idx].tolist())
                audio_cache = np.array(audio_dq, dtype=np.float32)
                audio_emb_cache = pipeline.preprocess_audio(audio_cache, sr=SAMPLE_RATE, fps=FPS)
                if audio_emb_cache is None:
                    raise RuntimeError("failed to extract audio embeddings")
                audio_emb_cache = audio_emb_cache.to(device=self.device, dtype=self.dtype)
                audio_ctx = lt._audio_context_from_embeddings_range(
                    audio_emb_cache, start_idx=audio_start_idx, end_idx=audio_end_idx, device=self.device, dtype=self.dtype
                )
                e["audio"][1].record()
                e["denoise"][0].record()
                counter.calls_this_chunk = 0
                pipeline.model = counter
                try:
                    x_final = lt._bridge_sample_one_chunk(
                        pipeline,
                        scheduler=scheduler,
                        ref_latent=x0,
                        audio_context=audio_ctx,
                        guidance_scale=guidance,
                        latent_motion_frames=latent_motion_frames,
                        clamp_latent_len=clamp_latent_len,
                        device=self.device,
                        dtype=self.dtype,
                    )
                finally:
                    pipeline.model = model
                e["denoise"][1].record()
                e["decode"][0].record()
                decoded_cthw = lt._decode_to_cthw(pipeline, x_final)
                e["decode"][1].record()
                e["color"][0].record()
                decoded_cthw = lt._maybe_apply_color_correction(pipeline, decoded_cthw)
                e["color"][1].record()
                e["hist"][0].record()
                latent_motion_frames = lt._encode_motion_prefix_from_decoded(
                    pipeline, decoded_video_cthw=decoded_cthw, motion_frames_num=motion_frames_num, device=self.device, dtype=self.dtype
                ).unsqueeze(0)
                e["hist"][1].record()
                clamp_latent_len = int(latent_motion_frames.shape[2])
                decoded_cthw = decoded_cthw[:, motion_frames_num:]  # stream mode always drops the overlap
                video_thwc = ((decoded_cthw + 1.0) / 2.0).permute(1, 2, 3, 0).clamp(0.0, 1.0).mul(255.0).contiguous()
                torch.cuda.synchronize()
                chunk_s = time.perf_counter() - c0
                # same conversion as the official writer: bf16 -> fp16 -> numpy -> uint8 (truncation)
                io0 = time.perf_counter()
                video_cpu = video_thwc.detach().cpu()
                if video_cpu.dtype == torch.bfloat16:
                    video_cpu = video_cpu.to(torch.float16)
                frames_np = video_cpu.numpy().astype(np.uint8)
                take = max(0, min(len(frames_np), out_frames - written))
                for fr in frames_np[:take]:
                    frame_hashes.append(hashlib.sha256(np.ascontiguousarray(fr).tobytes()).hexdigest())
                enc.stdin.write(np.ascontiguousarray(frames_np[:take]).tobytes())
                written += take
                if job["preview"]:
                    tmp = job_dir / "preview.tmp.jpg"
                    Image.fromarray(frames_np[-1]).save(tmp, quality=80)
                    tmp.replace(job_dir / "preview.jpg")
                io_s = time.perf_counter() - io0
                rec = {k: round(e[k][0].elapsed_time(e[k][1]) / 1000.0, 4) for k in e}
                rec.update(chunk_s=round(chunk_s, 4), io_s=round(io_s, 4), frames=int(len(frames_np)), written=int(take))
                per_chunk.append(rec)
                ev.emit("progress", chunk=chunk_idx + 1, chunks=num_chunks, frames=written, preview=bool(job["preview"]))
        except BaseException as exc:
            try:
                enc.stdin.close()
            except OSError:
                pass
            enc.kill()
            enc.wait(timeout=30)
            enc_log.close()
            out_path.unlink(missing_ok=True)
            if isinstance(exc, (BrokenPipeError, OSError)) and not isinstance(exc, Cancelled):
                tail = (job_dir / "ffmpeg.log").read_bytes()[-1500:].decode("utf-8", "replace")
                raise RuntimeError(f"the video encoder stopped: {exc}; ffmpeg: {tail}") from exc
            raise
        sampling_s = time.perf_counter() - t_loop
        self.mark_memory("chunk_loop", release_cache=False)
        t = time.perf_counter()
        enc.stdin.close()
        code = enc.wait(timeout=600)
        enc_log.close()
        timings["encode_finalize_s"] = round(time.perf_counter() - t, 3)
        if code != 0:
            tail = (job_dir / "ffmpeg.log").read_bytes()[-1500:].decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg failed (exit {code}): {tail}")

        t = time.perf_counter()
        check = verify_output(out_path, out_frames, orig_seconds)
        timings["verify_s"] = round(time.perf_counter() - t, 3)
        (job_dir / "frames_sha256.json").write_text(json.dumps(frame_hashes), encoding="utf-8")

        def mean(key, skip=2):
            xs = [c[key] for c in per_chunk[skip:]] or [c[key] for c in per_chunk]
            return round(sum(xs) / len(xs), 4)

        timings.update(
            sampling_loop_s=round(sampling_s, 3),
            chunk_mean_s=mean("chunk_s"),
            audio_encode_mean_s=mean("audio"),
            denoise_mean_s=mean("denoise"),
            decode_mean_s=mean("decode"),
            color_mean_s=mean("color"),
            history_mean_s=mean("hist"),
            frame_io_mean_s=mean("io_s"),
            chunk_timing_note="means exclude the first 2 chunks, as the official script does",
        )
        return {
            "status": "ok",
            "output": out_path.name,
            "sha256": _sha256_file(out_path),
            "video": check["video"],
            "audio": {
                "input_sample_rate": info.samplerate,
                "input_channels": info.channels,
                "input_samples": info.frames,
                "input_seconds": round(orig_seconds, 4),
                "encoder_input": "librosa.load(sr=16000, mono=True), as the official script",
                "samples_16k": n16,
                "padded_samples_16k": int(len(audio_all)),
                "peak_16k": round(peak, 4),
                "silent": peak < 1e-4,
                "muxed": check["audio"],
            },
            "chunks": num_chunks,
            "frames_per_chunk": slice_len,
            "overlap_frames_dropped_per_chunk": motion_frames_num,
            "generated_frames": num_chunks * slice_len,
            "output_frames": written,
            "trimmed_tail_frames": num_chunks * slice_len - written,
            "solver_steps": NUM_INFERENCE_STEPS,
            "audio_guidance": guidance,
            "forwards": {"total": forwards["conditional"] + forwards["unconditional"], **forwards},
            "frames_sha256": hashlib.sha256("".join(frame_hashes).encode()).hexdigest(),
            "per_chunk": per_chunk,
            "timings": timings,
            "memory": {
                "cuda_max_allocated_bytes": max(p["peak_allocated_bytes"] for p in self.memory_phases.values()),
                "cuda_max_reserved_bytes": max(p["peak_reserved_bytes"] for p in self.memory_phases.values()),
                "phases": dict(self.memory_phases),
            },
        }


def verify_output(path: Path, frames: int, audio_seconds: float) -> dict:
    """Decode the written MP4 completely and check both streams (an MP4 without audio is a failure)."""
    import av

    with av.open(str(path)) as c:
        vs = [s for s in c.streams if s.type == "video"]
        as_ = [s for s in c.streams if s.type == "audio"]
        if len(vs) != 1 or len(as_) != 1:
            raise RuntimeError(f"output must have exactly one video and one audio stream, got {len(vs)} video / {len(as_)} audio")
        v, a = vs[0], as_[0]
        meta = {
            "codec": v.codec_context.name,
            "width": v.codec_context.width,
            "height": v.codec_context.height,
            "fps": float(v.average_rate),
            "audio_codec": a.codec_context.name,
            "audio_rate": a.codec_context.sample_rate,
            "audio_channels": a.codec_context.channels,
        }
    n = 0
    with av.open(str(path)) as c:
        for _ in c.decode(video=0):
            n += 1
    samples = 0
    with av.open(str(path)) as c:
        for fr in c.decode(audio=0):
            samples += fr.samples
    audio_dec = samples / meta["audio_rate"]
    if n != frames:
        raise RuntimeError(f"output has {n} video frames, expected {frames}")
    if (meta["width"], meta["height"]) != (SIZE, SIZE) or abs(meta["fps"] - FPS) > 1e-6:
        raise RuntimeError(f"unexpected output video {meta}")
    # AAC encodes whole 1024-sample frames: allow up to 2 frames + 10 ms beyond the input length
    tol = 2 * 1024 / meta["audio_rate"] + 0.01
    if not (audio_seconds - 0.01 <= audio_dec <= audio_seconds + tol):
        raise RuntimeError(f"muxed audio lasts {audio_dec:.3f} s, input {audio_seconds:.3f} s")
    return {
        "video": {"frames": n, "fps": meta["fps"], "width": meta["width"], "height": meta["height"], "codec": meta["codec"], "seconds": round(n / FPS, 4)},
        "audio": {"codec": meta["audio_codec"], "sample_rate": meta["audio_rate"], "channels": meta["audio_channels"], "decoded_seconds": round(audio_dec, 4)},
    }


def versions() -> dict:
    import importlib.metadata as md

    out = {"python": sys.version.split()[0]}
    for name in ("torch", "diffusers", "transformers", "peft", "librosa", "av", "numpy"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-stdin", action="store_true", help="treat stdin EOF as 'the parent is gone' and stop")
    ap.add_argument("job")
    a = ap.parse_args(argv)
    job_path = Path(a.job).resolve()
    job_dir = job_path.parent
    events = Events(job_dir / "events.jsonl")
    t0 = time.time()
    events.emit("started", pid=os.getpid())
    try:
        job = load_job(job_path)
    except (JobError, OSError, ValueError) as exc:
        events.emit("invalid_job", error=str(exc)[:1000])
        return EXIT_INVALID
    ctl = Control(job_dir, a.watch_stdin)
    apply_env(job["env"], job_dir)
    try:
        engine = Engine(job, events, ctl)
        result = engine.generate(job, job_dir)
        result.update(
            job_id=job["job_id"],
            decoder=job["decoder"],
            attention_backend=engine.attention_backend,
            upstream=engine.upstream,
            weights=engine.weights,
            versions=versions(),
            process_seconds=round(time.time() - t0, 3),
            effective_config={
                "model_type": "pro",
                "lite": job["decoder"] == "lite_tae",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "shift_gamma": SHIFT_GAMMA,
                "noise_scale": NOISE_SCALE,
                "audio_guidance": job["audio_guidance"],
                "frame_num": FRAME_NUM,
                "motion_frames_latent_num": MOTION_FRAMES_LATENT_NUM,
                "cached_audio_duration": CACHED_AUDIO_DURATION,
                "audio_encode_mode": "stream",
                "history_update_mode": "roundtrip",
                "color_correction_strength": COLOR_CORRECTION_STRENGTH,
                "height": SIZE,
                "width": SIZE,
                "fps": FPS,
                "dtype": "bf16",
                "compile": False,
                "use_face_crop": False,
            },
        )
        _write_json(job_dir / "result.json", result)
        events.emit("done", seconds=round(time.time() - t0, 3))
        return EXIT_OK
    except Cancelled as exc:
        events.emit("cancelled", reason=str(exc))
        return EXIT_CANCELLED
    except JobError as exc:
        events.emit("failed", error_type="JobError", error=str(exc)[:2000])
        return EXIT_INVALID
    except BaseException as exc:  # noqa: BLE001
        events.emit("failed", error_type=type(exc).__name__, error=str(exc)[:2000])
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
