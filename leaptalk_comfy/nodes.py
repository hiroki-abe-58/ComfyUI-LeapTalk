"""ComfyUI nodes. Importing this module loads no CUDA, no model and no upstream code."""

from __future__ import annotations

import json
import time

from . import client, worker
from .config import CONFIG_ENV, CONFIG_FILENAME, ConfigError, config_path, load_runtimes

NO_RUNTIME = "(no runtime configured)"
RUNTIME_DEFAULT = "runtime default"
BACKEND_CHOICES = (RUNTIME_DEFAULT, "persistent", "one-shot")
BACKEND_HELP = (
    "one-shot: every job starts the runtime, loads the model and exits (v0.1 behaviour). "
    "persistent: the first job starts a worker that keeps the model loaded for later jobs, so they skip about "
    "20 s of start-up; it keeps GPU and system memory until it is unloaded (LeapTalk Worker node), idles out "
    "(worker_idle_seconds, default 120 s) or ComfyUI exits. 'runtime default' uses the runtime's configured "
    "backend (one-shot unless the administrator changed it)."
)
DECODER_HELP = (
    "lite_tae: LeapTalk's Lite decoder (taew2_1.pth, the official default). "
    "wan_vae: the standard Wan2.1 VAE from SoulX-FlashHead (slower; run without torch.compile here, the official script compiles it)."
)


def _runtime_ids() -> list[str]:
    try:
        ids = sorted(load_runtimes())
    except ConfigError:
        ids = []
    return ids or [NO_RUNTIME]


def _resolve(runtime_id: str):
    try:
        runtimes = load_runtimes()
    except ConfigError as exc:
        raise ValueError(f"LeapTalk runtime config is invalid: {exc}") from exc
    if runtime_id not in runtimes:
        where = config_path()
        raise ValueError(
            f"LeapTalk runtime {runtime_id!r} is not configured. An administrator registers runtimes in "
            f"{where if where else CONFIG_FILENAME} (or the file named by {CONFIG_ENV}); see the README."
        )
    return runtimes[runtime_id]


class LeapTalkRuntime:
    """Select an administrator-registered runtime (a separate Python environment with the LeapTalk code and weights)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"runtime_id": (_runtime_ids(), {"tooltip": f"Runtimes come from {CONFIG_FILENAME} (administrator config), never from the workflow."})},
            "optional": {"backend": (list(BACKEND_CHOICES), {"default": RUNTIME_DEFAULT, "tooltip": BACKEND_HELP})},
        }

    RETURN_TYPES = ("LEAPTALK_RUNTIME",)
    RETURN_NAMES = ("runtime",)
    FUNCTION = "select"
    CATEGORY = "LeapTalk"
    DESCRIPTION = "Pick a LeapTalk runtime registered by the administrator, and how jobs run on it (one-shot or a persistent worker)."

    def select(self, runtime_id, backend=RUNTIME_DEFAULT):
        rt = _resolve(runtime_id)
        if backend not in BACKEND_CHOICES:
            raise ValueError(f"backend must be one of {BACKEND_CHOICES}")
        return ({"runtime_id": rt.id, "backend": None if backend == RUNTIME_DEFAULT else backend},)


def _image_to_uint8(image):
    import numpy as np

    if image is None or not hasattr(image, "shape") or image.ndim != 4:
        raise ValueError("connect one IMAGE (batch x height x width x channels)")
    if image.shape[0] != 1:
        raise ValueError(f"LeapTalk takes exactly one reference image; this IMAGE batch has {image.shape[0]} (pick one with an image-batch node)")
    if image.shape[-1] not in (3, 4):
        raise ValueError("the IMAGE must have 3 (RGB) or 4 (RGBA) channels")
    return (image[0].detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)


def _audio_to_array(audio):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("connect an AUDIO input (for example Load Audio)")
    wav = audio["waveform"]
    if not hasattr(wav, "ndim") or wav.ndim != 3:
        raise ValueError("AUDIO waveform must be batch x channels x samples")
    if wav.shape[0] != 1:
        raise ValueError(f"LeapTalk takes one audio clip; this AUDIO batch has {wav.shape[0]}")
    return wav[0].detach().cpu().float().numpy(), int(audio["sample_rate"])


class LeapTalkGenerate:
    """Portrait + speech -> talking-head video with the official LeapTalk recipe (1 solver step per chunk)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime": ("LEAPTALK_RUNTIME",),
                "image": ("IMAGE", {"tooltip": "One reference portrait. The runtime resizes and center-crops it to 512x512, as the official script."}),
                "audio": ("AUDIO", {"tooltip": "Speech to animate. It is muxed into the output unchanged; the model hears it resampled to 16 kHz mono."}),
                "decoder": (list(client.DECODERS), {"default": "lite_tae", "tooltip": DECODER_HELP}),
                "audio_guidance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Audio classifier-free guidance. 1.0 = off (the official default, 1 model call per chunk); "
                        ">1 adds an unconditional call per chunk (2x model calls). Experimental: 2.0 gave visibly over-sharpened, "
                        "discoloured lips in our tests; keep 1.0 unless you are experimenting.",
                    },
                ),
                "preview": ("BOOLEAN", {"default": True, "tooltip": "Show the last frame of each finished chunk while the job runs."}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "generate"
    CATEGORY = "LeapTalk"
    DESCRIPTION = "512x512, 25 fps talking-head video with your audio, generated chunk by chunk by the LeapTalk runtime. Returns a VIDEO and a JSON report."

    def generate(self, runtime, image, audio, decoder, audio_guidance, preview):
        import comfy.model_management as mm
        import comfy.utils
        from comfy_api.latest import InputImpl

        if not isinstance(runtime, dict) or "runtime_id" not in runtime:
            raise ValueError("connect a LeapTalk Runtime node")
        rt = _resolve(runtime["runtime_id"])
        requested = runtime.get("backend")  # None: the runtime's configured default (one-shot unless the admin changed it)
        backend = requested or rt.backend
        img = _image_to_uint8(image)
        wav, sr = _audio_to_array(audio)
        pbar = comfy.utils.ProgressBar(1)

        def on_progress(done, total, preview_path):
            pil = client.read_preview(preview_path) if preview else None
            pbar.update_absolute(done, total, ("JPEG", pil, 512) if pil is not None else None)

        t0 = time.time()
        try:
            outcome = client.generate(
                rt,
                img,
                wav,
                sr,
                decoder=decoder,
                audio_guidance=float(audio_guidance),
                preview=bool(preview),
                interrupted=mm.processing_interrupted,
                on_progress=on_progress,
                backend=backend,
            )
        except client.JobCancelled as exc:  # the user's cancel
            raise mm.InterruptProcessingException() from exc
        except worker.MemoryGuardRefused as exc:  # "LeapTalk memory guard: did not start ... / stopped ..."
            raise RuntimeError(str(exc)) from exc
        except client.JobTimeout as exc:
            raise RuntimeError(f"LeapTalk timeout: {exc}") from exc
        except (client.RuntimeJobError, worker.WorkerError) as exc:
            raise RuntimeError(f"LeapTalk runtime error: {exc}") from exc
        report = _report(outcome, rt, time.time() - t0)
        report["backend"] = {"requested": requested or RUNTIME_DEFAULT, "runtime_default": rt.backend, "effective": outcome.result.get("backend", backend)}
        return (InputImpl.VideoFromFile(str(outcome.video)), json.dumps(report, indent=1, ensure_ascii=False))


def _report(outcome: client.JobOutcome, rt, wall_s: float) -> dict:
    r = outcome.result
    keep = (
        "decoder",
        "video",
        "audio",
        "chunks",
        "frames_per_chunk",
        "generated_frames",
        "output_frames",
        "trimmed_tail_frames",
        "solver_steps",
        "audio_guidance",
        "forwards",
        "attention_backend",
        "timings",
        "memory",
        "weights",
        "upstream",
        "versions",
        "effective_config",
        "frames_sha256",
        "reference_latent_sha256",
        "sha256",
        "process_seconds",
        "engine",
        "engine_init_timings",
        "job_memory",
        "worker",
    )
    rep = {"runtime_id": rt.id, "job": outcome.job_dir.name, "wall_seconds": round(wall_s, 2)}
    rep.update({k: r[k] for k in keep if k in r})
    secs = (r.get("audio") or {}).get("input_seconds")
    if secs:
        rep["wall_seconds_per_audio_second"] = round(wall_s / secs, 3)
    rep["host"] = r.get("host")
    return rep


class LeapTalkDoctor:
    """Check a runtime: upstream files at the pinned commit, model files, CUDA, attention backend, ffmpeg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime_id": (_runtime_ids(),),
                "verify_sha256": ("BOOLEAN", {"default": False, "tooltip": "Hash every model file (about 7 GB of reads)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    CATEGORY = "LeapTalk"
    OUTPUT_NODE = True
    DESCRIPTION = "Diagnose a LeapTalk runtime without generating a video."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def check(self, runtime_id, verify_sha256):
        import comfy.model_management as mm

        where = config_path()
        try:
            runtimes = load_runtimes()
        except ConfigError as exc:
            report = {"overall": "fail", "config": str(where), "error": str(exc)}
            return {"ui": {"text": [f"config error: {exc}"]}, "result": (json.dumps(report, indent=1),)}
        if runtime_id not in runtimes:
            report = {
                "overall": "fail",
                "config": str(where),
                "error": f"no runtime {runtime_id!r}; create {CONFIG_FILENAME} (see examples/ and the README)",
                "configured": sorted(runtimes),
            }
            return {"ui": {"text": [report["error"]]}, "result": (json.dumps(report, indent=1),)}
        try:
            report = client.doctor(runtimes[runtime_id], verify_sha256=verify_sha256, interrupted=mm.processing_interrupted)
        except client.JobCancelled as exc:
            raise mm.InterruptProcessingException() from exc
        lines = [f"{runtime_id}: {report.get('overall')}"] + [f"{c['status']:>4}  {c['name']}" for c in report.get("checks", [])]
        return {"ui": {"text": ["\n".join(lines)]}, "result": (json.dumps(report, indent=1, ensure_ascii=False),)}


class LeapTalkWorker:
    """Status or unload of the persistent worker (the loaded LeapTalk model) of this ComfyUI process."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["status", "unload"], {"tooltip": "status: show the worker without starting it. unload: stop it and free its GPU/system memory."}),
            },
            "optional": {
                "after": ("STRING", {"forceInput": True, "tooltip": "Optional: connect a Generate report to run this after that job."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "LeapTalk"
    OUTPUT_NODE = True
    DESCRIPTION = "Show or unload the persistent LeapTalk worker. Status never starts a worker and does not extend its idle time."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()  # always execute; never answer from ComfyUI's cache

    def run(self, action, after=None):
        if action == "unload":
            out = worker.MANAGER.unload("unload requested from the LeapTalk Worker node")
            out["now"] = worker.MANAGER.status()
        elif action == "status":
            out = worker.MANAGER.status()
        else:
            raise ValueError("action must be status or unload")
        text = json.dumps(out, indent=1, ensure_ascii=False, default=str)
        w = (out.get("now") or out).get("worker")
        if w:
            line = f"{action}: worker {w['worker_instance_id']} is {w['state']} with the model loaded ({w['decoder']}, {w['jobs_done']} jobs)"
            if w.get("idle_seconds_left") is not None:
                line += f"; it unloads after {w['idle_seconds_left']:.0f} more idle seconds"
        else:
            line = f"{action}: no worker loaded (no LeapTalk model in memory)"
        if action == "unload":
            line += f" | unload: {out.get('status')}"
            if out.get("detail"):
                line += f" - {out['detail']}"
        return {"ui": {"text": [line]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "LeapTalkRuntime": LeapTalkRuntime,
    "LeapTalkGenerate": LeapTalkGenerate,
    "LeapTalkDoctor": LeapTalkDoctor,
    "LeapTalkWorker": LeapTalkWorker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LeapTalkRuntime": "LeapTalk Runtime",
    "LeapTalkGenerate": "LeapTalk Generate (portrait + speech)",
    "LeapTalkDoctor": "LeapTalk Doctor",
    "LeapTalkWorker": "LeapTalk Worker (status / unload)",
}
