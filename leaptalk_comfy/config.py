"""Administrator-registered LeapTalk runtimes.

A workflow can only *name* a runtime (``runtime_id``). Everything that is executed - the runtime's
Python, the upstream checkout, the model folders, ffmpeg - comes from a local JSON file that the
ComfyUI web API cannot write to:

1. the path in the ``COMFYUI_LEAPTALK_CONFIG`` environment variable, else
2. ``<ComfyUI user directory>/leaptalk.runtimes.json`` (a file at the root of the user directory,
   outside every per-user folder that the userdata API serves).

See ``examples/leaptalk.runtimes.example.json``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_ENV = "COMFYUI_LEAPTALK_CONFIG"
CONFIG_FILENAME = "leaptalk.runtimes.json"
SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

# Settings forwarded into the runtime process (mirrors runtime/leaptalk_job.py ENV_KEYS).
RUNTIME_ENV_KEYS = frozenset({"CUDA_VISIBLE_DEVICES", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"})
MODEL_KEYS = ("soulx_dir", "wav2vec_dir", "leaptalk_dir")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_RUNTIME_KEYS = {"kind", "python", "upstream_dir", "models", "ffmpeg", "env", "timeout_minutes", "max_audio_seconds", "jobs_dir", "description"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Runtime:
    id: str
    kind: str  # "native": a separate Python environment on this machine (Windows or Linux)
    python: str
    upstream_dir: str
    models: dict  # soulx_dir / wav2vec_dir / leaptalk_dir -> absolute path
    ffmpeg: str
    env: dict = field(default_factory=dict)
    timeout_minutes: int = 60
    max_audio_seconds: int = 600
    jobs_dir: str | None = None  # host path; default <ComfyUI temp>/leaptalk
    description: str = ""


def _abs(value, what: str, rid: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1024 or any(c in value for c in "\0\n\r"):
        raise ConfigError(f"runtime {rid!r}: {what} must be a non-empty path")
    p = Path(value)
    if not p.is_absolute():
        raise ConfigError(f"runtime {rid!r}: {what} must be an absolute path")
    if ".." in p.parts:
        raise ConfigError(f"runtime {rid!r}: {what} must not contain '..'")
    return value


def parse_runtime(rid: str, data: dict) -> Runtime:
    if not isinstance(rid, str) or not _ID_RE.fullmatch(rid):
        raise ConfigError(f"invalid runtime id {rid!r}")
    if not isinstance(data, dict):
        raise ConfigError(f"runtime {rid!r} must be an object")
    unknown = set(data) - _RUNTIME_KEYS
    if unknown:
        raise ConfigError(f"runtime {rid!r}: unknown keys {sorted(unknown)}")
    kind = data.get("kind", "native")
    if kind != "native":
        raise ConfigError(f"runtime {rid!r}: kind must be 'native' (a separate Python environment on this machine)")
    python = _abs(data.get("python"), "python", rid)
    upstream = _abs(data.get("upstream_dir"), "upstream_dir", rid)
    ffmpeg = _abs(data.get("ffmpeg"), "ffmpeg", rid)
    models = data.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_KEYS):
        raise ConfigError(f"runtime {rid!r}: models must have exactly {list(MODEL_KEYS)}")
    models = {k: _abs(v, f"models.{k}", rid) for k, v in models.items()}
    env = data.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ConfigError(f"runtime {rid!r}: env must map strings to strings")
    bad = sorted(set(env) - RUNTIME_ENV_KEYS)
    if bad:
        raise ConfigError(f"runtime {rid!r}: env keys not allowed: {bad} (allowed: {sorted(RUNTIME_ENV_KEYS)})")
    timeout = data.get("timeout_minutes", 60)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 24 * 60:
        raise ConfigError(f"runtime {rid!r}: timeout_minutes must be an integer in [1, 1440]")
    max_audio = data.get("max_audio_seconds", 600)
    if isinstance(max_audio, bool) or not isinstance(max_audio, int) or not 1 <= max_audio <= 1800:
        raise ConfigError(f"runtime {rid!r}: max_audio_seconds must be an integer in [1, 1800]")
    jobs_dir = data.get("jobs_dir")
    if jobs_dir is not None:
        jobs_dir = _abs(jobs_dir, "jobs_dir", rid)
    desc = data.get("description", "")
    if not isinstance(desc, str) or len(desc) > 500:
        raise ConfigError(f"runtime {rid!r}: description must be a short string")
    return Runtime(
        id=rid,
        kind=kind,
        python=python,
        upstream_dir=upstream,
        models=models,
        ffmpeg=ffmpeg,
        env=dict(env),
        timeout_minutes=timeout,
        max_audio_seconds=max_audio,
        jobs_dir=jobs_dir,
        description=desc,
    )


def config_path() -> Path | None:
    explicit = os.environ.get(CONFIG_ENV)
    if explicit:
        return Path(explicit)
    try:
        import folder_paths
    except ImportError:
        return None
    return Path(folder_paths.get_user_directory()) / CONFIG_FILENAME


def load_runtimes(path: Path | None = None) -> dict[str, Runtime]:
    """Return {} when no config file exists; raise ConfigError for a broken one."""
    path = path or config_path()
    if path is None or not path.is_file():
        return {}
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ConfigError(f"{path.name} is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("runtimes"), dict):
        raise ConfigError(f"{path.name} needs schema_version {SCHEMA_VERSION} and a 'runtimes' object")
    return {rid: parse_runtime(rid, rt) for rid, rt in data["runtimes"].items()}
