# Testing

## CPU tests (CI and local)

```sh
python -m pip install pytest==9.1.1 ruff==0.16.10
COMFYUI_PATH=/path/to/ComfyUI python -m pytest -q -rs tests
```

No GPU, no weights: `tests/fake_runtime/fake_leaptalk_job.py` stands in for the LeapTalk runtime. It
validates jobs with the real runner's `load_job` and `chunk_plan` and writes real MP4 files (H.264 +
AAC from the job's audio). Covered:

- runtime config validation (kind, absolute paths, `..`, unknown keys, environment allowlist,
  limits) and the config file location;
- the inputs the node writes: the audio as an IEEE-float WAV, bit-exact for mono/stereo and 16/44.1/48
  kHz; rejection of empty, too short, too long (no silent truncation), NaN/Inf, >8 channels and bad
  sample rates; silence accepted and flagged; image size/alpha handling;
- the runner's own validation of `job.json` (tampered fields, files outside the job directory, secret
  environment keys), the upstream hash check (line endings ignored), the per-decoder model file check,
  removal of credential-like variables;
- the official stream-mode padding/chunking (`chunk_plan`) against the counts measured on real runs
  (9.35 s -> 9 chunks / 234 frames, 33.975 s -> 31 chunks / 850 frames) and edge cases (0.5 s, exact
  chunk multiples, 60 s);
- real subprocesses: success with progress and previews, runtime failure, cancel and timeout (the
  whole process tree including a grandchild is gone), and parent death: killing the "ComfyUI" process
  ends the runtime tree (Windows job object; on Linux the child stops on stdin EOF);
- inside a real ComfyUI checkout (CPU), loaded from a folder named differently from the repository
  and the Registry id (with spaces, a quote and non-ASCII characters): node registration without heavy
  imports, `validate_prompt` with `LoadImage`/`LoadAudio` inputs whose file names contain spaces,
  quotes and non-ASCII characters, a `VIDEO` output with frames and the original stereo audio,
  rejection of image/audio batches and NaN audio, ComfyUI interrupts mapped to a job stop.

Tests that need ComfyUI are marked `comfy` and fail (not skip) without `COMFYUI_PATH`. CI runs
everything on Ubuntu and Windows with ComfyUI v0.38.0 and CPU PyTorch.

## GPU end-to-end (maintainer, real runtime)

`scripts/gpu_e2e.py` starts its own ComfyUI on a free `127.0.0.1` port (`--cpu`; the runtime does
the GPU work) and drives it through the HTTP API with real weights:

| Step | Pass criteria |
| --- | --- |
| generate / long / cases | `LoadImage` + `LoadAudio` -> **LeapTalk Generate** -> `SaveVideo`; the saved MP4 is decoded completely: exactly one video and one audio stream, 512x512 at 25 fps, frame count equal to the report (`ceil(audio seconds x 25)`), LoRA 480/480 tensors applied |
| doctor | **LeapTalk Doctor** reports `ok` |
| preview | (separate check over ComfyUI's websocket) one JPEG progress preview per finished chunk reaches the client before the job ends |
| error | a runtime with a missing model folder fails with the runner's message, nothing left |
| cancel | `/interrupt` during the chunk loop -> `execution_interrupted`, runner tree gone, GPU memory back to idle |
| timeout | `timeout_minutes: 1` with a 97 s clip -> error, nothing left |
| crash | ComfyUI killed (no cleanup code runs in it) during the chunk loop -> the runner tree (launcher, interpreter, ffmpeg) is gone, GPU memory back to idle |

Results of the release run: `docs/results/gpu_e2e_report.json`, summarised in docs/BENCHMARKS.md.

## Official reference

`inference.py` of LeapTalk was run unchanged (single process, `--compile off --num_inference_steps 1
--lite`) with observation-only hooks (count generator calls, record latents and every frame handed
to the video writer). This package's runtime must produce the same 8-bit frames for the same inputs;
see "Same frames as the official script" in docs/BENCHMARKS.md.

## GUI check

`scripts/export_gui_workflows.py` (needs Playwright) loads each API workflow into the real ComfyUI
frontend, checks that every node type is known and that `graphToPrompt()` reproduces it, and writes
the UI-format files in `workflows/`.
