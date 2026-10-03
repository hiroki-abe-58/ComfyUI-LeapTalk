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

Persistent worker (v0.2), with the real `runtime/leaptalk_worker.py` and a fake Engine
(`tests/fake_runtime/fake_engine_job.py`: it re-exports the real job module and replaces only the
model, so job validation, events and result fields are the real ones; its frames depend on the
portrait, so a stale portrait would show):

- reuse across jobs: one worker and one Engine (same ids, PIDs, init / LoRA-merge counters 1),
  portrait A -> B -> A gives A's frames again, every job has its own folder, events and result;
- worker identity: a decoder change starts a new worker and reports why; Status never starts a
  worker and does not refresh the idle timer; Unload, Unload with nothing loaded;
- idle timeout, and that neither the idle timer nor Unload cuts a running job (Unload answers
  `busy`); three concurrent requests start exactly one worker;
- cancel and timeout keep or replace the worker and the next job succeeds; a worker killed during a
  job (the job fails, the next job gets a fresh worker); an Engine that fails to load; a worker that
  answers something that is not the protocol; nothing is left running after any of these;
- killing the "ComfyUI" process while the worker idles ends the worker tree (job object / stdin EOF);
- the worker's environment holds no token/secret variables;
- the memory guard with fixed snapshots: refusing a load (commit, unknown memory), counting only the
  measured extra peak for a warm job, refusing a warm job near the stop level, the stop rule with
  its duration and hysteresis (including the 93-95 % pattern of a normal job on the measured machine);
- a one-shot job unloads an idle worker first; the v0.1 config and the v0.1 workflow stay one-shot;
- talking to the worker directly: ids outside the allowed characters, a job folder outside the jobs
  root, a missing job, a duplicate request id, status and shutdown;
- inside a real ComfyUI: the Worker node is registered, never cached (`IS_CHANGED`), Status starts
  nothing, Generate with `backend = persistent` reuses one worker over three jobs and reports the
  requested/effective backend, Unload frees it; every shipped API workflow (v0.1 and new) validates;
- the files the worker needs are not excluded from the Registry package (`.comfyignore`).

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

Results of the v0.1.0 release run: `docs/results/gpu_e2e_report.json`, summarised in docs/BENCHMARKS.md.

### Persistent worker series (v0.2)

`scripts/gpu_worker_e2e.py` drives its own ComfyUI the same way, started with `--cache-none` so every
queued prompt really executes (a cached output would prove nothing about reuse). It listens on the
websocket (previews are counted per prompt and compared with that job's own `preview.jpg`), samples
the system commit charge, available memory and GPU memory, counts the worker processes that descend
from its ComfyUI, and interrupts the prompt if commit stays at or above 95 % for 5 s. For every job it
records the prompt id, job id, backend, worker instance id and PIDs, Engine instance id, init /
LoRA-merge counters, generator calls, frame hashes, timings and memory. Steps: the v0.1 workflow
(one-shot) vs persistent first and warm jobs on the same inputs, ten jobs on one worker (portraits
A/B/A, speech changes, 0.5 s, silence, stereo 48 kHz, ~34 s), ~96 s, a fresh worker, decoder switch
and back, cancel / timeout / runtime error / worker killed during a job / worker launcher killed while
idle, each followed by a normal job, idle timeout and explicit Unload, prompts queued at once with an
Unload between them, and ComfyUI killed while the worker idles and while it generates. The v0.1.0
baseline was run with the same driver against `git archive v0.1.0` installed alone in the same test
ComfyUI. The `shipped` step queues the API workflows of an installed copy (`--shipped-dir`) with only the
runtime id and input names replaced; it was run against `git archive` of the release commit installed
under a different folder name. Results: `docs/results/v020/`, summarised in docs/BENCHMARKS.md.

Release assets: `.github/workflows/verify-release.yml` (run by hand) downloads every asset of a release
anonymously and checks them with `sha256sum -c SHA256SUMS`.

## Official reference

`inference.py` of LeapTalk was run unchanged (single process, `--compile off --num_inference_steps 1
--lite`) with observation-only hooks (count generator calls, record latents and every frame handed
to the video writer). This package's runtime must produce the same 8-bit frames for the same inputs;
see "Same frames as the official script" in docs/BENCHMARKS.md.

## GUI check

`scripts/export_gui_workflows.py` (needs Playwright) loads each API workflow into the real ComfyUI
frontend, checks that every node type is known and that `graphToPrompt()` reproduces it, and writes
the UI-format files in `workflows/`.
