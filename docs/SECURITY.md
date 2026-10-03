# Security and trust model

ComfyUI-LeapTalk runs LeapTalk in a separate Python process (the runtime): a new process for every
job (`one-shot`, the default), or one worker process that keeps the model loaded between jobs
(`persistent`, see below). This page describes what that process can be told to do and how it is
stopped. It is **not a sandbox**: the runtime runs with the rights of the user who started ComfyUI,
and it executes the LeapTalk code and model files the administrator installed.

## Who decides what runs

- The administrator registers runtimes in `leaptalk.runtimes.json` (ComfyUI user directory root, or
  the file named by `COMFYUI_LEAPTALK_CONFIG`). That file names the runtime's Python, the LeapTalk
  checkout, the model folders and ffmpeg. The ComfyUI web API does not serve or write this location.
- A workflow can only pick a `runtime_id` from that file and pass data: one image, one audio clip,
  the decoder (`lite_tae` / `wan_vae`), the audio guidance (1.0–4.0) and the preview switch.
- No workflow field becomes a command, a path or an environment variable.

## How a job runs

1. The node writes `input.png`, `input.wav` (the audio unchanged, as IEEE-float WAV) and a
   schema-checked `job.json` into a fresh job directory under ComfyUI's temp folder.
2. It starts `[runtime python, <package>/runtime/leaptalk_job.py, job.json]` as an argument list
   (never through a shell) with an allowlisted environment: system variables needed by Windows,
   `PATH` limited to the runtime's Python folder and the system folder, a temp folder inside the job
   directory, offline flags for Hugging Face. ComfyUI's other variables (tokens, keys) are not passed.
3. The runner validates `job.json` again, refuses unknown keys, relative paths, `..`, and environment
   keys outside a short allowlist, removes credential-like variables from its own environment, checks
   the LeapTalk files against the pinned commit and the model files' sizes, and only then imports the
   model code.
4. Checkpoints are loaded with `torch.load(..., weights_only=True)` (audio projection) and safetensors
   (base model, LoRA, wav2vec2). The Lite decoder (`taew2_1.pth`) is loaded by upstream code with
   `weights_only=True`. The Wan VAE (`decoder = wan_vae` only) is loaded by upstream code with `torch.load`'s
   default, which is `weights_only=True` in torch 2.6 and later (the tested runtime uses 2.7.1).
5. ComfyUI reads back only fixed names inside the job directory (`events.jsonl`, `result.json`,
   `output.mp4`, `preview.jpg`), checks the recorded sha256 of the video and its frame count.

## Persistent worker

With `backend = persistent` (chosen on the LeapTalk Runtime node, or as the runtime's default in
`leaptalk.runtimes.json`), the first job starts `runtime/leaptalk_worker.py` with the runtime's
Python and the same allowlisted environment. It loads the model once and then waits for further jobs
from this ComfyUI process. It is a child process connected by pipes: it opens no port and serves no
HTTP.

- **What it accepts.** One JSON message per line (protocol version 1, at most 64 KiB, no pickle).
  A job request carries only a request id and a job id; the worker derives the job folder from its
  own jobs root, refuses ids with other characters, folders outside the root, symbolic links and
  junctions, duplicate request ids, and jobs whose upstream/model/ffmpeg/decoder/environment
  settings differ from the ones it was started with. Inputs and outputs travel as files in the job
  folder (as in one-shot mode), not through the pipe.
- **What it may use** is fixed when it starts (`<jobs root>/sessions/<worker id>/worker.json`,
  written by ComfyUI from the administrator's config) together with a frozen copy of the runtime
  scripts; editing the package while a worker runs does not change that worker.
- **When it is replaced.** Before each job ComfyUI compares the worker's identity (runtime id, Python,
  upstream folder and pinned commit, model folders and the size/modification time of every model file
  used, ffmpeg, environment, decoder, the runtime scripts' hash, protocol version) with the request
  and starts a new worker on any difference, reporting why. The worker itself checks before each job
  that its model files (size and modification time) and the pinned upstream files (content hashes)
  are unchanged and that the merged weights still have the fingerprint taken after loading; otherwise
  it refuses the job and exits. Full SHA-256 checks of the model files stay with **LeapTalk Doctor**.
- **When it is not reused.** Any unexpected error in a job (CUDA error, out of memory, an encoder
  failure, a protocol violation, a fingerprint change) ends the worker; the next job starts a new one.
  Nothing is retried automatically, and a failed persistent job never falls back to one-shot.
- **One at a time.** One lock in ComfyUI serialises worker start, jobs, Unload, the idle timer and
  one-shot jobs of this plugin; a one-shot job unloads an idle worker first. Other custom nodes and
  other ComfyUI processes are not coordinated.
- **Memory guard.** Before loading a model (a new worker) ComfyUI checks the system commit charge,
  available memory and the projected commit; before a job on a loaded worker it checks only that
  job's measured extra peak. While a job runs or the worker idles, a commit charge at or above
  `stop_commit_pct` for `stop_sustain_seconds` stops the job and the worker. A value that cannot be
  measured refuses the load. These checks reduce the risk of exhausting memory; they do not guarantee
  it cannot happen. One-shot jobs behave as in v0.1 (no guard; the process exits after the job).

## Stopping

- Windows: the runtime is placed in a Job Object created with `KILL_ON_JOB_CLOSE`. The process is
  created suspended, assigned to the job object and only then resumed, so the venv launcher and the
  interpreter it starts are both inside it (the persistent worker's interpreter reports its PID and
  ComfyUI checks that PID is in the job before using the worker). Cancel and timeout first ask the
  job to stop between chunks (`CANCEL` file; for the worker also a cancel message), then end the job
  object, which ends every process in it (the venv launcher, the interpreter, ffmpeg). If ComfyUI
  exits or crashes, Windows closes the job handle and ends the tree. No process is looked up or
  stopped by name. The job object is a lifetime control, not a security boundary.
- Linux/macOS: the runtime starts a new process group; cancel/timeout signal that group. The runner
  also stops when its stdin (held open by ComfyUI) closes, i.e. when ComfyUI goes away; the persistent
  worker exits on stdin EOF in the same way.
- Persistent worker: also ended by the **LeapTalk Worker** node (`unload`), by the idle timeout
  (`worker_idle_seconds`, default 120 s, counted from the end of the last job and never while a job
  runs), by memory pressure while idle, and when ComfyUI exits. Unload never cuts a running job off: it
  waits briefly and otherwise answers `busy`.

Verified on Windows with real jobs: cancel (stopped in about 2 s), timeout, a runtime error and a
hard-killed ComfyUI process left no runtime process behind and GPU memory returned to idle
(docs/TESTING.md). The POSIX paths are covered by CPU tests in CI, not by GPU runs.

## What this does not protect against

- A malicious LeapTalk checkout, Python package or model file: the runtime executes them.
- Anyone who can edit `leaptalk.runtimes.json` can make ComfyUI start any program.
- Generated videos can show a real person's face saying things they never said. Only use portraits
  and voices you have the rights and consent to use, and label generated media as AI-generated.
