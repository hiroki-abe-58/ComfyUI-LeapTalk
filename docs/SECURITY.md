# Security and trust model

ComfyUI-LeapTalk starts a separate Python process (the runtime) for every job. This page describes
what that process can be told to do and how it is stopped. It is **not a sandbox**: the runtime runs
with the rights of the user who started ComfyUI, and it executes the LeapTalk code and model files
the administrator installed.

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

## Stopping

- Windows: the runtime is placed in a Job Object created with `KILL_ON_JOB_CLOSE`. Cancel and
  timeout first create a `CANCEL` file (the runner stops between chunks), then end the job object,
  which ends every process in it (the venv launcher, the interpreter, ffmpeg). If ComfyUI exits or
  crashes, Windows closes the job handle and ends the tree. No PID lookup is involved.
- Linux/macOS: the runtime starts a new process group; cancel/timeout signal that group. The runner
  also stops when its stdin (held open by ComfyUI) closes, i.e. when ComfyUI goes away.

Verified on Windows with real jobs: cancel (stopped in about 2 s), timeout, a runtime error and a
hard-killed ComfyUI process left no runtime process behind and GPU memory returned to idle
(docs/TESTING.md). The POSIX paths are covered by CPU tests in CI, not by GPU runs.

## What this does not protect against

- A malicious LeapTalk checkout, Python package or model file: the runtime executes them.
- Anyone who can edit `leaptalk.runtimes.json` can make ComfyUI start any program.
- Generated videos can show a real person's face saying things they never said. Only use portraits
  and voices you have the rights and consent to use, and label generated media as AI-generated.
