# Setup

ComfyUI-LeapTalk adds three nodes to ComfyUI and runs LeapTalk itself in a **separate Python
environment** (the "runtime"). Nothing is installed into ComfyUI's Python, and nothing is downloaded
while a workflow runs.

Tested: Windows 11 host, ComfyUI v0.38.0, runtime venv on the same Windows machine (native), NVIDIA
RTX 5090 32 GB, driver 595.95. Linux hosts use the same steps with Linux paths but were **not tested
with real weights**. WSL2 runtimes are not supported in this release.

## 1. Install the node

```text
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-LeapTalk
```

No Python packages are added to ComfyUI.

## 2. Create the runtime environment

Any Python 3.12 works; the published results used `uv`:

```text
uv venv D:\leaptalk\venv --python 3.12
uv pip install --python D:\leaptalk\venv\Scripts\python.exe torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python D:\leaptalk\venv\Scripts\python.exe -r ComfyUI/custom_nodes/ComfyUI-LeapTalk/runtime/requirements-runtime.txt
```

- torch 2.7.1+cu128 is LeapTalk's pinned version with CUDA 12.8 wheels, which include RTX 50
  series (sm_120) kernels.
- flash-attn / SageAttention are optional upstream speed-ups and were **not** used for the
  results here (LeapTalk falls back to PyTorch scaled-dot-product attention).
- `torch.compile` is not used: the published LoRA does not need it (see docs/BENCHMARKS.md).

## 3. Get LeapTalk at the pinned commit

```text
git clone https://github.com/zhangrongxiang/LeapTalk D:\leaptalk\LeapTalk
git -C D:\leaptalk\LeapTalk checkout 5d8fef8dac3d1c9d3b3d5d9cceb6c76acc81e2c1
```

Keep the checkout unmodified and in its own folder. The runtime checks 14 upstream files against
the pinned commit (line endings are ignored) and refuses to run on a modified checkout.

## 4. Download the weights (pinned revisions)

```text
hf download z-rx/leaptalk --revision b33b788fd1c9b3df11627e00c6a0415272d829dc --local-dir D:\leaptalk\models\leaptalk
hf download facebook/wav2vec2-base-960h --revision 22aad52d435eb6dbaf354bdad9b0da84ce7d6156 --local-dir D:\leaptalk\models\wav2vec2-base-960h --include config.json preprocessor_config.json model.safetensors
hf download Soul-AILab/SoulX-FlashHead-1_3B --revision 59119b6c681230c3eeee157e224ae1941746711e --local-dir D:\leaptalk\models\SoulX-FlashHead-1_3B --include "Model_Pro/*" "VAE_Wan/*"
```

| File | Size | SHA-256 |
| --- | --- | --- |
| `leaptalk/lora/adapter_model.safetensors` | 188,803,712 | `30fdb8ff…` |
| `leaptalk/audio_proj_step_10400.pt` | 173,651,301 | `40b9b53c…` |
| `leaptalk/taew2_1.pth` (Lite decoder) | 22,679,486 | `f986092b…` |
| `SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors` | 6,030,864,656 | `e47e61b9…` |
| `SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth` (only for `decoder = wan_vae`) | 507,609,880 | `38071ab5…` |
| `wav2vec2-base-960h/model.safetensors` | 377,607,901 | `8aa76ab2…` |

Full hashes are in `runtime/leaptalk_job.py` (`MODEL_FILES`); **LeapTalk Doctor** with
`verify_sha256` checks them. `Model_Lite` and `VAE_LTX` of SoulX-FlashHead are not used.

## 5. ffmpeg

The runtime encodes the video (libx264) and muxes your audio (AAC) with ffmpeg. Any ffmpeg with
`libx264` and `aac` works; the results used the ffmpeg 7.1 build that `imageio-ffmpeg` 0.6.0 installs
(copy `...\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe` to e.g.
`D:\leaptalk\bin\ffmpeg.exe`).

## 6. Register the runtime

Create `ComfyUI/user/leaptalk.runtimes.json` (template:
[examples/leaptalk.runtimes.example.json](../examples/leaptalk.runtimes.example.json)) or point
`COMFYUI_LEAPTALK_CONFIG` at a file elsewhere:

```json
{
 "schema_version": 1,
 "runtimes": {
  "windows-native": {
   "kind": "native",
   "python": "D:\\leaptalk\\venv\\Scripts\\python.exe",
   "upstream_dir": "D:\\leaptalk\\LeapTalk",
   "models": {
    "soulx_dir": "D:\\leaptalk\\models\\SoulX-FlashHead-1_3B",
    "wav2vec_dir": "D:\\leaptalk\\models\\wav2vec2-base-960h",
    "leaptalk_dir": "D:\\leaptalk\\models\\leaptalk"
   },
   "ffmpeg": "D:\\leaptalk\\bin\\ffmpeg.exe",
   "env": {"CUDA_VISIBLE_DEVICES": "0"},
   "timeout_minutes": 30,
   "max_audio_seconds": 600
  }
 }
}
```

Only an administrator should edit this file: it decides which program ComfyUI starts. Workflows can
only pick a `runtime_id`. `max_audio_seconds` (up to 1800) rejects longer clips instead of cutting
them; `timeout_minutes` stops a job that runs too long.

Restart ComfyUI and run **LeapTalk Doctor** (workflow `workflows/leaptalk_doctor.json`).

## 7. Generate

Load `workflows/leaptalk_portrait_speech.json`, pick your runtime, a portrait (one image, ideally a
frontal head-and-shoulders photo; it is resized and center-cropped to 512x512) and a speech clip, and
queue it.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `runtime 'x' is not configured` | create `leaptalk.runtimes.json` (step 6) and restart ComfyUI |
| Doctor: `upstream_files fail` | the LeapTalk checkout is not at `5d8fef8` or was modified |
| Doctor: model file `size …, expected …` | incomplete download; download again with the pinned revision |
| `CUDA is not available` | the runtime venv has a CPU-only torch; reinstall with the cu128 index |
| `the audio is … s long; this runtime accepts up to …` | raise `max_audio_seconds` (max 1800) or split the audio |
| `LeapTalk takes exactly one reference image` | the IMAGE input is a batch; pick one image |
| Windows: memory pressure | see "Memory" in docs/BENCHMARKS.md: CUDA allocations count against the Windows commit limit |
