# Benchmarks, comparison and evaluation

Everything below is from one machine, one day (2026-10-03), with the setup in docs/SETUP.md. Raw
records: `docs/results/gpu_e2e_report.json` (ComfyUI HTTP end-to-end), `docs/results/reference_check.json`
(official script vs this package), `docs/results/quality_eval.json` (frame review and automatic
proxies), `docs/results/demo_manifest.json` (demo inputs and outputs).

## Setup

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 32 GB, driver 595.95, shared with the Windows desktop (about 2.4 GB in use at idle) |
| Host | Windows 11, 64 GB RAM, ComfyUI v0.38.0 (CPU-only PyTorch in ComfyUI's own venv) |
| Runtime | separate venv on the same Windows machine: Python 3.12.13, torch 2.7.1+cu128, diffusers 0.38.0, transformers 4.57.3, PEFT 0.19.1; attention: PyTorch SDPA (no flash-attn / SageAttention); no torch.compile |
| Upstream | LeapTalk `5d8fef8`, unmodified |
| Weights | z-rx/leaptalk `b33b788` (LoRA, `audio_proj_step_10400.pt`, `taew2_1.pth`), SoulX-FlashHead-1_3B `59119b6` (`Model_Pro`, `VAE_Wan`), wav2vec2-base-960h `22aad52` |
| Recipe | official defaults: 1 solver step per chunk (ViBT, shift 5), audio guidance 1.0, 33-frame window, 2 history latents (5 overlap frames), 28 new frames per chunk, 25 fps, 512x512, bf16, history update by decoder round-trip, color correction 1.0, Lite TAE decoder |
| Inputs | two fictional portraits (SDXL) and English speech (Kokoro-82M) made for this test, see docs/LICENSING.md |

## What runs (checked on every job)

- **Base**: SoulX-FlashHead-1_3B `Model_Pro` (`WanModelAudioProject`, bf16).
- **LeapTalk LoRA**: 480/480 checkpoint tensors present in the model and loaded with their exact
  values (r 64, alpha 64, q/k/v/o of self- and cross-attention in all 30 blocks), then merged
  (relative change of `blocks.0.self_attn.q.weight`: 0.156). The published LoRA has plain keys (no
  `._orig_mod.`), so the official script does not force `torch.compile`; this package does not compile
  either.
- **LeapTalk audio projection**: `audio_proj_step_10400.pt` loaded with `strict=True` and
  `weights_only=True`; values verified (relative change of `proj1.weight` vs the base model: 0.638).
- **Scheduler**: `ViBTScheduler` with timesteps `[1000]` (1 step; with one step the stochastic term is
  exactly 0, so the output does not depend on a seed), the reference-anchored source and the clamped
  history prefix of `_bridge_sample_one_chunk`.
- **Decoder**: LeapTalk Lite TAE (`taew2_1.pth`, TAEHV) or, optionally, the Wan2.1 VAE.
- **Generator calls**: 1 per chunk at audio guidance 1.0 (2 per chunk with audio CFG > 1), counted.

## Same frames as the official script

The official `inference.py` was run unchanged in a single process (`--compile off
--num_inference_steps 1 --lite`, which is what `inf.sh` with `torchrun --nproc_per_node=1` runs without
multi-GPU) with observation-only hooks recording every frame passed to its video writer.

| Input | Official frames | This package (direct) | This package through ComfyUI (clean install) |
| --- | --- | --- | --- |
| portrait A + 9.35 s speech | 252 written, MP4 cut to the audio by `-shortest` | first 234 frames **identical** (uint8, before encoding) | 234/234 **identical** |
| portrait B + 33.98 s speech | 868 written | – | 850/850 **identical** |

The package keeps `ceil(audio seconds x 25)` frames (the official script writes all generated
frames and lets ffmpeg's `-shortest` cut them) and muxes the original audio as AAC instead of MP3.
The MP4 bytes therefore differ; the generated frames do not. The ComfyUI path includes ComfyUI's
`LoadAudio` decoding and this package's float WAV hand-off, so that path is exact too.

## Speed (RTX 5090, Windows native runtime)

Through ComfyUI's HTTP API, one job per queue item (each job starts the runtime process and loads
the models), clean install:

| Speech | Output frames | Chunks | Queue -> done (wall) | Wall / audio seconds |
| --- | --- | --- | --- | --- |
| 0.5 s | 13 | 2 | 30.3 s | – |
| 9.35 s (A) | 234 | 9 | 34.7 s | 3.7 |
| 10.40 s (B) | 260 | 10 | 30.3 s | 2.9 |
| 30.78 s (A) | 770 | 28 | 40.3 s | 1.31 |
| 33.98 s (B) | 850 | 31 | 43.4 s | 1.28 |
| 96.5 s | 2414 | 87 | 70.6 s | 0.73 |

Where the time goes (runtime process, 34 s clip): Python imports 18.6 s (torch, diffusers,
transformers, PEFT, xfuser, mediapipe on Windows), model load 2.3 s (warm file cache; 6.0 s cold),
LoRA load + merge 1.1 s, reference encode 0.2 s, chunk loop 15.7 s, encoder finalize 0.2 s, output
check 0.5 s.

Per chunk (28 new frames, mean over chunks 3+ as the official script reports):

| | Lite TAE | Wan VAE | Lite TAE, audio CFG 2.0 |
| --- | --- | --- | --- |
| audio encoding (wav2vec2 over the 8 s window) | 0.013 s | 0.013 s | 0.012 s |
| generator (1 step) | 0.358 s | 0.359 s | 0.714 s (2 calls) |
| decode | 0.049 s | 0.830 s | 0.048 s |
| color correction + history round-trip | 0.011 s | – | 0.011 s |
| frame conversion + hand-off to the encoder | 0.058 s | 0.061 s | 0.059 s |
| **chunk total** | **0.43 s** | **1.28 s** | **0.79 s** |
| generation throughput | about 65 frames/s | about 22 frames/s | about 36 frames/s |

The output plays at 25 fps; "frames/s" is how fast frames are produced once the models are loaded,
not the end-to-end time of a job. These numbers are not comparable with the paper's figures (other
GPU, other measurement). No streaming to a viewer is implemented: the video is returned when the job
ends (previews of finished chunks are shown in ComfyUI while it runs).

## Memory

| | Lite TAE | Wan VAE |
| --- | --- | --- |
| CUDA after model load (allocated / reserved) | 3.2 / 3.4 GiB | 3.2 / 3.4 GiB |
| CUDA peak in the chunk loop (allocated / reserved) | 8.1 / 11.1 GiB | 6.1 / 7.6 GiB |
| whole GPU (incl. ~2.4 GB desktop) | 14.5 GB | – |
| runtime process tree RSS (peak, direct run) | about 5.1 GiB | – |

The peak does not grow with the length of the speech (frames are streamed to the encoder chunk by
chunk; a 96.5 s clip used the same peak). The Lite TAE decodes all frames of a chunk at once, which is
the peak; the Wan VAE path uses about 3.5 GiB less GPU memory but is about 3x slower per chunk.

**Windows commit charge.** On this Windows machine, CUDA memory counted against the system commit
charge almost 1:1 (WDDM). With ComfyUI and the runtime running, commit rose from about 73 % to a peak
of 92–93 % of the 93.3 GB commit limit, with other desktop applications open. If your commit charge
is already high, close other GPU applications first or use `decoder = wan_vae`. These are observed
peaks for this configuration, not minimum requirements.

## Output review

All four portrait/speech combinations (2 portraits x a ~10 s and a ~30 s script) were reviewed frame by
frame at word timings, plus the 0.5 s clip, 3 s of silence, a stereo 48 kHz input and a 96.5 s clip.
How: frames at the start/middle of the first word, two words in the middle, the last word and 0.2 s
after the end of speech, chunk-boundary frame pairs, and the first vs last frame. Nobody listened to
the outputs; the speech fixtures were checked with ASR instead (word error rate 3–6 %).

- The mouth moves from the first word ("Hello" at 0.3–0.5 s); the last word ("watching", "Goodbye")
  is visible with open mouth shapes; the mouth closes after the end of speech. Rounded vowels ("for",
  "open") show rounded lips.
- Identity stays stable over 34 s and 96.5 s (first vs last frame mean difference 3–4 of 255); blinks
  and small head motion appear.
- Chunk boundaries: the frame-to-frame change at a boundary averages 2.2–3.2 (of 255) vs 1.4–1.8
  inside chunks, maximum 6.3. That is a small measurable step; no visible seam in the reviewed pairs.
- Automatic proxy (not a lip-sync metric): correlation between mouth-region change and speech energy
  per frame 0.14–0.32 at a lag of 0–1 frames; mouth-region change is larger during speech than in
  silence (19.6–21.4 vs 14.0–17.9).
- Audio guidance 2.0 (experimental): strong artifacts (over-sharpened, discoloured lips, exaggerated
  features). Keep the default 1.0.
- Silence produces a mostly still, closed-mouth face; the 0.5 s clip gives 13 frames.

![Portrait A: input and frames at word timings](img/frames_a_short.jpg)

![Portrait B: input and frames at word timings](img/frames_b_short.jpg)

![Same frames: Lite TAE, Wan VAE, and audio CFG 2.0](img/decoders_and_cfg.jpg)

These are observations on two synthetic portraits and two synthetic voices, not a quality benchmark.
