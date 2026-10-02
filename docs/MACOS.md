# macOS / Apple Silicon: not supported (porting notes)

This release does **not** run on macOS. Nothing here was run on a Mac (no MPS or MLX test). These notes
come from reading LeapTalk `5d8fef8` and record what a port would have to change.

## Blockers found in the upstream code

| Where | What | Why it matters on Apple Silicon (MPS) |
| --- | --- | --- |
| `flash_head/src/modules/flash_head_model.py`, `rope_apply` / `precompute_freqs_cis` | rotary embeddings computed in float64 / complex128 (`view_as_complex(x.to(torch.float64))`) | MPS has no float64; a float32 path would change the numerics and has to be validated against CUDA outputs |
| `vibt/scheduler.py`, `set_parameters` | `torch.Generator("cuda")` whenever a seed is given (the inference script always passes one) | fails without CUDA; needs the pipeline device |
| `inference.py` chunk loop | `torch.cuda.synchronize()` and `torch.cuda.Event` timing (guarded by `use_cuda_timing`) | guarded, so it degrades to no timing; this package's runner uses CUDA events unconditionally and would need a host-timer fallback |
| `flash_head_model.py`, `AudioProjModel` | `torch.cuda.amp.autocast(dtype=torch.float32)` around a LayerNorm | CUDA autocast; on MPS it is a no-op with a warning, the result should be checked |
| `flash_head_model.py` imports | `xfuser` (multi-GPU) imported unconditionally | must be installable on macOS or replaced by a stub for the single-device path |
| attention | sageattention / flash-attn are CUDA-only | the PyTorch SDPA fallback (used for all results here) is the path to use |
| runtime process control | this package uses a Windows job object or POSIX process groups | the POSIX path applies to macOS but has no GPU test |

## Porting plan (not started)

1. Run the official script on CPU with float32 for one chunk to get a device-independent reference
   for the LeapTalk path (slow, but establishes expected tensors).
2. Patch, as tracked patches outside the upstream checkout: device-aware scheduler generator, a
   float32 rotary-embedding path, host timers. Compare MPS outputs with the CUDA/CPU reference per
   chunk (latents and decoded frames) and report the differences instead of claiming equality.
3. Measure memory on 32 GB / 64 GB unified memory: on CUDA the peak was about 8 GiB allocated with the
   Lite TAE decoder and about 6 GiB with the Wan VAE (docs/BENCHMARKS.md).
4. Only then add a `kind` for macOS runtimes and document it as experimental.
