# Changelog

## 0.2.1 (not released - work in progress)

**Status:** not released. The real-GPU check of this version could not run on the test machine: with the
memory load of its other applications (about 70 GiB of the 93.3 GiB commit limit before ComfyUI started),
the stricter admission below refused every load, including `wan_vae` (projected 95.1-95.3 % against the
95 % stop level, about 0.1-0.3 GiB short) - see `docs/results/v021/clean_install_22bce0d_blocked.json`.
Untested runtime changes are not released.

Memory-guard fixes and an explicit lower-memory workflow. Generation is unchanged (same model, LoRA,
scheduler, decoders, audio handling); node inputs and outputs, workflow and runtime-config formats are
unchanged.

**Behaviour change: one-shot jobs are now protected by the memory guard as well.** With
`memory_guard.enabled` true (the default), a one-shot job - including the v0.1 workflow - is refused
before it starts, refused after its model has loaded, or stopped while it loads or generates when memory
is too tight, exactly like a persistent job. In v0.1.0/v0.2.0 one-shot jobs were not checked. A refused
job is not retried and never switched to another backend or decoder.

- **No limit was loosened.** Defaults stay `start_max_commit_pct` 90, `max_projected_commit_pct` 97,
  `stop_commit_pct` 95, `stop_sustain_seconds` 5, and every estimated peak gets a fixed 25 % safety margin.
- **Admission is stricter and consistent with the stop rule.** The projected commit (commit now +
  estimated additional peak + 25 %) must stay below the lower of `max_projected_commit_pct` and
  `stop_commit_pct`: a job that is predicted to reach the stop level is no longer started (0.2.0 allowed
  projections up to 97 %).
- **Separate cold and warm estimates per decoder.** Cold (a model load plus its first job, also every
  one-shot job) and warm (one job on a loaded worker) are estimated separately for `lite_tae` and
  `wan_vae` (and audio guidance > 1). The estimate never goes below the highest value measured on the
  reference machine, and rises to the highest peak measured in the running ComfyUI process (Windows
  commit change around each load/job and the job's CUDA reserved peak). 0.2.0 used one 16 GiB value for
  every load without a margin, and the last job's CUDA peak alone for the next job.
- **The first job of a new worker is checked again** once the model is loaded, against the measured
  commit charge.
- **Start-up is watched.** While a persistent worker or a one-shot runtime imports and loads (when it may
  not answer), the stop rule runs in ComfyUI and ends the process tree through the job object.
- Refusal messages name what is missing (projected commit, the limit, the estimated shortfall), what the
  loaded worker holds, and the options (Unload, the lower-memory workflow, closing other applications).
  Memory-guard refusal, memory-guard stop, user cancel, timeout and runtime errors are reported as
  different errors.
- New `workflows/leaptalk_persistent_lower_memory.json`: persistent backend with `decoder = wan_vae`
  (lower peak memory, slower decoding, frames differ from Lite TAE), Worker status after each job. The
  existing workflows and node defaults stay Lite TAE.
- Worker status shows the measured estimates and the commit the loaded worker holds.

Limits that remain: on a machine whose other applications already use most of the commit limit,
Lite TAE jobs (and possibly even `wan_vae` loads) are refused; see docs/BENCHMARKS.md. Nobody has
listened to the generated videos with sound yet.

## 0.2.0 (pre-release)

Optional persistent worker (`backend = persistent`, LeapTalk Worker node), one-shot by default. See the
v0.2.0 release notes and docs/BENCHMARKS.md.

## 0.1.0 (pre-release)

First release: LeapTalk Generate / Runtime / Doctor, one-shot runtime per job.
