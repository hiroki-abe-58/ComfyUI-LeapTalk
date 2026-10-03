"""Host memory snapshot, admission check and run-time stop rule for LeapTalk runtime processes.

Measurements. Windows: ``GetPerformanceInfo`` - ``CommitTotal`` (the system commit charge) and
``CommitLimit`` (physical memory plus page files), ``PhysicalAvailable``, all in pages. On the machine the
results were measured on, CUDA allocations counted against the commit charge almost 1:1 (an observation
on that machine, not a CUDA rule), so a model load can push commit toward its limit even when "free RAM"
looks large. Linux: ``MemAvailable`` and ``Committed_AS`` / ``CommitLimit`` from /proc/meminfo
(CommitLimit is only enforced with strict overcommit; otherwise only available memory is used). A value
that cannot be measured is "unknown", never "unlimited": loads and jobs are refused.

Everything is kept in bytes; GiB and percentages are only for messages and reports.

Admission (before a model load, and again before every job):

    estimate  = the expected additional peak of what is about to start, for its profile
                (decoder, audio guidance 1.0 or > 1):
                  cold = loading a runtime + its first job (one-shot jobs are always cold)
                  warm = one job on an already loaded worker (the loaded model is already in "commit now")
                = max(built-in or configured fallback, highest value measured in this ComfyUI process)
    margin    = 25 % of the estimate (fixed)
    projected = commit now + estimate + margin
    admitted  if projected < min(max_projected_commit_pct, stop_commit_pct) x commit limit
              and, for a load, commit now < start_max_commit_pct and available >= min_available_gib
              and, for a job on a loaded worker, commit now < stop_commit_pct

so a job whose projected peak reaches the run-time stop level is not started. The built-in fallbacks
are the highest values measured on the reference machine (Windows 11, RTX 5090, 64 GB; v0.2.0 runs); a
profile that was never measured gets a larger fallback.

Run time: a job (one-shot or persistent, including model loading) or an idle worker is stopped when
commit stays at or above ``stop_commit_pct`` for ``stop_sustain_seconds`` without a sample below it.

The thresholds are conservative values for one 64 GB workstation, not requirements; meeting them does
not guarantee that memory cannot run out.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass, field

GIB = 1024**3
SAFETY_MARGIN = 0.25  # added to every estimated peak; fixed, not a setting

DEFAULTS = {
    "enabled": True,
    "start_max_commit_pct": 90.0,  # no model load at or above this commit charge
    "min_available_gib": 8.0,  # ... or with less available physical memory
    "max_projected_commit_pct": 97.0,  # projected commit must stay below this (and below stop_commit_pct)
    "stop_commit_pct": 95.0,  # running job / loading / idle worker: stop when commit stays at or above this ...
    "stop_sustain_seconds": 5.0,  # ... for this long without a sample below it
}
# Optional administrator overrides of the fallback estimates (GiB), used for every decoder:
#   expected_worker_gib - cold (load + first job), expected_job_gib - warm (one job on a loaded worker)

# (decoder, guidance) -> GiB of commit added: the highest value measured on the reference machine in the
# v0.2.0 GPU runs (to 0.1 GiB); cold = commit high-water from before the load to the end of the first job,
# warm = high-water during a job minus the commit at its start
FALLBACK_GIB = {
    ("lite_tae", "single"): {"cold": 16.6, "warm": 8.1},
    ("wan_vae", "single"): {"cold": 13.1, "warm": 4.1},
}


class MemoryGuardRefused(RuntimeError):
    """The memory guard refused to start (kind "admission") or stopped (kind "stopped") a load or job."""

    def __init__(self, message: str, *, kind: str = "admission", report: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.report = report or {}


# --------------------------------------------------------------------------- measurement


def snapshot() -> dict:
    """{'measurable', 'source', 'commit_enforced', 't', '<x>_bytes' for total/available/commit/commit_limit,
    'commit_pct', and the same values in GiB (rounded, for display)}"""
    snap: dict = {"measurable": False, "source": None, "t": time.monotonic()}
    try:
        if os.name == "nt":
            from ctypes import wintypes

            class PI(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("CommitTotal", ctypes.c_size_t),
                    ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t),
                    ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wintypes.DWORD),
                    ("ProcessCount", wintypes.DWORD),
                    ("ThreadCount", wintypes.DWORD),
                ]

            pi = PI()
            pi.cb = ctypes.sizeof(pi)
            if ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(pi), pi.cb):
                ps = pi.PageSize  # the counts are in pages
                snap.update(
                    source="GetPerformanceInfo",
                    total_bytes=pi.PhysicalTotal * ps,
                    available_bytes=pi.PhysicalAvailable * ps,
                    commit_bytes=pi.CommitTotal * ps,
                    commit_limit_bytes=pi.CommitLimit * ps,
                )
        elif os.path.exists("/proc/meminfo"):
            vals = {}
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    k, _, rest = line.partition(":")
                    vals[k] = int(rest.split()[0]) * 1024
            try:
                with open("/proc/sys/vm/overcommit_memory", encoding="ascii") as f:
                    strict = f.read().strip() == "2"
            except OSError:
                strict = False
            snap.update(
                source="/proc/meminfo",
                total_bytes=vals["MemTotal"],
                available_bytes=vals["MemAvailable"],
                commit_bytes=vals["Committed_AS"],
                commit_limit_bytes=vals["CommitLimit"],
                commit_enforced=strict,  # heuristic overcommit: CommitLimit is not a limit, Committed_AS may exceed it
            )
    except Exception:  # noqa: BLE001 - any failure means "unknown", never "unlimited"
        snap = {"measurable": False, "source": None, "t": snap["t"]}
    if snap.get("commit_limit_bytes"):
        snap["commit_pct"] = round(100.0 * snap["commit_bytes"] / snap["commit_limit_bytes"], 2)
        snap["measurable"] = True
    snap.setdefault("commit_enforced", os.name == "nt")
    for k in ("total", "available", "commit", "commit_limit"):
        if f"{k}_bytes" in snap:
            snap[f"{k}_gib"] = round(snap[f"{k}_bytes"] / GIB, 2)
    return snap


def _b(snap: dict, key: str) -> int | None:
    """``<key>_bytes`` of a snapshot (or ``<key>_gib`` converted, for hand-made snapshots)."""
    if snap.get(f"{key}_bytes") is not None:
        return int(snap[f"{key}_bytes"])
    if snap.get(f"{key}_gib") is not None:
        return int(snap[f"{key}_gib"] * GIB)
    return None


def _pct(snap: dict) -> float | None:
    if snap.get("commit_pct") is not None:
        return float(snap["commit_pct"])
    c, lim = _b(snap, "commit"), _b(snap, "commit_limit")
    return 100.0 * c / lim if c is not None and lim else None


def gib(nbytes: float | None) -> str:
    return "?" if nbytes is None else f"{nbytes / GIB:.1f} GiB"


def settings(cfg: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update(cfg or {})
    return out


def _floor_bytes(cfg: dict) -> int:
    """Available memory below which a running job / an idle worker counts as under pressure."""
    return int(max(0.5, cfg["min_available_gib"] / 4) * GIB)


# --------------------------------------------------------------------------- estimates


def profile_of(job: dict) -> tuple:
    g = float(job.get("audio_guidance", 1.0))
    return (job["decoder"], "single" if g <= 1.0 else "cfg")


def profile_name(profile: tuple) -> str:
    return f"{profile[0]}, audio guidance {'1.0' if profile[1] == 'single' else '> 1'}"


def fallback_bytes(profile: tuple, kind: str) -> int:
    known = FALLBACK_GIB.get(profile)
    if known is not None:
        return int(known[kind] * GIB)
    # never measured: guidance > 1 runs two generator passes per chunk - count the job part twice
    base = FALLBACK_GIB.get((profile[0], "single")) or FALLBACK_GIB[("lite_tae", "single")]
    extra = base["warm"]
    return int((base[kind] + extra) * GIB)


class PeakEstimates:
    """Highest additional peak measured in this ComfyUI process, per (profile, cold|warm). Values only
    go up, so a small job (e.g. a 0.5 s clip whose peak fell between samples) never lowers the estimate
    for a larger one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hw: dict = {}

    def observe(self, profile: tuple, kind: str, nbytes: int | None, source: str) -> None:
        if nbytes is None or nbytes <= 0:
            return
        with self._lock:
            rec = self._hw.setdefault((profile, kind), {"bytes": 0, "samples": 0, "sources": {}})
            rec["samples"] += 1
            rec["sources"][source] = max(rec["sources"].get(source, 0), int(nbytes))
            rec["bytes"] = max(rec["bytes"], int(nbytes))

    def estimate(self, profile: tuple, kind: str, cfg: dict) -> dict:
        key = "expected_worker_gib" if kind == "cold" else "expected_job_gib"
        if key in cfg:
            base, base_src = int(float(cfg[key]) * GIB), f"runtime config memory_guard.{key}"
        else:
            base = fallback_bytes(profile, kind)
            base_src = (
                "built-in (highest value measured on the reference machine)" if profile in FALLBACK_GIB else "built-in (profile never measured: conservative)"
            )
        with self._lock:
            hw = dict(self._hw.get((profile, kind)) or {})
        use_hw = bool(hw) and hw["bytes"] > base
        return {
            "profile": profile_name(profile),
            "kind": kind,
            "bytes": hw["bytes"] if use_hw else base,
            "source": f"measured in this ComfyUI process (high-water of {hw['samples']} job(s))" if use_hw else base_src,
            "fallback_bytes": base,
            "high_water_bytes": hw.get("bytes"),
            "samples": hw.get("samples", 0),
        }

    def describe(self) -> list:
        with self._lock:
            return [
                {"profile": profile_name(p), "kind": k, "high_water_gib": round(v["bytes"] / GIB, 2), "samples": v["samples"]}
                for (p, k), v in sorted(self._hw.items())
            ]


# --------------------------------------------------------------------------- admission


def admit(snap: dict, cfg: dict, est: dict, kind: str) -> dict:
    """Admission decision for a load (``kind="cold"``) or a job on a loaded worker (``"warm"``).
    Returns a report dict; ``report["admitted"]`` is the decision, ``report["reason"]`` the message."""
    est_b = int(est["bytes"])
    margin_b = int(round(est_b * SAFETY_MARGIN))
    rep: dict = {
        "admitted": False,
        "kind": kind,
        "profile": est.get("profile"),
        "guard_enabled": bool(cfg["enabled"]),
        "source": snap.get("source"),
        "commit_enforced": snap.get("commit_enforced"),
        "commit_now_bytes": _b(snap, "commit"),
        "commit_limit_bytes": _b(snap, "commit_limit"),
        "commit_now_pct": _pct(snap),
        "physical_available_bytes": _b(snap, "available"),
        "estimated_additional_peak_bytes": est_b,
        "estimate_source": est.get("source"),
        "estimate_samples": est.get("samples", 0),
        "safety_margin_bytes": margin_b,
        "safety_margin_pct": SAFETY_MARGIN * 100,
        "start_limit_pct": cfg["start_max_commit_pct"],
        "stop_limit_pct": cfg["stop_commit_pct"],
        "projected_limit_pct": min(cfg["max_projected_commit_pct"], cfg["stop_commit_pct"]),
        "projected_commit_bytes": None,
        "projected_commit_pct": None,
        "shortfall_bytes": 0,
    }
    what = "load a model and run its first job" if kind == "cold" else "start a job on the loaded worker"
    if not cfg["enabled"]:
        rep.update(admitted=True, reason="memory guard disabled by the runtime config")
        return rep
    if not snap.get("measurable"):
        rep["reason"] = f"cannot measure host memory (commit charge unknown); refusing to {what}"
        return rep
    avail = rep["physical_available_bytes"]
    need_avail = int(cfg["min_available_gib"] * GIB) if kind == "cold" else _floor_bytes(cfg)
    if not snap.get("commit_enforced"):
        # Linux without strict overcommit: the commit charge is not a limit; only available memory counts
        if avail < need_avail:
            rep.update(shortfall_bytes=need_avail - avail, reason=f"only {gib(avail)} of memory available (needs {gib(need_avail)} to {what})")
            return rep
        rep.update(admitted=True, reason=f"ok: {gib(avail)} available (commit not enforced on this system)")
        return rep
    limit_b = rep["commit_limit_bytes"]
    now_b = rep["commit_now_bytes"]
    now_pct = rep["commit_now_pct"]
    projected_b = now_b + est_b + margin_b
    rep["projected_commit_bytes"] = projected_b
    rep["projected_commit_pct"] = round(100.0 * projected_b / limit_b, 2)
    if kind == "cold" and now_pct >= cfg["start_max_commit_pct"]:
        rep.update(
            shortfall_bytes=int(now_b - cfg["start_max_commit_pct"] / 100.0 * limit_b),
            reason=f"system commit charge is {now_pct:.1f} %; no model is loaded at or above {cfg['start_max_commit_pct']:.0f} %",
        )
        return rep
    if kind == "warm" and now_pct >= cfg["stop_commit_pct"]:
        rep.update(
            shortfall_bytes=int(now_b - cfg["stop_commit_pct"] / 100.0 * limit_b),
            reason=f"system commit charge is {now_pct:.1f} %, at or above the stop level {cfg['stop_commit_pct']:.0f} %",
        )
        return rep
    if avail is None or avail < need_avail:
        rep.update(shortfall_bytes=need_avail - (avail or 0), reason=f"only {gib(avail)} of physical memory available (needs {gib(need_avail)} to {what})")
        return rep
    eff_limit_b = int(rep["projected_limit_pct"] / 100.0 * limit_b)
    if projected_b >= eff_limit_b:
        which = "the stop level" if cfg["stop_commit_pct"] <= cfg["max_projected_commit_pct"] else "max_projected_commit_pct"
        rep.update(
            shortfall_bytes=projected_b - eff_limit_b,
            reason=(
                f"commit would reach about {rep['projected_commit_pct']:.1f} % ({gib(now_b)} now + {gib(est_b)} estimated peak "
                f"+ {gib(margin_b)} safety margin, of {gib(limit_b)}); the limit is {rep['projected_limit_pct']:.0f} % ({which}), "
                f"about {gib(projected_b - eff_limit_b)} short (an estimate; freeing that much does not guarantee success)"
            ),
        )
        return rep
    rep.update(admitted=True, reason=f"ok: commit {now_pct:.1f} %, projected {rep['projected_commit_pct']:.1f} % (limit {rep['projected_limit_pct']:.0f} %)")
    return rep


def check_start(snap: dict, cfg: dict, expected_add_gib: float, *, load: bool = True) -> tuple[bool, str]:
    """Compatibility wrapper: admission with a fixed estimate (GiB)."""
    est = {"bytes": int(expected_add_gib * GIB), "source": "fixed", "profile": None}
    rep = admit(snap, cfg, est, "cold" if load else "warm")
    return rep["admitted"], rep["reason"]


# --------------------------------------------------------------------------- run time


@dataclass
class SustainedPressure:
    """True once commit has stayed at/above ``stop_commit_pct`` (or memory was unmeasurable) for
    ``stop_sustain_seconds`` without interruption; any sample below the stop level restarts the timer,
    so momentary peaks do not stop a job.

    The hysteresis is between stopping and loading again: a stopped worker is not restarted
    automatically, and the next load needs commit below ``start_max_commit_pct`` (default 90 %, five
    points under the default stop level) plus room for its projected peak."""

    cfg: dict
    since: float | None = None
    last: dict = field(default_factory=dict)

    def update(self, snap: dict) -> bool:
        self.last = snap
        if not self.cfg["enabled"]:
            return False
        now = snap.get("t", time.monotonic())
        if snap.get("measurable") and not snap.get("commit_enforced"):
            high = (_b(snap, "available") or 0) < _floor_bytes(self.cfg)  # Linux heuristic overcommit: watch available memory instead
        else:
            pct = _pct(snap)
            high = (not snap.get("measurable")) or pct is None or pct >= self.cfg["stop_commit_pct"]
        if not high:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.cfg["stop_sustain_seconds"]


class Tracker:
    """Commit high-water between two points (start of a load or job, its end), for the estimates."""

    def __init__(self, snap: dict):
        self.start = _b(snap, "commit")
        self.max = self.start

    def update(self, snap: dict) -> None:
        c = _b(snap, "commit")
        if c is not None and (self.max is None or c > self.max):
            self.max = c

    def delta(self) -> int | None:
        return None if self.start is None or self.max is None else self.max - self.start
