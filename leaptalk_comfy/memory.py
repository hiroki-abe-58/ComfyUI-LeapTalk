"""Host memory snapshot and the start/run guard used for runtime processes.

Windows: physical memory and the system commit charge from ``GetPerformanceInfo``. On the machine
the results were measured on, CUDA allocations counted against the commit charge almost 1:1, so a
model load can push commit toward its limit even when "free RAM" looks large. Linux: ``MemAvailable``
and ``Committed_AS`` / ``CommitLimit`` from /proc/meminfo (CommitLimit is only enforced with strict
overcommit). A value that cannot be measured is reported as unknown and is never treated as
"unlimited": the guard then refuses to start a new runtime unless the administrator disabled it.

The default thresholds are conservative settings for a 64 GB Windows workstation, not requirements.
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass, field

GIB = 1024**3

DEFAULTS = {
    "enabled": True,
    "start_max_commit_pct": 90.0,  # do not start a new runtime above this commit charge
    "min_available_gib": 8.0,  # ... or below this much available physical memory
    "max_projected_commit_pct": 97.0,  # commit now + the expected addition must stay below this
    "expected_worker_gib": 16.0,  # commit added by a new runtime (measured: ~16-18 GiB incl. CUDA)
    "expected_job_gib": 8.0,  # extra commit of one job on a loaded runtime, until measured
    "stop_commit_pct": 95.0,  # running job / idle worker: stop when commit stays at or above this ...
    "stop_sustain_seconds": 5.0,  # ... for this long
}


def snapshot() -> dict:
    """{'measurable': bool, 'source': str, 'total_gib', 'available_gib', 'commit_gib', 'commit_limit_gib', 'commit_pct', 't'}"""
    snap = {"measurable": False, "source": None, "t": time.monotonic()}
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
                ps = pi.PageSize
                snap.update(
                    source="GetPerformanceInfo",
                    total_gib=pi.PhysicalTotal * ps / GIB,
                    available_gib=pi.PhysicalAvailable * ps / GIB,
                    commit_gib=pi.CommitTotal * ps / GIB,
                    commit_limit_gib=pi.CommitLimit * ps / GIB,
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
                total_gib=vals["MemTotal"] / GIB,
                available_gib=vals["MemAvailable"] / GIB,
                commit_gib=vals["Committed_AS"] / GIB,
                commit_limit_gib=vals["CommitLimit"] / GIB,
                commit_enforced=strict,  # heuristic overcommit: CommitLimit is not a limit, Committed_AS may exceed it
            )
    except Exception:  # noqa: BLE001 - any failure means "unknown", never "unlimited"
        snap["source"] = None
    if snap.get("commit_limit_gib"):
        snap["commit_pct"] = 100.0 * snap["commit_gib"] / snap["commit_limit_gib"]
        snap["measurable"] = True
    snap.setdefault("commit_enforced", os.name == "nt")
    for k in ("total_gib", "available_gib", "commit_gib", "commit_limit_gib", "commit_pct"):
        if k in snap:
            snap[k] = round(snap[k], 2)
    return snap


def settings(cfg: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update(cfg or {})
    return out


def _floor_gib(cfg: dict) -> float:
    """Available memory below which a running job / an idle worker counts as under pressure."""
    return max(0.5, cfg["min_available_gib"] / 4)


def check_start(snap: dict, cfg: dict, expected_add_gib: float, *, load: bool = True) -> tuple[bool, str]:
    """May a new runtime load a model (``load=True``), or a job start on an already loaded worker
    (``load=False``), adding about ``expected_add_gib`` of commit?

    A load must start below ``start_max_commit_pct`` with ``min_available_gib`` free. A warm job loads
    nothing; the resident model is already in the commit charge, so it only needs to stay below the
    stop level and keep its own extra peak under ``max_projected_commit_pct``.
    """
    if not cfg["enabled"]:
        return True, "memory guard disabled by the runtime config"
    what = "load a model" if load else "start a job"
    if not snap.get("measurable"):
        return False, f"cannot measure host memory (commit charge unknown); refusing to {what} - set memory_guard.enabled=false only if you accept that"
    need_avail = cfg["min_available_gib"] if load else _floor_gib(cfg)
    if not snap.get("commit_enforced"):
        # Linux without strict overcommit: the commit charge is not a limit; only available memory counts
        if snap["available_gib"] < need_avail:
            return False, f"only {snap['available_gib']:.1f} GiB of memory available (needs {need_avail:.1f} GiB to {what})"
        return True, f"ok: available {snap['available_gib']:.1f} GiB (commit not enforced on this system)"
    if load and snap["commit_pct"] >= cfg["start_max_commit_pct"]:
        return False, f"system commit charge is {snap['commit_pct']:.1f} % (limit {cfg['start_max_commit_pct']:.0f} % before a load)"
    if not load and snap["commit_pct"] >= cfg["stop_commit_pct"]:
        return False, f"system commit charge is {snap['commit_pct']:.1f} % (at or above the stop level {cfg['stop_commit_pct']:.0f} %)"
    if snap["available_gib"] < need_avail:
        return False, f"only {snap['available_gib']:.1f} GiB of physical memory available (needs {need_avail:.1f} GiB to {what})"
    projected = 100.0 * (snap["commit_gib"] + expected_add_gib) / snap["commit_limit_gib"]
    if projected >= cfg["max_projected_commit_pct"]:
        return False, (
            f"commit would reach about {projected:.1f} % ({snap['commit_gib']:.1f} + {expected_add_gib:.1f} GiB of "
            f"{snap['commit_limit_gib']:.1f} GiB); limit {cfg['max_projected_commit_pct']:.0f} %"
        )
    return True, f"ok: commit {snap['commit_pct']:.1f} %, available {snap['available_gib']:.1f} GiB, projected {projected:.1f} %"


@dataclass
class SustainedPressure:
    """True once commit has stayed at/above ``stop_commit_pct`` (or memory was unmeasurable) for
    ``stop_sustain_seconds`` without interruption; any sample below the stop level restarts the timer,
    so momentary peaks do not stop a job.

    The hysteresis is between stopping and loading again: a stopped worker is not restarted
    automatically, and the next load needs commit below ``start_max_commit_pct`` (default 90 %, five
    points under the default stop level), so a job near the limit cannot cause a stop/restart loop."""

    cfg: dict
    since: float | None = None
    last: dict = field(default_factory=dict)

    def update(self, snap: dict) -> bool:
        self.last = snap
        if not self.cfg["enabled"]:
            return False
        now = snap.get("t", time.monotonic())
        if snap.get("measurable") and not snap.get("commit_enforced"):
            high = snap["available_gib"] < _floor_gib(self.cfg)  # Linux heuristic overcommit: watch available memory instead
        else:
            high = (not snap.get("measurable")) or snap["commit_pct"] >= self.cfg["stop_commit_pct"]
        if not high:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.cfg["stop_sustain_seconds"]
