"""Start and stop one runtime process tree, owned by this ComfyUI process.

Windows: the runner is assigned to a Job Object created with KILL_ON_JOB_CLOSE. Every process it
starts (the venv launcher's interpreter, ffmpeg) is in the same job, ``terminate()`` ends exactly
that tree (no PID lookup, so no PID-reuse risk), and if ComfyUI exits or crashes the operating system
closes the job handle and ends the tree. Linux/macOS: the runner starts a new session (process
group); ``terminate()`` signals that group, and the runner watches its stdin, which ComfyUI holds
open, so it stops when ComfyUI goes away.

This is process ownership, not a sandbox: the runtime runs with the rights of the ComfyUI user.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from pathlib import Path

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    from ctypes import wintypes

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("Read", "Write", "Other", "ReadT", "WriteT", "OtherT")]

    class _EXT_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMIT),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _ACCOUNTING(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x0200
    _ExtendedLimitInformation = 9
    _BasicAccountingInformation = 1


class ProcessTree:
    """One runtime process (and its children) owned by the caller."""

    def __init__(self, argv: list[str], *, cwd: Path, env: dict, stdout: Path, stderr: Path, memory_limit_bytes: int | None = None):
        self.job = None
        self.peak_job_memory = None
        if IS_WINDOWS:
            self.job = _k32.CreateJobObjectW(None, None)
            if not self.job:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
            info = _EXT_LIMIT()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if memory_limit_bytes:
                info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_JOB_MEMORY
                info.JobMemoryLimit = int(memory_limit_bytes)
            if not _k32.SetInformationJobObject(self.job, _ExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
                err = ctypes.get_last_error()
                _k32.CloseHandle(self.job)
                raise OSError(err, "SetInformationJobObject failed")
        kwargs: dict = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        with open(stdout, "wb") as out, open(stderr, "wb") as err:
            self.proc = subprocess.Popen(  # noqa: S603 - argv list from the admin config + this package's script, no shell
                argv, stdin=subprocess.PIPE, stdout=out, stderr=err, env=env, cwd=str(cwd), close_fds=True, **kwargs
            )
        if IS_WINDOWS:
            # The runner starts its own children (ffmpeg) only after model loading, long after this call;
            # children created after assignment belong to the job automatically.
            if not _k32.AssignProcessToJobObject(self.job, int(self.proc._handle)):
                err = ctypes.get_last_error()
                self.proc.kill()
                self.close()
                raise OSError(err, "AssignProcessToJobObject failed")
            self.pgid = None
        else:
            self.pgid = os.getpgid(self.proc.pid)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def poll(self) -> int | None:
        return self.proc.poll()

    def close_stdin(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass

    def active_processes(self) -> int:
        """Processes of this tree that are still alive (best effort on POSIX)."""
        if IS_WINDOWS:
            if not self.job:
                return 0
            acc = _ACCOUNTING()
            if not _k32.QueryInformationJobObject(self.job, _BasicAccountingInformation, ctypes.byref(acc), ctypes.sizeof(acc), None):
                return -1
            return int(acc.ActiveProcesses)
        try:
            os.killpg(self.pgid, 0)
            return 1
        except ProcessLookupError:
            return 0
        except PermissionError:
            return -1

    def _read_peak_memory(self) -> None:
        if IS_WINDOWS and self.job:
            info = _EXT_LIMIT()
            if _k32.QueryInformationJobObject(self.job, _ExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info), None):
                self.peak_job_memory = int(info.PeakJobMemoryUsed)

    def terminate(self, grace_s: float = 5.0) -> dict:
        """End the whole tree now. Returns what was left before/after."""
        before = self.active_processes()
        self.close_stdin()
        if IS_WINDOWS:
            if self.job:
                _k32.TerminateJobObject(self.job, 1)
        else:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.pgid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + grace_s
                while time.monotonic() < deadline and self.active_processes() > 0:
                    time.sleep(0.1)
                if self.active_processes() == 0:
                    break
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and self.active_processes() > 0:
            time.sleep(0.1)
        return {"active_before": before, "active_after": self.active_processes()}

    def close(self) -> None:
        """Release the tree. On Windows, closing the job handle also ends anything still running in it."""
        self.close_stdin()
        if IS_WINDOWS and self.job:
            self._read_peak_memory()
            _k32.CloseHandle(self.job)
            self.job = None
