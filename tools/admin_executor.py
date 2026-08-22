"""Windows elevated command execution via UAC.

Provides the ability to run terminal commands with administrator privileges
on Windows by using ``ShellExecuteExW`` with the ``runas`` verb (the standard
UAC elevation prompt), then waiting on the returned process handle.

Safety guarantees:
- No UAC bypass: uses the standard ShellExecuteExW("runas") contract
- No credential storage: each execution is an independent process
- No silent elevation: UAC prompt is always shown to the user
- Explicit opt-in: elevated=True must be passed explicitly
- Sanitized environment: the elevated child receives the SAME scrubbed
  environment as a normal local terminal subprocess (reuses
  tools.environments.local.build_subprocess_env), never the raw Hermes
  service-process environment.  It is passed through a structured JSON file
  to a minimal trusted helper (tools/elevated_helper.py) and applied via
  CreateProcessW(lpEnvironment=...) — never re-parsed through shell text.
- Process lifecycle: ShellExecuteExW returns an hProcess handle
  (SEE_MASK_NOCLOSEPROCESS); we wait on it with WaitForSingleObject and close
  it on every path.  The done marker file is only a secondary signal.
"""

from __future__ import annotations

import codecs
import ctypes
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
from ctypes import wintypes


_TIMEOUT_S = 120
_POLL_INTERVAL_S = 0.5

# Chunk size for the bounded streaming char-count pass (_count_utf8_chars).
# Any fixed power of two; the pass must stay memory-bounded regardless of
# how large the elevated output file grows.
_COUNT_CHUNK_BYTES = 64 * 1024

# Fallback cap if tools.tool_output_limits cannot be imported (e.g. isolated
# unit-test import). Mirrors the historical terminal output cap.
_FALLBACK_OUTPUT_MAX_CHARS = 50_000

# Allowlist for the working directory that gets embedded into the elevated
# .cmd script (mirrors tools/terminal_tool._WORKDIR_SAFE_RE so the two layers
# enforce the same rule — no second, inconsistent policy).  The cwd is
# interpolated inside `cd /d "..."`, so shell metacharacters there could
# escape the quotes and inject into the script.  terminal_tool already
# validates the explicit ``workdir`` argument; this is a cheap defense-in-
# depth check for the resolved/session cwd path too.
_WORKDIR_SAFE_RE = re.compile(r"^[A-Za-z0-9/\\:_\- .~ +@=,]+$")


def _validate_cwd_for_script(cwd: str) -> str | None:
    """Return an error message if *cwd* is unsafe to embed in a .cmd script."""
    if not cwd:
        return None
    if not _WORKDIR_SAFE_RE.match(cwd):
        for ch in cwd:
            if not _WORKDIR_SAFE_RE.match(ch):
                return (
                    f"Blocked: working directory contains disallowed character "
                    f"{repr(ch)}. Use a simple filesystem path without shell "
                    f"metacharacters."
                )
        return "Blocked: working directory contains disallowed characters."
    return None


def _output_max_chars() -> int:
    """Return the model-visible output cap, mirroring the terminal pipeline.

    Reuses tools.tool_output_limits so elevated output honors the same
    cap/truncation semantics as normal terminal output. Falls back to a fixed
    constant only if that module cannot be imported (isolated tests).
    """
    try:
        from tools.tool_output_limits import get_max_bytes

        return get_max_bytes()
    except Exception:
        return _FALLBACK_OUTPUT_MAX_CHARS


def _read_output_bounded(output_file: str, max_chars: int) -> tuple[str, int]:
    """Read *output_file* with a single bounded character-level streaming pass.

    Returns ``(output, total_chars)``.  ``total_chars`` is the exact Unicode
    character count of the decoded stream (never a byte count), accumulated
    through an incremental UTF-8 decoder over fixed-size binary reads — no
    ``os.path.getsize()`` gate and no byte-offset ``seek()`` for the tail.

    - When ``total_chars <= max_chars`` the FULL original text is returned
      (bounded by ``max_chars`` characters), without truncation or overlap.
    - When the output overflows, only the 40% head / 60% tail window is
      retained (matching the terminal pipeline's truncation shape) using a
      bounded deque for the tail; the two windows are disjoint by
      construction (the tail starts after the head because the total exceeds
      ``head_chars + tail_chars``), so no characters are duplicated or
      dropped and no replacement character appears at the tail from a
      mid-codepoint cut.
    """
    from collections import deque

    head_chars = int(max_chars * 0.4)
    tail_chars = max_chars - head_chars

    full: list[str] = []        # complete text while total <= max_chars
    head: list[str] = []        # first head_chars characters (overflow path)
    head_len = 0
    tail: deque[str] = deque()  # last tail_chars characters (overflow path)
    tail_len = 0
    total_chars = 0
    overflow = False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _ingest(text: str) -> None:
        nonlocal total_chars, head_len, tail_len, overflow
        if not text:
            return
        total_chars += len(text)
        if not overflow:
            if total_chars <= max_chars:
                full.append(text)
                return
            # Transition into overflow: fold everything seen so far (at most
            # max_chars chars) plus this chunk into the head/tail windows.
            overflow = True
            combined = "".join(full) + text
            head.append(combined[:head_chars])
            head_len = min(head_chars, len(combined))
            if len(combined) > tail_chars:
                tail.append(combined[-tail_chars:])
                tail_len = tail_chars
            else:
                tail.append(combined)
                tail_len = len(combined)
            full.clear()
            return
        # Overflow: the head is already fixed; keep only the bounded tail.
        tail.append(text)
        tail_len += len(text)
        while tail_len > tail_chars:
            first = tail[0]
            if len(first) <= tail_len - tail_chars:
                tail.popleft()
                tail_len -= len(first)
            else:
                excess = tail_len - tail_chars
                tail[0] = first[excess:]
                tail_len -= excess

    try:
        with open(output_file, "rb") as f:
            while True:
                chunk = f.read(_COUNT_CHUNK_BYTES)
                if not chunk:
                    break
                _ingest(decoder.decode(chunk))
        _ingest(decoder.decode(b"", final=True))
    except OSError:
        return "", 0

    if not overflow:
        return "".join(full), total_chars
    return "".join(head) + "".join(tail), total_chars


def is_windows() -> bool:
    """Return True on Windows."""
    return sys.platform == "win32"


def is_running_as_admin() -> bool:
    """Return True when the current Windows process is elevated."""
    if not is_windows():
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Windows API adapter — a thin, injectable wrapper over the ctypes calls so
# tests can substitute a fake and exercise every lifecycle path (cancelled
# UAC, access denied, WAIT_TIMEOUT, WAIT_FAILED, handle close) without a real
# UAC prompt.
# ---------------------------------------------------------------------------


class _SHELLEXECUTEINFOW(ctypes.Structure):
    """SHELLEXECUTEINFOW (Unicode) — see shellapi.h."""

    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("fMask", wintypes.ULONG),
        ("hwnd", wintypes.HWND),
        ("lpVerb", wintypes.LPCWSTR),
        ("lpFile", wintypes.LPCWSTR),
        ("lpParameters", wintypes.LPCWSTR),
        ("lpDirectory", wintypes.LPCWSTR),
        ("nShow", ctypes.c_int),
        ("hInstApp", wintypes.HINSTANCE),
        ("lpIDList", ctypes.c_void_p),
        ("lpClass", wintypes.LPCWSTR),
        ("hkeyClass", wintypes.HKEY),
        ("dwHotKey", wintypes.DWORD),
        ("hIcon", ctypes.c_void_p),
        ("hProcess", wintypes.HANDLE),
    ]


# SEE_MASK_NOCLOSEPROCESS: keep the process handle so we can wait on it.
_SEE_MASK_NOCLOSEPROCESS = 0x00000040
_SW_HIDE = 0
_INVALID_HANDLE_VALUE = -1


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _SECURITY_DESCRIPTOR(ctypes.Structure):
    _fields_ = [
        ("Revision", wintypes.BYTE),
        ("Sbz1", wintypes.BYTE),
        ("Control", wintypes.WORD),
        ("Owner", ctypes.c_void_p),
        ("Group", ctypes.c_void_p),
        ("Sacl", ctypes.c_void_p),
        ("Dacl", ctypes.c_void_p),
    ]


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_ulong),
        ("InternalHigh", ctypes.c_ulong),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_ulonglong),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]

# Named-pipe constants (kernel32)
_PIPE_ACCESS_DUPLEX = 0x00000003
_PIPE_TYPE_MESSAGE = 0x00000004
_PIPE_READMODE_MESSAGE = 0x00000002
_PIPE_WAIT = 0x00000000
_PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
_FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_TERMINATE = 0x0001
_PROCESS_CREATE_PROCESS = 0x0080
_PROCESS_DUP_HANDLE = 0x0040
_TH32CS_SNAPPROCESS = 0x00000002

# Security descriptor / DACL construction
_SECURITY_DESCRIPTOR_REVISION = 1
_ACL_REVISION = 2
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_ACCESS_SYSTEM_SECURITY = 0x01000000
_WRITE_DAC = 0x00040000
_WRITE_OWNER = 0x00080000
_READ_CONTROL = 0x00020000
_SYNCHRONIZE = 0x00100000
_STANDARD_RIGHTS_REQUIRED = 0x000F0000
_PIPE_ACCESS_DUPLEX_FULL = (
    _GENERIC_READ | _GENERIC_WRITE | _WRITE_DAC | _WRITE_OWNER
    | _READ_CONTROL | _SYNCHRONIZE
)

# Timeouts for the control-pipe protocol.
_CONNECT_TIMEOUT_S = 60  # UAC prompt can keep the helper idle for a while
_CANCEL_ACK_TIMEOUT_S = 10  # helper must ack a cancel within this window


class _WindowsElevationApi:
    """Thin adapter over the Windows APIs used by elevated execution.

    All handle-returning operations are wrapped here so the caller can close
    every handle on success, cancel, error and timeout paths uniformly, and
    so tests can substitute a fake adapter (see test_admin_executor).
    """

    # ShellExecuteExW / GetLastError classes
    ERROR_FILE_NOT_FOUND = 2
    ERROR_PATH_NOT_FOUND = 3
    ERROR_ACCESS_DENIED = 5
    ERROR_CANCELLED = 1223
    ERROR_UNKNOWN = -1

    # WaitForSingleObject results
    WAIT_OBJECT_0 = 0x00000000
    WAIT_ABANDONED = 0x00000080
    WAIT_TIMEOUT = 0x00000102
    WAIT_FAILED = 0xFFFFFFFF
    INFINITE = 0xFFFFFFFF

    # GetExitCodeProcess
    STILL_ACTIVE = 259

    def __init__(self) -> None:
        self._shell32 = None
        self._kernel32 = None

    def _shell(self):
        if self._shell32 is None:
            self._shell32 = ctypes.WinDLL("shell32", use_last_error=True)
            self._shell32.ShellExecuteExW.argtypes = [
                ctypes.POINTER(_SHELLEXECUTEINFOW),
            ]
            self._shell32.ShellExecuteExW.restype = wintypes.BOOL
        return self._shell32

    def _kernel(self):
        if self._kernel32 is None:
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._kernel32.WaitForSingleObject.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
            ]
            self._kernel32.WaitForSingleObject.restype = wintypes.DWORD
            self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            self._kernel32.CloseHandle.restype = wintypes.BOOL
            self._kernel32.TerminateProcess.argtypes = [
                wintypes.HANDLE,
                wintypes.UINT,
            ]
            self._kernel32.TerminateProcess.restype = wintypes.BOOL
            self._kernel32.GetExitCodeProcess.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            ]
            self._kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            self._kernel32.GetProcessId.argtypes = [wintypes.HANDLE]
            self._kernel32.GetProcessId.restype = wintypes.DWORD
            self._kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            self._kernel32.OpenProcess.restype = wintypes.HANDLE
            self._kernel32.CreateNamedPipeW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
            ]
            self._kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
            self._kernel32.ConnectNamedPipe.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
            ]
            self._kernel32.ConnectNamedPipe.restype = wintypes.BOOL
            self._kernel32.GetNamedPipeClientProcessId.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            ]
            self._kernel32.GetNamedPipeClientProcessId.restype = wintypes.BOOL
            self._kernel32.PeekNamedPipe.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(wintypes.DWORD),
            ]
            self._kernel32.PeekNamedPipe.restype = wintypes.BOOL
            self._kernel32.ReadFile.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.c_void_p,
            ]
            self._kernel32.ReadFile.restype = wintypes.BOOL
            self._kernel32.WriteFile.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.c_void_p,
            ]
            self._kernel32.WriteFile.restype = wintypes.BOOL
            self._kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
            self._kernel32.FlushFileBuffers.restype = wintypes.BOOL
        return self._kernel32

    # -- named pipe helpers ---------------------------------------------------

    def build_pipe_security_attributes(self):
        """Build SECURITY_ATTRIBUTES with a tight DACL for the pipe server.

        Grants connect/read/write to ONLY: the current user (whose elevated
        helper token carries the same primary SID), SYSTEM, and the local
        Administrators group.  NEVER a broad Everyone ACE.  Returns a
        ctypes SECURITY_ATTRIBUTES (or None on construction failure — the
        caller must fail closed and not create a pipe without it).
        """
        if not is_windows():
            return None
        try:
            adv = ctypes.WinDLL("advapi32", use_last_error=True)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)

            # Declare argtypes so 64-bit pointers are marshalled correctly.
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL
            k32.LocalFree.argtypes = [wintypes.HLOCAL]
            k32.LocalFree.restype = wintypes.HLOCAL
            adv.OpenProcessToken.argtypes = [
                wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
            ]
            adv.OpenProcessToken.restype = wintypes.BOOL
            adv.GetTokenInformation.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            ]
            adv.GetTokenInformation.restype = wintypes.BOOL
            adv.ConvertStringSidToSidW.argtypes = [
                wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p),
            ]
            adv.ConvertStringSidToSidW.restype = wintypes.BOOL
            adv.InitializeAcl.argtypes = [
                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
            ]
            adv.InitializeAcl.restype = wintypes.BOOL
            adv.AddAccessAllowedAce.argtypes = [
                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
            ]
            adv.AddAccessAllowedAce.restype = wintypes.BOOL
            adv.InitializeSecurityDescriptor.argtypes = [
                ctypes.c_void_p, wintypes.DWORD,
            ]
            adv.InitializeSecurityDescriptor.restype = wintypes.BOOL
            adv.SetSecurityDescriptorDacl.argtypes = [
                ctypes.c_void_p, wintypes.BOOL, ctypes.c_void_p, wintypes.BOOL,
            ]
            adv.SetSecurityDescriptorDacl.restype = wintypes.BOOL

            # 1. Current user SID via the process token (TokenUser = 1).
            token = wintypes.HANDLE()
            if not adv.OpenProcessToken(
                k32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
            ):
                return None
            try:
                size = wintypes.DWORD()
                adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
                buf = ctypes.create_string_buffer(size.value)
                if not adv.GetTokenInformation(
                    token, 1, buf, size.value, ctypes.byref(size)
                ):
                    return None
                # TOKEN_USER begins with SID_AND_ATTRIBUTES whose first
                # member is the PSID (pointer); the buffer returned by
                # GetTokenInformation starts directly with that PSID.
                user_sid = ctypes.cast(
                    buf, ctypes.POINTER(ctypes.c_void_p)
                ).contents.value
            finally:
                k32.CloseHandle(token)

            # 2. SYSTEM and Administrators well-known SIDs.
            system_sid = ctypes.c_void_p()
            admin_sid = ctypes.c_void_p()
            if not adv.ConvertStringSidToSidW(
                "S-1-5-18", ctypes.byref(system_sid)
            ):
                return None
            if not adv.ConvertStringSidToSidW(
                "S-1-5-32-544", ctypes.byref(admin_sid)
            ):
                k32.LocalFree(system_sid)
                return None

            # 3. Build an ACL with exactly three ACEs.
            acl_buf = ctypes.create_string_buffer(1024)
            if not adv.InitializeAcl(acl_buf, 1024, 2):
                k32.LocalFree(system_sid)
                k32.LocalFree(admin_sid)
                return None
            ace_mask = (
                _GENERIC_READ | _GENERIC_WRITE | _SYNCHRONIZE
                | _READ_CONTROL
            )
            ok = adv.AddAccessAllowedAce(
                acl_buf, 2, ace_mask, user_sid
            ) and adv.AddAccessAllowedAce(
                acl_buf, 2, ace_mask, system_sid
            ) and adv.AddAccessAllowedAce(
                acl_buf, 2, ace_mask, admin_sid
            )
            k32.LocalFree(system_sid)
            k32.LocalFree(admin_sid)
            if not ok:
                return None

            # 4. SECURITY_DESCRIPTOR with the DACL.
            sd_buf = ctypes.create_string_buffer(
                ctypes.sizeof(_SECURITY_DESCRIPTOR)
            )
            sd = ctypes.cast(sd_buf, ctypes.POINTER(_SECURITY_DESCRIPTOR))
            if not adv.InitializeSecurityDescriptor(sd, 1):
                return None
            if not adv.SetSecurityDescriptorDacl(
                sd, True, ctypes.cast(acl_buf, ctypes.c_void_p), False
            ):
                return None

            sa = _SECURITY_ATTRIBUTES()
            sa.nLength = ctypes.sizeof(_SECURITY_ATTRIBUTES)
            sa.lpSecurityDescriptor = ctypes.cast(
                sd_buf, ctypes.c_void_p
            )
            sa.bInheritHandle = False
            # Keep the descriptor + ACL buffers alive: ctypes releases the
            # create_string_buffer objects when this function returns, which
            # would leave lpSecurityDescriptor dangling and make
            # CreateNamedPipeW fail.  Attaching them to the returned object
            # pins their lifetime to the SECURITY_ATTRIBUTES.
            sa._sd_buf = sd_buf
            sa._acl_buf = acl_buf
            return sa
        except Exception:  # pragma: no cover - defensive
            return None

    def create_named_pipe(self, name: str, message_mode: bool, security_attributes=None) -> int:
        """Create the server end of a one-shot named pipe.

        Uses ``PIPE_REJECT_REMOTE_CLIENTS`` (the pipe can only be opened by
        a process on this machine — never over SMB) and
        ``FILE_FLAG_FIRST_PIPE_INSTANCE`` (a second ``CreateNamedPipeW``
        with the same name fails instead of silently creating a second
        instance that a rogue client could race to).  The caller passes the
        tight DACL from :meth:`build_pipe_security_attributes`; the default
        ``None`` is only safe for the real API because the parent always
        builds one first (fail closed if construction fails).
        """
        k32 = self._kernel()
        open_mode = (
            _PIPE_ACCESS_DUPLEX
            | _FILE_FLAG_FIRST_PIPE_INSTANCE
        )
        pipe_mode = _PIPE_WAIT | _PIPE_REJECT_REMOTE_CLIENTS
        if message_mode:
            pipe_mode |= _PIPE_TYPE_MESSAGE | _PIPE_READMODE_MESSAGE
        handle = k32.CreateNamedPipeW(
            name,
            open_mode,
            pipe_mode,
            1,  # one instance
            65536,  # output buffer
            65536,  # input buffer
            0,  # default timeout
            ctypes.byref(security_attributes)
            if security_attributes is not None
            else None,
        )
        if handle in (None, -1, _INVALID_HANDLE_VALUE):
            raise OSError(f"CreateNamedPipeW failed (error {ctypes.get_last_error()})")
        return handle

    def connect_named_pipe(self, handle) -> None:
        ok = self._kernel().ConnectNamedPipe(handle, None)
        # ERROR_PIPE_CONNECTED (535) means a client connected between
        # CreateNamedPipe and ConnectNamedPipe — also a successful connect.
        if not ok and ctypes.get_last_error() != 535:
            raise OSError(
                f"ConnectNamedPipe failed (error {ctypes.get_last_error()})"
            )

    def connect_named_pipe_bounded(self, handle, timeout_s: float) -> bool:
        """Connect the server end of the pipe (blocking).

        The BOUND is enforced by ``_ElevatedPipeChannel.wait_connect``: it
        runs this on worker threads, waits for both pipes / the helper
        process / the deadline, then CLOSES the pipe handles to abort any
        still-pending ``ConnectNamedPipe`` (closing the server handle makes
        a pending connect fail) and joins the workers — so no blocking
        thread is left behind and no overlapped-I/O flag is needed on the
        pipe (byte-stream ReadFile later stays synchronous).
        """
        self.connect_named_pipe(handle)
        return True

    def is_trusted_descendant(self, ancestor_pid: int, child_pid: int, max_depth: int = 16) -> bool:
        """Return True if *child_pid* is a descendant of *ancestor_pid*.

        Walks the live process table (Toolhelp snapshot) parent->child
        links up to *max_depth* hops.  A venv launcher (the process the
        ShellExecuteExW handle points at) spawns the real helper as its
        child, so this proves the helper belongs to THIS launch without
        trusting a self-reported PID.
        """
        if not ancestor_pid or not child_pid or ancestor_pid == child_pid:
            return ancestor_pid == child_pid and bool(ancestor_pid)
        try:
            k32 = self._kernel()
            k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
            k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            k32.Process32FirstW.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W),
            ]
            k32.Process32FirstW.restype = wintypes.BOOL
            k32.Process32NextW.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W),
            ]
            k32.Process32NextW.restype = wintypes.BOOL
            snap = k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
            if snap in (None, -1, _INVALID_HANDLE_VALUE):
                return False
            try:
                pe = _PROCESSENTRY32W()
                pe.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
                parents: dict[int, int] = {}
                if not k32.Process32FirstW(snap, ctypes.byref(pe)):
                    return False
                while True:
                    parents[int(pe.th32ProcessID)] = int(pe.th32ParentProcessID)
                    if not k32.Process32NextW(snap, ctypes.byref(pe)):
                        break
            finally:
                self.close_handle(snap)
            # Walk child_pid upward: each hop must land on a parent we saw.
            pid = child_pid
            for _ in range(max_depth):
                parent = parents.get(pid)
                if parent is None:
                    return False
                if parent == ancestor_pid:
                    return True
                pid = parent
            return False
        except Exception:  # pragma: no cover - defensive
            return False

    def get_named_pipe_client_process_id(self, handle) -> int:
        pid = wintypes.DWORD()
        ok = self._kernel().GetNamedPipeClientProcessId(handle, ctypes.byref(pid))
        return int(pid.value) if ok else 0

    def peek_pipe(self, handle) -> int:
        avail = wintypes.DWORD()
        self._kernel().PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None)
        return int(avail.value)

    def read_pipe_bytes(self, handle, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        nread = wintypes.DWORD()
        ok = self._kernel().ReadFile(handle, buf, size, ctypes.byref(nread), None)
        got = int(nread.value)
        if not ok:
            # ERROR_BROKEN_PIPE (109) / ERROR_PIPE_NOT_CONNECTED (233) = EOF.
            return b""
        return buf.raw[:got]

    def write_pipe_bytes(self, handle, data: bytes) -> None:
        buf = ctypes.create_string_buffer(data)
        nwritten = wintypes.DWORD()
        if not self._kernel().WriteFile(
            handle, buf, len(data), ctypes.byref(nwritten), None
        ):
            raise OSError(f"WriteFile failed (error {ctypes.get_last_error()})")
        if int(nwritten.value) != len(data):
            raise OSError(
                f"WriteFile partial write: {nwritten.value} of {len(data)} "
                f"(error {ctypes.get_last_error()})"
            )

    def get_process_id(self, handle) -> int:
        return int(self._kernel().GetProcessId(handle))

    def open_process(self, access: int, pid: int):
        handle = self._kernel().OpenProcess(access, False, pid)
        if not handle:
            return None
        return handle

    # -- ShellExecuteExW -----------------------------------------------------

    def shellexecute_ex(
        self,
        verb: str,
        file: str,
        parameters: str,
        directory: str | None,
    ) -> tuple[int, int]:
        """Call ShellExecuteExW(runas).  Returns ``(hProcess, last_error)``.

        On success ``hProcess`` is the elevated process handle (non-NULL when
        ``SEE_MASK_NOCLOSEPROCESS`` is set) and ``last_error`` is 0.  On
        failure ``hProcess`` is None and ``last_error`` is the Win32 code
        (ERROR_CANCELLED=1223 for a dismissed UAC prompt, ERROR_ACCESS_DENIED
        for access-denied, ERROR_FILE/PATH_NOT_FOUND, or another system code).
        """
        sei = _SHELLEXECUTEINFOW()
        sei.cbSize = ctypes.sizeof(_SHELLEXECUTEINFOW)
        sei.fMask = _SEE_MASK_NOCLOSEPROCESS
        sei.lpVerb = verb
        sei.lpFile = file
        sei.lpParameters = parameters
        sei.lpDirectory = directory
        sei.nShow = _SW_HIDE
        ok = bool(self._shell().ShellExecuteExW(ctypes.byref(sei)))
        if not ok:
            err = ctypes.get_last_error()
            return None, err
        return sei.hProcess, 0

    # -- kernel32 ------------------------------------------------------------

    def wait_for_single_object(self, handle, timeout_ms: int) -> int:
        return int(self._kernel().WaitForSingleObject(handle, timeout_ms))

    def close_handle(self, handle) -> None:
        if handle:
            try:
                self._kernel().CloseHandle(handle)
            except Exception:
                pass

    def terminate_process(self, handle, exit_code: int = 1) -> bool:
        return bool(self._kernel().TerminateProcess(handle, exit_code))

    def get_exit_code(self, handle) -> "tuple[bool, int]":
        code = wintypes.DWORD()
        ok = bool(self._kernel().GetExitCodeProcess(handle, ctypes.byref(code)))
        return ok, int(code.value)

    @staticmethod
    def format_last_error(err: int) -> str:
        try:
            return ctypes.FormatError(err) or f"Win32 error {err}"
        except Exception:
            return f"Win32 error {err}"


# Human-readable classification of ShellExecuteExW failures.  The UAC cancel
# case (ERROR_CANCELLED) is a distinct, user-meaningful outcome and must not
# be lumped with launch success or with a generic system failure.
def _classify_launch_error(last_error: int) -> dict:
    if last_error == _WindowsElevationApi.ERROR_CANCELLED:
        return {
            "error": "Elevated launch cancelled: the UAC prompt was dismissed "
            "(ERROR_CANCELLED=1223).",
            "error_kind": "cancelled",
        }
    if last_error == _WindowsElevationApi.ERROR_ACCESS_DENIED:
        return {
            "error": "Elevated launch failed: access denied "
            "(ERROR_ACCESS_DENIED=5).",
            "error_kind": "access_denied",
        }
    if last_error in (
        _WindowsElevationApi.ERROR_FILE_NOT_FOUND,
        _WindowsElevationApi.ERROR_PATH_NOT_FOUND,
    ):
        return {
            "error": "Elevated launch failed: file or path not found "
            f"(Win32 error {last_error}).",
            "error_kind": "not_found",
        }
    return {
        "error": f"Elevated launch failed: system error {last_error} "
        f"({_WindowsElevationApi.format_last_error(last_error)}).",
        "error_kind": "other",
    }


def _default_elevation_api() -> "_WindowsElevationApi":
    return _WindowsElevationApi()


def can_elevate() -> bool:
    """Return True if elevation is possible on this platform.

    Elevation requires:
    - Windows platform
    - ShellExecuteExW available
    - Not already running as admin (elevation would be a no-op)
    """
    if not is_windows():
        return False
    if is_running_as_admin():
        return False
    try:
        return hasattr(ctypes.WinDLL("shell32", use_last_error=True), "ShellExecuteExW")
    except Exception:
        return False


def execute_elevated(
    command: str,
    cwd: str | None = None,
    timeout: int = _TIMEOUT_S,
    *,
    _api: "_WindowsElevationApi | None" = None,
) -> dict:
    """Execute a command with Windows administrator privileges via UAC.

    Flow:
    1. Build a structured request with a SANITIZED environment and write it
       as UTF-8 JSON to a restricted temp file.
    2. ShellExecuteExW("runas") launches python.exe running the minimal
       trusted helper (tools/elevated_helper.py), which reads the JSON and
       creates the real command process with CreateProcessW.
    3. WaitForSingleObject on the returned process handle (the done marker is
       only a secondary signal).
    4. Return {"output": str, "exit_code": int, "error": str|None}.

    Args:
        command: The shell command to execute
        cwd: Working directory (optional)
        timeout: Max seconds to wait for completion
        _api: injectable Windows API adapter (tests only)

    Returns:
        dict with "output", "exit_code", "error" keys
    """
    if not is_windows():
        return {
            "output": "",
            "exit_code": -1,
            "error": "Elevated execution is only supported on Windows",
        }

    if is_running_as_admin():
        return {
            "output": "",
            "exit_code": -1,
            "error": (
                "Already running as administrator. "
                "Use normal terminal execution instead of elevated."
            ),
        }

    # Defense-in-depth: the resolved cwd is passed to the elevated helper via
    # lpCurrentDirectory (CreateProcessW) — it is no longer interpolated into
    # a .cmd script — but we keep the same allowlist validation as
    # terminal_tool's workdir check so a malicious/session-derived cwd cannot
    # smuggle metacharacters into the ShellExecuteExW parameters either.
    #
    # When *cwd* is omitted the executor falls back to os.getcwd(); that
    # default must be validated too, so resolve effective_cwd up front and
    # validate it uniformly.  Nothing (temp dir, request, ShellExecuteExW)
    # may happen before this check passes.
    effective_cwd = cwd or os.getcwd()
    cwd_error = _validate_cwd_for_script(effective_cwd)
    if cwd_error:
        return {
            "output": "",
            "exit_code": -1,
            "error": cwd_error,
        }

    api = _api or _default_elevation_api()
    return _execute_elevated_impl(command, effective_cwd, timeout, api)


# Raw staging dirs created by _stage_raw_output.  A hard crash / power loss
# between staging and sanitization can leave one behind (the normal finally
# path cannot run); stale dirs older than the TTL are reaped on the NEXT
# execution — never in real time.
_RAW_DIR_PREFIX = "hermes_elevated_raw_"
_RAW_DIR_TTL_SECONDS = 24 * 3600


def _cleanup_stale_raw_dirs(ttl_seconds: int | None = None) -> None:
    """Reap stale ``hermes_elevated_raw_*`` staging dirs from earlier runs.

    Only directories with our exact prefix under the system temp root are
    touched; foreign temp dirs are left alone.  Cleanup failures are
    swallowed — reaping must never block the normal elevated execution path.
    """
    ttl = ttl_seconds if ttl_seconds is not None else _RAW_DIR_TTL_SECONDS
    now = time.time()
    try:
        entries = os.listdir(tempfile.gettempdir())
    except OSError:
        return
    for name in entries:
        if not name.startswith(_RAW_DIR_PREFIX):
            continue
        path = os.path.join(tempfile.gettempdir(), name)
        try:
            if not os.path.isdir(path):
                continue
            if now - os.path.getmtime(path) > ttl:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            # Permission/race issues must never block execution.
            continue


def _stage_raw_output(output_file: str) -> str | None:
    """Move the raw elevated output to a controlled staging location.

    The raw file must NEVER land in the persistent spill directory
    (``cache/terminal-output``): a crash between raw generation and
    sanitization would leave an unredacted file in the durable cache.  The
    raw output is instead moved to a throwaway directory under the system
    temp area, and the terminal pipeline streams it from there into the
    sanitized spill, then deletes it.  Returns the raw path (which is an
    internal protocol field, never returned as ``full_output_path``), or
    None on failure (the pipeline then degrades to the visible window only).
    """
    # Opportunistically reap raw staging dirs left by a previous hard crash
    # (best-effort; see _cleanup_stale_raw_dirs).
    _cleanup_stale_raw_dirs()
    raw_dir = None
    try:
        raw_dir = tempfile.mkdtemp(prefix=_RAW_DIR_PREFIX)
        raw_path = os.path.join(raw_dir, "output.raw")
        shutil.move(output_file, raw_path)
        # Restrict the raw file to the current user (best-effort; on POSIX
        # this is 0o600, on Windows os.chmod only honors the read-only bit
        # and the default temp ACL is already user-scoped).
        try:
            os.chmod(raw_path, 0o600)
        except OSError:
            pass
        return raw_path
    except Exception:
        if raw_dir:
            shutil.rmtree(raw_dir, ignore_errors=True)
        return None


def _elevated_helper_path() -> str:
    """Absolute path to the minimal trusted elevated helper (repo file)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "elevated_helper.py")


class _ElevatedPipeChannel:
    """Named-pipe control channel between the parent and the elevated helper.

    round-7: the guarded command travels over a one-shot, process-bound named
    pipe instead of a user-writable JSON file.  The parent creates the pipes
    with a random name + nonce, ``ShellExecuteExW`` passes ONLY the pipe names
    and the nonce to the helper, the helper's client PID is verified against
    the returned process handle, and only THEN is the request (command / cwd /
    sanitized env) sent.  The helper never writes files: stdout/stderr stream
    back over the output pipe into a file created by the PARENT (medium
    integrity), and the exit code / status travel on the control pipe.

    round-8 hardening:
    - **Bounded connect**: ``ConnectNamedPipe`` runs on worker threads with
      completion events; ``wait_connect`` waits for BOTH pipes, the helper
      process handle, or ``_CONNECT_TIMEOUT_S`` — whichever comes first.
      A single connected pipe, a helper that dies mid-connect, or a rogue
      client squatting one pipe all fail closed within the bound, and every
      pipe/event/process handle is closed.  No blocking thread is left
      behind (the worker threads are daemon + joined).
    - **Real identity binding**: the launched python is the base executable
      (``sys._base_executable``) so ``GetProcessId(hProcess)`` IS the helper
      PID; the control-pipe client PID, the output-pipe client PID and the
      helper-reported pid must all agree.  When a launcher shim is used the
      real helper PID must be a *trusted descendant* of the hProcess PID
      (walked from the live process table), never an arbitrary self-report.
    - **Pipe hardening**: both pipes use ``PIPE_REJECT_REMOTE_CLIENTS``,
      ``FILE_FLAG_FIRST_PIPE_INSTANCE`` and a tight DACL (current user +
      SYSTEM + Administrators only — never Everyone).
    - **Output file lifecycle**: the raw output file is parent-owned and
      deleted on EVERY path (small output read-then-delete, overflow staged
      to a throwaway raw dir for the sanitizer, launch/protocol/cancel/error
      delete).  Nothing relies on ``channel.close`` making the file vanish.

    A second client cannot connect (one instance), cannot forge the nonce, and
    cannot replay an old request against a fresh launch.  If the client PID
    check fails the channel refuses to send anything (fail closed).
    """

    def __init__(self, api: "_WindowsElevationApi") -> None:
        self._api = api
        self.nonce = secrets.token_hex(16)
        token = secrets.token_hex(8)
        self.control_name = f"\\\\.\\pipe\\hermes-elevated-ctl-{os.getpid()}-{token}"
        self.output_name = f"\\\\.\\pipe\\hermes-elevated-out-{os.getpid()}-{token}"
        self._h_control = None
        self._h_output = None
        self._security_attributes = None
        self.output_path: "str | None" = None
        self._drain_thread: "threading.Thread | None" = None
        self._drain_error: "Exception | None" = None

    # -- lifecycle -----------------------------------------------------------

    def create(self) -> None:
        # Fail closed: without a tight DACL no pipe may be created.  The
        # elevated helper runs under the SAME user token (elevated), so the
        # current-user SID is exactly the connecting principal.
        self._security_attributes = self._api.build_pipe_security_attributes()
        if not self._security_attributes and is_windows():
            raise OSError(
                "failed to build pipe security descriptor; refusing to "
                "create the elevated pipes (fail closed)"
            )
        self._h_control = self._api.create_named_pipe(
            self.control_name,
            message_mode=True,
            security_attributes=self._security_attributes,
        )
        self._h_output = self._api.create_named_pipe(
            self.output_name,
            message_mode=False,
            security_attributes=self._security_attributes,
        )
        # The output file is created by the PARENT (medium integrity, its own
        # temp dir) — the elevated helper never touches a path for it.  It is
        # deleted on every result path (see _collect_elevated_result and
        # close()); it never outlives the call.
        fd, self.output_path = tempfile.mkstemp(
            prefix="hermes_elevated_out_", suffix=".txt"
        )
        os.close(fd)

    def launch_parameters(self, helper_path: str) -> str:
        """ShellExecuteExW parameters: helper + pipes + nonce.  The command,
        cwd and env are deliberately NOT here."""
        return (
            f'"{helper_path}" "{self.control_name}" "{self.output_name}" '
            f"{self.nonce}"
        )

    def wait_connect(
        self, timeout_s: int = _CONNECT_TIMEOUT_S, h_process=None
    ) -> None:
        """Connect BOTH pipes within *timeout_s*, watching the helper process.

        Runs each ``ConnectNamedPipe`` on a short-lived worker thread whose
        completion is signalled through a threading.Event, so the parent can
        wait on pipe completion, helper-process exit, or the timeout —
        whichever happens first.  Raises TimeoutError when the bound expires
        or the helper exits before both pipes connected; every handle is
        closed by the caller on that path (no blocking thread is left: the
        workers are daemon threads and are joined after the bound).
        """
        import threading as _threading

        deadline = time.monotonic() + timeout_s
        both_done = _threading.Event()
        results: dict[str, "tuple[bool, str]"] = {}
        lock = _threading.Lock()
        remaining_workers = 2

        def _connect(pipe_key: str, handle, message_mode: bool) -> None:
            nonlocal remaining_workers
            try:
                ok = self._api.connect_named_pipe_bounded(
                    handle, timeout_s=timeout_s
                )
                with lock:
                    results[pipe_key] = (ok, "")
            except Exception as exc:  # noqa: BLE001
                with lock:
                    results[pipe_key] = (False, str(exc))
            finally:
                with lock:
                    remaining_workers -= 1
                    if remaining_workers == 0:
                        both_done.set()

        workers = [
            _threading.Thread(
                target=_connect,
                args=("control", self._h_control, True),
                daemon=True,
            ),
            _threading.Thread(
                target=_connect,
                args=("output", self._h_output, False),
                daemon=True,
            ),
        ]
        for w in workers:
            w.start()

        # Wait for BOTH pipes, helper exit, or the deadline.
        while not both_done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Poll helper liveness (h_process may be a fake handle in tests;
            # treat a dead helper as an immediate fail).
            if h_process is not None:
                try:
                    wait_rc = self._api.wait_for_single_object(h_process, 0)
                    if wait_rc == self._api.WAIT_OBJECT_0:
                        break  # helper exited before both pipes connected
                except Exception:  # noqa: BLE001 - fake APIs may not support
                    pass
            both_done.wait(timeout=min(0.1, remaining))

        if not both_done.is_set():
            # Bound expired (or helper exited): abort the pending blocking
            # ConnectNamedPipe calls by closing the server pipe handles.
            # Closing the server end makes a pending connect fail, so the
            # workers exit promptly and can be joined — no blocked thread
            # survives this method.
            for handle in (self._h_control, self._h_output):
                if handle:
                    try:
                        self._api.close_handle(handle)
                    except Exception:  # noqa: BLE001 - best-effort abort
                        pass
            self._h_control = None
            self._h_output = None

        for w in workers:
            w.join(timeout=1.0)

        with lock:
            control_ok, control_err = results.get("control", (False, "not started"))
            output_ok, output_err = results.get("output", (False, "not started"))

        if not control_ok or not output_ok:
            # One or both pipes never connected: fail closed with a precise
            # diagnostic (which pipe, what error, helper state).
            detail = []
            if not control_ok:
                detail.append(f"control pipe: {control_err or 'timeout'}")
            if not output_ok:
                detail.append(f"output pipe: {output_err or 'timeout'}")
            if h_process is not None:
                try:
                    wait_rc = self._api.wait_for_single_object(h_process, 0)
                    if wait_rc == self._api.WAIT_OBJECT_0:
                        detail.append("helper process exited before connecting")
                except Exception:
                    pass
            raise TimeoutError(
                "elevated helper did not connect both pipes within "
                f"{timeout_s}s ({'; '.join(detail)})"
            )

    def verify_client(
        self, h_process, helper_pid: "int | None", control_pid: "int | None" = None
    ) -> bool:
        """Bind BOTH pipe clients to the launched helper process.

        Checks, all of which must hold (fail closed):
        1. ``GetProcessId(hProcess)`` (the pid of the launched python) must
           equal the helper-reported pid — the launcher is the base
           executable so they are the same process; when a shim is in play
           the helper must be a TRUSTED DESCENDANT of the hProcess pid.
        2. The control-pipe client pid must equal the helper-reported pid.
        3. The output-pipe client pid must equal the helper-reported pid.

        Any mismatch (rogue client on either pipe, non-descendant pid, or a
        missing pid) refuses the connection and nothing is sent.
        """
        if not helper_pid:
            return False
        # 1. hProcess identity (real python on the normal path).
        h_pid = 0
        try:
            if h_process is not None:
                h_pid = self._api.get_process_id(h_process)
        except Exception:  # noqa: BLE001 - fake APIs may not implement it
            h_pid = 0
        if h_pid and h_pid != helper_pid:
            # Launcher shim path: the real helper must be a descendant of the
            # process the ShellExecuteExW handle refers to — never an
            # arbitrary self-reported PID.
            if not self._api.is_trusted_descendant(h_pid, helper_pid):
                return False
        # 2/3. Both pipe clients must be the helper itself.
        try:
            control_client = self._api.get_named_pipe_client_process_id(
                self._h_control
            )
            output_client = self._api.get_named_pipe_client_process_id(
                self._h_output
            )
        except Exception:  # noqa: BLE001 - fake APIs may not implement it
            return False
        if control_client != helper_pid or output_client != helper_pid:
            return False
        return True

    # -- control protocol -----------------------------------------------------

    def send(self, message: dict) -> None:
        from tools.elevated_protocol import pack_message

        self._api.write_pipe_bytes(self._h_control, pack_message(message))

    def read_message(self, timeout_ms: "int | None" = None) -> dict:
        from tools.elevated_protocol import ProtocolError, unpack_frame

        deadline = time.monotonic() + (
            timeout_ms / 1000.0 if timeout_ms is not None else 1e9
        )
        buf = b""
        while True:
            avail = self._api.peek_pipe(self._h_control)
            if avail > 0:
                chunk = self._api.read_pipe_bytes(self._h_control, avail)
                if not chunk:
                    raise ProtocolError(
                        "pipe reported data but read returned none"
                    )
                buf += chunk
                if buf:
                    try:
                        message, _consumed = unpack_frame(buf)
                        return message
                    except ProtocolError as exc:
                        if "truncated" in str(exc):
                            continue  # wait for the rest of the frame
                        raise
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"no control message within {timeout_ms}ms"
                )
            time.sleep(0.02)

    # -- output stream --------------------------------------------------------

    def start_output_drain(self) -> None:
        """Drain the output pipe into the parent-owned output file.  Must be
        started before the helper begins emitting child output."""
        self._drain_thread = threading.Thread(
            target=self._drain_output, daemon=True
        )
        self._drain_thread.start()

    def _drain_output(self) -> None:
        try:
            with open(self.output_path, "wb") as f:
                while True:
                    chunk = self._api.read_pipe_bytes(self._h_output, 65536)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as exc:  # noqa: BLE001
            self._drain_error = exc

    def join_output(self, timeout_s: float = 5.0) -> None:
        if self._drain_thread:
            self._drain_thread.join(timeout_s)

    # -- cleanup ---------------------------------------------------------------

    def close(self) -> None:
        """Close pipe/event handles AND the parent-owned output file.

        The raw output file never outlives the channel: on every path
        (success small output, overflow staged to the raw dir, launch/
        protocol/cancel/error, timeout) ``close()`` deletes it if it still
        exists.  When the output overflowed, ``_stage_raw_output`` already
        MOVED the file to the throwaway raw dir (owned by the sanitizer), so
        this unlink is a no-op — the staged raw is consumed and deleted by
        the terminal pipeline, never by the channel.
        """
        if self.output_path:
            try:
                os.unlink(self.output_path)
            except OSError:
                pass
            self.output_path = None
        for handle in (self._h_control, self._h_output):
            if handle:
                self._api.close_handle(handle)
        self._h_control = None
        self._h_output = None


def _process_is_dead(api: "_WindowsElevationApi", pid: int) -> bool:
    """Best-effort check whether a PID is no longer running."""
    if not pid:
        return False
    h = api.open_process(_PROCESS_QUERY_LIMITED_INFORMATION, pid)
    if h is None:
        return True  # OpenProcess failed -> process is gone
    try:
        ok, code = api.get_exit_code(h)
        return ok and code != api.STILL_ACTIVE
    finally:
        api.close_handle(h)


def _handle_elevated_timeout(
    api: "_WindowsElevationApi",
    channel: "_ElevatedPipeChannel",
    h_process,
    child_pid: "int | None",
    job_bound: bool,
    timeout: int,
) -> dict:
    """Timeout path: cancel via the control pipe, then terminate if needed.

    A terminated-tree claim is ONLY made when there is evidence: the helper
    acknowledged the cancel with ``terminated=true`` (it called
    TerminateJobObject), or the helper is dead AND the job was bound AND the
    child PID is verified gone.  Anything less reports ``may still be
    running``.

    round-8: with CREATE_SUSPENDED + mandatory job binding, the child never
    runs outside the job, so the "may still be running" branch is reached
    only when the helper itself vanished without a cancel ack AND the job
    was never bound.  The raw output file is deleted by ``channel.close``
    on every path (never kept, never orphaned in the system temp area).
    """
    from tools.elevated_protocol import KIND_CANCELLED, make_cancel

    ack = None
    try:
        channel.send(make_cancel(channel.nonce))
        ack = channel.read_message(timeout_ms=_CANCEL_ACK_TIMEOUT_S * 1000)
    except Exception:
        ack = None
    if ack is not None and ack.get("kind") == KIND_CANCELLED:
        channel.join_output()
        if ack.get("terminated"):
            return {
                "output": "",
                "exit_code": -1,
                "error": (
                    f"Elevated command timed out after {timeout}s and was "
                    "terminated (job tree confirmed dead)."
                ),
                "error_kind": "timeout",
            }
        return {
            "output": "",
            "exit_code": -1,
            "error": (
                f"Elevated command timed out after {timeout}s. "
                "The command may still be running; it could not be "
                "confirmed terminated."
            ),
            "error_kind": "timeout",
        }

    # The helper did not acknowledge the cancel.  Terminate it; the job
    # (KILL_ON_JOB_CLOSE) kills the tree only if the job was actually bound.
    terminated = api.terminate_process(h_process, 1)
    helper_dead = False
    if terminated:
        ok, code = api.get_exit_code(h_process)
        helper_dead = ok and code != api.STILL_ACTIVE
    if helper_dead and job_bound and child_pid and _process_is_dead(api, child_pid):
        channel.join_output()
        return {
            "output": "",
            "exit_code": -1,
            "error": (
                f"Elevated command timed out after {timeout}s and was "
                "terminated (helper exited, job-bound child verified gone)."
            ),
            "error_kind": "timeout",
        }
    return {
        "output": "",
        "exit_code": -1,
        "error": (
            f"Elevated command timed out after {timeout}s. "
            "The command may still be running; it could not be confirmed "
            "terminated."
        ),
        "error_kind": "timeout",
    }


def _collect_elevated_result(channel: "_ElevatedPipeChannel", rc: int) -> dict:
    """Read the parent-owned output file written from the output pipe.

    Small output: the raw file is read and IMMEDIATELY deleted (the file
    must not outlive the call).  Overflow: the raw file is safely MOVED to
    a throwaway raw staging dir for the sanitizer (``channel.close`` then
    finds the file gone and its unlink is a no-op).  Either way the raw
    output file never remains in the system temp area.
    """
    output_file = channel.output_path
    max_chars = _output_max_chars()
    output, output_total_chars = _read_output_bounded(output_file, max_chars)

    result: dict = {
        "output": output,
        "exit_code": rc,
        "error": None,
    }

    # Hand the raw output to the pipeline for sanitized spilling.
    # ``raw_output_path`` is an INTERNAL field: it never leaves the pipeline
    # as ``full_output_path`` — only the sanitized spill path does, and only
    # after redaction succeeds.
    if output_total_chars > max_chars:
        raw_path = _stage_raw_output(output_file)
        if raw_path:
            result["output_total_chars"] = output_total_chars
            result["raw_output_path"] = raw_path
    else:
        # Small output: delete the raw file right now.  Do not rely on
        # channel.close() doing it later — this is the only path that reads
        # the file, so the delete belongs here.
        try:
            os.unlink(output_file)
        except OSError:
            pass
    return result


def _python_executable_for_elevation() -> str:
    """Return the stable Python executable to elevate.

    Prefers ``sys._base_executable`` (the real interpreter, not the venv
    launcher shim) so ``GetProcessId(hProcess)`` from ShellExecuteExW is the
    actual helper PID and the pipe-client identity check can require a
    direct match instead of a descendant walk.  Falls back to
    ``sys.executable`` only when the base executable is unavailable.
    """
    base = getattr(sys, "_base_executable", None)
    if base and os.path.isfile(base):
        return base
    return sys.executable


def _execute_elevated_impl(
    command: str,
    cwd: str | None,
    timeout: int,
    api: "_WindowsElevationApi",
) -> dict:
    """Internal implementation — runs inside a try/finally that cleans tmp_dir.

    round-7 flow:
    1. Build the SANITIZED environment (centralized build_subprocess_env).
    2. Create the control + output named pipes and a parent-owned output file.
    3. ShellExecuteExW(runas) launches the helper with ONLY pipe names + nonce.
    4. Verify the helper client PID against the returned process handle.
    5. Send the guarded request over the control pipe; drain the output pipe.
    6. Wait for done / handle timeout with cancel + verified termination.

    round-8 additions:
    - Launch the BASE python executable so the hProcess PID is the helper
      PID itself (identity binding, see ``_ElevatedPipeChannel.verify_client``).
    - Bounded two-pipe connect (worker threads + helper-exit watch).
    - Every helper message must carry the launch nonce; the message sequence
      is validated against a strict state machine; both pipe clients must be
      the verified helper.
    - The parent-owned output file is deleted on every path (small output
      read-then-delete in ``_collect_elevated_result``; overflow staged to
      the raw dir for the sanitizer; launch/protocol/pid-mismatch/cancel/
      error paths delete it via ``channel.close``).
    """
    from tools.elevated_protocol import (
        KIND_CONNECTED,
        KIND_DONE,
        KIND_READY,
        ProtocolError,
        _HelperStateMachine,
        make_request,
        validate_helper_message,
    )

    try:
        from tools.environments.local import build_subprocess_env

        env = build_subprocess_env()
    except Exception as e:  # pragma: no cover - defensive
        return {
            "output": "",
            "exit_code": -1,
            "error": f"Failed to build sanitized environment: {e}",
        }

    channel = _ElevatedPipeChannel(api)
    try:
        channel.create()
        helper_path = _elevated_helper_path()
        python_exe = _python_executable_for_elevation()
        parameters = channel.launch_parameters(helper_path)

        h_process, last_error = api.shellexecute_ex(
            "runas", python_exe, parameters, cwd,
        )
        if h_process is None:
            classified = _classify_launch_error(last_error)
            return {
                "output": "",
                "exit_code": -1,
                "error": classified["error"],
                "error_kind": classified["error_kind"],
            }

        try:
            # UAC approved -> helper connects to both pipes, within a hard
            # bound and watching the helper process for early exit.
            try:
                channel.wait_connect(h_process=h_process)
            except TimeoutError as exc:
                return {
                    "output": "",
                    "exit_code": -1,
                    "error": f"Elevated helper connect failed: {exc}",
                    "error_kind": "connect_timeout",
                }

            # Strict helper-message validation: nonce + required fields +
            # types + legal state transitions.
            state_machine = _HelperStateMachine()
            try:
                first = channel.read_message(
                    timeout_ms=_CONNECT_TIMEOUT_S * 1000
                )
                validate_helper_message(first, channel.nonce)
                state_machine.transition(first["kind"])
            except ProtocolError as exc:
                return {
                    "output": "",
                    "exit_code": -1,
                    "error": f"Elevated helper protocol error: {exc}",
                    "error_kind": "protocol",
                }
            if first.get("kind") != KIND_CONNECTED:
                return {
                    "output": "",
                    "exit_code": -1,
                    "error": (
                        f"Unexpected first helper message: "
                        f"{first.get('kind')!r} (expected connected)."
                    ),
                    "error_kind": "protocol",
                }

            # Bind the pipe client to the helper: the helper reports the pid
            # that actually opened the pipe; the OS cross-check comes from
            # GetNamedPipeClientProcessId on BOTH pipes, and the hProcess
            # PID (base python) must be the same process (or a trusted
            # ancestor of a launcher shim).
            if not channel.verify_client(h_process, first.get("pid")):
                return {
                    "output": "",
                    "exit_code": -1,
                    "error": (
                        "Elevated helper client PID does not match the "
                        "launched process on both pipes; refusing to send "
                        "the request (fail closed)."
                    ),
                    "error_kind": "pid_mismatch",
                }

            # Start draining BEFORE sending the request so no output is lost.
            channel.start_output_drain()
            channel.send(make_request(channel.nonce, command, cwd, env))

            # Wait for ready (job_bound / child_pid) or an immediate done
            # (e.g. CreateProcessW failed).
            job_bound = False
            child_pid: "int | None" = None
            try:
                progress = channel.read_message(timeout_ms=30_000)
                validate_helper_message(progress, channel.nonce)
                state_machine.transition(progress["kind"])
                if progress.get("kind") == KIND_READY:
                    job_bound = bool(progress.get("job_bound"))
                    child_pid = progress.get("child_pid")
                elif progress.get("kind") == KIND_DONE:
                    channel.join_output()
                    return _collect_elevated_result(
                        channel, int(progress.get("rc", -1))
                    )
                else:
                    return {
                        "output": "",
                        "exit_code": -1,
                        "error": (
                            "Unexpected helper progress message: "
                            f"{progress.get('kind')!r}."
                        ),
                        "error_kind": "protocol",
                    }
            except ProtocolError as exc:
                return {
                    "output": "",
                    "exit_code": -1,
                    "error": f"Elevated helper protocol error: {exc}",
                    "error_kind": "protocol",
                }
            except TimeoutError:
                return _handle_elevated_timeout(
                    api, channel, h_process, child_pid, job_bound, timeout
                )

            # Wait for done within the user timeout; then cancel flow.
            try:
                done = channel.read_message(timeout_ms=int(timeout * 1000))
                validate_helper_message(done, channel.nonce)
                state_machine.transition(done["kind"])
            except ProtocolError as exc:
                return {
                    "output": "",
                    "exit_code": -1,
                    "error": f"Elevated helper protocol error: {exc}",
                    "error_kind": "protocol",
                }
            except TimeoutError:
                return _handle_elevated_timeout(
                    api, channel, h_process, child_pid, job_bound, timeout
                )
            if done.get("kind") != KIND_DONE:
                return {
                    "output": "",
                    "exit_code": -1,
                    "error": (
                        "Unexpected helper final message: "
                        f"{done.get('kind')!r}."
                    ),
                    "error_kind": "protocol",
                }
            channel.join_output()
            return _collect_elevated_result(channel, int(done.get("rc", -1)))
        finally:
            # Close the elevated process handle on every path (success,
            # cancel, error, timeout).
            api.close_handle(h_process)
    finally:
        channel.close()
