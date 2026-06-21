"""Windows elevated command execution via UAC.

Provides the ability to run terminal commands with administrator privileges
on Windows by using ShellExecuteW with the "runas" verb, which triggers
the standard UAC (User Account Control) elevation prompt.

Safety guarantees:
- No UAC bypass: uses standard Windows ShellExecuteW("runas")
- No credential storage: each execution is an independent process
- No silent elevation: UAC prompt is always shown to the user
- Explicit opt-in: elevated=True must be passed explicitly
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import time


_TIMEOUT_S = 120
_POLL_INTERVAL_S = 0.5


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


def can_elevate() -> bool:
    """Return True if elevation is possible on this platform.

    Elevation requires:
    - Windows platform
    - ShellExecuteW available
    - Not already running as admin (elevation would be a no-op)
    """
    if not is_windows():
        return False
    if is_running_as_admin():
        return False
    try:
        return hasattr(ctypes.windll.shell32, "ShellExecuteW")
    except Exception:
        return False


def execute_elevated(
    command: str,
    cwd: str | None = None,
    timeout: int = _TIMEOUT_S,
) -> dict:
    """Execute a command with Windows administrator privileges via UAC.

    Flow:
    1. Write command to a temp .cmd script that redirects output to a file
    2. Call ShellExecuteW("runas", "cmd.exe", "/c script.cmd") -> UAC prompt
    3. Poll for the output file to appear
    4. Return {"output": str, "exit_code": int, "error": str|None}

    Args:
        command: The shell command to execute
        cwd: Working directory (optional)
        timeout: Max seconds to wait for completion

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

    tmp_dir = tempfile.mkdtemp(prefix="hermes_elevated_")
    output_file = os.path.join(tmp_dir, "output.txt")
    rc_file = os.path.join(tmp_dir, "rc.txt")
    done_file = os.path.join(tmp_dir, "done.txt")

    effective_cwd = cwd or os.getcwd()
    script_content = (
        f"@echo off\r\n"
        f"cd /d \"{effective_cwd}\"\r\n"
        f"{command} > \"{output_file}\" 2>&1\r\n"
        f"echo %ERRORLEVEL% > \"{rc_file}\"\r\n"
        f"echo done > \"{done_file}\"\r\n"
    )

    script_path = os.path.join(tmp_dir, "elevated.cmd")
    try:
        with open(script_path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(script_content)
    except OSError as e:
        return {
            "output": "",
            "exit_code": -1,
            "error": f"Failed to write elevated command script: {e}",
        }

    try:
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            "cmd.exe",
            f'/c "{script_path}"',
            effective_cwd,
            0,
        )
    except Exception as exc:
        return {
            "output": "",
            "exit_code": -1,
            "error": f"ShellExecuteW failed: {exc}",
        }

    if result <= 32:
        error_map = {
            2: "File not found",
            5: "Access denied",
            8: "Out of memory",
            26: "Sharing violation",
            27: "File name association incomplete",
            28: "DDE timeout",
            29: "DDE transaction failed",
            30: "DDE busy",
            31: "No association for file type",
            1223: "User cancelled the UAC prompt",
        }
        error_msg = error_map.get(result, f"Unknown error (code {result})")
        return {
            "output": "",
            "exit_code": -1,
            "error": f"Elevated launch failed: {error_msg} (ShellExecuteW={result})",
        }

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(done_file):
            output = ""
            exit_code = -1
            try:
                with open(output_file, "r", encoding="utf-8", errors="replace") as f:
                    output = f.read()
            except FileNotFoundError:
                pass
            try:
                with open(rc_file, "r") as f:
                    exit_code = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                pass
            return {
                "output": output,
                "exit_code": exit_code,
                "error": None,
            }
        time.sleep(_POLL_INTERVAL_S)

    return {
        "output": "",
        "exit_code": -1,
        "error": f"Elevated command timed out after {timeout}s. UAC may have been dismissed or the command is still running.",
    }
