"""Refresh ``os.environ["PATH"]`` / ``os.environ["PATHEXT"]`` from the Windows
Registry so that tools installed since the current process started (WinGet, MSI,
etc.) are discoverable without a full restart.

This module mirrors the pattern from
``kimix/utils/windows_env.py`` / ``kimi-cli/src/kimi_cli/utils/environment.py``
in the upstream kimi-agent project.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    import ctypes
    import winreg


def ps_with_utf8(command: str) -> str:
    """Wrap a PowerShell command with UTF-8 encoding directives.

    Prepends ``[Console]::OutputEncoding`` and ``$OutputEncoding`` settings
    so PowerShell emits UTF-8 to stdout regardless of the system code page.
    No-op on non-Windows.

    Usage::

        cmd = ps_with_utf8("Get-ChildItem")
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], ...)

    The helper is **idempotent** — if a preamble is already present it won't
    double-prepend.  This is safe to call even on strings that may have been
    wrapped by an earlier code path.
    """
    if sys.platform != "win32":
        return command
    preamble = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        "$OutputEncoding=[System.Text.Encoding]::UTF8;"
    )
    if command.startswith(preamble):
        return command
    return preamble + command


def _expand_registry_string(value: str) -> str:
    """Expand a ``REG_EXPAND_SZ`` value using the Windows API.

    ``os.path.expandvars`` only expands against the current process
    environment, which may be stale.  The Windows API
    ``ExpandEnvironmentStringsW`` performs a fresh expansion against
    the *system* and *user* environment blocks, giving the correct
    result even for variables that were changed externally.
    """
    if "%" not in value:
        return value
    try:
        _ExpandEnvironmentStringsW = (
            ctypes.windll.kernel32.ExpandEnvironmentStringsW
        )
        nchars = _ExpandEnvironmentStringsW(value, None, 0)
        if nchars == 0:
            return value
        buf = ctypes.create_unicode_buffer(nchars)
        _ExpandEnvironmentStringsW(value, buf, nchars)
        return buf.value
    except Exception:
        return os.path.expandvars(value)


def _read_registry_value(
    hive: int, subkey: str, name: str
) -> tuple[str | None, int | None]:
    """Read a named value from the registry.

    Returns ``(value, reg_type)``.  *value* may be ``None`` when the
    value does not exist or cannot be read.  *reg_type* is the Windows
    registry type constant (e.g. ``winreg.REG_SZ``).
    """
    try:
        with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ) as key:
            val, reg_type = winreg.QueryValueEx(key, name)
            if isinstance(val, str):
                return val, reg_type
            return None, None
    except (FileNotFoundError, OSError):
        return None, None


def _merge_dedup_paths(*sources: str) -> str:
    """Merge semicolon-separated *sources*, deduplicating case-insensitively."""
    seen: set[str] = set()
    merged: list[str] = []
    for src in sources:
        for part in src.split(";"):
            part = part.strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                merged.append(part)
    return ";".join(merged)


def refresh_env_from_registry() -> None:
    """Refresh ``os.environ["PATH"]`` and ``os.environ["PATHEXT"]``
    from the Windows registry.

    Reads both the system (HKLM) and user (HKCU) values,
    expands ``REG_EXPAND_SZ`` entries via the Windows API, and
    merges them into the current process environment.

    After calling this function, ``shutil.which`` and
    ``subprocess.Popen`` can locate binaries installed by
    external package managers (WinGet, MSI, etc.) without
    restarting the process.

    This is a no-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        return

    # --- PATH ---
    sys_val, sys_type = _read_registry_value(
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        "Path",
    )
    usr_val, usr_type = _read_registry_value(
        winreg.HKEY_CURRENT_USER,
        r"Environment",
        "Path",
    )

    path_parts: list[str] = []
    if sys_val:
        if sys_type == winreg.REG_EXPAND_SZ:
            sys_val = _expand_registry_string(sys_val)
        path_parts.append(sys_val)
    if usr_val:
        if usr_type == winreg.REG_EXPAND_SZ:
            usr_val = _expand_registry_string(usr_val)
        path_parts.append(usr_val)

    if path_parts:
        os.environ["PATH"] = _merge_dedup_paths(*path_parts)

    # --- PATHEXT ---
    sys_val, sys_type = _read_registry_value(
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        "PATHEXT",
    )
    usr_val, usr_type = _read_registry_value(
        winreg.HKEY_CURRENT_USER,
        r"Environment",
        "PATHEXT",
    )

    pathext_parts: list[str] = []
    if sys_val:
        if sys_type == winreg.REG_EXPAND_SZ:
            sys_val = _expand_registry_string(sys_val)
        pathext_parts.append(sys_val)
    if usr_val:
        if usr_type == winreg.REG_EXPAND_SZ:
            usr_val = _expand_registry_string(usr_val)
        pathext_parts.append(usr_val)

    if pathext_parts:
        os.environ["PATHEXT"] = _merge_dedup_paths(*pathext_parts)
