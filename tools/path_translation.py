"""Path translation helpers for local Windows/MSYS/WSL interop."""

from __future__ import annotations

import platform
import re
from typing import Any


_MSYS_DRIVE_RE = re.compile(r"^/([A-Za-z])(?:/(.*))?$")
_WSL_DRIVE_RE = re.compile(r"^/mnt/([A-Za-z])(?:/(.*))?$")
_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")


def is_windows_host() -> bool:
    return platform.system() == "Windows"


def _host_enabled(host_is_windows: bool | None) -> bool:
    return is_windows_host() if host_is_windows is None else bool(host_is_windows)


def _windows_drive_path(drive: str, tail: str | None) -> str:
    suffix = (tail or "").replace("/", "\\")
    return f"{drive.upper()}:\\" + suffix if suffix else f"{drive.upper()}:\\"


def normalize_windows_host_path(path: Any, *, host_is_windows: bool | None = None) -> str:
    """Return a native Windows path for local Windows-host POSIX drive forms.

    Converts MSYS/Git Bash paths like ``/c/Users`` and WSL paths like
    ``/mnt/c/Users`` to ``C:\\Users``. Non-Windows hosts, remote/container
    paths, UNC paths, and already-native Windows paths are returned unchanged.
    """

    text = str(path or "")
    if not _host_enabled(host_is_windows) or not text:
        return text
    wsl = _WSL_DRIVE_RE.match(text)
    if wsl:
        return _windows_drive_path(wsl.group(1), wsl.group(2))
    msys = _MSYS_DRIVE_RE.match(text)
    if msys:
        return _windows_drive_path(msys.group(1), msys.group(2))
    return text


def windows_path_to_git_bash_path(path: Any, *, host_is_windows: bool | None = None) -> str:
    """Return a Git Bash/MSYS drive path for a local Windows-host path."""

    text = str(path or "")
    if not _host_enabled(host_is_windows) or not text:
        return text
    normalized = normalize_windows_host_path(text, host_is_windows=True)
    match = _WINDOWS_DRIVE_RE.match(normalized)
    if not match:
        return text
    tail = match.group(2).replace("\\", "/")
    root = f"/{match.group(1).lower()}"
    return f"{root}/{tail}" if tail else root


def windows_path_to_wsl_path(path: Any, *, host_is_windows: bool | None = None) -> str:
    """Return a WSL ``/mnt/<drive>`` path for a local Windows-host path."""

    text = str(path or "")
    if not _host_enabled(host_is_windows) or not text:
        return text
    normalized = normalize_windows_host_path(text, host_is_windows=True)
    match = _WINDOWS_DRIVE_RE.match(normalized)
    if not match:
        return text
    tail = match.group(2).replace("\\", "/")
    root = f"/mnt/{match.group(1).lower()}"
    return f"{root}/{tail}" if tail else root


def missing_path_hint(original: Any, normalized: Any) -> str:
    """Build a concise normalized-path miss message without directory scans."""

    original_text = str(original or "")
    normalized_text = str(normalized or "")
    if original_text and normalized_text and original_text != normalized_text:
        return (
            f"Path not found after normalization: {normalized_text} "
            f"(normalized from {original_text}). Verify that the drive/folder exists "
            "or pass the existing native Windows path."
        )
    return f"Path not found: {normalized_text or original_text}"
