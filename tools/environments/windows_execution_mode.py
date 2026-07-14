"""Shared Windows execution-mode selection for Desktop PTYs and agent commands."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hermes_cli._subprocess_compat import windows_hide_flags

VALID_MODES = frozenset({"smart", "wsl2", "windows-native"})


@dataclass(frozen=True)
class WindowsExecutionMode:
    configured: str
    resolved: str
    distribution: str | None = None


def normalize_mode(value: object) -> str:
    return value if isinstance(value, str) and value in VALID_MODES else "smart"


def _config_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "desktop-terminal.json"
    except Exception:
        return Path.home() / ".hermes" / "desktop-terminal.json"


def read_configured_mode() -> str:
    override = os.environ.get("HERMES_WINDOWS_EXECUTION_MODE")
    if override:
        return normalize_mode(override)

    try:
        data = json.loads(_config_path().read_text(encoding="utf-8"))
        return normalize_mode(data.get("mode"))
    except Exception:
        return "smart"


def _decode_wsl_output(raw: bytes) -> str:
    if b"\x00" in raw:
        for encoding in ("utf-16-le", "utf-8"):
            try:
                return raw.decode(encoding).replace("\x00", "")
            except UnicodeError:
                pass
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


def parse_wsl_distributions(raw: bytes | str) -> list[str]:
    text = _decode_wsl_output(raw) if isinstance(raw, bytes) else raw.replace("\x00", "")
    result: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        name = re.sub(r"^\*\s*", "", line).strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result


def wsl_executable() -> str | None:
    return shutil.which("wsl.exe") or shutil.which("wsl")


def list_wsl_distributions() -> list[str]:
    executable = wsl_executable()
    if not executable:
        return []
    try:
        completed = subprocess.run(
            [executable, "--list", "--quiet"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
    except OSError:
        return []
    return parse_wsl_distributions(completed.stdout) if completed.returncode == 0 else []


def distribution_from_wsl_path(cwd: str) -> str | None:
    match = re.match(r"^\\\\(?:wsl\.localhost|wsl\$)\\([^\\/]+)(?:[\\/]|$)", cwd or "", re.I)
    return match.group(1) if match else None


def _matching_distribution(distributions: list[str], requested: str | None) -> str | None:
    if not requested:
        return None
    requested_key = requested.casefold()
    return next((name for name in distributions if name.casefold() == requested_key), None)


def resolve_windows_execution_mode(
    cwd: str = "",
    *,
    configured: str | None = None,
    distributions: list[str] | None = None,
    is_windows: bool | None = None,
) -> WindowsExecutionMode:
    if is_windows is None:
        is_windows = os.name == "nt"
    configured_mode = normalize_mode(configured if configured is not None else read_configured_mode())
    available = list_wsl_distributions() if distributions is None and is_windows else (distributions or [])

    if not is_windows or configured_mode == "windows-native":
        return WindowsExecutionMode(configured_mode, "windows-native")

    path_distribution = _matching_distribution(available, distribution_from_wsl_path(cwd))
    if configured_mode == "smart":
        if path_distribution:
            return WindowsExecutionMode(configured_mode, "wsl2", path_distribution)
        return WindowsExecutionMode(configured_mode, "windows-native")

    distribution = path_distribution or (available[0] if available else None)
    if distribution:
        return WindowsExecutionMode(configured_mode, "wsl2", distribution)
    return WindowsExecutionMode(configured_mode, "windows-native")


def windows_to_wsl_path(value: str, distribution: str | None = None) -> str:
    raw = (value or "").strip()
    if not raw:
        return "~"

    unc = re.match(r"^\\\\(?:wsl\.localhost|wsl\$)\\([^\\/]+)(?:[\\/](.*))?$", raw, re.I)
    if unc and (not distribution or unc.group(1).casefold() == distribution.casefold()):
        tail = (unc.group(2) or "").replace("\\", "/").lstrip("/")
        return f"/{tail}" if tail else "/"

    drive = re.match(r"^([a-zA-Z]):[\\/]*(.*)$", raw)
    if drive:
        tail = (drive.group(2) or "").replace("\\", "/").lstrip("/")
        suffix = f"/{tail}" if tail else ""
        return f"/mnt/{drive.group(1).lower()}{suffix}"

    return raw.replace("\\", "/")


def wsl_to_windows_path(value: str, distribution: str | None) -> str:
    raw = (value or "").strip()
    mount = re.match(r"^/mnt/([a-zA-Z])(?:/(.*))?$", raw)
    if mount:
        tail = (mount.group(2) or "").replace("/", "\\")
        return f"{mount.group(1).upper()}:\\{tail}" if tail else f"{mount.group(1).upper()}:\\"
    if raw.startswith("/") and distribution:
        tail = raw.lstrip("/").replace("/", "\\")
        base = f"\\\\wsl.localhost\\{distribution}"
        return f"{base}\\{tail}" if tail else base
    return raw


def build_wsl_bash_args(command: str, cwd: str, *, login: bool, distribution: str) -> list[str]:
    executable = wsl_executable()
    if not executable:
        raise RuntimeError("WSL2 mode is selected, but wsl.exe is not available")
    shell_flag = "-lc" if login else "-c"
    return [
        executable,
        "--distribution",
        distribution,
        "--cd",
        windows_to_wsl_path(cwd, distribution),
        "--exec",
        "bash",
        shell_flag,
        command,
    ]
