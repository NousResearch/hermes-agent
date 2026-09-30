"""PM-backed cua-driver selection, MCP discovery and the runtime contract gate.
Config-derived policy (``_cua_no_overlay``, ``_run_driver`` ...) is looked up lazily through the facade."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tools.computer_use.cua_backend")

# PM owns the pinned binary; an explicit override remains externally owned.
_CUA_DRIVER_CMD_ENV = "HERMES_CUA_DRIVER_CMD"
_CUA_DRIVER_DEFAULT_CMD = "cua-driver"
_CUA_DRIVER_ARGS = ["mcp"]  # stdio MCP; fallback when the driver has no `manifest` verb
_CUA_DRIVER_RUNTIME_CONTRACT_MIN = (0, 20, 0)
_CUA_DRIVER_RUNTIME_CONTRACT_ARGS = {  # key order feeds the "manifest is missing" text
    "mcp": {"--socket", "--grant"},
    "serve": {"--socket", "--permission-mode", "--capability-manifest", "--approve-capability-manifest", "--embedded"},
    "stop": {"--socket"},
}
_SEMVER_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?")


def _cb():
    """Facade module (config/policy helpers), looked up lazily to avoid the import cycle."""
    from tools.computer_use import cua_backend
    return cua_backend

def _driver_json(driver_cmd: str, *args: str, timeout: float) -> Optional[Dict[str, Any]]:
    """Run a driver verb and parse its stdout as a JSON object; None on spawn failure, empty stdout (older drivers
    print usage to stderr), unparseable or non-object output, or a non-zero exit."""
    proc = _cb()._run_driver(driver_cmd, *args, timeout=timeout, swallow=Exception)
    out = (proc.stdout or "").strip() if proc is not None else ""
    return None if proc is None or not out or proc.returncode != 0 else _json_object(out)

def _json_object(text: str) -> Optional[Dict[str, Any]]:
    """``json.loads`` that yields a dict or None (unparseable / non-object)."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None

def _valid_mcp_args(invocation: Any) -> Optional[List[str]]:
    """``mcp_invocation.args`` when it is a list of strings (possibly empty), else None."""
    args = invocation.get("args") if isinstance(invocation, dict) else None
    return args if isinstance(args, list) and all(isinstance(a, str) for a in args) else None

def _has_path_separator(value: str) -> bool:
    return os.sep in value or (os.altsep is not None and os.altsep in value)

def _wsl_windows_path_to_posix(path: str) -> str:
    """Translate a Windows absolute manifest command to its DrvFS ``/mnt/<drive>/...`` form when Hermes runs in WSL
    (a Windows cua-driver manifest can report ``C:\\...`` while Hermes spawns via POSIX). Non-Windows paths and
    non-WSL hosts are returned unchanged."""
    if not re.match(r"^[A-Za-z]:[\\/]", path):
        return path
    try:
        from hermes_constants import is_wsl
        wsl = is_wsl()
    except Exception:
        wsl = False
    win = PureWindowsPath(path)
    drive = (win.drive or "").rstrip(":").lower()
    return "/".join(["/mnt", drive, *win.parts[1:]]) if wsl and drive else path

def resolve_cua_driver_cmd(override: Optional[str] = None) -> Optional[str]:
    """Read PM's selected binary without installing; never replace an explicit override."""
    configured = (override if override is not None else os.environ.get(_CUA_DRIVER_CMD_ENV, "")).strip()
    if configured:
        expanded = os.path.expanduser(configured)
        resolved = shutil.which(expanded)
        return expanded if resolved and _has_path_separator(expanded) else resolved
    from pm import installed_package

    installed = installed_package("cua-driver")
    return str(installed.binary) if installed and installed.binary else None

def cua_driver_binary_available() -> bool:
    """True if PM or an explicit override selects a local driver."""
    return resolve_cua_driver_cmd() is not None

def cua_driver_install_hint() -> str:
    return ("cua-driver is not installed. Install the pinned driver with:\n  hermes computer-use install\n"
            "Or run `hermes tools` and enable the Computer Use toolset to install it automatically.")

def _mcp_args_with_overlay_flag(args: List[str], driver_cmd: str = _CUA_DRIVER_DEFAULT_CMD) -> List[str]:
    """Return *args* with ``--no-overlay`` appended when configured and supported."""
    on = _cb()._cua_no_overlay() and _cua_driver_supports_no_overlay(driver_cmd)
    return [*args, "--no-overlay"] if on else list(args)

@functools.lru_cache(maxsize=1)
def _cua_driver_supports_no_overlay(driver_cmd: str) -> bool:
    """True if ``<driver> --help`` mentions ``--no-overlay`` (probed once); older drivers reject unknown flags, which
    would crash the MCP spawn."""
    try:
        proc = _cb()._run_driver(driver_cmd, "--help", timeout=3.0)
        return "--no-overlay" in (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return False

def _resolve_mcp_invocation(driver_cmd: str, *, timeout: float = 6.0) -> Tuple[str, List[str]]:
    """``(command, args)`` that spawn cua-driver's stdio MCP server, asked of the driver itself via ``cua-driver
    manifest`` (``mcp_invocation``) so a subcommand rename keeps working. Falls back to ``(driver_cmd, ["mcp"])``
    on older drivers or any discovery failure — the wrapper must not refuse to start over a failed discovery hop.
    ``--no-overlay`` appended when allowed.

    Surface 8 of NousResearch/hermes-agent#47072: instead of hardcoding ``["mcp"]`` we ask the driver itself
    via ``cua-driver manifest`` (trycua/cua#1961). The manifest carries a stable ``mcp_invocation`` pointer
    with both ``command`` and ``args``, so a future cua-driver that renames or relocates the subcommand
    keeps working without a Hermes patch.
    When ``computer_use.no_overlay`` is enabled (or auto-detected — macOS, headless/WSL2/X11 Linux),
    ``--no-overlay`` is appended to suppress the cursor overlay rendering loop that can consume CPU
    indefinitely when idle (#28152, #47032). Older drivers that don't recognise the flag will reject it;
    callers should fall back to the no-overlay invocation on spawn failure.
    """
    manifest = _driver_json(driver_cmd, "manifest", timeout=timeout) or {}
    invocation = manifest.get("mcp_invocation")
    args = _valid_mcp_args(invocation)
    command = invocation.get("command") if args is not None and isinstance(invocation, dict) else None
    args = list(_CUA_DRIVER_ARGS) if args is None else args
    # Translate a Windows ``C:\...`` command for WSL BEFORE the separator check (backslash is not a separator on
    # POSIX). A generic ``cua-driver`` name would lose the resolved user-local path under a GUI's thin PATH, so only
    # a concrete (path-bearing) command replaces the one we verified — and THAT binary is probed for `--no-overlay`,
    # not the system one.
    command = _wsl_windows_path_to_posix(command) if isinstance(command, str) and command else ""
    command = command if command and _has_path_separator(command) else driver_cmd
    return command, _mcp_args_with_overlay_flag(args, driver_cmd=command)

def _manifest_contract_reason(manifest: Optional[Dict[str, Any]]) -> str:
    """Why a parsed manifest fails the 0.20 contract, or ``""`` when it passes (version floor, MCP launch
    command, then the ``"<verb> <flag>"`` entries the advertised subcommands lack)."""
    if manifest is None:
        return "driver manifest is missing or invalid"
    match = _SEMVER_RE.fullmatch(str(manifest.get("binary_version") or "").strip())
    if not match:
        return "driver manifest does not report a semantic version"
    if tuple(int(part) for part in match.groups()) < _CUA_DRIVER_RUNTIME_CONTRACT_MIN:
        return "Hermes computer use requires cua-driver 0.20.0 or newer"
    if not _valid_mcp_args(manifest.get("mcp_invocation")):
        return "driver manifest does not provide an MCP launch command"
    advertised: Dict[str, set[str]] = {
        command["name"]: {arg["name"] for arg in command.get("args") or []
                          if isinstance(arg, dict) and isinstance(arg.get("name"), str)}
        for command in manifest.get("subcommands") or []
        if isinstance(command, dict) and isinstance(command.get("name"), str)
    }
    missing = [f"{command} {arg}" for command, required in _CUA_DRIVER_RUNTIME_CONTRACT_ARGS.items()
               for arg in sorted(required - advertised.get(command, set()))]
    return "driver manifest is missing: " + ", ".join(missing) if missing else ""

def cua_driver_runtime_contract_status(binary: Optional[str] = None) -> Dict[str, Any]:
    """Report whether a local driver can host Hermes' 0.20 integration."""
    resolved = binary or resolve_cua_driver_cmd()
    version: Optional[str] = None
    reason = "cua-driver is not installed"
    if resolved:
        try:
            result = _cb()._run_driver(resolved, "manifest", timeout=15.0 if sys.platform == "win32" else 5.0)
        except (OSError, subprocess.SubprocessError) as exc:
            result, reason = None, f"manifest check failed: {exc}"
        if result is not None and result.returncode != 0:
            result, reason = None, (result.stderr or result.stdout or "manifest command failed").strip().splitlines()[-1][:200]
        if result is not None:
            manifest = _json_object(result.stdout or "")
            reason = _manifest_contract_reason(manifest)
            version = str(manifest.get("binary_version") or "").strip() or None if manifest is not None else None
    return {"ready": not reason, "binary": resolved, "version": version, "reason": reason}

def cua_driver_update_check(*, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Historical import: upstream release polling is retired; PM owns the pin."""
    return None
