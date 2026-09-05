"""cua-driver binary resolution, MCP-invocation discovery, the 0.20 runtime contract gate, and the update check.
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

# No version *pin* knob on purpose: the upstream installer always fetches the latest release, so a pin
# var would only LOOK like it pinned. Point HERMES_CUA_DRIVER_CMD at a specific binary instead.
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
_UPSTREAM_SCRIPTS = "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts"

def _cb():
    """Facade module (config/policy helpers), looked up lazily to avoid the import cycle."""
    from tools.computer_use import cua_backend
    return cua_backend

def _driver_json(driver_cmd: str, *args: str, timeout: float, require_ok: bool, execution_context=None) -> Optional[Dict[str, Any]]:
    """Run a driver verb and parse its stdout as a JSON object; None on spawn failure, empty stdout (older drivers
    print usage to stderr), unparseable or non-object output — and, with ``require_ok``, on a non-zero exit."""
    proc = _cb()._run_driver(driver_cmd, *args, timeout=timeout, swallow=Exception,
                            **({"execution_context": execution_context} if execution_context else {}))
    out = (proc.stdout or "").strip() if proc is not None else ""
    return None if not out or (require_ok and proc.returncode != 0) else _json_object(out)

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
    return os.path.join("/mnt", drive, *(str(part) for part in win.parts[1:])) if wsl and drive else path

def _candidate_cua_driver_commands(override: Optional[str] = None) -> List[str]:
    """Candidate commands in resolution order. ``override`` / a non-empty ``HERMES_CUA_DRIVER_CMD`` is authoritative
    (if wrong, report the driver missing rather than silently picking another binary). Otherwise PATH, then
    canonical installer locations — Finder/Dock-launched apps inherit a narrow PATH without ``~/.local/bin``;
    fresh Windows sessions inherit a stale one."""
    configured = (override if override is not None else os.environ.get(_CUA_DRIVER_CMD_ENV, "")).strip()
    if configured:
        return [configured]
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return [_CUA_DRIVER_DEFAULT_CMD, os.path.join(local_app_data, "Programs", "Cua", "cua-driver", "bin", "cua-driver.exe"),
                os.path.join(home, ".local", "bin", "cua-driver.exe"), os.path.join(home, ".local", "bin", "cua-driver")]
    return [_CUA_DRIVER_DEFAULT_CMD, os.path.join(home, ".local", "bin", "cua-driver"),
            os.path.join(home, ".cargo", "bin", "cua-driver"), "/opt/homebrew/bin/cua-driver", "/usr/local/bin/cua-driver"]

def resolve_cua_driver_cmd(override: Optional[str] = None) -> Optional[str]:
    """Resolve the cua-driver executable for every runtime/status surface; an override is never silently replaced."""
    for expanded in map(os.path.expanduser, _candidate_cua_driver_commands(override)):
        resolved = shutil.which(expanded)
        if resolved:
            return expanded if _has_path_separator(expanded) else resolved
    return None

def cua_driver_binary_available() -> bool:
    """True if `cua-driver` resolves via env, PATH, or known install paths."""
    return resolve_cua_driver_cmd() is not None

def cua_driver_install_hint() -> str:
    installer = (f"  irm {_UPSTREAM_SCRIPTS}/install.ps1 | iex" if sys.platform == "win32"
                 else f'  /bin/bash -c "$(curl -fsSL {_UPSTREAM_SCRIPTS}/install.sh)"')
    return ("cua-driver is not installed. Install with one of:\n  hermes computer-use install\n"
            f"Or run the upstream installer directly:\n{installer}\n"
            "Or run `hermes tools` and enable the Computer Use toolset to install it automatically.")

def _mcp_args_with_overlay_flag(args: List[str], driver_cmd: str = _CUA_DRIVER_DEFAULT_CMD, *, execution_context=None) -> List[str]:
    """Return *args* with ``--no-overlay`` appended when configured and supported."""
    if execution_context is not None:
        launch = execution_context.context.computer_use
        no_overlay = launch.no_overlay if launch and launch.no_overlay is not None else _cb()._cua_no_overlay()
        # Context-selected binaries pass the runtime contract; pass explicit policy
        # without falling back to a process-global cached probe/driver.
        return [*args, "--no-overlay"] if no_overlay else list(args)
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

def _resolve_mcp_invocation(driver_cmd: str, *, timeout: float = 6.0, execution_context=None) -> Tuple[str, List[str]]:
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
    manifest = _driver_json(driver_cmd, "manifest", timeout=timeout, require_ok=True,
                            **({"execution_context": execution_context} if execution_context else {})) or {}
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
    if execution_context is not None:
        # Explicit selection is authoritative; don't execute a manifest-selected stranger.
        command = driver_cmd
        return command, _mcp_args_with_overlay_flag(args, driver_cmd=command, execution_context=execution_context)
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

def cua_driver_runtime_contract_status(binary: Optional[str] = None, *, execution_context=None) -> Dict[str, Any]:
    """Report whether a local driver can host Hermes' 0.20 integration."""
    resolved = binary or resolve_cua_driver_cmd()
    version: Optional[str] = None
    reason = "cua-driver is not installed"
    if resolved:
        try:
            result = _cb()._run_driver(resolved, "manifest", timeout=15.0 if sys.platform == "win32" else 5.0,
                                       **({"execution_context": execution_context} if execution_context else {}))
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
    """cua-driver's native ``check-update`` verb compares the installed binary against the latest GitHub release
    (cached ~20h); we prefer it over a hardcoded floor. Returns the ``check-update --json`` payload (``{current_version, latest_version, update_available, ...}``),
    or ``None`` when the binary is missing, the driver predates the verb, the GitHub check failed (``error`` set)
    or the output didn't parse. Never raises. ``timeout`` defaults to 8s on POSIX / 25s on Windows: first spawn of
    the exe routinely eats seconds in Defender scanning, and callers treat ``None`` as indeterminate (the upgrade
    path used to fall through to a full reinstall on a false timeout).

    See #1734.
    """
    timeout = (25.0 if sys.platform == "win32" else 8.0) if timeout is None else timeout
    driver_cmd = resolve_cua_driver_cmd()
    data = _driver_json(driver_cmd, "check-update", "--json", timeout=timeout, require_ok=False) if driver_cmd else None
    return None if data is None or data.get("error") else data

def cua_driver_update_nudge() -> Optional[str]:
    """One-line "an update is available" message, or ``None`` when up to date, indeterminate, or driver too old."""
    state = cua_driver_update_check()
    if not state or not state.get("update_available"):
        return None
    return (f"cua-driver {state.get('latest_version') or '?'} is available "
            f"(you have {state.get('current_version') or '?'}); update with `hermes computer-use install --upgrade`.")
