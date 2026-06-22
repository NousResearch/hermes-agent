"""
macOS Accessibility + Screen Recording permission helpers for Computer Use.

cua-driver 0.5+ owns the permission model. Crucially, the grants attach to
cua-driver's OWN TCC identity (``com.trycua.driver`` — the installed
``CuaDriver.app``), NOT the terminal, the Hermes CLI, or the Hermes desktop
app. So:

  * ``cua-driver permissions status --json`` reports the driver daemon's real
    grant state, independent of who asks.
  * ``cua-driver permissions grant`` launches CuaDriver via LaunchServices so
    the macOS dialog is attributed to ``com.trycua.driver`` — the process that
    actually does the work.

Because the permission lives with the cua-driver binary, the Hermes desktop
app needs no Accessibility / Screen Recording entitlements of its own. This is
a thin, testable client driven by the ``hermes computer-use permissions`` CLI
and the desktop ``/api/tools/computer-use/status`` endpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

_BOOLS = ("accessibility", "screen_recording", "screen_recording_capturable")


def _driver_cmd(override: Optional[str]) -> str:
    if override:
        return override
    try:
        from hermes_cli.tools_config import _cua_driver_cmd

        return _cua_driver_cmd()
    except Exception:
        return os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip() or "cua-driver"


def _child_env() -> Dict[str, str]:
    """cua-driver child env honoring the Hermes telemetry opt-in policy."""
    try:
        from tools.computer_use.cua_backend import cua_driver_child_env

        return cua_driver_child_env()
    except Exception:
        return dict(os.environ)


def _run(binary: str, *args: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_child_env(),
    )


def permissions_status(driver_cmd: Optional[str] = None) -> Dict[str, Any]:
    """Computer Use install + macOS permission state for the desktop card.

    ``None`` permission values mean "unknown" — the driver binary is missing,
    the platform has no TCC model, or no CuaDriver daemon is running to answer
    for its own identity yet.
    """
    binary = shutil.which(_driver_cmd(driver_cmd))
    out: Dict[str, Any] = {
        "platform_supported": sys.platform == "darwin",
        "installed": bool(binary),
        "version": None,
        "source": None,
        "error": None,
        **{k: None for k in _BOOLS},
    }
    if not binary:
        return out

    try:
        out["version"] = (_run(binary, "--version", timeout=5).stdout or "").strip() or None
    except Exception:
        pass

    # Permissions are a macOS concept; cua-driver only exposes the subcommand there.
    if sys.platform != "darwin":
        return out

    try:
        raw = (_run(binary, "permissions", "status", "--json", timeout=10).stdout or "").strip()
        data = json.loads(raw) if raw else {}
    except subprocess.TimeoutExpired:
        out["error"] = "cua-driver permissions status timed out"
        return out
    except Exception as exc:  # spawn failure or malformed JSON
        out["error"] = f"cua-driver permissions status failed: {exc}"
        return out

    if isinstance(data, dict):
        out.update({k: data[k] for k in _BOOLS if isinstance(data.get(k), bool)})
        if isinstance(data.get("source"), dict):
            out["source"] = data["source"]
    return out


def request_permissions_grant(driver_cmd: Optional[str] = None) -> int:
    """Run ``cua-driver permissions grant`` (macOS); stream its output.

    Launches CuaDriver via LaunchServices so the TCC dialog is attributed to
    ``com.trycua.driver``, then waits for the grant. Returns the driver's exit
    code (0 ok), 2 if the binary is missing, 64 on an unsupported platform.
    """
    if sys.platform != "darwin":
        print("Computer Use permissions are managed on macOS only.")
        return 64

    binary = shutil.which(_driver_cmd(driver_cmd))
    if not binary:
        print("cua-driver: not installed. Run: hermes computer-use install")
        return 2

    print(
        "Requesting Accessibility + Screen Recording for CuaDriver.\n"
        "macOS will show a dialog attributed to CuaDriver (com.trycua.driver) — "
        "approve it, then return here."
    )
    try:
        return int(subprocess.run([binary, "permissions", "grant"], env=_child_env()).returncode)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
    except Exception as exc:  # pragma: no cover - defensive
        print(f"cua-driver permissions grant failed: {exc}", file=sys.stderr)
        return 2
