"""
Cross-platform Computer Use readiness + macOS permission helpers.

cua-driver runs on macOS, Windows, and Linux, but "ready to drive" means
something different on each:

  * macOS — explicit TCC grants (Accessibility + Screen Recording). cua-driver
    reports/requests them via ``permissions status`` / ``permissions grant``.
    The grants attach to cua-driver's OWN identity (``com.trycua.driver`` /
    the installed ``CuaDriver.app``), NOT Hermes — so no Hermes entitlement is
    involved, and ``grant`` launches CuaDriver via LaunchServices so the macOS
    dialog is attributed correctly.
  * Windows — no TCC toggles; the UIAccess worker (``cua-driver-uia.exe``) may
    trip a SmartScreen prompt on first run. Readiness == driver health.
  * Linux — assistive control via the X11/XWayland stack. Readiness == driver
    health.

The universal signal on every platform is ``cua-driver doctor --json`` (binary
integrity + platform support). ``computer_use_status`` folds that together with
the macOS permission detail into one payload for the desktop card, the
``hermes computer-use permissions`` CLI, and ``/api/tools/computer-use/status``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Platforms with a cua-driver runtime backend (mirrors the toolset platform_gate).
_RUNTIME_PLATFORMS = frozenset({"darwin", "win32", "linux"})
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


def _json_out(binary: str, *args: str, timeout: float) -> Any:
    """Run ``binary args`` and parse stdout as JSON, or ``None`` on any failure."""
    raw = (_run(binary, *args, timeout=timeout).stdout or "").strip()
    return json.loads(raw) if raw else None


def _doctor(binary: str) -> Optional[Dict[str, Any]]:
    """``cua-driver doctor --json`` → ``{ok, checks:[{label,status,message}]}``."""
    try:
        data = _json_out(binary, "doctor", "--json", timeout=12)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    checks: List[Dict[str, str]] = [
        {
            "label": str(p.get("label", "")),
            "status": str(p.get("status", "")),
            "message": str(p.get("message", "")),
        }
        for p in data.get("probes", [])
        if isinstance(p, dict)
    ]
    return {"ok": bool(data.get("ok")), "checks": checks}


def _mac_permissions(binary: str, out: Dict[str, Any]) -> None:
    """Fold ``cua-driver permissions status --json`` booleans into ``out``."""
    try:
        data = _json_out(binary, "permissions", "status", "--json", timeout=10)
    except subprocess.TimeoutExpired:
        out["error"] = "cua-driver permissions status timed out"
        return
    except Exception as exc:  # spawn failure or malformed JSON
        out["error"] = f"cua-driver permissions status failed: {exc}"
        return
    if isinstance(data, dict):
        out.update({k: data[k] for k in _BOOLS if isinstance(data.get(k), bool)})
        if isinstance(data.get("source"), dict):
            out["source"] = data["source"]


def computer_use_status(driver_cmd: Optional[str] = None) -> Dict[str, Any]:
    """Unified, OS-aware Computer Use readiness for the desktop card.

    ``ready`` is the single signal the UI keys off: on macOS it's both TCC
    grants; elsewhere it's driver health (no TCC model). ``None`` means
    unknown (binary missing / probe failed). ``can_grant`` is macOS-only.
    """
    plat = sys.platform
    binary = shutil.which(_driver_cmd(driver_cmd))
    out: Dict[str, Any] = {
        "platform": plat,
        "platform_supported": plat in _RUNTIME_PLATFORMS,
        "installed": bool(binary),
        "version": None,
        "ready": None,
        "can_grant": plat == "darwin",
        "checks": [],
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

    doctor = _doctor(binary)
    if doctor is not None:
        out["checks"] = doctor["checks"]

    if plat == "darwin":
        _mac_permissions(binary, out)
        if out["error"] is None:
            out["ready"] = out["accessibility"] is True and out["screen_recording"] is True
    elif doctor is not None:
        # No TCC model off macOS — readiness is driver health.
        out["ready"] = doctor["ok"]
    return out


def request_permissions_grant(driver_cmd: Optional[str] = None) -> int:
    """Run ``cua-driver permissions grant`` (macOS); stream its output.

    Launches CuaDriver via LaunchServices so the TCC dialog is attributed to
    ``com.trycua.driver``, then waits for the grant. Returns the driver's exit
    code (0 ok), 2 if the binary is missing, 64 on a non-macOS platform (which
    has no TCC permission model to grant).
    """
    if sys.platform != "darwin":
        print("Computer Use permissions are a macOS concept; nothing to grant here.")
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
