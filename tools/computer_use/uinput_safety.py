"""Guard against trycua/cua#2618 on Linux/X11 computer-use sessions.

``ensure_master_pointer`` in cua-driver-rs created an XInput master pointer
pair *before* opening ``/dev/uinput``. When that device wasn't read+write
accessible, every real-pointer input attempt (click/drag/scroll/keyboard)
created and abandoned a fresh XInput master, and a sustained session
eventually hit Xorg's ``BadAlloc`` once too many masters had piled up,
killing the driver connection outright.

Fixed upstream by trycua/cua#2631 (released as ``cua-driver-rs-v0.13.1``):
the driver now checks uinput accessibility before touching the XInput
hierarchy at all. This module is Hermes' front-line guard for users still on
an older driver (Hermes #74148) — it detects the exact known-unsafe
combination and lets both ``hermes computer-use doctor``
(:mod:`tools.computer_use.doctor`) and the cua-driver backend
(:mod:`tools.computer_use.cua_backend`) react to it consistently from one
tested predicate.

Deliberately conservative: any input this module can't positively confirm as
unsafe (wrong platform, no X11 DISPLAY, an unparseable/missing driver
version, an already-fixed driver, or an accessible device) preserves current
(no-guard) behavior rather than guessing.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

from tools.computer_use.backend import ActionResult

UINPUT_DEVICE_PATH = "/dev/uinput"
UINPUT_LEAK_RISK_CODE = "linux_uinput_xinput_leak_risk"

# First cua-driver-rs release containing the trycua/cua#2631 fix.
_FIXED_VERSION: Tuple[int, int, int] = (0, 13, 1)

_VERSION_RE = re.compile(r"^\s*v?(\d+)\.(\d+)\.(\d+)\s*$")


def parse_driver_version(version: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Parse a bare semver-ish string ("0.13.1", "v0.9.0") into a comparable
    ``(major, minor, patch)`` tuple.

    Returns ``None`` for missing / non-string / malformed input (extra
    suffixes, missing components, non-numeric parts, ...) so callers treat
    it as "unknown" rather than guessing a side.
    """
    if not isinstance(version, str):
        return None
    m = _VERSION_RE.match(version)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def driver_version_is_vulnerable(version: Optional[str]) -> Optional[bool]:
    """True if ``version`` predates the trycua/cua#2631 fix (< 0.13.1),
    False if it's fixed (>= 0.13.1), None if unparseable/unknown."""
    parsed = parse_driver_version(version)
    if parsed is None:
        return None
    return parsed < _FIXED_VERSION


def uinput_read_write_accessible(path: str = UINPUT_DEVICE_PATH) -> bool:
    """True if this process can open ``path`` for both reading and writing.

    Mirrors cua-driver's own ``uinput_accessible()`` gate (trycua/cua#2631):
    an actual read+write ``open()`` attempt, not just a permission-bit
    check, since that's the real operation the vulnerable code path
    performs. Any failure (missing device, permission denied, ...) is
    "not accessible" — this function never raises.
    """
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    else:
        os.close(fd)
        return True


def uinput_leak_risk_possible(
    *,
    platform: str,
    display: Optional[str],
    driver_version: Optional[str],
) -> bool:
    """True if platform, display and driver version alone leave the
    trycua/cua#2618 risk open — i.e. Linux, an X11 session (``DISPLAY``
    set), and a parseable cua-driver version below 0.13.1.

    This is exactly the device-free half of :func:`detect_uinput_leak_risk`,
    factored out so callers can skip the ``/dev/uinput`` probe entirely when
    it could not change the answer. A False here means no device state can
    produce risk, so a fixed/current driver (or a non-Linux, display-less,
    or unknown-version host) never opens the device unnecessarily.
    """
    if platform != "linux":
        return False
    if not display:
        return False
    return driver_version_is_vulnerable(driver_version) is True


def detect_uinput_leak_risk(
    *,
    platform: str,
    display: Optional[str],
    driver_version: Optional[str],
    uinput_accessible: bool,
) -> Optional[Dict[str, Any]]:
    """Return a risk descriptor when the known-unsafe combination from
    trycua/cua#2618 is present, else ``None``.

    Unsafe combination: Linux, an X11 session (``DISPLAY`` set), a
    parseable cua-driver version below 0.13.1, and ``/dev/uinput`` not
    read+write accessible. Every other combination — non-Linux, no
    ``DISPLAY``, an unparseable/missing version, a fixed version, or an
    accessible device — preserves current (no-guard) behavior.

    Callers that must pay to learn ``uinput_accessible`` should gate on
    :func:`uinput_leak_risk_possible` first; this function stays total so
    it can also be called with a value already in hand.
    """
    if not uinput_leak_risk_possible(
        platform=platform, display=display, driver_version=driver_version
    ):
        return None
    if uinput_accessible:
        return None
    return {
        "code": UINPUT_LEAK_RISK_CODE,
        "driver_version": driver_version,
        "uinput_path": UINPUT_DEVICE_PATH,
    }


def uinput_leak_risk_message(risk: Dict[str, Any]) -> str:
    version = risk.get("driver_version") or "unknown"
    path = risk.get("uinput_path", UINPUT_DEVICE_PATH)
    return (
        f"cua-driver {version} on Linux/X11 can leak XInput master pointers "
        f"when {path} is not read+write accessible, eventually crashing the "
        "session (trycua/cua#2618, fixed in trycua/cua#2631; Hermes #74148). "
        "Upgrade cua-driver to >=0.13.1: `hermes computer-use install --upgrade`."
    )


def uinput_leak_risk_hint() -> str:
    return (
        "Upgrade cua-driver to >=0.13.1 (`hermes computer-use install --upgrade`). "
        "This is a driver-side fix (trycua/cua#2631) — do not work around it by "
        "changing /dev/uinput's permissions or running as a more privileged user."
    )


def uinput_leak_risk_check(risk: Dict[str, Any]) -> Dict[str, Any]:
    """Build a ``health_report``-shaped check dict for
    ``hermes computer-use doctor`` to append to its checks list."""
    return {
        "name": "linux_uinput_xinput_leak_risk",
        "status": "fail",
        "message": uinput_leak_risk_message(risk),
        "hint": uinput_leak_risk_hint(),
        "data": {
            "driver_version": risk.get("driver_version"),
            "uinput_path": risk.get("uinput_path"),
            "upstream_issue": "trycua/cua#2618",
            "upstream_fix": "trycua/cua#2631",
            "hermes_issue": "#74148",
        },
    }


def uinput_leak_risk_action_result(action: str, risk: Dict[str, Any]) -> ActionResult:
    """Build the refusal ``ActionResult`` an input action returns when
    ``detect_uinput_leak_risk`` reports risk, instead of declaring an input
    session or invoking the driver."""
    return ActionResult(
        ok=False,
        action=action,
        message=uinput_leak_risk_message(risk),
        code=UINPUT_LEAK_RISK_CODE,
        meta={
            "driver_version": risk.get("driver_version"),
            "uinput_path": risk.get("uinput_path"),
            "upstream_issue": "trycua/cua#2618",
            "upstream_fix": "trycua/cua#2631",
            "hermes_issue": "#74148",
        },
    )
