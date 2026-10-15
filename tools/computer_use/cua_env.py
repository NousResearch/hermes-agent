"""Pure cua-driver child-env builder: no config, no application imports.

``cua_backend`` (config-aware) and the PM verify probe (``pm/packages.py``, which must not pull
``hermes_cli.config`` and with it provider discovery) both build the child env here, so the
telemetry gate and the Bot Desktop screen policy cannot drift between them.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

CUA_TELEMETRY_ENV_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"  # "0" disables; absent => ON upstream
CUA_NATIVE_WAYLAND_ENV_VAR = "CUA_DRIVER_RS_ENABLE_WAYLAND"


def build_child_env(
    base_env: Optional[dict[str, str]] = None, *, telemetry_disabled: bool, native_wayland: bool
) -> dict[str, str]:
    """``base_env`` (default ``os.environ``) plus the profile's Bot Desktop screen when one is
    running, telemetry off when ``telemetry_disabled``, and the native-Wayland bridge when
    ``native_wayland`` is requested and the child has a Wayland display (Linux only)."""
    env = dict(os.environ if base_env is None else base_env)
    # A running Bot Desktop for this profile owns the agent's screen: DISPLAY/XAUTHORITY/DBUS point there so
    # cua-driver never acts on a seat the human is sitting at (#90374 class) and headless hosts get a display.
    from tools.bot_desktop.runtime import desktop_env
    env = desktop_env(env)
    if telemetry_disabled:
        env[CUA_TELEMETRY_ENV_VAR] = "0"
    if native_wayland and sys.platform == "linux" and env.get("WAYLAND_DISPLAY"):
        env[CUA_NATIVE_WAYLAND_ENV_VAR] = "1"
    return env
