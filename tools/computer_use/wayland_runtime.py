"""Install Hermes' Linux Wayland policy at the cua-driver spawn-env seam.

Current Computer Use has several cua-driver launch paths (standard, bounded,
unrestricted/YOLO, runtime-contract probes, and CLI fallback). They all flow
through ``cua_backend.cua_driver_child_env`` before crossing the process
boundary. Wrapping that single function keeps native-Wayland selection aligned
across the evolved upstream lifecycle without copying or replacing permission
logic from ``cua_backend.py``.

The wrappers are Linux-only, idempotent, and fail-closed. Non-Linux imports are
a no-op. If capability diagnosis fails, the child explicitly receives
``CUA_DRIVER_RS_ENABLE_WAYLAND=0`` rather than inheriting a stale opt-in.
Feature-manifest probes also pass through Hermes' subprocess sanitizer before
the third-party driver is launched.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Any

from tools.computer_use.linux_wayland import WAYLAND_ENABLE_ENV

logger = logging.getLogger(__name__)

_INSTALLED_ATTR = "__hermes_native_wayland_policy__"
_PROBE_INSTALLED_ATTR = "__hermes_sanitized_wayland_probe__"


def _wrap_feature_probe_for_safety(wayland: Any):
    """Sanitize the environment before ``cua-driver manifest`` feature probes."""
    current = wayland.probe_driver_features
    if getattr(current, _PROBE_INSTALLED_ATTR, False):
        return current

    @functools.wraps(current)
    def wrapped(driver_cmd, env=None):
        if not driver_cmd:
            return current(driver_cmd, env)
        try:
            from tools.environments.local import _sanitize_subprocess_env

            source_env = dict(os.environ if env is None else env)
            probe_env = _sanitize_subprocess_env(source_env)
        except Exception:
            # The feature probe is optional evidence used to admit native
            # Wayland in auto mode. If we cannot scrub the environment, do not
            # spawn a third-party binary and do not claim support.
            logger.debug(
                "computer_use: could not sanitize native Wayland feature probe",
                exc_info=True,
            )
            return wayland.CuaDriverFeatures()
        return current(driver_cmd, probe_env)

    setattr(wrapped, _PROBE_INSTALLED_ATTR, True)
    wayland.probe_driver_features = wrapped
    return wrapped


def _wrap_child_env_for_wayland(backend: Any):
    """Wrap ``backend.cua_driver_child_env`` once and return the active callable.

    Kept separate from :func:`install_wayland_runtime_policy` so the routing
    contract can be tested with a tiny dummy backend rather than importing the
    entire Computer Use runtime.
    """
    current = backend.cua_driver_child_env
    if getattr(current, _INSTALLED_ATTR, False):
        return current

    @functools.wraps(current)
    def wrapped(base_env=None):
        env = current(base_env)
        try:
            from tools.computer_use.linux_wayland import native_wayland_child_env

            driver_cmd = backend.resolve_cua_driver_cmd()
            config = backend._computer_use_cfg()
            return native_wayland_child_env(driver_cmd, config, env)
        except Exception:
            # Native Wayland is an authority/capability decision, not a hint.
            # Never let a stale inherited opt-in bypass a failed policy probe.
            logger.debug(
                "computer_use: native Wayland child-env policy failed closed",
                exc_info=True,
            )
            env[WAYLAND_ENABLE_ENV] = "0"
            return env

    setattr(wrapped, _INSTALLED_ATTR, True)
    backend.cua_driver_child_env = wrapped
    return wrapped


def install_wayland_runtime_policy() -> bool:
    """Install the Linux cua-driver Wayland policy once.

    Returns ``True`` when this call installed the child-env wrapper, ``False``
    when the platform is not Linux or the wrapper was already present. The
    feature-probe sanitizer is independently idempotent.
    """
    if sys.platform != "linux":
        return False

    # Import lazily so non-Linux platforms do not pull the CUA backend merely
    # by importing the package. Importing these siblings during package
    # initialization is cycle-safe: cua_backend depends on backend/browser
    # siblings, not on tools.computer_use.tool.
    from tools.computer_use import cua_backend
    from tools.computer_use import linux_wayland

    _wrap_feature_probe_for_safety(linux_wayland)
    if getattr(cua_backend.cua_driver_child_env, _INSTALLED_ATTR, False):
        return False
    _wrap_child_env_for_wayland(cua_backend)
    return True
