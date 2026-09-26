"""Shims to suppress old updater work until relaunch. New code must not use these."""

import sys
from typing import NoReturn

from hermes_cli._old_updater import stop_for_relaunch


def ensure(feature: str, *, prompt: bool = True) -> NoReturn:
    # Shim to suppress old updater work until relaunch. Do not claim readiness.
    # Preserve the dependency-unavailable failure without claiming a completed install.
    raise ImportError("Dependencies are unknown to this old updater. Please relaunch Hermes.")


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300) -> NoReturn:
    # Plugins still call this retired API during normal agent construction.
    # Only an actual updater call stack may transfer control to the updater;
    # argv can still say "serve" or "gateway" when /update runs in-process.
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if (frame.f_globals.get("__name__") in ("hermes_cli.update_cmd", "hermes_cli.main")
                    and frame.f_code.co_name in ("_cmd_update_impl", "cmd_update")):
                stop_for_relaunch()
            frame = frame.f_back
    finally:
        del frame
    raise ImportError(
        "tools.lazy_deps.install_specs is retired; runtime dependency installation "
        "is unavailable. Update the plugin to use Hermes package management."
    )
