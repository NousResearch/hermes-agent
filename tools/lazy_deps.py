"""Shims to suppress old updater work until relaunch. New code must not use these."""

from typing import NoReturn


def ensure(feature: str, *, prompt: bool = True) -> NoReturn:
    # Shim to suppress old updater work until relaunch. Do not claim readiness.
    # Preserve the dependency-unavailable failure without claiming a completed install.
    raise ImportError("Dependencies are unknown to this old updater. Please relaunch Hermes.")


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300) -> NoReturn:
    # Shim to suppress old updater work until relaunch. Do not install or report success.
    # Fail catchably, like ensure(): callers cannot know they are talking to a shim, and
    # stop_for_relaunch() both runs the update-takeover machinery and exits the process —
    # inside a worker or agent-build thread that kills the thread and the session never
    # becomes ready ("Available Tools" empty until /new). An exception lets the caller
    # degrade; a caller that genuinely needs the handoff should request it explicitly.
    raise ImportError("Dependencies are unknown to this old updater. Please relaunch Hermes.")
