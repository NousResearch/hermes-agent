"""Shims to suppress old updater work until relaunch. New code must not use these."""

from typing import NoReturn

from hermes_cli._old_updater import in_historical_update, stop_for_relaunch


def ensure(feature: str, *, prompt: bool = True) -> NoReturn:
    # Shim to suppress old updater work until relaunch. Do not claim readiness.
    # Preserve the dependency-unavailable failure without claiming a completed install.
    raise ImportError("Dependencies are unknown to this old updater. Please relaunch Hermes.")


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300,
                  constraints: list | None = None, dry_run: bool = False) -> NoReturn:
    # Plugins still call this retired API during normal agent construction.
    # Only an actual updater call stack may transfer control to the updater;
    # argv can still say "serve" or "gateway" when /update runs in-process.
    # Historical updaters also passed constraints/dry_run; accepted and ignored.
    if in_historical_update():
        # never returns: hands off to the takeover child and exits
        stop_for_relaunch()
    raise ImportError(
        "tools.lazy_deps.install_specs is retired; runtime dependency "
        "installation is unavailable."
    )
