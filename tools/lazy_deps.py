"""Shims to suppress old updater work until relaunch. New code must not use these."""

from typing import NoReturn

from hermes_cli._old_updater import stop_historical_updater


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
    stop_historical_updater("tools.lazy_deps.install_specs", ImportError)
