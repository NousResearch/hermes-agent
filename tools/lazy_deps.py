"""Shims to suppress old updater work until relaunch. New code must not use these."""

from typing import NoReturn

from hermes_cli._old_updater import in_historical_update, stop_for_relaunch


def ensure(feature: str, *, prompt: bool = True) -> NoReturn:
    # Shim to suppress old updater work until relaunch. Do not claim readiness.
    # Preserve the dependency-unavailable failure without claiming a completed install.
    # Relaunching installs nothing: name the durable command so catchers that
    # surface this text stop directing users at a bare (soon-pruned) uv install.
    raise ImportError(
        "Dependencies are unknown to this old updater. Install them durably with "
        "`hermes pm install` (or `hermes pm install --extra NAME` for a declared "
        "extra); a bare `uv pip install` is pruned on the next environment sync."
    )


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300,
                  constraints: list | None = None, dry_run: bool = False) -> NoReturn:
    # Plugins still call this retired API during normal agent construction.
    # Only an actual updater call stack may transfer control to the updater;
    # argv can still say "serve" or "gateway" when /update runs in-process.
    # Historical updaters also passed constraints/dry_run; accepted and ignored.
    if in_historical_update():
        # never returns: hands off to the takeover child and exits
        stop_for_relaunch()
    # specs arrive as pip names; only an extra NAME records a selection that
    # later syncs keep, so point there rather than asserting unavailability (#135131).
    raise ImportError(
        "tools.lazy_deps.install_specs is retired; runtime dependency "
        "installation is unavailable from the agent. Declare the dependency as "
        "a project extra and enable it with `hermes pm install --extra NAME`; "
        "a bare `uv pip install` is pruned on the next environment sync."
    )
