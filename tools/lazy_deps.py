"""Shims to suppress old updater work until relaunch. New code must not use these."""

from typing import NoReturn

from hermes_cli._old_updater import in_historical_update, stop_for_relaunch


def ensure(feature: str, *, prompt: bool = True) -> NoReturn:
    # Shim to suppress old updater work until relaunch. Do not claim readiness.
    # Preserve the dependency-unavailable failure without claiming a completed install.
    raise ImportError("Dependencies are unknown to this old updater. Please relaunch Hermes.")


def _pkg_name_from_spec(spec: str) -> str:
    """Distribution name of a requirement spec ("chromadb>=0.4" -> "chromadb")."""
    for sep in ("<", ">", "=", "!", "~", "[", " ", ";"):
        spec = spec.split(sep, 1)[0]
    return spec.strip()


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300,
                  constraints: list | None = None, dry_run: bool = False) -> None:
    # Plugins still call this retired API during normal agent construction.
    # Only an actual updater call stack may transfer control to the updater;
    # argv can still say "serve" or "gateway" when /update runs in-process.
    # Historical updaters also passed constraints/dry_run; accepted and ignored.
    if in_historical_update():
        # never returns: hands off to the takeover child and exits
        stop_for_relaunch()
    # The install half is retired, but the ANSWER must still be true (#135131): plugins
    # called this to make a dependency exist, and a blanket "unavailable" made a memory
    # provider drop its driver even when the package was importable through the paths
    # PM installs into. Satisfied specs return; only genuinely missing ones raise.
    import importlib.util

    missing = [
        spec for spec in specs
        if importlib.util.find_spec(_pkg_name_from_spec(spec)) is None
    ]
    if missing:
        raise ImportError(
            "tools.lazy_deps.install_specs is retired; runtime dependency installation "
            f"is unavailable and these are not importable: {', '.join(missing)}"
        )
