"""systemd user-scope argv construction, runtime-dir checks, and degrade wording.

Split out of ``tools/process_registry`` so the facade keeps shrinking; the probe
verdicts and their cached reason stay there, written under its double-checked
lock. See #70716 and #137222."""

import os
import stat
from pathlib import Path


def _default_user_runtime_dir() -> Path:
    """``/run/user/<uid>``; a function so tests can point it at a temp dir with a real socket."""
    return Path(f"/run/user/{os.getuid()}")  # windows-footgun: ok — only reached behind the _IS_LINUX gate in systemd_user_bus_env


def _secure_user_runtime_dir(path: Path) -> bool:
    """Accept only an absolute, owned, non-writable real directory."""
    try:
        metadata = path.lstat()
        return (
            path.is_absolute()
            and stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == os.getuid()  # windows-footgun: ok — only reached behind the _IS_LINUX gate in systemd_user_bus_env
            and metadata.st_mode & 0o022 == 0
        )
    except OSError:
        return False


def _build_systemd_scope_argv(shell_argv: list[str], unit_suffix: str) -> list[str]:
    """Wrap *shell_argv* in a ``systemd-run --user --scope`` invocation with its own
    memory accounting, so an OOM in the worker cannot kill the gateway cgroup.

    ``--collect`` makes the transient scope self-clean after exit; ``--unit`` gives it a recognisable name
    for ``systemctl --user status`` / journalctl. See #70716.
    """
    import shutil

    from tools.process_registry import _systemd_scope_argv

    binary = shutil.which("systemd-run")
    if binary is None:
        # Caller should have probed availability; never pass None into Popen anyway.
        return shell_argv
    return _systemd_scope_argv(binary, f"hermes-worker-{unit_suffix}", *shell_argv)


def _systemd_scope_unavailable_detail() -> str:
    """Degrade detail for a False scope verdict that names the actual cause (#137222):
    a missing ``systemd-run`` binary is a service-PATH problem, not a user-bus problem,
    and the two remedies differ."""
    from tools.process_registry import _SYSTEMD_SCOPE_UNAVAILABLE_REASON

    reason = _SYSTEMD_SCOPE_UNAVAILABLE_REASON or "no probe verdict is cached yet"
    if reason.startswith("systemd-run was not found"):
        return (
            f"{reason}. The gateway needs the systemd-run binary to place children in "
            "restart-safe user scopes; install systemd or add it to the service unit PATH "
            "(NixOS Home Manager: services.hermes-agent.extraPackages = [ pkgs.systemd ])"
        )
    return (
        f"systemd-run --user --scope is unavailable: {reason} (usually no reachable user "
        f"D-Bus session at /run/user/{os.getuid()}/bus). On a system-level service install, run "  # windows-footgun: ok — the caller returns before this on non-Linux
        "`sudo loginctl enable-linger <gateway-user>` and restart the gateway."
    )
