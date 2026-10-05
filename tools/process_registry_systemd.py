"""systemd user-scope unit control for terminal workers: enumerate and stop scopes.

Split out of ``process_registry`` to keep that facade under its size cap; callers
late-import from here. See #70716.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


def list_systemd_user_scope_units(pattern: str) -> list[str]:
    """Unit names the user manager currently has loaded as scopes matching ``pattern``.

    Fail-closed, like the other systemd seams here: no ``systemctl``, an unreachable
    manager, or a failed call returns ``[]``, which callers read as "nothing to do".
    ``--all`` is deliberate — a scope whose processes are gone can still be loaded.
    """
    import shutil

    from tools.process_registry import systemd_user_bus_env

    binary = shutil.which("systemctl")
    if binary is None:
        return []
    try:
        result = subprocess.run(
            [
                binary,
                "--user",
                "list-units",
                "--type=scope",
                "--all",
                "--plain",
                "--no-legend",
                "--no-pager",
                pattern,
            ],
            capture_output=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
            env=systemd_user_bus_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    units: list[str] = []
    for line in (result.stdout or b"").decode(errors="replace").splitlines():
        fields = line.split()
        if fields and fields[0].endswith(".scope"):
            units.append(fields[0])
    return units


def _stop_systemd_unit(unit_name: str, *, no_block: bool = False) -> bool:
    """Stop a transient systemd user scope by unit name.
    Reaps the *entire* cgroup — catching double-forked descendants reparented to init
    inside the scope that survive a plain PID signal (SIGTERM all, SIGKILL after
    ``TimeoutStopSec``). True if stopped or already gone; False if ``systemctl`` is
    unavailable or the stop failed.

    ``no_block`` enqueues the job and returns without waiting for it (``systemctl
    --no-block``), for shutdown and restart paths that must not spend the unit's
    ``TimeoutStopSec`` on an escapee that ignores SIGTERM.

    See #70716.
    """
    import shutil

    from tools.process_registry import systemd_user_bus_env

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, "--user", *(("--no-block",) if no_block else ()), "stop", unit_name],
            capture_output=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
            env=systemd_user_bus_env(),
        )
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(errors="replace").strip()
            if any(marker in stderr.lower() for marker in ("not loaded", "not found", "does not exist")):
                return True
            logger.debug("systemctl --user stop %s exited %d: %s", unit_name, result.returncode, stderr)
            return False
        return True
    except Exception as exc:
        logger.debug("systemctl --user stop %s failed: %s", unit_name, exc)
        return False
