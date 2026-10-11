"""Kernel-enforced secret isolation for agent-driven children.

Every spawn that runs agent-chosen code on the Hermes host (foreground terminal, background/PTY
processes, the execute_code kernel, cron scripts) passes its argv through :func:`wrap_argv`. On
Linux the command then starts under ``landlock_exec.py``, which applies a Landlock ruleset hiding
the secret stores listed by ``agent.file_safety.terminal_protected_paths`` and freezing the
directory entries around them before exec'ing the command.

``security.terminal_secret_isolation`` (config.yaml):

- ``auto`` (default): Linux children are sandboxed when Landlock is usable; otherwise they run
  unprotected and a one-time warning is logged.
- ``require``: Linux children are always sandboxed; a host without usable Landlock refuses the
  command (fail closed). Recommended wherever the agent reads untrusted content.
- ``off``: no isolation.

Other OSes have no implementation: commands run unprotected with a one-time warning, so run the
agent on Linux (or a remote backend) where this matters. Unknown values and an unreadable config
fail closed to ``require``.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_HELPER = str(Path(__file__).with_name("landlock_exec.py"))
_MODES = ("require", "auto", "off")
_ALIASES = {"true": "require", "on": "require", "enforce": "require", "false": "off", "disabled": "off", "none": "off"}
_PR_SET_DUMPABLE = 4
_WARNED: set[str] = set()
_PARENT_HARDENED = False


def _security_config() -> dict:
    """The ``security`` section; pinned to ``require`` when config cannot be read, so an
    unreadable config fails closed instead of weakening isolation."""
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly().get("security") or {}
    except Exception:
        logger.warning("terminal secret isolation: config unreadable, enforcing 'require'", exc_info=True)
        return {"terminal_secret_isolation": "require"}


def isolation_mode() -> str:
    raw = _security_config().get("terminal_secret_isolation", "auto")
    if isinstance(raw, bool):
        return "require" if raw else "off"
    mode = str(raw).strip().lower()
    mode = _ALIASES.get(mode, mode)
    return mode if mode in _MODES else "require"


def _harden_parent() -> None:
    """Make the spawning process non-dumpable once: its ``/proc/<pid>/environ``, ``mem`` and
    ``fd/`` become unreadable and ptrace-attach is refused for same-UID processes, so a child
    cannot lift secrets out of the gateway's own environment or memory."""
    global _PARENT_HARDENED
    if _PARENT_HARDENED:
        return
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        logger.warning("terminal secret isolation: prctl(PR_SET_DUMPABLE, 0) failed (errno %s)", ctypes.get_errno())
        return
    _PARENT_HARDENED = True


def wrap_argv(argv: list[str]) -> list[str]:
    """Return *argv* wrapped in the Landlock helper per the configured mode."""
    if not sys.platform.startswith("linux"):
        if sys.platform not in _WARNED:
            _WARNED.add(sys.platform)
            logger.warning("terminal secret isolation is not implemented on %s: the agent's shell can "
                           "read Hermes secrets on this host", sys.platform)
        return list(argv)
    mode = isolation_mode()
    if mode == "off":
        return list(argv)
    if mode == "auto" and "landlock" not in _WARNED:
        _WARNED.add("landlock")
        from tools.environments.landlock_exec import abi_version
        if abi_version() < 1:
            logger.warning("terminal secret isolation: Landlock is unavailable on this host, so agent commands "
                           "run unprotected; set security.terminal_secret_isolation: require to refuse them")
    from agent.file_safety import terminal_protected_paths
    _harden_parent()
    # When a real secret exists, HERMES_HOME is frozen (no new top-level entries), so the terminal
    # session-snapshot cannot create its cache dir there on a first run. Pre-create it (and it is then
    # enumerated as a writable subdir), so the snapshot works under the frozen root. Best-effort.
    from hermes_constants import get_hermes_home
    try:
        (Path(get_hermes_home()) / "cache" / "terminal").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("terminal secret isolation: cannot pre-create the snapshot cache dir: %s", exc)
    no_access, read_only = terminal_protected_paths()
    wrapped = [sys.executable, "-I", "-S", _HELPER, "--mode", mode]
    for path in no_access:
        wrapped += ["--no-access", path]
    for path in read_only:
        wrapped += ["--read-only", path]
    return [*wrapped, "--", *argv]
