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
    from hermes_cli.config import load_config_readonly
    return load_config_readonly().get("security") or {}


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
    mode = isolation_mode()
    if mode == "off":
        return list(argv)
    if not sys.platform.startswith("linux"):
        if sys.platform not in _WARNED:
            _WARNED.add(sys.platform)
            logger.warning("terminal secret isolation is not implemented on %s: the agent's shell can "
                           "read Hermes secrets on this host", sys.platform)
        return list(argv)
    from agent.file_safety import terminal_protected_paths
    _harden_parent()
    no_access, read_only = terminal_protected_paths()
    wrapped = [sys.executable, "-I", "-S", _HELPER, "--mode", mode]
    for path in no_access:
        wrapped += ["--no-access", path]
    for path in read_only:
        wrapped += ["--read-only", path]
    return [*wrapped, "--", *argv]
