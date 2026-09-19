"""Machine-wide ``cua-driver serve`` daemon diagnostics: is a daemon configured, is it reachable, and does the
unit / desktop entry that starts it still point at a binary that exists?

The driver BINARY and its daemon are separate failure domains. Hermes' runtime contract only inspects the binary
(version floor, manifest verbs), so a systemd/launchd/desktop reference left pointing into a pruned
``packages/releases/<version>/`` directory kills ``computer_use`` while version/platform/session/AX checks stay green
(issue #114748: 46h / 45k journal lines, doctor still said "binary healthy"). This module starts and repairs
nothing — it reports the state and names the fix.
"""

from __future__ import annotations

import glob
import logging
import os
import shlex
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional

from tools.computer_use.cua_backend_driver import _has_path_separator, resolve_cua_driver_cmd

logger = logging.getLogger("tools.computer_use.cua_backend")

# Hermes' stable launcher: the upstream installer repoints `packages/current` on every upgrade and prunes the
# versioned release directories it replaced (last 5 kept), so a `releases/<version>/...` reference has a bounded
# lifetime. Anything Hermes prints into user-visible config must name THIS path.
STABLE_CUA_DRIVER_LAUNCHER = "~/.cua-driver/packages/current/cua-driver"
_RELEASES_DIR = "~/.cua-driver/packages/releases"
_PROBE_ERRORS = (OSError, subprocess.SubprocessError, ValueError, TypeError)
_NO_STATUS_VERB = ("unrecognized", "unknown subcommand", "unexpected argument", "invalid subcommand")


def _cb():
    """Facade module (``_run_quiet``), looked up lazily to avoid the import cycle."""
    from tools.computer_use import cua_backend
    return cua_backend


def _reference_globs() -> List[str]:
    """Unit / desktop / launchd files that can carry a ``cua-driver serve`` command. Windows has none: there the
    driver is registered as a scheduled task (``cua-driver autostart``), not a config file."""
    if sys.platform == "win32":
        return []
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return [os.path.join(home, "Library", "LaunchAgents", "*.plist"),
                "/Library/LaunchAgents/*.plist", "/Library/LaunchDaemons/*.plist"]
    return [os.path.join(home, ".config", "systemd", "user", "*.service"),
            "/etc/systemd/system/*.service",
            os.path.join(home, ".config", "autostart", "*.desktop"),
            os.path.join(home, ".local", "share", "applications", "*.desktop")]


def _expand(token: str) -> str:
    """Expand ``~`` and systemd's ``%h`` (home) specifier — the two forms hand-written units use."""
    return os.path.expanduser(token.replace("%h", os.path.expanduser("~")))


def _command_lines(text: str) -> List[str]:
    """Command strings of a systemd unit (``ExecStart=``), desktop entry (``Exec=``) or launchd plist
    (``<string>`` under ProgramArguments). Comments are skipped, so a commented-out unit is not a reference."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        for key in ("ExecStart=", "Exec="):
            if line.startswith(key):
                out.append(line[len(key):].strip())
        if line.startswith("<string>") and line.endswith("</string>"):
            out.append(line[len("<string>"):-len("</string>")].strip())
    return out


def _argv(command: str) -> List[str]:
    """Best-effort argv, minus systemd's ``-``/``@``/``+``/``!`` execution-control prefixes."""
    command = command.strip()
    if command[:1] in {"-", "@", "+", "!", ":", "|"}:
        command = command[1:].strip()
    try:
        # posix=False on Windows: these files (systemd/desktop/launchd) only exist on POSIX, and there posix-mode
        # shlex is the right parser — on Windows it would eat the backslashes of a ``C:\...`` path we still want
        # to report on (e.g. a reference discovered via a test or a synced config).
        return [tok.strip("\"'") for tok in shlex.split(command, posix=sys.platform != "win32") if tok]
    except ValueError:  # unbalanced quote — split on whitespace rather than lose the reference
        return command.split()


def _flag_value(argv: List[str], flag: str) -> Optional[str]:
    """Value of ``--flag VALUE`` / ``--flag=VALUE``, or None."""
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return _expand(argv[index + 1])
        if token.startswith(flag + "="):
            return _expand(token[len(flag) + 1:])
    return None


def _reference(path: str, command: str) -> Optional[Dict[str, Any]]:
    """One serve reference: ``{path, command, binary, binary_exists, socket}``, or None when *command* is not
    ``cua-driver … serve`` (comments, unrelated services)."""
    argv = _argv(_expand(command))
    if "serve" not in [tok.lower() for tok in argv]:
        return None
    binary = next((tok for tok in argv if "cua-driver" in os.path.basename(tok).lower()), "")
    if not binary:
        return None
    is_path = _has_path_separator(binary)
    return {"path": path, "command": command, "binary": binary, "binary_is_path": is_path,
            "binary_exists": (not is_path) or os.path.exists(binary),  # a bare name still resolves via PATH
            "socket": _flag_value(argv, "--socket")}


def cua_driver_serve_references() -> List[Dict[str, Any]]:
    """Every unit / desktop entry / launchd plist on this host that starts a ``cua-driver serve`` daemon."""
    found: List[Dict[str, Any]] = []
    for pattern in _reference_globs():
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError as exc:
                logger.debug("cua daemon reference %s unreadable: %s", path, exc)
                continue
            found += [ref for command in _command_lines(text) if (ref := _reference(path, command))]
    return found


def daemon_probe(binary: Optional[str], socket_path: Optional[str] = None) -> Optional[bool]:
    """Is a ``serve`` daemon reachable? ``cua-driver status [--socket P]`` exits 0 when one accepts connections.
    None = cannot tell: no driver to ask, the spawn failed, or this driver predates the ``status`` verb — an
    indeterminate probe must never be reported as a dead daemon."""
    driver = binary or resolve_cua_driver_cmd()
    if driver:
        args = [driver, "status", *(["--socket", socket_path] if socket_path else [])]
        proc = _cb()._run_quiet(args, timeout=5.0, swallow=_PROBE_ERRORS)
        if proc is not None:
            if proc.returncode == 0:
                return True
            text = ((proc.stdout or "") + (proc.stderr or "")).lower()
            return None if any(word in text for word in _NO_STATUS_VERB) else False
    if socket_path and hasattr(socket, "AF_UNIX") and os.path.exists(socket_path):
        try:  # no driver to ask, but a connectable unix socket is proof enough
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                return sock.connect_ex(socket_path) == 0
        except (OSError, ValueError):
            return None
    return None


def _freshness(references: List[Dict[str, Any]], probe: Optional[bool]) -> str:
    """State name from the references + probe combination (see ``cua_driver_daemon_status``)."""
    if not references:
        return "not_configured"
    if any(not ref["binary_exists"] for ref in references):
        return "not_running"  # the unit's target is gone — it can never start
    if resolve_cua_driver_cmd() is None:
        return "not_installed"
    if probe is True:
        return "running"
    return "not_running" if probe is False else "unknown"


def cua_driver_daemon_status(*, binary: Optional[str] = None) -> Dict[str, Any]:
    """Diagnose the machine-wide daemon: configured? reachable? does the unit that starts it still exist?

    ``state`` is one of ``not_configured`` (nothing starts a daemon here — the MCP runtime spawns its own on
    demand), ``not_installed`` (a unit references a driver that is not installed), ``not_running`` (configured but
    dead — stale ``ExecStart`` target or nothing listening), ``running``, ``unknown`` (indeterminate probe).
    Never raises; a diagnostics failure must not become a new failure the operator cannot act on.
    """
    try:
        references = cua_driver_serve_references()
        missing = [ref for ref in references if ref["binary_is_path"] and not ref["binary_exists"]]
        sockets = [ref["socket"] for ref in references if ref["socket"]]
        # Only probe when there is something to probe FOR: with no serve reference nothing on this host starts a
        # daemon (the MCP runtime spawns its own), and with a missing target the daemon is provably dead. Probing
        # anyway would add a `cua-driver status` spawn to every healthy doctor run for no information.
        probe = daemon_probe(binary, sockets[0] if sockets else None) if references and not missing else None
        state = _freshness(references, probe)
    except Exception as exc:  # pragma: no cover - defensive: probes are best-effort by contract
        logger.debug("cua daemon status probe failed: %s", exc)
        return {"state": "unknown", "configured": False, "running": None, "socket": None, "references": [],
                "missing_targets": [], "reason": f"daemon probe failed: {exc}", "hint": ""}
    where = sockets[0] if sockets else None
    reason, hint = "", ""
    if state == "not_running":
        if missing:
            reason = f"{missing[0]['path']} starts a daemon from a path that does not exist: {missing[0]['binary']}"
            hint = (f"Point it at the stable launcher ({STABLE_CUA_DRIVER_LAUNCHER}) and restart it "
                    f"({_restart_hint(missing[0]['path'])}); the installer prunes old {_RELEASES_DIR}/<version>/ "
                    "directories. Reinstalling the driver will NOT fix this — the binary itself is fine.")
        else:
            reason = "a serve unit exists but no cua-driver daemon is listening" + (f" on {where}" if where else "")
            hint = f"Inspect why it will not start: {_logs_hint(references[0]['path'])}"
    elif state == "not_installed":
        reason = f"{references[0]['path']} starts cua-driver, but no cua-driver binary is installed"
        hint = "Run: hermes computer-use install"
    elif state == "unknown":
        reason = "could not determine whether a cua-driver serve daemon is reachable"
        hint = "Ask the driver directly: cua-driver status" + (f" --socket {where}" if where else "")
    return {"state": state, "configured": bool(references), "running": state == "running", "socket": where,
            "references": references, "missing_targets": [ref["binary"] for ref in missing],
            "reason": reason, "hint": hint}


def _systemd_scope(unit_path: str) -> str:
    """``--user `` for a unit under ``~/.config/systemd/user``, ``""`` for a system unit (which needs sudo)."""
    user_dir = os.path.normcase(os.path.join(os.path.expanduser("~"), ".config", "systemd", "user"))
    return "--user " if os.path.normcase(os.path.abspath(unit_path)).startswith(user_dir) else ""

def _restart_hint(unit_path: str) -> str:
    """How to restart the unit at *unit_path*, per its own kind."""
    name = os.path.basename(unit_path)
    if unit_path.endswith(".plist"):
        return f"launchctl kickstart -k gui/$(id -u)/{name[:-len('.plist')]}"
    if unit_path.endswith(".service"):
        scope = _systemd_scope(unit_path)
        sudo = "" if scope else "sudo "
        return f"{sudo}systemctl {scope}daemon-reload && {sudo}systemctl {scope}restart {name}"
    return f"edit {unit_path} and re-launch the session"  # desktop entry: picked up at next login


def _logs_hint(unit_path: str) -> str:
    """Where to read why a unit died (the crash-loop symptom the issue only saw in the journal)."""
    name = os.path.basename(unit_path)
    if unit_path.endswith(".plist"):
        return 'log show --last 5m --predicate \'process == "cua-driver"\''
    if unit_path.endswith(".service"):
        return f"journalctl {_systemd_scope(unit_path)}-u {name} -n 50"
    return f"the daemon's own log (it is started by {name})"
