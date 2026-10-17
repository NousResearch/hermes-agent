"""Cross-process MCP subprocess ownership diagnostic.

Three Hermes surfaces (gateway daemon, dashboard, an interactive TUI/CLI
session) can each hold their own independent stdio connections to the same
configured ``mcp_servers``, per the per-process connection model documented in
``references/native-mcp.md``. Each connecting process also lazily spawns its
own ``tools/mcp_death_supervisor.py`` watchdog (see that module and
``tools/mcp_tool.py::_update_death_supervisor``), one per Hermes process, not
one per host.

Seeing several ``mcp_death_supervisor.py --parent-pgid <N>`` processes on a
box therefore is NOT, by itself, evidence of a leak: it is expected whenever
more than one Hermes surface is alive concurrently. The one signal that IS a
genuine leak is a supervisor whose ``--parent-pgid`` process is dead — that
supervisor should have reaped its registered groups and exited on pipe EOF;
if it is still running, something kept it alive past its parent's death.

This module only reads ``/proc`` / process tables (via psutil when available,
otherwise a Linux ``/proc`` fallback) — it never signals, kills, or otherwise
mutates any process. Safe to run against a live production box at any time.
"""

from __future__ import annotations

import os
from typing import Any, Optional

# Reuse the existing, hardened "is this PID alive" check (zombie-aware,
# cross-platform) rather than reimplementing pid-liveness semantics here.
from gateway.status import _pid_exists

_SUPERVISOR_BASENAME = "mcp_death_supervisor.py"
_PARENT_PGID_FLAG = "--parent-pgid"

# Token-set / substring markers used to label which kind of Hermes surface a
# parent process is. Best-effort only: an unrecognized parent still gets a
# full alive/dead + server report, just no role label.
_ROLE_TOKEN_MARKERS: tuple[tuple[frozenset, str], ...] = (
    (frozenset({"gateway", "run"}), "gateway"),
    (frozenset({"dashboard"}), "dashboard"),
)
_ROLE_SUBSTRING_MARKERS: tuple[tuple[str, str], ...] = (
    ("tui_gateway.entry", "tui"),
    ("ui-tui/dist/entry.js", "tui"),
    ("ui-tui" + os.sep + "dist" + os.sep + "entry.js", "tui"),
)

# Substrings identifying the two bundled stdio MCP servers configured on this
# box (see config.yaml `mcp_servers:`). Purely cosmetic labeling; unresolved
# servers still show up with their pgid, just no name.
_SERVER_NAME_SUBSTRING_MARKERS: tuple[tuple[str, str], ...] = (
    ("wikipedia-mcp", "wikipedia"),
    ("mcp-searxng", "searxng"),
)


def _iter_processes():
    """Yield ``(pid, ppid, pgid_or_none, sid_or_none, cmdline_list)`` for every readable process.

    psutil first (handles Windows/macOS too); a Linux ``/proc`` fallback if
    psutil is unavailable, matching the fallback style already used in
    ``tools/mcp_tool_lifecycle.py::_snapshot_child_pids``. ``sid`` (session id) is
    read so descendant-search can stop at a nested session boundary — see
    ``_descendants``.
    """
    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is not None:
        for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
            try:
                pid = proc.info["pid"]
                ppid = proc.info["ppid"]
                cmdline = proc.info["cmdline"] or []
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
            try:
                pgid: Optional[int] = os.getpgid(pid)
            except (ProcessLookupError, PermissionError, OSError):
                pgid = None
            try:
                sid: Optional[int] = os.getsid(pid)
            except (ProcessLookupError, PermissionError, OSError, AttributeError):
                sid = None
            yield pid, ppid, pgid, sid, cmdline
        return

    try:
        candidates = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return
    for pid in candidates:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
                stat = f.read()
            # comm is parenthesized and may itself contain ')'; split on the LAST one so
            # the remaining fields (starting with state) parse regardless of comm content.
            _, _, rest = stat.rpartition(")")
            ppid = int(rest.split()[1])
        except (OSError, ValueError, IndexError):
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read()
            cmdline = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        except OSError:
            cmdline = []
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None
        try:
            sid = os.getsid(pid)
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            sid = None
        yield pid, ppid, pgid, sid, cmdline


def _extract_parent_pgid(cmdline: list) -> Optional[int]:
    """Pull the ``--parent-pgid`` value out of a death-supervisor's argv."""
    for i, part in enumerate(cmdline):
        if part == _PARENT_PGID_FLAG and i + 1 < len(cmdline):
            try:
                return int(cmdline[i + 1])
            except ValueError:
                return None
        if part.startswith(_PARENT_PGID_FLAG + "="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _role_for_cmdline(cmdline: list) -> Optional[str]:
    tokens = frozenset(cmdline)
    for marker, role in _ROLE_TOKEN_MARKERS:
        if marker <= tokens:
            return role
    joined = " ".join(cmdline)
    for marker, role in _ROLE_SUBSTRING_MARKERS:
        if marker in joined:
            return role
    return None


def _server_name_for_cmdline(cmdline: list) -> Optional[str]:
    joined = " ".join(cmdline)
    for marker, name in _SERVER_NAME_SUBSTRING_MARKERS:
        if marker in joined:
            return name
    return None


def list_mcp_subprocess_owners() -> list[dict]:
    """Report every running ``mcp_death_supervisor.py`` and whether its parent is alive.

    Read-only: only inspects the process table (``/proc`` / psutil), never signals
    or kills anything.

    Returns a list of dicts, one per supervisor found::

        {
            "supervisor_pid": int,       # the mcp_death_supervisor.py process itself
            "parent_pid": int,           # value passed as --parent-pgid
            "alive": bool,               # is a process with that pid currently running
            "role": str | None,          # "gateway" / "dashboard" / "tui" / None if unrecognized
            "server_names": [str, ...],  # e.g. ["searxng", "wikipedia"], best-effort
            "server_pgids": [int, ...],  # process groups of the MCP children under this parent
            "leak_signal": bool,         # True only when alive is False (see module docstring)
        }

    A supervisor whose parent is dead (``alive: False``) but is still running IS the
    one genuine orphan/leak signal — it should have reaped its groups and exited on
    pipe EOF. Every other row (``alive: True``) is a legitimate, currently-in-use
    connection owned by a live Hermes process; do not kill its MCP children without
    stopping that process (or issuing /reload-mcp for its profile) first.
    """
    procs = list(_iter_processes())
    cmdline_by_pid = {pid: cmdline for pid, _ppid, _pgid, _sid, cmdline in procs}

    children_by_ppid: dict = {}
    for pid, ppid, pgid, sid, cmdline in procs:
        children_by_ppid.setdefault(ppid, []).append((pid, pgid, sid, cmdline))

    def _descendants(root_pid: int) -> list:
        """All (pid, pgid, cmdline) descending from root_pid, stopping at session boundaries.

        A direct-children-only lookup misses servers spawned by an intermediate
        process (e.g. the TUI's ``tui_gateway.entry`` backend, itself a child of the
        pgid-leader ``ui-tui`` node process recorded as --parent-pgid), so this walks
        the ppid tree instead of just direct children.

        But it must NOT cross into a nested session: the dashboard's "Desktop remote
        gateway" launches TUI sessions as OS children of the dashboard process, yet
        each TUI session (``start_new_session``-style: its own pid == its own sid) is
        a distinct Hermes surface with its own death supervisor. Walking past that
        boundary would attribute the TUI's MCP children to the dashboard too. A child
        is a boundary (included in the results, but not descended into) when its own
        sid equals its own pid; unresolvable sid (permission/timing) is treated the
        same way — safer to under-attribute than to falsely merge two surfaces.
        """
        found: list = []
        stack = list(children_by_ppid.get(root_pid, []))
        visited = {root_pid}
        while stack:
            child_pid, child_pgid, child_sid, child_cmdline = stack.pop()
            if child_pid in visited:
                continue
            visited.add(child_pid)
            found.append((child_pid, child_pgid, child_cmdline))
            is_session_boundary = child_sid is None or child_sid == child_pid
            if not is_session_boundary:
                stack.extend(children_by_ppid.get(child_pid, []))
        return found

    owners: list = []
    for pid, _ppid, _pgid, _sid, cmdline in procs:
        if not any(part.endswith(_SUPERVISOR_BASENAME) for part in cmdline):
            continue
        parent_pid = _extract_parent_pgid(cmdline)
        if parent_pid is None:
            continue

        alive = _pid_exists(parent_pid)
        role = _role_for_cmdline(cmdline_by_pid.get(parent_pid, []))

        server_names: set = set()
        server_pgids: set = set()
        for child_pid, child_pgid, child_cmdline in _descendants(parent_pid):
            if child_pid == pid:
                continue  # the supervisor itself is also a descendant of parent_pid
            name = _server_name_for_cmdline(child_cmdline)
            if name:
                server_names.add(name)
                if child_pgid is not None:
                    server_pgids.add(child_pgid)

        owners.append({
            "supervisor_pid": pid,
            "parent_pid": parent_pid,
            "alive": alive,
            "role": role,
            "server_names": sorted(server_names),
            "server_pgids": sorted(server_pgids),
            "leak_signal": not alive,
        })

    return owners
