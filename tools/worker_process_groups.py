"""Reclaim the process groups a worker opened, so a dead worker cannot keep running its work.

Why this exists
---------------
Every worker-spawned child that needs its own session (the terminal tool's bash, the
execute_code kernel, a docker-less background command) is created with
``start_new_session=True`` — it leaves the worker's process group on purpose, so a
terminal command's own timeout can tear down that command's whole tree without
signalling the worker. The price is that the worker's death does not reach those
groups: ``_die_with_owner`` (agent/managed_worker.py) can only kill the group the
worker *leads*, and a SIGKILLed or crashed worker runs no code at all.

The observed failure, 2026-11-03: one blocked card was retried while its previous
attempt's child was still decoding frames. Two copies of a frame-decoding pipeline
ran side by side (18.8 GB + 11.0 GB RSS), filled ``hermes-gateway.service`` past
MemoryHigh, and the starved owner could no longer answer a worker's claim RPC — every
new card was recorded as ``Kanban owner refused: timed out``. Killing the two
processes by hand took the cgroup from 28.4 GB to 2.2 GB and cards spawned again.

How it works
------------
A worker records each group it opens in a small JSON ledger named after its own pid
(``$HERMES_PROCESS_GROUPS_DIR/worker-<pid>.json``, default
``<HERMES_HOME>/run/process-groups``). Three paths then reclaim it:

1. **normal exit** — ``install()`` registers ``atexit`` in the worker;
2. **crash / SIGTERM / SIGHUP** — ``install()`` registers a signal handler that
   reclaims and then re-raises the default disposition;
3. **SIGKILL (unreachable from inside)** — the owner calls ``reclaim_worker(pid)``
   once its child is reaped (``gateway.session_managed_worker.ManagedWorker.close``),
   and ``hermes kanban orphans`` sweeps anything that still escaped.

Reclaim is TERM → grace → KILL over the group, guarded by process identity, and it
never touches a group recorded with ``persist=True`` (a background process the user
asked to outlive the turn).

POSIX only. On Windows every entry point is a no-op: ``start_new_session`` does not
exist there and the terminal tool already kills process trees with ``taskkill /T``.
"""

from __future__ import annotations

import argparse
import atexit
import errno
import json
import os
import signal
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Iterable, Optional

# Opt out with HERMES_PROCESS_GROUPS=0 (a worker supervisor restart needs no code change).
GROUP_ENV = "HERMES_PROCESS_GROUPS"
DIR_ENV = "HERMES_PROCESS_GROUPS_DIR"
# A worker killed with SIGKILL runs no handler, so its ledger outlives it; the owner's
# reclaim and the CLI sweep are what remove these.
LEDGER_PREFIX = "worker-"
LEDGER_SUFFIX = ".json"
LOCK_SUFFIX = ".lock"
# TERM→KILL grace. Long enough for a decoder to flush its current frame, short enough
# that a retried card is never running beside its own predecessor.
GRACE_SECONDS = 10.0
# Ctrl-C in an interactive session is impatient; the terminal tool's own timeout teardown
# is what normally ends these children, this is only the session-exit backstop.
SIGINT_GRACE_SECONDS = 3.0
CLK_TCK = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0

_INSTALLED = False


# --------------------------------------------------------------------------- helpers

def enabled() -> bool:
    """Is per-worker group reclamation active in this process?"""
    if os.name != "posix":  # windows-footgun: ok — POSIX-only mechanism (see module docstring)
        return False
    return os.environ.get(GROUP_ENV, "1").strip().lower() not in {"0", "off", "false", "no"}


def ledger_dir() -> Path:
    """Directory holding one ledger per worker pid."""
    override = os.environ.get(DIR_ENV)
    if override:
        return Path(override)
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:
        home = Path(os.path.expanduser("~/.hermes"))
    return home / "run" / "process-groups"


def ledger_path(worker_pid: Optional[int] = None) -> Path:
    return ledger_dir() / f"{LEDGER_PREFIX}{int(worker_pid or os.getpid())}{LEDGER_SUFFIX}"


def _ledger_dirs() -> list[Path]:
    """Every ledger directory this host may hold: the active one plus sibling profiles'.

    The owner and its worker must agree on the directory, so the owner passes DIR_ENV at
    spawn. A sweep, however, runs from whichever profile asked and must still see the
    ledgers the gateway wrote — hence the union (root home first, then per-profile homes).
    """
    dirs = [ledger_dir()]
    home = Path(os.path.expanduser("~/.hermes"))
    roots = [home] + sorted(p for p in (home / "profiles").glob("*") if p.is_dir())
    for root in roots:
        dirs.append(root / "run" / "process-groups")
    seen, out = set(), []
    for path in dirs:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            out.append(path)
    return out


def _read_stat(pid: int) -> Optional[dict]:
    """/proc/<pid>/stat as a dict, or None when the process is gone/unreadable.

    ``comm`` (field 2) may contain spaces and parentheses, so fields resume after the
    LAST ``)``. Fields are 1-based in proc(5); ``fields[0]`` here is field 3 (state).
    """
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    try:
        rest = data[data.rindex(b")") + 2:].split()
        return {
            "pid": int(pid),
            "state": rest[0].decode("ascii", "replace"),
            "ppid": int(rest[1]),
            "pgrp": int(rest[2]),
            "start": float(rest[19]),
        }
    except (ValueError, IndexError):
        return None


def _all_stats() -> dict[int, dict]:
    """Every readable /proc/<pid>/stat, keyed by pid (one readdir, one read each)."""
    out: dict[int, dict] = {}
    with suppress(OSError):
        for name in os.listdir("/proc"):
            if name.isdigit():
                stat = _read_stat(int(name))
                if stat is not None:
                    out[stat["pid"]] = stat
    return out


def _group_members(pgid: int, stats: Optional[dict[int, dict]] = None) -> list[int]:
    stats = _all_stats() if stats is None else stats
    return [pid for pid, stat in stats.items() if stat["pgrp"] == pgid and stat["state"] != "Z"]


def _cwd(pid: int) -> Optional[str]:
    try:
        return os.readlink(f"/proc/{int(pid)}/cwd")
    except OSError:
        return None


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as fh:
            raw = fh.read(4096)
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def _rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{int(pid)}/statm", "r") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _uptime() -> float:
    try:
        with open("/proc/uptime") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def _age_seconds(stat: dict) -> float:
    """Process age from the kernel's monotonic start tick, not the wall clock."""
    return max(0.0, _uptime() - stat["start"] / CLK_TCK)


def _under_roots(path: Optional[str], roots: Iterable[str]) -> bool:
    """Is *path* inside one of *roots*?

    Lexical comparison only. ``/proc/<pid>/cwd`` is a readlink of the task's real dentry, so it is
    already free of symlinks, and ``_roots_for`` canonicalises every root. Resolving the candidate
    again would only add one realpath() walk per process on the host — which is also exactly what
    makes a sweep expensive on a busy farm.
    """
    if not path:
        return False
    candidate = os.path.normpath(path)
    for root in roots:
        if not root:
            continue
        base = os.path.normpath(os.path.expanduser(str(root)))
        if candidate == base or candidate.startswith(base.rstrip(os.sep) + os.sep):
            return True
    return False


def _default_roots() -> list[str]:
    """Where a leaked worker child's cwd lands: the card workspace, the scratch tree."""
    roots = [os.environ.get(key) for key in ("HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_WORKSPACES_ROOT")]
    roots = [root for root in roots if root]
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:
        home = Path(os.path.expanduser("~/.hermes"))
    roots += [str(home / "kanban" / "workspaces"), str(home / "cache" / "scratch")]
    return roots


def _roots_for(extra: Optional[Iterable[str]] = None) -> list[str]:
    """Deduplicated union of the caller's roots and this profile's defaults."""
    seen, out = set(), []
    for root in list(extra or []) + _default_roots():
        if not root:
            continue
        real = os.path.realpath(str(root))
        if real not in seen:
            seen.add(real)
            out.append(str(root))
    return out


# --------------------------------------------------------------------------- ledger io

@contextmanager
def _locked(path: Path):
    """Exclusive advisory lock beside *path* (threads and sibling processes both)."""
    lock_path = path.with_name(path.name + LOCK_SUFFIX)
    with suppress(OSError):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    try:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        with suppress(Exception):
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _load(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("version") == 1 else None


def _store(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------- registering groups

def note_group(leader_pid: int, pgid: Optional[int] = None, *, label: str = "",
               persist: bool = False, roots: Optional[Iterable[str]] = None) -> bool:
    """Record a child process group this worker just opened.

    *persist* marks a process the caller deliberately asked to outlive the turn
    (``persist_on_release``): reclaim skips it. Never raises — a registry failure must
    not break the tool call that spawned the child.
    """
    if not enabled():
        return False
    try:
        worker_pid = os.getpid()
        stat = _read_stat(int(leader_pid))
        if stat is None:
            return False
        group = int(pgid if pgid is not None else stat["pgrp"])
        if group <= 1 or group == os.getpgrp():
            # Our own group: the caller did not detach the child; _die_with_owner covers it,
            # and signalling here would take the worker down with it.
            return False
        path = ledger_path(worker_pid)
        roots = _roots_for(roots)
        with _locked(path):
            data = _load(path) or {
                "version": 1,
                "worker": {"pid": worker_pid, "birth": (_read_stat(worker_pid) or {}).get("start"),
                           "started": time.time(), "profile": os.environ.get("HERMES_PROFILE", ""),
                           "roots": roots},
                "groups": [],
            }
            recorded = [str(r) for r in ((data.get("worker") or {}).get("roots") or [])]
            missing = [r for r in roots if os.path.realpath(r) not in {os.path.realpath(x) for x in recorded}]
            if missing:
                data.setdefault("worker", {})["roots"] = recorded + missing
            groups = [g for g in data.get("groups", []) if isinstance(g, dict)]
            if not any(int(g.get("pgid", -1)) == group for g in groups):
                groups.append({"pgid": group, "leader": int(leader_pid), "leader_birth": stat["start"],
                               "label": (label or "")[:200], "persist": bool(persist),
                               "registered": time.time()})
                data["groups"] = groups
                _store(path, data)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------ reclaiming

def _group_is_ours(record: dict, roots: Iterable[str], stats: dict[int, dict]) -> bool:
    """Is *record*'s group still the one we opened, and not a recycled pid/pgid?

    A live leader must still be the same incarnation. Once the leader is gone the group
    can only be recognised by where its surviving members run: every candidate must sit
    under a root this worker's ledger owns. That check is what keeps a recycled pgid from
    reaching an unrelated process — and what still reaches the orphan in the incident,
    whose leader was itself and whose cwd was the card's scratch dir.
    """
    pgid = int(record["pgid"])
    leader = int(record.get("leader", pgid))
    leader_stat = stats.get(leader)
    if leader_stat is not None and leader_stat["pgrp"] == pgid:
        return abs(leader_stat["start"] - float(record.get("leader_birth", -1))) < 1.0
    members = _group_members(pgid, stats)
    if not members:
        return False
    return all(_under_roots(_cwd(pid), roots) for pid in members)


def _terminate_group(record: dict, roots: Iterable[str], *, grace: float, kill: bool) -> bool:
    """TERM the group (KILL only when *kill*), wait out the grace, then KILL survivors."""
    pgid = int(record["pgid"])
    if pgid <= 1 or pgid == os.getpgrp() or not _group_is_ours(record, roots, _all_stats()):
        return False
    signalled = False
    for sig, budget in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0 if kill else 0.0)):
        if not kill and sig == signal.SIGKILL:
            break
        try:
            os.killpg(pgid, sig)  # windows-footgun: ok — POSIX only (enabled() gate)
            signalled = True
        except (ProcessLookupError, PermissionError):
            break
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            if not _group_members(pgid):
                return True
            time.sleep(0.05)
        if not kill:
            return signalled
    return signalled and not _group_members(pgid)


def reclaim_ledger(path: Path, *, grace: float = GRACE_SECONDS, force: bool = False,
                   reason: str = "") -> dict:
    """Reclaim every group *path* still lists; drop the groups that are gone.

    Returns ``{"ledger": str, "reclaimed": [pgid...], "skipped": [pgid...]}``. The ledger
    file is removed when nothing reclaimable remains — a worker that left live groups keeps
    it so the CLI sweep can still find them.
    """
    summary = {"ledger": str(path), "reclaimed": [], "skipped": [], "reason": reason}
    data = _load(path)
    if data is None:
        with suppress(OSError):
            path.unlink()  # unreadable/corrupt ledger for a dead worker: nothing to honour
        return summary
    roots = [str(r) for r in (data.get("worker", {}).get("roots") or _default_roots())]
    remaining, stats = [], _all_stats()
    for record in data.get("groups", []):
        if not isinstance(record, dict) or "pgid" not in record:
            continue
        pgid = int(record["pgid"])
        if record.get("persist") and not force:
            # Asked to outlive the turn; only an explicit sweep (force) may take it.
            remaining.append(record)
            summary["skipped"].append(pgid)
            continue
        if not _group_members(pgid, stats):
            continue  # already gone: drop the record
        if not _group_is_ours(record, roots, stats):
            remaining.append(record)
            summary["skipped"].append(pgid)
            continue
        if _terminate_group(record, roots, grace=grace, kill=True):
            summary["reclaimed"].append(pgid)
        else:
            remaining.append(record)
            summary["skipped"].append(pgid)
    if remaining:
        data["groups"] = remaining
        with suppress(Exception):
            with _locked(path):
                _store(path, data)
    else:
        with suppress(OSError):
            path.unlink()
        with suppress(OSError):
            path.with_name(path.name + LOCK_SUFFIX).unlink()
    return summary


def reclaim_worker(worker_pid: int, *, grace: float = GRACE_SECONDS, force: bool = False,
                   reason: str = "") -> dict:
    """Owner side: reclaim the groups a (now dead) worker opened."""
    path = ledger_path(worker_pid)
    if not path.exists():
        return {"ledger": str(path), "reclaimed": [], "skipped": [], "reason": reason or "no_ledger"}
    return reclaim_ledger(path, grace=grace, force=force, reason=reason or f"worker {worker_pid} exited")


# --------------------------------------------------------------------- worker install

def _reclaim_own(*, grace: float, reason: str) -> None:
    with suppress(Exception):
        reclaim_ledger(ledger_path(os.getpid()), grace=grace, reason=reason)


def _signal_handler(signum: int, frame, previous):
    """Reclaim, then hand the signal back to whoever owned the disposition before us.

    A worker killed by SIGTERM must still leave no children: Python's default SIGTERM
    disposition exits without running ``atexit`` handlers, so the reclaim has to happen
    here. The previous disposition is restored and re-raised so an existing handler (the
    CLI's agent interrupt, a test's stop_event) still sees the signal.
    """
    grace = SIGINT_GRACE_SECONDS if signum == signal.SIGINT else GRACE_SECONDS
    _reclaim_own(grace=grace, reason=f"signal {signum}")
    with suppress(Exception):
        signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
    try:
        if previous in (signal.SIG_IGN,):
            return
        if callable(previous):
            previous(signum, frame)
            return
        os.kill(os.getpid(), signum)  # default disposition: exit like the signal asked
    except Exception:
        os._exit(128 + int(signum))


def install(roots: Optional[Iterable[str]] = None) -> bool:
    """Arm exit-time reclaim in this process (the worker side of the contract).

    Idempotent; returns True when this call armed it. ``roots`` (workspace / scratch
    trees) are recorded in the ledger so a reclaim that outlives the leader can still tell
    our descendants from a stranger's.
    """
    global _INSTALLED
    if _INSTALLED or not enabled():
        return False
    _INSTALLED = True
    os.environ.setdefault(DIR_ENV, str(ledger_dir()))
    roots = _roots_for(roots)
    # Touch the ledger early: an owner that loses this worker before it spawns anything
    # then finds a ledger naming it (birth included), not silence.
    with suppress(Exception):
        path = ledger_path()
        if not path.exists():
            with _locked(path):
                _store(path, {"version": 1, "groups": [], "worker": {
                    "pid": os.getpid(), "birth": (_read_stat(os.getpid()) or {}).get("start"),
                    "started": time.time(), "profile": os.environ.get("HERMES_PROFILE", ""),
                    "roots": roots}})
    atexit.register(lambda: _reclaim_own(grace=GRACE_SECONDS, reason="worker exit"))
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        with suppress(ValueError, OSError):  # not the main thread: atexit still covers us
            previous = signal.getsignal(sig)
            signal.signal(sig, lambda signum, frame, _prev=previous: _signal_handler(signum, frame, _prev))
    return True


# ------------------------------------------------------------------------ orphan scan

def scan_orphans(roots: Optional[Iterable[str]] = None, *, min_age: float = 60.0,
                 include_persisted_ledgers: bool = True) -> list[dict]:
    """Reparented processes still running in a workspace/scratch tree.

    ``PPID=1`` is the signature of a leaked worker child: its spawner exited and init
    adopted it. Only processes whose cwd sits under one of *roots* are reported, so a
    legitimate init-owned daemon is never listed. Also returns ledgers whose worker pid is
    gone while groups remain (``source: "ledger"``) — the SIGKILL case the scan cannot see
    by parentage alone.
    """
    roots = [str(r) for r in (roots or _default_roots())]
    stats = _all_stats()
    me = os.getpid()
    rows: list[dict] = []
    for pid, stat in stats.items():
        if pid == me or stat["state"] == "Z":
            continue
        cwd = _cwd(pid)
        if not _under_roots(cwd, roots):
            # A ledger-recorded group member whose cwd we may not read still counts as ours.
            if not _owned_by_ledger(stat["pgrp"]):
                continue
        age = _age_seconds(stat)
        if age < min_age:
            continue
        if stat["ppid"] == 1:
            source = "ppid1"
        elif _owned_by_live_worker(pid, stats):
            continue  # a worker is running this right now: not a leak
        else:
            source = "unowned"
        rows.append({"source": source, "pid": pid, "ppid": stat["ppid"], "pgid": stat["pgrp"],
                     "leader_birth": stat["start"],
                     "age_seconds": round(age, 1), "rss_bytes": _rss_bytes(pid), "cwd": cwd,
                     "cmdline": _cmdline(pid)[:300]})
    if include_persisted_ledgers:
        reported = {row["pid"] for row in rows}
        for ledger_file in sorted(path for directory in _ledger_dirs() for path in directory.glob(f"{LEDGER_PREFIX}*{LEDGER_SUFFIX}")):
            data = _load(ledger_file) or {}
            worker = data.get("worker") or {}
            worker_pid = int(worker.get("pid") or 0)
            if worker_pid and _read_stat(worker_pid) is not None:
                continue  # its owner is still alive and will reclaim on its own exit
            for record in data.get("groups", []):
                pgid = int(record.get("pgid", 0))
                if not pgid or not _group_members(pgid, stats):
                    continue
                provenance = {"ledger": str(ledger_file), "label": record.get("label", ""),
                              "persist": bool(record.get("persist"))}
                if pgid in reported:
                    # The cwd scan already saw a member of this group (as "unowned"): the ledger is
                    # the stronger evidence of whose it is, so upgrade the row instead of dropping
                    # the provenance the operator needs to judge a kill.
                    for row in rows:
                        if row["pgid"] == pgid:
                            row["source"] = "ledger"
                            row.update(provenance)
                            break
                    continue
                reported.add(pgid)
                rows.append({"source": "ledger", "pid": pgid, "ppid": None, "pgid": pgid,
                             "age_seconds": round(max((_age_seconds(s) for p, s in stats.items()
                                                       if s["pgrp"] == pgid), default=0.0), 1),
                             "rss_bytes": sum(_rss_bytes(p) for p, s in stats.items() if s["pgrp"] == pgid),
                             "cwd": _cwd(pgid), "cmdline": _cmdline(pgid)[:300], **provenance})
    return rows


def _owned_by_ledger(pgid: int) -> bool:
    for directory in _ledger_dirs():
        for path in directory.glob(f"{LEDGER_PREFIX}*{LEDGER_SUFFIX}"):
            data = _load(path) or {}
            if any(int(g.get("pgid", -1)) == pgid for g in data.get("groups", []) if isinstance(g, dict)):
                return True
    return False


WORKER_MARKERS = ("kanban_worker_client", "agent.managed_worker")


def _ancestors(pid: int, stats: dict[int, dict]) -> list[int]:
    """The ppid chain above *pid* (bounded, cycle-safe)."""
    chain, seen, current = [], {pid}, stats.get(pid)
    while current is not None and len(chain) < 64:
        parent = int(current["ppid"])
        if parent <= 0 or parent in seen:
            break
        chain.append(parent)
        seen.add(parent)
        current = stats.get(parent)
    return chain


def _owned_by_live_worker(pid: int, stats: dict[int, dict]) -> bool:
    """Is a live kanban worker *pid* itself, or an ancestor of it?

    A command a worker is running right now has the worker in its ancestry. A leak from a
    worker that died does not (its ancestors are init or a subreaper). This is what separates
    "the card is working" from "the card was retried and this is the previous attempt".

    The process itself counts: a running worker is long-lived with a workspace cwd, and calling
    it an orphan would put every live card on the report.
    """
    if any(marker in _cmdline(pid) for marker in WORKER_MARKERS):
        return True
    for ancestor in _ancestors(pid, stats):
        cmdline = _cmdline(ancestor)
        if any(marker in cmdline for marker in WORKER_MARKERS):
            return True
    return False


def sweep(rows: Optional[list[dict]] = None, *, roots: Optional[Iterable[str]] = None,
          grace: float = GRACE_SECONDS, include_unowned: bool = False) -> dict:
    """Kill what ``scan_orphans`` found. Returns ``{"killed": [...], "kept": [...]}``.

    ``source: "unowned"`` rows are only signalled when *include_unowned* is set: their cwd
    is under a workspace, but nothing proves a dead worker owned them (a hand-run command in
    the same directory looks identical), so the default sweep leaves them alone.
    """
    rows = scan_orphans(roots) if rows is None else rows
    killed, kept = [], []
    for row in rows:
        if row.get("persist"):
            kept.append(row)
            continue
        if row.get("source") == "unowned" and not include_unowned:
            kept.append(row)
            continue
        if row.get("source") == "ledger" and row.get("ledger"):
            summary = reclaim_ledger(Path(row["ledger"]), grace=grace, reason="orphan sweep")
            (killed if summary["reclaimed"] or not summary["skipped"] else kept).append(row)
            continue
        record = {"pgid": row["pgid"], "leader": row["pid"], "leader_birth": row.get("leader_birth", -1)}
        if _terminate_group(record, roots or _default_roots(), grace=grace, kill=True):
            killed.append(row)
        else:
            kept.append(row)
    return {"killed": killed, "kept": kept}


# ------------------------------------------------------------------------------ cli

def _format_bytes(value: int) -> str:
    size = float(value or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}TiB"


def _format_age(seconds: float) -> str:
    seconds = int(seconds or 0)
    if seconds >= 86400:
        return f"{seconds // 86400}d{seconds % 86400 // 3600}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60}s"
    return f"{seconds}s"


def main(argv: Optional[list[str]] = None) -> int:
    """``hermes kanban orphans`` — report (default) or reclaim leaked worker children."""
    parser = argparse.ArgumentParser(
        prog="hermes kanban orphans",
        description="List (default) or reclaim processes a dead worker left running in a "
                    "workspace/scratch tree. Reports PPID=1 processes whose cwd is under a "
                    "root, plus groups named by the ledger of a worker whose pid is gone.")
    parser.add_argument("--kill", action="store_true",
                        help="Reclaim what is listed (SIGTERM the group, SIGKILL after the grace). "
                             "Without this flag nothing is signalled.")
    parser.add_argument("--min-age", type=float, default=60.0, metavar="SECONDS",
                        help="Only report processes at least this old (default: 60). A live "
                             "foreground command is not an orphan.")
    parser.add_argument("--root", action="append", default=None, metavar="DIR",
                        help="Workspace/scratch root to scan (repeatable; default: this "
                             "profile's kanban workspaces + cache/scratch).")
    parser.add_argument("--grace", type=float, default=GRACE_SECONDS, metavar="SECONDS",
                        help=f"TERM→KILL grace for --kill (default: {GRACE_SECONDS:g}).")
    parser.add_argument("--include-unowned", action="store_true",
                        help="Also reclaim rows marked 'unowned' (cwd under a root but no proof a "
                             "dead worker owned them). Off by default: a hand-run command there "
                             "looks the same.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args(argv)

    rows = scan_orphans(args.root, min_age=args.min_age)
    if args.kill:
        result = sweep(rows, roots=args.root, grace=args.grace, include_unowned=args.include_unowned)
        rows = result["killed"]
        if not args.json:
            for row in rows:
                print(f"reclaimed pid {row['pid']} (pgid {row['pgid']}, {_format_age(row['age_seconds'])}) "
                      f"{row.get('cmdline', '')[:120]}")
            for row in result["kept"]:
                print(f"kept      pid {row['pid']} (pgid {row['pgid']}) — still alive or persisted")
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif not args.kill:
        if not rows:
            print("no leaked worker processes")
        else:
            print(f"{'PID':>8}  {'PGID':>8}  {'AGE':>7}  {'RSS':>9}  SOURCE  CWD / COMMAND")
            for row in rows:
                print(f"{row['pid']:>8}  {row['pgid']:>8}  {_format_age(row['age_seconds']):>7}  "
                      f"{_format_bytes(row['rss_bytes']):>9}  {row['source']:<6}  "
                      f"{(row.get('cwd') or '?')} :: {(row.get('cmdline') or '')[:100]}")
            print(f"\n{len(rows)} process(es). Nothing was signalled — rerun with --kill to reclaim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())