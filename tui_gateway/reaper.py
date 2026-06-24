"""Stage 3 — supervisor reaper: orphan + dedup + heartbeat housekeeping.

WHY THIS EXISTS (and what it deliberately does NOT duplicate)
------------------------------------------------------------
The orchestrator (orchestrator.py) already manages ITS OWN gateway+renderer
lifecycle (respawn on exit, bounded budget). The gateway (server.py) already has
_schedule_ws_orphan_reap for INTRA-gateway session teardown. This reaper is a
THIRD, distinct concern at the PROCESS/host layer that neither covers:

  1. ORPHAN debris — renderer/gateway/ws_host processes left behind by a PRIOR
     orchestrator that crashed/was SIGKILLed (its children reparent to init and
     leak; on this host that debris fills swap — documented host problem
     [constrained-host-memory-triage]). poll() can't see them: they were never
     this orchestrator's children.
  2. HEARTBEAT liveness — a renderer that is ALIVE (poll() is None) but FROZEN
     (event loop wedged, not touching its heartbeat file). The orchestrator's
     poll()-based loop never catches this; only a heartbeat-age check does.
  3. DEDUP — more than one live renderer attached to the same gateway (e.g. a
     respawn race left two). Exactly one should survive; the rest are reaped.

DESIGN: the dangerous "what to kill" decision is a PURE function
(plan_reap) over an injected process snapshot + the orchestrator's own known
pids. That makes it fully unit-testable WITHOUT touching real processes. A thin
scan_processes() feeds it from /proc; execute_reap() does the actual kills, and
is the only part with side effects.

SAFETY INVARIANTS (encoded + tested):
  - NEVER reap our own current gateway/renderer/orchestrator pids.
  - NEVER reap a process we cannot positively identify as Hermes-TUI debris
    (must carry our marker env key). No PID-pattern guessing, no pgrep
    self-match [durable-long-run-orchestration footgun].
  - Orphan = carries the marker AND its orchestrator parent is gone AND it is
    not one of the live pids we were told to keep.
"""
from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


# Env key every orchestrator-spawned child carries so the reaper can POSITIVELY
# identify Hermes-TUI debris (never a heuristic PID match). The orchestrator
# stamps this with its OWN pid so a child's owning-orchestrator is self-describing.
MARKER_ENV = "HERMES_TUI_ORCH_OWNER_PID"

# A renderer whose heartbeat file is older than this is considered frozen.
DEFAULT_HEARTBEAT_STALE_S = 90.0


@dataclass(frozen=True)
class ProcInfo:
    """A minimal, snapshot-friendly view of a process. All fields come from a
    /proc read (or are injected in tests). `owner_pid` is the value of
    MARKER_ENV (the orchestrator pid that spawned it), or None if unmarked."""

    pid: int
    owner_pid: Optional[int]
    role: Optional[str]  # "gateway" | "renderer" | None
    heartbeat_age_s: Optional[float] = None  # None when not heartbeat-tracked


@dataclass
class ReapPlan:
    """What execute_reap should kill, with a reason per pid for observability."""

    orphans: list[tuple[int, str]] = field(default_factory=list)
    frozen: list[tuple[int, str]] = field(default_factory=list)
    dupes: list[tuple[int, str]] = field(default_factory=list)

    @property
    def all_pids(self) -> list[int]:
        seen: dict[int, None] = {}
        for pid, _ in [*self.orphans, *self.frozen, *self.dupes]:
            seen.setdefault(pid, None)
        return list(seen)

    def is_empty(self) -> bool:
        return not (self.orphans or self.frozen or self.dupes)


def plan_reap(
    snapshot: Iterable[ProcInfo],
    *,
    my_orchestrator_pid: int,
    live_pids: Iterable[int],
    alive_orchestrator_pids: Iterable[int],
    heartbeat_stale_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> ReapPlan:
    """Pure decision function. Returns a ReapPlan; performs NO side effects.

    Args:
      snapshot: all candidate processes (already filtered to carry MARKER_ENV;
        scan_processes guarantees this, but plan_reap re-checks owner_pid).
      my_orchestrator_pid: this orchestrator's pid — its children are NEVER orphans.
      live_pids: pids this orchestrator currently owns+tracks (its gateway +
        current renderer). NEVER reaped, regardless of heartbeat (the orchestrator
        owns their lifecycle; the reaper must not race it).
      alive_orchestrator_pids: pids of orchestrators currently alive on the host
        (including ours). A child whose owner_pid is NOT in this set is orphaned.
      heartbeat_stale_s: a tracked renderer older than this is "frozen".
    """
    live = set(live_pids)
    alive_orchs = set(alive_orchestrator_pids)
    plan = ReapPlan()

    # Bucket the marked, non-live children by (owner_pid) so dedup can see
    # multiple renderers under the same gateway/owner.
    renderers_by_owner: dict[int, list[ProcInfo]] = {}

    for p in snapshot:
        # Defensive: only ever consider processes that positively self-identify
        # as orchestrator children. owner_pid None ⇒ not ours ⇒ never touched.
        if p.owner_pid is None:
            continue
        # NEVER touch a pid the orchestrator owns/tracks right now.
        if p.pid in live:
            continue
        # NEVER touch our own orchestrator process.
        if p.pid == my_orchestrator_pid:
            continue

        # (1) ORPHAN: its owning orchestrator is gone.
        if p.owner_pid not in alive_orchs:
            plan.orphans.append((p.pid, f"orphan: owner orchestrator {p.owner_pid} gone"))
            continue

        # Owner is alive but this isn't a live tracked pid. It belongs to a
        # DIFFERENT live orchestrator (multi-instance) — not ours to reap on the
        # orphan path. But we still consider it for dedup/frozen ONLY if it's
        # owned by US (my_orchestrator_pid); another orchestrator manages its own.
        if p.owner_pid != my_orchestrator_pid:
            continue

        # (3) collect for dedup (renderers owned by us but not the live one).
        if p.role == "renderer":
            renderers_by_owner.setdefault(p.owner_pid, []).append(p)

        # (2) FROZEN: a tracked renderer with a stale heartbeat.
        if p.role == "renderer" and p.heartbeat_age_s is not None and p.heartbeat_age_s > heartbeat_stale_s:
            plan.frozen.append((p.pid, f"frozen: heartbeat {p.heartbeat_age_s:.0f}s > {heartbeat_stale_s:.0f}s"))

    # (3) DEDUP: if we (somehow) own >1 non-live renderer for the same owner,
    # they're all stale leftovers (the live one is excluded above) — reap them.
    for owner, procs in renderers_by_owner.items():
        if len(procs) >= 1:
            for p in procs:
                # don't double-list a pid already flagged frozen
                if p.pid not in {pid for pid, _ in plan.frozen}:
                    plan.dupes.append((p.pid, f"dupe: extra renderer for owner {owner}"))

    return plan


def execute_reap(
    plan: ReapPlan,
    *,
    kill: Callable[[int, int], None] = os.kill,
    log: Optional[Callable[[str], None]] = None,
) -> list[int]:
    """Apply a ReapPlan with SIGTERM. Returns the pids signalled. `kill` and
    `log` are injectable for tests. Each kill is best-effort (a pid may have
    already exited between snapshot and now — ProcessLookupError is benign)."""
    signalled: list[int] = []
    for pid, reason in [*plan.orphans, *plan.frozen, *plan.dupes]:
        try:
            kill(pid, signal.SIGTERM)
            signalled.append(pid)
            if log:
                log(f"reaped pid={pid} ({reason})")
        except ProcessLookupError:
            pass  # already gone — fine
        except PermissionError:
            if log:
                log(f"reap SKIP pid={pid}: not permitted ({reason})")
    return signalled


# ── Real /proc scanner (the only part that touches the host) ────────────────

def _read_proc_environ(pid: int) -> dict[str, str]:
    """Read /proc/<pid>/environ. Returns {} on any error (process gone, EPERM)."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            raw = fh.read()
    except (OSError, PermissionError):
        return {}
    env: dict[str, str] = {}
    for chunk in raw.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        k, _, v = chunk.partition(b"=")
        env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return env


def _heartbeat_age(env: dict[str, str], now: float) -> Optional[float]:
    """Age of the renderer's heartbeat file, or None if not tracked/missing."""
    hb = env.get("HERMES_TUI_HEARTBEAT_FILE")
    if not hb:
        return None
    try:
        return now - os.stat(hb).st_mtime
    except OSError:
        return None  # missing file ⇒ unknown, NOT auto-frozen (fail safe)


def scan_processes(now: Optional[float] = None) -> list[ProcInfo]:
    """Scan /proc for orchestrator-marked children. Only processes carrying
    MARKER_ENV are returned — everything else on the host is ignored entirely,
    so the reaper can never touch an unrelated process."""
    now = time.time() if now is None else now
    out: list[ProcInfo] = []
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return out
    for pid in pids:
        env = _read_proc_environ(pid)
        owner_raw = env.get(MARKER_ENV)
        if not owner_raw:
            continue  # not an orchestrator child — ignore
        try:
            owner = int(owner_raw)
        except ValueError:
            continue
        role = env.get("HERMES_TUI_ORCH_ROLE")  # "gateway" | "renderer" | None
        out.append(
            ProcInfo(
                pid=pid,
                owner_pid=owner,
                role=role,
                heartbeat_age_s=_heartbeat_age(env, now) if role == "renderer" else None,
            )
        )
    return out


def pid_alive(pid: int) -> bool:
    """True if pid exists (signal 0 probe). Used to compute alive orchestrators."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours to signal
