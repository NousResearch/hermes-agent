"""P-0106: non-fatal delegate_task sequential-deadline handling.

``delegate_task`` is exempt from the sequential-call deadline on main
(``_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS``, agent/tool_executor.py, upstream
#107255) — that exemption is the primary fix. This module is regression
insurance for the class "a deadline applies to ``delegate_task`` again":
under a 420s deadline every real batch "timed out" while its children kept
running on daemon threads, the orchestrator re-dispatched and multiplied
children and spend, and the timeout branch cancelled the worker and
interrupted its thread. When a timeout does fire, the executor consults
this module and treats it as non-fatal while the batch's live transcripts
show recent activity.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tools.delegate_tool")

# Do not treat a delegate_task sequential timeout as fatal unless the batch has
# gone quiet this long on EVERY task log. Covers model API pauses (reasoning
# models routinely pause >120s), provider retries, and long terminal waits;
# per-child hard caps come from delegation.child_timeout_seconds (default off).
# Not config: a timeout policy with one consumer gets no env/config surface.
_DELEGATION_QUIET_SECONDS = 600.0

# Worker thread id -> the batch it dispatched. The worker thread itself
# registers/unregisters; the deadline thread only reads. Bounded naturally:
# one entry per in-flight delegate_task worker, popped in its finally.
_ACTIVE_DELEGATIONS: "dict[int, _DelegationState]" = {}


class _DelegationState:
    """One in-flight delegate_task batch seen from the deadline side."""

    __slots__ = ("delegation_id", "dir_path")

    def __init__(self, delegation_id: str, dir_path: Optional[Path]) -> None:
        self.delegation_id = delegation_id
        self.dir_path = dir_path

    def task_log_paths(self) -> list:
        """task-<n>.log paths for this batch, sorted for stable output."""
        if self.dir_path is None:
            return []
        try:
            return sorted(
                p for p in self.dir_path.iterdir()
                if p.name.startswith("task-") and p.name.endswith(".log")
            )
        except OSError:
            return []

    def most_recent_log_activity_ns(self) -> Optional[int]:
        """Newest wall-clock mtime (ns) across the batch's task logs, or None
        when the live side channel is unavailable (creation failed, dir
        pruned). None means "cannot prove liveness" -> fatal, not assumed-alive."""
        stamps = []
        for log in self.task_log_paths():
            try:
                stamps.append(log.stat().st_mtime_ns)
            except OSError:
                continue
        return max(stamps) if stamps else None

    def recently_active(self, quiet_seconds: float = _DELEGATION_QUIET_SECONDS) -> bool:
        """True when at least one task log changed within ``quiet_seconds``."""
        latest = self.most_recent_log_activity_ns()
        if latest is None:
            return False
        return (time.time_ns() - latest) <= quiet_seconds * 1_000_000_000


def register_delegation(delegation_id: Optional[str], live_dir: Optional[str]) -> None:
    """Record the batch the calling worker thread dispatched (live-transcript dir).

    Self-keyed on the caller's thread id; paired with :func:`unregister_delegation`
    in the same call frame for sync batches."""
    if not delegation_id:
        return
    dir_path = Path(live_dir) if live_dir else None
    _ACTIVE_DELEGATIONS[threading.get_ident()] = _DelegationState(delegation_id, dir_path)


def unregister_delegation(worker_tid: int) -> None:
    """Drop the worker's entry when delegate_task returns (any outcome)."""
    _ACTIVE_DELEGATIONS.pop(worker_tid, None)


def active_delegation_id(worker_tid: int) -> Optional[str]:
    """The batch id registered for this worker tid, or None (unknown/finished)."""
    state = _ACTIVE_DELEGATIONS.get(worker_tid)
    return state.delegation_id if state is not None else None


def maybe_defer_sequential_timeout(worker_tid: int) -> bool:
    """True when a delegate_task sequential timeout should NOT be fatal: the
    worker's batch still shows recent live-transcript activity, so children are
    progressing and must not be re-dispatched or torn down."""
    state = _ACTIVE_DELEGATIONS.get(worker_tid)
    if state is None:
        return False
    try:
        return state.recently_active()
    except Exception:
        logger.debug(
            "delegate_task timeout guard failed for batch %s", state.delegation_id, exc_info=True
        )
        return False


def timeout_notice(elapsed_s: float, delegation_id: Optional[str]) -> str:
    """What the model sees instead of a flat timeout error when the batch is alive."""
    live = f"cache/delegation/live/{delegation_id}/ (task-N.log)" if delegation_id and delegation_id != "?" \
        else "cache/delegation/live/<delegation_id>/ (task-N.log)"
    return (
        f"delegate_task hit the sequential tool deadline after {elapsed_s:.0f}s, but the batch's "
        f"children are still active (live transcripts updated recently). The batch was NOT "
        f"re-dispatched and its children keep running. Do NOT re-delegate these tasks; poll "
        f"instead:\n"
        f"1. Read the live transcripts under the Hermes home: {live}\n"
        f"2. Or call delegate_task with action='list' for ids/goals/status."
    )
