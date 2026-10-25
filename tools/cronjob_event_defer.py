"""Bounded deferral queue for event-triggered cron fires that lose the at-most-once claim.

``cron_job`` webhook routes fire ONE existing job per incoming event, and each fire holds the
``fire_claim`` CAS for the whole agent run (minutes). A burst — one GitHub PR open fans out to
pull_request + check_suite + issue_comment within seconds — therefore collides with the claim:
every event landing in the in-flight window is refused with "Job is already being fired by the
scheduler" and, without this queue, was discarded (a 2026-10 live gateway dropped PR-open
events exactly this way). Event fires are queueable by nature — the job re-reads state each
run — so the refused context is parked here and one merged follow-up run replays the whole
batch once the in-flight run releases the claim.

In-memory by design: the queue lives exactly as long as the gateway process that owns the
route, each deferral is logged, and the bound (``MAX_PENDING_PER_JOB`` × ``MAX_CONTEXT_CHARS``)
caps what a storm can park. Durable replay across a gateway restart is out of scope — the
in-flight run the events were waiting on dies with the same process anyway.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_PENDING_PER_JOB = 50
MAX_CONTEXT_CHARS = 6000
MAX_MERGED_CHARS = 48000

_HEADER_ONLY_NOTE = "(body omitted: merged replay hit its size cap; the newest events keep theirs)"
_MERGE_HEADER = (
    "Deferred webhook events replayed in one run: {count} event(s) arrived while a previous "
    "run held the fire claim. Per-event contexts follow, oldest first; each keeps a header, "
    "and beyond the size cap older bodies are omitted in favour of newer ones.")


@dataclass
class DeferredEvent:
    """One parked event fire: the rendered per-run context plus routing diagnostics."""

    context: str
    event_type: str = ""
    delivery_id: str = ""


_pending: dict[str, deque[DeferredEvent]] = {}
# The first deferrer after an idle queue (or after the previous replayer stood down) owns the
# replay; election and stand-down share the lock with enqueue so no deferrer is ever orphaned.
_replay_owner_alive: dict[str, bool] = {}
_lock = threading.Lock()


def defer_event(job_id: str, context: str, *, event_type: str = "",
                delivery_id: str = "") -> tuple[int, bool]:
    """Park one event context for ``job_id``; returns ``(depth_after, became_replay_owner)``.

    Drop-oldest at ``MAX_PENDING_PER_JOB``: a storm must evict stale heads, never grow the
    queue without bound. ``became_replay_owner`` is True only for the caller whose enqueue
    found no live replayer — that caller starts the drain loop; every other caller returns
    immediately with its event already in the batch.
    """
    with _lock:
        queue = _pending.setdefault(job_id, deque())
        if len(queue) >= MAX_PENDING_PER_JOB:
            dropped = queue.popleft()
            logger.warning(
                "[cron-event] defer queue for job %s at capacity (%d); dropping oldest "
                "event (type=%s, delivery=%s)", job_id, MAX_PENDING_PER_JOB,
                dropped.event_type or "unknown", dropped.delivery_id or "-")
        queue.append(DeferredEvent(str(context)[:MAX_CONTEXT_CHARS], event_type, delivery_id))
        owner = not _replay_owner_alive.get(job_id, False)
        if owner:
            _replay_owner_alive[job_id] = True
        depth = len(queue)
    logger.info("[cron-event] deferred job=%s depth=%d%s", job_id, depth,
                " — this caller owns the replay" if owner else "")
    return depth, owner


def take_pending(job_id: str) -> list[DeferredEvent]:
    """Pop the whole batch for ``job_id`` (caller replays or re-queues it)."""
    with _lock:
        queue = _pending.get(job_id)
        if not queue:
            return []
        events = list(queue)
        queue.clear()
        return events


def push_front(job_id: str, events: list[DeferredEvent]) -> None:
    """Re-queue a batch a replay attempt could not fire yet, oldest first.

    New events deferred meanwhile sit BEHIND the re-queued batch: one merged replay should
    carry all of them, and arrival order within a merged batch is informational only."""
    if not events:
        return
    with _lock:
        queue = _pending.setdefault(job_id, deque())
        overflow = len(events) + len(queue) - MAX_PENDING_PER_JOB
        for _ in range(max(0, overflow)):
            if not queue:
                break
            queue.pop()  # evict the newest of the OLD batch first — the pushed batch replays now
        queue.extendleft(reversed(events))


def pending_count(job_id: str) -> int:
    with _lock:
        return len(_pending.get(job_id, ()))


def finish_or_continue(job_id: str) -> bool:
    """Replayer exit gate: True when new events arrived while the last fire ran (keep
    replaying); False after atomically standing down, so a deferrer landing after this
    point elects itself instead of trusting a dead owner."""
    with _lock:
        if _pending.get(job_id):
            return True
        _replay_owner_alive[job_id] = False
        return False


def mark_replayer_stood_down(job_id: str) -> None:
    """Release the owner slot on ANY replayer exit (crash, give-up) so the next deferrer
    can elect itself; idempotent with ``finish_or_continue``'s stand-down."""
    with _lock:
        _replay_owner_alive[job_id] = False


def clear(job_id: str | None = None) -> None:
    """Test/maintenance hook: drop one job's batch, or everything."""
    with _lock:
        if job_id is None:
            _pending.clear()
            _replay_owner_alive.clear()
        else:
            _pending.pop(job_id, None)
            _replay_owner_alive.pop(job_id, None)


def merge_contexts(events: list[DeferredEvent]) -> str:
    """Render a batch as one bounded ``extra_prompt``: a header with the count, then a
    labelled section per event. Bodies are allocated NEWEST-first so the cap squeezes the
    stalest heads (a merge gate mostly needs the latest event plus awareness of the rest)."""
    if not events:
        return ""
    if len(events) == 1:
        return events[0].context

    header = _MERGE_HEADER.format(count=len(events))
    labels = []
    for i, ev in enumerate(events):
        tag = f"type={ev.event_type or 'unknown'}"
        if ev.delivery_id:
            tag += f", delivery={ev.delivery_id}"
        labels.append(f"--- deferred event {i + 1}/{len(events)} ({tag}) ---")

    budget = max(0, MAX_MERGED_CHARS - len(header) - sum(len(l) for l in labels)
                 - len(_HEADER_ONLY_NOTE))
    bodies: list[str | None] = [None] * len(events)
    for i in range(len(events) - 1, -1, -1):  # newest first
        body = events[i].context[:MAX_CONTEXT_CHARS]
        if len(body) <= budget:
            bodies[i] = body
            budget -= len(body)
        elif budget > 0:
            bodies[i] = body[:budget]
            budget = 0
        if budget <= 0:
            break

    parts = [header]
    for label, body in zip(labels, bodies):
        parts.append(label)
        parts.append(body if body is not None else _HEADER_ONLY_NOTE)
    return "\n\n".join(parts)
