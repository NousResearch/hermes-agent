"""Token-free workload counts, not a machine/process utilization monitor."""
import sys
import time

from hermes_constants import hermes_home_key
from tui_gateway.usage_telemetry import envelope

MAX_RECORDS = 1024


def category(count, source, reason=None):
    return {"supported": True, "status": "known" if count is not None else "unknown",
            "count": count, "source": source, "reason": reason}


def load_level(categories):
    known = [c["count"] for c in categories.values() if c["status"] == "known"]
    total = sum(known) if known else None
    level = "unknown" if total is None else "heavy" if total >= 4 else "busy" if total else "calm"
    return {"level": level, "known_count": total, "basis": "known_category_counts",
            "thresholds": {"busy": 1, "heavy": 4}}


def _subagent_count(owners):
    registry = sys.modules.get("tools.delegate_tool_registry")
    if registry is None or owners is None or not registry._active_subagents_lock.acquire(blocking=False):
        return None
    try:
        records = registry._active_subagents
        if len(records) > MAX_RECORDS:
            return None
        # Exact live record identity, not public ids (which may be reused).
        return sum(id(r.get("owner_session_record")) in owners for r in records.values())
    finally:
        registry._active_subagents_lock.release()


def _cron_count(home):
    scheduler = sys.modules.get("cron.scheduler")
    if scheduler is None or not scheduler._running_lock.acquire(blocking=False):
        return None
    try:
        fires = scheduler._running_fire_owners
        # Legacy claim-only jobs carry no profile identity. Do not guess their home.
        if len(fires) > MAX_RECORDS or scheduler._running_job_ids - fires.keys():
            return None
        count = scanned = 0
        for executions in fires.values():
            scanned += len(executions)
            if scanned > MAX_RECORDS:
                return None
            count += sum(hermes_home_key(profile) == hermes_home_key(home)
                         for _, profile in executions.values())
        return count
    finally:
        scheduler._running_lock.release()


def active_work(home, sessions, sessions_lock):
    as_of_us = time.time_ns() // 1000
    counts = {
        "agent_turns": category(None, "tui_gateway_sessions", "snapshot_unavailable"),
        "subagents": category(None, "delegate_registry", "registry_not_observed"),
        "cron_executions": category(None, "cron_scheduler", "registry_not_observed"),
        "queued_work": category(None, "tui_gateway_prompt_queue", "snapshot_unavailable"),
    }
    owners = None
    if sessions_lock.acquire(timeout=0.05):
        try:
            records = tuple(sessions.values()) if len(sessions) <= MAX_RECORDS else None
        finally:
            sessions_lock.release()
        if records is not None:
            owners = set()
            running = queued = 0
            readable = True
            for session in records:
                if hermes_home_key(session.get("profile_home") or home) != hermes_home_key(home):
                    continue
                owners.add(id(session))
                lock = session.get("history_lock")
                if lock is None or not lock.acquire(blocking=False):
                    readable = False
                    break
                try:
                    if not isinstance(session.get("running"), bool):
                        readable = False
                        break
                    running += int(session["running"])
                    queued += int(session.get("queued_prompt") is not None) + len(session.get("queued_prompts") or [])
                finally:
                    lock.release()
            if readable:
                counts["agent_turns"] = category(running, "tui_gateway_sessions")
                counts["queued_work"] = category(queued, "tui_gateway_prompt_queue")
            if not readable:
                owners = None
    counts["subagents"] = category(_subagent_count(owners), "owned_tui_delegations", "other_parents_not_observed")
    counts["cron_executions"] = category(_cron_count(home), "profile_scoped_inprocess_cron", "other_processes_not_observed")
    return {**envelope(home, as_of_us), "categories": counts, "load": load_level(counts),
            "coverage": {"status": "partial", "scope": "gateway_process",
                         "reason": "other_processes_and_unobserved_categories_excluded"},
            "freshness": {"status": "fresh", "stale_after_seconds": 10}}
