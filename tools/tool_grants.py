"""Tool grant engine: single source of grant/revoke logic for the tool broker.

Mirrors ``tools.skill_grants`` exactly, but over tool names instead of skill
names. Used by the agent-facing ``tool_request`` tool, the runtime
toolset-scope fence (``agent.tool_executor``), and the kanban task-completion
hook. All grant state lives in the central append-only profile activity
ledger as ``tool.borrowed`` / ``tool.revoked`` events; this module is the only
place that interprets them.

A grant is temporary: it is revoked when its linked kanban task completes
(``revoke_grants_for_task``) or, for grants with no live task, after a TTL
(``sweep_expired_grants``).
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from hermes_cli.profile_activity_ledger import append_event, query_events

logger = logging.getLogger(__name__)

EV_BORROW = "tool.borrowed"
EV_DENY = "tool.denied"
EV_REVOKE = "tool.revoked"
GRANT_SOURCE = "tool.broker"

# Default cap on temporary borrows of one tool by one profile per 30 days
# before the broker stops auto-granting and escalates to Denji for a permanent
# decision.
FREQUENCY_LIMIT = 6
# Grants with no live/identifiable completed task expire after this many hours.
DEFAULT_TTL_HOURS = 24

# Curated, never-grant tool NAMES. Tools are more dangerous than skills (they
# act directly rather than just informing), so this list is conservative: it
# covers profile-lifecycle/governance mutators, skill management, and arbitrary
# command execution. These require KENSEI/Denji, never the temporary broker.
# terminal/process are ALSO covered via NEVER_GRANT_TOOLSETS, but are listed
# here as a hardcoded floor so a toolsets import/resolution failure can never
# shrink the deny set and let arbitrary command execution through.
NEVER_GRANT_TOOLS = {
    "kanban_profile_edit",
    "kanban_profile_rollback",
    "kanban_request_subprofile",
    "kanban_request_human_approval",
    "skill_manage",
    "terminal",
    "process",
}

# Toolsets that are too dangerous to grant tool-by-tool via the temporary
# broker: every tool resolved from these toolsets is added to the effective
# deny set. "terminal" grants arbitrary command execution. This is additive
# defence-in-depth over the hardcoded floor above (so a future tool added to
# the terminal toolset is denied without needing a manual NEVER_GRANT_TOOLS
# update). Only toolsets that actually exist in toolsets.py are resolved; a
# name with no matching toolset is silently skipped (not an error).
NEVER_GRANT_TOOLSETS = {
    "terminal",
}

# WS-7 style Denji loop closure: governance profiles are allowed certain
# normally-denied tools — these are Denji's own tools. The check below still
# denies non-governance profiles. Mirrors skill_grants.GOVERNANCE_GRANT_WHITELIST;
# left empty unless/until a concrete need is identified.
GOVERNANCE_GRANT_WHITELIST: dict[str, set] = {}


def _now() -> int:
    return int(time.time())


def _new_event_id(prefix: str = "borrow") -> str:
    """Readable, collision-proof id: <prefix>-YYYYMMDD-<rand>.

    The random suffix alone guarantees uniqueness, so we avoid scanning the
    whole ledger to compute a sequence number (that was an unbounded query on
    the grant hot path)."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}-{today}-{uuid.uuid4().hex[:8]}"


def _effective_never_grant() -> set:
    """Union of NEVER_GRANT_TOOLS and every tool name resolved from
    NEVER_GRANT_TOOLSETS via ``toolsets.resolve_toolset``.

    Computed lazily so an import error degrades safely: on failure this falls
    back to the curated NEVER_GRANT_TOOLS set alone (fail closed on at least
    that set, rather than raising).  Resolution failures are logged at
    WARNING (not DEBUG) so a silently unresolvable toolset is visible in
    normal logs; the hardcoded floor above remains the primary deny set."""
    deny = set(NEVER_GRANT_TOOLS)
    try:
        from toolsets import resolve_toolset

        for toolset_name in NEVER_GRANT_TOOLSETS:
            try:
                deny.update(resolve_toolset(toolset_name) or [])
            except Exception as e:  # noqa: BLE001
                logger.warning("resolve_toolset(%s) failed: %s", toolset_name, e)
    except Exception as e:  # noqa: BLE001
        logger.warning("_effective_never_grant: toolsets import failed: %s", e)
    return deny


def revoked_borrow_ids(*, since: Optional[int] = None) -> set:
    """event_ids of borrows that have a matching revoke.

    When ``since`` is provided (epoch seconds), only revokes on or after
    that timestamp are scanned; bounds the hot-path query so it does not
    grow with the total historical ledger size.
    """
    return {
        (e.get("payload") or {}).get("borrow_event_id")
        for e in query_events(event_types=[EV_REVOKE], since=since)
    }


def has_active_grant(
    profile: Optional[str], tool: str, *, since: Optional[int] = None
) -> bool:
    """True if ``profile`` holds a live (unrevoked) grant for ``tool``.

    Bounded query contract: when ``since`` is None the lookback defaults
    to ``_GRANT_SCAN_WINDOW_SECONDS`` (7 days) so the hot-path scan does
    not grow with total historical ledger size.  Because a revoke can only
    occur *after* its borrow, the same ``since`` bound applied to the
    revoke query safely covers every revoke for every borrow within the
    window — a revoke cannot fall outside the window when its borrow is
    inside it.  Grants older than the window are assumed to have been
    swept by the TTL fallback.

    Fails CLOSED: any ledger error returns False so a broken ledger never
    silently unblocks access under enforcement."""
    if not profile:
        return False
    try:
        if since is None:
            since = int(time.time() - _GRANT_SCAN_WINDOW_SECONDS)
        borrows = query_events(
            event_types=[EV_BORROW], target_profile=profile, object_id=tool,
            since=since,
        )
        if not borrows:
            return False
        revoked = {
            (e.get("payload") or {}).get("borrow_event_id")
            for e in query_events(
                event_types=[EV_REVOKE], target_profile=profile, object_id=tool,
                since=since,
            )
        }
        return any(b.get("event_id") not in revoked for b in borrows)
    except Exception as e:  # noqa: BLE001
        logger.debug("has_active_grant failed for %s/%s: %s", profile, tool, e)
        return False


def borrow_count(profile: str, tool: str, days: int = 30) -> int:
    since = int(time.time() - days * 86400)
    return len(query_events(event_types=[EV_BORROW], target_profile=profile, object_id=tool, since=since))


def record_deny(profile: str, tool: str, task_id: str, reason: str, board: str = "ops") -> dict:
    eid = _new_event_id("deny")
    append_event(
        source=GRANT_SOURCE, event_type=EV_DENY, event_id=eid,
        actor_profile="tool-broker", target_profile=profile,
        object_type="tool", object_id=tool, board=board,
        summary=f"Denied {tool} for {profile}: {reason}",
        payload={"task_id": task_id, "task_result": "denied", "recommendation": f"denied: {reason}"},
    )
    return {"decision": "deny", "granted": False, "event_id": eid, "reason": reason}


def grant_tool(profile: str, tool: str, task_id: str, reason: str = "", board: str = "ops") -> dict:
    """Evaluate and (if safe) record a temporary task-scoped grant.

    Safety gates, in order: a non-empty task_id (grants are task-scoped, so a
    grant with no task could only ever be cleaned up by the TTL sweep), the
    tool must exist in the registry, the effective NEVER_GRANT deny set
    (curated tool names plus every tool resolved from NEVER_GRANT_TOOLSETS),
    then a per-tool per-profile frequency cap (escalate to Denji rather than
    keep auto-granting). Returns a decision dict; never raises on the happy
    path."""
    if not task_id:
        return {"decision": "deny", "granted": False, "event_id": None,
                "reason": "a task_id is required; grants are scoped to a task"}

    # Normalise so casing/whitespace variants cannot slip past the deny set on
    # a registry that ever accepts them.
    tool = (tool or "").strip()
    if not tool:
        return {"decision": "deny", "granted": False, "event_id": None,
                "reason": "a tool name is required"}

    try:
        from tools.registry import discover_builtin_tools, registry as _registry

        # Builtin tools self-register on import; ensure that has happened so
        # a cold-started broker doesn't deny tools it just hasn't seen yet.
        discover_builtin_tools()
        if _registry.get_entry(tool) is None:
            return record_deny(profile, tool, task_id,
                               f"'{tool}' is not a registered tool", board)
    except Exception as e:  # noqa: BLE001
        # Warn (not debug): a systemic registry failure denies every request,
        # which should be visible in normal logs rather than only at debug.
        logger.warning("tool registry lookup failed for %s: %s", tool, e)
        return record_deny(profile, tool, task_id,
                           f"could not verify '{tool}' exists in the registry", board)

    deny_set = _effective_never_grant()
    if tool in deny_set:
        allowed = GOVERNANCE_GRANT_WHITELIST.get(profile, set())
        if tool not in allowed:
            return record_deny(profile, tool, task_id,
                               f"'{tool}' is on the NEVER_GRANT list (requires KENSEI/Denji)", board)

    freq = borrow_count(profile, tool)
    if freq >= FREQUENCY_LIMIT:
        return record_deny(profile, tool, task_id,
                           f"frequency limit: {freq} borrows in 30 days; escalate to Denji for permanent enable",
                           board)
    eid = _new_event_id()
    append_event(
        source=GRANT_SOURCE, event_type=EV_BORROW, event_id=eid,
        actor_profile="tool-broker", target_profile=profile,
        object_type="tool", object_id=tool, board=board,
        summary=f"{profile} borrowed {tool} for {task_id}",
        payload={"task_id": task_id, "grant_type": "temporary",
                 "grant_expiry": "task_completion", "reason": reason, "granted_at": _now()},
    )
    result = {"decision": "grant_task_only", "granted": True, "event_id": eid, "reason": reason}
    if freq >= FREQUENCY_LIMIT - 3:
        result["warning"] = f"{freq + 1} borrows this month — nearing the permanent-enable threshold"
    return result


def _revoke(borrow: dict, result: str, reason: str) -> None:
    eid = borrow.get("event_id")
    # Single revocation encoding (mirrors skill_grants G-2 fix): the payload
    # carries the borrow event_id as the canonical link; the revoke gets its
    # own independent event_id (not derived from the borrow's) to avoid dual
    # encoding.
    revoke_eid = _new_event_id("revoke")
    append_event(
        source=GRANT_SOURCE, event_type=EV_REVOKE, event_id=revoke_eid,
        actor_profile="tool-broker", target_profile=borrow.get("target_profile"),
        object_type="tool", object_id=borrow.get("object_id"), board=borrow.get("board"),
        summary=f"Revoked {borrow.get('object_id')} grant {eid} ({reason})",
        payload={"borrow_event_id": eid, "task_result": result, "revoke_reason": reason, "revoked_at": _now()},
    )


def revoke_by_event_id(event_id: str, result: str = "completed") -> dict:
    """Revoke a single grant by its borrow event_id (used by the broker CLI)."""
    borrows = {e.get("event_id"): e for e in query_events(event_types=[EV_BORROW])}
    borrow = borrows.get(event_id)
    if borrow is None:
        return {"error": f"borrow event_id {event_id} not found"}
    if event_id in revoked_borrow_ids():
        return {"error": f"event_id {event_id} already revoked"}
    _revoke(borrow, result, "manual")
    return {"event_id": event_id, "status": "revoked"}


# Hot-path window for grant scans: grants are task-scoped and revoked on
# completion, so a grant older than this many seconds is either already
# revoked or swept by the TTL.  Bounds the ledger scan so it does not
# grow with total historical events.
_GRANT_SCAN_WINDOW_SECONDS = 7 * 86400  # 7 days


def revoke_grants_for_task(task_id: str, result: str = "completed") -> int:
    """Revoke every live grant linked to ``task_id``. Returns the count revoked.
    Best-effort: never raises (called from the kanban completion path)."""
    try:
        since = int(time.time() - _GRANT_SCAN_WINDOW_SECONDS)
        revoked = revoked_borrow_ids(since=since)
        live = [
            b for b in query_events(event_types=[EV_BORROW], since=since)
            if (b.get("payload") or {}).get("task_id") == task_id and b.get("event_id") not in revoked
        ]
        for b in live:
            _revoke(b, result, "task_completed")
        return len(live)
    except Exception:  # noqa: BLE001
        return 0


def sweep_expired_grants(ttl_hours: int = DEFAULT_TTL_HOURS) -> int:
    """Revoke live grants older than the TTL (fallback for grants whose task
    never reached a completion event). Returns the count revoked. Best-effort:
    never raises, consistent with the other engine entry points."""
    try:
        cutoff = int(time.time() - ttl_hours * 3600)
        since = int(time.time() - _GRANT_SCAN_WINDOW_SECONDS)
        revoked = revoked_borrow_ids(since=since)
        expired = [
            b for b in query_events(event_types=[EV_BORROW], since=since)
            if b.get("event_id") not in revoked and (b.get("occurred_at") or 0) < cutoff
        ]
        for b in expired:
            _revoke(b, "expired", f"ttl_{ttl_hours}h")
        return len(expired)
    except Exception as e:  # noqa: BLE001
        logger.debug("sweep_expired_grants failed: %s", e)
        return 0
