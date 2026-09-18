"""Read-only Desktop "Agent Inbox" aggregation.

``inbox.list`` returns, for the ACTIVE connection + profile only, a bounded, deny-listed
list of sessions that currently need the operator or carry persisted automation state.

Authoritative, non-fabricated sources:
  - persisted automation (goal/loop/heartbeat) via the SAME snapshots ``session.control.read``
    returns (``_snapshot_control``), so sessions that are NOT open are still covered — the
    aggregation never resumes a session or hydrates a transcript (no side effects).
  - live pending approval from the gateway's in-process queue, redacted on egress.
  - live pending clarify for OPEN sessions (in-memory on this gateway process).
Coverage is declared outright: this is an ACTIVE-profile view, never a global inbox.

This module is READ-ONLY. Opening or dismissing anything never resolves an approval or a
clarify prompt; resolution stays in the owning session's UI through the existing RPCs.
"""

from __future__ import annotations

import logging
import os
import time

from .method_ctx import HandlerRegistry, bind_module
from hermes_state_sessions import INTERNAL_LISTING_SOURCES

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

logger = logging.getLogger(__name__)

# Same deny-list as the session sidebar (INTERNAL_LISTING_SOURCES): sub-agent runs,
# kanban workers and one-shot runs are not human-facing inbox items. Import the
# canonical tuple — a hand-rolled copy here silently overrode the sidebar's set,
# because every method module's top-level names land in the shared server namespace.
_INBOX_DENY_SOURCES = frozenset(INTERNAL_LISTING_SOURCES)

_DEFAULT_LIMIT = 200
_MAX_LIMIT = 1000

# Valid categories for the interactive inbox left navigation.
VALID_CATEGORIES = frozenset({"goals", "loops", "heartbeats", "subagents", "background_tasks", "other"})


def _inbox_denied_source(row) -> bool:
    return (row.get("source") or "").strip().lower() in _INBOX_DENY_SOURCES


# ── pure lane classification ──────────────────────────────────────────────────
def classify_lanes(
    control: dict | None, pending_approval: bool, pending_clarify: bool, now: float | None = None
) -> list[str]:
    """Ordered, deduplicated lanes for one session from its allowed control snapshot.

    Honest rules — no inference from turn counts:
      needs_you  - a pending approval or clarify prompt for this session.
      running    - an ACTIVE goal with no wait barrier, or a loop awaiting its response.
      waiting    - a paused goal / wait barrier, or a loop deferred by a goal or paused.
      scheduled  - an active heartbeat, or an active loop not awaiting a response whose
                   next due time is in the future.
    ``control`` is the ``session.control.read`` snapshot shape; absent state returns [].
    """
    lanes: list[str] = []
    if pending_approval or pending_clarify:
        lanes.append("needs_you")
    control = control or {}
    goal = control.get("goal")
    loop = control.get("loop")
    heartbeat = control.get("heartbeat")
    if now is None:
        now = time.time()

    if bool(goal and goal.get("status") == "active" and not goal.get("wait_barrier")) or bool(
        loop and loop.get("status") == "active" and loop.get("awaiting_response")
    ):
        lanes.append("running")
    if bool(goal and (goal.get("status") == "paused" or goal.get("wait_barrier"))) or bool(
        loop and (loop.get("status") == "paused" or loop.get("deferred_by_goal"))
    ):
        lanes.append("waiting")
    if bool(heartbeat and heartbeat.get("status") == "active") or bool(
        loop
        and loop.get("status") == "active"
        and not loop.get("awaiting_response")
        and (loop.get("next_due_at") or 0) > now
    ):
        lanes.append("scheduled")
    return lanes


def _count_lanes(items: list[dict]) -> dict:
    counts = {lane: 0 for lane in ("needs_you", "running", "waiting", "scheduled", "total")}
    for item in items:
        counts["total"] += 1
        for lane in item.get("lanes", []):
            if lane in counts:
                counts[lane] += 1
    return counts


def classify_categories(control: dict | None) -> list[str]:
    """Categories for the left navigation from a control snapshot.

    Returns ALL matching categories (overlapping membership).  A session with
    both a goal and a loop appears in both ``goals`` and ``loops``.
    Sessions with no recognized automation state return ``["other"]``.
    Completed/done automation types are included when persisted state exists,
    with separate activity state indicated by the goal/loop/heartbeat status.
    """
    control = control or {}
    goal = control.get("goal")
    loop = control.get("loop")
    heartbeat = control.get("heartbeat")
    cats: list[str] = []
    if goal and goal.get("status") in ("active", "paused", "done"):
        cats.append("goals")
    if loop and loop.get("status") in ("active", "paused", "done"):
        cats.append("loops")
    if heartbeat and heartbeat.get("status") in ("active", "paused", "done"):
        cats.append("heartbeats")
    if not cats:
        cats.append("other")
    return cats


def _count_categories(items: list[dict]) -> dict:
    counts = {cat: 0 for cat in ("goals", "loops", "heartbeats", "subagents", "background_tasks", "other")}
    for item in items:
        for cat in item.get("categories", []):
            if cat in counts:
                counts[cat] += 1
    return counts


def _subagent_counts_by_owner() -> tuple[dict[str, int], str | None]:
    """session_key → active subagent count from the live registry.

    Uses ``list_active_subagents()`` which returns a snapshot-safe copy of all
    running children across all sessions. Only the count is surfaced — never the
    goal text or transcript content.

    Returns ``(counts, error)`` where *error* is ``None`` on success or a safe
    error message on failure — never ``{}`` misinterpreted as "zero subagents".
    """
    from tools.delegate_tool_registry import list_active_subagents

    try:
        records = list_active_subagents()
    except Exception as exc:
        return {}, f"subagent enumeration failed: {_safe_error_message(exc)}"
    counts: dict[str, int] = {}
    for r in records:
        owner_key = str(r.get("owner_agent_session_id") or "")
        if owner_key:
            counts[owner_key] = counts.get(owner_key, 0) + 1
    return counts, None


def _background_task_counts_by_session(session_keys: list[str]) -> tuple[dict[str, int], str | None]:
    """session_key → running background-process count from the process registry.

    Uses ``process_registry.list_sessions(session_key=key)`` for each bounded
    allowed key to respect the PUBLIC scoped API.  Background processes are
    terminal processes spawned with ``background=true`` (NOT subagents — those
    are tracked separately via ``delegate_tool_registry``).  Only the count is
    surfaced — never command text or output content.

    Returns ``(counts, error)`` where *error* is ``None`` on success or a safe
    error message on failure — never ``{}`` misinterpreted as "zero processes".
    """
    from tools.process_registry import process_registry

    counts: dict[str, int] = {}
    try:
        for key in session_keys:
            if not key:
                continue
            try:
                sessions = process_registry.list_sessions(session_key=key)
            except Exception as exc:
                return counts, f"bg-process query failed for {_safe_error_message(exc)}"
            running = sum(1 for s in sessions if s.get("status") == "running")
            if running > 0:
                counts[key] = running
    except Exception as exc:
        return counts, f"bg-process enumeration failed: {_safe_error_message(exc)}"
    return counts, None


def badge_state(items: list[dict], errors: list = ()) -> str:
    """Amber when anything needs the operator; red on any data-source failure/error state
    (so an unsupported/error view is never labelled all-clear); muted otherwise."""
    if errors:
        return "red"
    if any("needs_you" in item.get("lanes", []) for item in items):
        return "amber"
    return "none"


# ── live sources ──────────────────────────────────────────────────────────────
def _live_clarify_by_session_key(profile_home: str | None) -> tuple[dict[str, int], list[str]]:
    """session_key → pending clarify count for OPEN sessions owned by this profile.

    Uses the public ``server_requests.open_requests(sid)`` reader (NOT the private
    ``_open`` dict) for pending clarify detection.  Only the count is surfaced — never
    the question text — so credential-shaped clarify payloads are never egressed.

    Returns (counts, errors) so callers can surface enumeration/query failures as
    coverage errors and a red badge instead of silently swallowing them.
    """
    out: dict[str, int] = {}
    errors: list[str] = []
    try:
        with _sessions_lock:
            snapshot = list(_sessions.items())
    except Exception as exc:
        logger.debug("inbox live-session enumeration failed", exc_info=True)
        errors.append(f"live-session enumeration failed: {_safe_error_message(exc)}")
        return out, errors
    from tui_gateway import server_requests

    # When profile_home is None (launch profile), match sessions whose profile_home
    # is the launch profile's home (_hermes_home). When it's a foreign profile,
    # match the explicit path.  normcase ensures Windows drive-letter and casing
    # differences don't cause a false mismatch.
    want_home = os.path.normcase(str(profile_home) if profile_home is not None else str(_hermes_home))
    for sid, record in snapshot:
        if not isinstance(record, dict):
            continue
        if os.path.normcase(str(record.get("profile_home") or "")) != want_home:
            continue
        key = str(record.get("session_key") or "")
        if not key:
            continue
        try:
            reqs = server_requests.open_requests(sid)
        except Exception as exc:
            errors.append(f"{key}: clarify query failed: {_safe_error_message(exc)}")
            continue
        clarify_count = sum(1 for r in reqs if r.get("method") == "clarify")
        if clarify_count > 0:
            out[key] = clarify_count
    return out, errors


def _safe_error_message(exc: Exception) -> str:
    """Sanitize an exception for RPC egress: safe code only, never raw internal text."""
    return f"{type(exc).__name__}"


def _list_inbox(rid, params: dict) -> dict:
    profile = (params.get("profile") or "").strip() or None
    try:
        profile_home_raw = _profile_home(profile)
        profile_home = str(profile_home_raw) if profile_home_raw is not None else None
    except Exception as exc:
        # ProfileUnavailableError propagates to the outer handler which turns it into
        # JSON-RPC 4064. Any other resolution failure is also fatal: silently falling
        # back to the launch profile would return data from the wrong user.
        from tui_gateway.server import ProfileUnavailableError
        if isinstance(exc, ProfileUnavailableError):
            raise
        return _err(rid, 5031, f"profile resolution failed: {_safe_error_message(exc)}")
    try:
        limit = int(params.get("limit", _DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    cap = max(1, min(limit, _MAX_LIMIT))

    # Fetch cap+1 rows so truncation can be detected from the raw row count
    # before deny-list filtering reduces the scan.  The extra row is never processed.
    fetch_limit = min(cap + 1, _MAX_LIMIT + 1)
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5031)
        try:
            rows = _listing_rows(db, fetch_limit)
        except Exception as exc:
            logger.warning("inbox.list scan failed: %s", exc)
            return _err(rid, 5031, "inbox.list scan failed")

    clarify_by_key, clarify_errors = _live_clarify_by_session_key(profile_home)
    subagent_by_key, subagent_error = _subagent_counts_by_owner()

    # Collect session keys from the allowed rows for bounded bg-process queries
    session_keys = []
    for row in rows[:cap]:
        if _inbox_denied_source(row):
            continue
        key = str(row.get("id") or "")
        if key:
            session_keys.append(key)
    bg_task_by_key, bg_task_error = _background_task_counts_by_session(session_keys)

    items: list[dict] = []
    errors: list[str] = []
    if subagent_error:
        errors.append(subagent_error)
    if bg_task_error:
        errors.append(bg_task_error)
    scanned = 0
    for row in rows[:cap]:
        if _inbox_denied_source(row):
            continue
        key = str(row.get("id") or "")
        if not key:
            continue
        scanned += 1
        try:
            control = _snapshot_control(key)
        except Exception as exc:
            errors.append(f"{key}: snapshot read failed: {_safe_error_message(exc)}")
            control = {}
        approval = None
        try:
            pending = _pending_approval_request_payload(key, strict=True)
            if isinstance(pending, dict):
                # Metadata-only approval — never copy arbitrary description text
                # that could contain credential-shaped values or raw exception text.
                approval = {
                    "count": 1,
                    "description": "pending approval",
                    "command_redacted": True,
                }
        except Exception as exc:
            errors.append(f"{key}: approval read failed: {_safe_error_message(exc)}")
        clarify_count = clarify_by_key.get(key, 0)
        clarify = {"count": clarify_count} if clarify_count > 0 else None
        lanes = classify_lanes(control, approval is not None, clarify is not None)
        cats = classify_categories(control)
        subagent_count = subagent_by_key.get(key, 0)
        bg_task_count = bg_task_by_key.get(key, 0)
        # Sessions with active subagents appear in the subagents category (overlapping)
        if subagent_count > 0 and "subagents" not in cats:
            cats.append("subagents")
        # Sessions with background processes appear in the background_tasks category
        if bg_task_count > 0 and "background_tasks" not in cats:
            cats.append("background_tasks")
        items.append({
            "session_key": key,
            "title": str(row.get("title") or ""),
            "source": str(row.get("source") or ""),
            "cwd": str(row.get("cwd") or ""),
            "lanes": lanes,
            "goal": control.get("goal"),
            "loop": control.get("loop"),
            "heartbeat": control.get("heartbeat"),
            "pending_approval": approval,
            "pending_clarify": clarify,
            "categories": cats,
            "subagent_count": subagent_count,
            "subagent_count_unavailable": subagent_error is not None,
            "background_task_count": bg_task_count,
            "background_task_count_unavailable": bg_task_error is not None,
        })

    # Honest truncation: we fetched cap+1 rows; if we got more than cap raw rows
    # (before deny filtering), the result is truncated regardless of how many survived filtering.
    # scanned_sessions is capped at cap to honestly reflect how many were fully processed.
    raw_truncated = len(rows) > cap
    errors.extend(clarify_errors)
    coverage = {
        "profile": profile or _response_profile_name(profile),
        "connection_scope": "active connection and profile only",
        "scanned_sessions": min(scanned, cap),
        "partial": raw_truncated,
        "approval_scope": "live gateway approval queue",
        "clarify_scope": "live open sessions only",
        "errors": errors[:20],
    }
    return _ok(rid, {
        "inbox": {
            "coverage": coverage,
            "items": items,
            "counts": _count_lanes(items),
            "categories": _count_categories(items),
            "badge": badge_state(items, errors),
        }
    })


@method("inbox.list")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Read-only cross-session inbox aggregation for the active profile."""
    from tui_gateway.server import ProfileUnavailableError
    try:
        return _list_inbox(rid, params)
    except ProfileUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed, never fabricate a partial badge
        logger.debug("inbox.list failed: %s", exc, exc_info=True)
        return _err(rid, 5031, "inbox.list failed")


def register(server) -> None:
    """Rebind this module's handlers onto the server namespace."""
    bind_module(globals(), server, skip=("_",))
