"""Desktop "Agent Inbox" aggregation.

``inbox.list`` returns, for the ACTIVE connection + profile only, a bounded, deny-listed
list of sessions that currently need the operator or carry persisted automation state.

Authoritative, non-fabricated sources:
  - persisted automation (goal/loop/heartbeat) via the SAME snapshots ``session.control.read``
    returns (``_snapshot_control``), so sessions that are NOT open are still covered — the
    aggregation never resumes a session or hydrates a transcript (no side effects).
  - live pending approval from the gateway's in-process queue, redacted on egress.
  - live pending clarify for OPEN sessions (in-memory on this gateway process).
  - durably recorded EXPIRED requests (timed out / withdrawn without an answer), so a
    prompt that died on its own is still visible afterwards — what it was for, that it
    expired, and a Redo — instead of vanishing silently.
Coverage is declared outright: this is an ACTIVE-profile view, never a global inbox.

Request resolution stays in the owning session's UI through the existing RPCs; the only
writes that originate here are the inbox's own expired-request records (and their
dismissal), never an approval/clarify decision.
"""

from __future__ import annotations

import json
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
    control: dict | None, pending_approval: bool, pending_clarify: bool, now: float | None = None,
    pending_expired: bool = False,
) -> list[str]:
    """Ordered, deduplicated lanes for one session from its allowed control snapshot.

    Honest rules — no inference from turn counts:
      needs_you  - a pending approval or clarify prompt for this session, or an expired
                   request still awaiting the operator's redo/dismiss decision.
      running    - an ACTIVE goal with no wait barrier, or a loop awaiting its response.
      waiting    - a paused goal / wait barrier, or a loop deferred by a goal or paused.
      scheduled  - an active heartbeat, or an active loop not awaiting a response whose
                   next due time is in the future.
    ``control`` is the ``session.control.read`` snapshot shape; absent state returns [].
    """
    lanes: list[str] = []
    if pending_approval or pending_clarify or pending_expired:
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


# ── expired requests (durable) ────────────────────────────────────────────────
# A request that ends without an answer (timeout, withdrawal, session teardown) used to
# disappear completely: no trace of what it was for and no way to re-raise it. Each one is
# recorded in the owning session's store (state_meta — survives the turn, the session close
# and an app restart), pruned to a bounded window, and rendered by the panel with a Redo.
_EXPIRED_PREFIX = "inbox.expired."
_EXPIRED_KEEP_PER_SESSION = 10
_EXPIRED_MAX_AGE_S = 7 * 24 * 3600


def _expired_record_key(session_key: str, request_id: str) -> str:
    return f"{_EXPIRED_PREFIX}{session_key}.{request_id}"


def _parse_expired_record(raw) -> dict | None:
    try:
        entry = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(entry, dict) or not entry.get("request_id"):
        return None
    return entry


def record_expired_request(db, session_key: str, payload: dict, outcome: str) -> bool:
    """Persist one request that ended without an answer; True when written.

    Only what the panel needs to say *what it was for* is kept, each field bounded. The
    command text is whatever the surface already redacted for display, so a
    credential-shaped value never reaches this store.
    """
    if db is None or not session_key:
        return False
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        return False
    entry = {
        "request_id": request_id,
        "session_key": session_key,
        "kind": "approval",
        "command": str(payload.get("command") or "")[:500],
        "description": str(payload.get("description") or "")[:300],
        "pattern_keys": [str(k) for k in (payload.get("pattern_keys") or [])][:8],
        "ended_at": time.time(),
        "outcome": str(outcome or "unknown")[:40],
    }
    try:
        db.set_meta(_expired_record_key(session_key, request_id), json.dumps(entry))
        _prune_expired_requests(db, session_key)
    except Exception:
        logger.warning("failed to record expired request %s", request_id, exc_info=True)
        return False
    return True


def _prune_expired_requests(db, session_key: str) -> None:
    """Keep the newest N per session and drop anything past the age window."""
    cutoff = time.time() - _EXPIRED_MAX_AGE_S
    for index, entry in enumerate(load_expired_requests(db, session_key)):  # newest first
        too_old = float(entry.get("ended_at") or 0) < cutoff
        if too_old or index >= _EXPIRED_KEEP_PER_SESSION:
            db.delete_meta(_expired_record_key(session_key, str(entry.get("request_id") or "")))


def load_expired_requests(db, session_key: str) -> list[dict]:
    """Expired requests for one session, newest first."""
    if db is None or not session_key:
        return []
    try:
        rows = db.list_meta_prefix(f"{_EXPIRED_PREFIX}{session_key}.")
    except Exception:
        logger.debug("expired-request read failed", exc_info=True)
        return []
    entries = [entry for _key, raw in rows if (entry := _parse_expired_record(raw)) is not None]
    entries.sort(key=lambda e: float(e.get("ended_at") or 0), reverse=True)
    return entries


def load_expired_request_counts(db) -> dict[str, int]:
    """session_key → expired-request count for every session, from ONE prefix scan."""
    counts: dict[str, int] = {}
    try:
        rows = db.list_meta_prefix(_EXPIRED_PREFIX)
    except Exception:
        logger.debug("expired-request scan failed", exc_info=True)
        return counts
    for _key, raw in rows:
        entry = _parse_expired_record(raw)
        if entry is None:
            continue
        key = str(entry.get("session_key") or "")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def clear_expired_request(db, session_key: str, request_id: str) -> bool:
    """Drop one expired-request record (Dismiss). False when it was not there."""
    if db is None or not session_key or not request_id:
        return False
    key = _expired_record_key(session_key, request_id)
    if db.get_meta(key) is None:
        return False
    db.delete_meta(key)
    return True


# ── live sources ──────────────────────────────────────────────────────────────
def _inbox_home_key(home) -> str:
    """Normalized comparison key for "which profile home owns this runtime session".

    A session created under the launch profile stores ``profile_home = None``
    (``server._add_session``: "None = launch"), and ``_profile_home()`` returns None for
    that same profile — so comparing the raw values against a home path never matched,
    and every launch-profile session vanished from the live joins. The panel then listed
    a pending approval under Needs attention while the detail read returned nothing to
    act on (live, 2026-09-18). Both sides resolve through here; a foreign profile still
    matches only its own normalized path.
    """
    return os.path.normcase(str(home) if home is not None else str(_hermes_home))


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

    # Both sides normalize through _inbox_home_key: None means the launch profile's home
    # (see its docstring for why raw-value comparison dropped every launch session).
    want_home = _inbox_home_key(profile_home)
    for sid, record in snapshot:
        if not isinstance(record, dict):
            continue
        if _inbox_home_key(record.get("profile_home")) != want_home:
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
    # One prefix scan for every session's expired-request count (state_meta), so a request
    # that died without an answer stays visible after its turn, session and app restart.
    expired_by_key = load_expired_request_counts(db)

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
        expired_count = expired_by_key.get(key, 0)
        lanes = classify_lanes(control, approval is not None, clarify is not None,
                               pending_expired=expired_count > 0)
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
            "expired_request_count": expired_count,
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
