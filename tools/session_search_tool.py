#!/usr/bin/env python3
"""
Session Search Tool - Long-Term Conversation Recall

Single-shape tool with three calling modes (inferred from args, no explicit
mode parameter):

  1. DISCOVERY — pass ``query``. Runs FTS5, dedupes hits by session lineage,
     returns top N sessions each with: snippet, ±5 message window around the
     match, plus bookend_start (first 3 user+assistant msgs of session) and
     bookend_end (last 3). Zero LLM cost.

  2. SCROLL — pass ``session_id`` + ``around_message_id``. Returns a window
     of ±window messages centered on the anchor, no FTS5, no bookends. To
     scroll forward / backward, re-anchor on the last / first message id of
     the returned window.

  3. BROWSE — no args. Returns recent sessions chronologically (titles,
     previews, timestamps).

All three modes operate on the SQLite session DB via the FTS5 index and
the get_anchored_view / get_messages_around primitives in hermes_state.
No LLM calls anywhere — every shape returns source DB messages, but recall
payloads are bounded before injection into the active model context: old
compaction-summary bodies are omitted, large message bodies are truncated,
and oversized tool-call payloads are summarized with metadata.

History: PR #20238 (JabberELF) seeded a fast/summary dual-mode split; the
toolkit expansion in PR #26419 (yoniebans) added the anchored drill-down,
bookends, and sort. This module merges all of that into a single calling
shape with no mode parameter, no summary LLM path, and explicit scroll
support.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

# Sources that are excluded from session browsing/searching by default.
# Third-party integrations tag their sessions with HERMES_SESSION_SOURCE=tool;
# delegate subagent runs are tagged "subagent" — neither belongs in the
# user's session history.
_HIDDEN_SESSION_SOURCES = ("subagent", "tool")

# Automation sources that are kept searchable but DEMOTED below interactive
# sessions in discover ranking. Cron jobs run on a schedule and accumulate
# large volumes of repetitive vocabulary (recurring project names, dates,
# "session", summaries); under bare BM25 they dominate the top-N FTS rows and
# starve out the user's own interactive sessions, producing "recall blindness"
# where only cron sessions surface (#19434). Demoting — not excluding — keeps
# cron content reachable when it's the only match, while interactive sessions
# always win when both match.
_DEMOTED_SESSION_SOURCES = ("cron",)

# How many FTS rows discover scans before dedup-by-lineage. The interactive
# vs automation split below only helps if enough rows are in hand to find
# interactive matches buried under a wall of cron hits, so this is well above
# the handful of distinct sessions a typical query returns.
_DISCOVER_SCAN_LIMIT = 300

# ``session_search`` output is injected straight back into the active model
# context. A single historical compaction handoff can be tens of thousands of
# chars and often contains stale "Active Task"/"Remaining Work" text. Returning
# it verbatim from discovery/bookends re-inflates the new session with exactly
# the compressed history the user was trying to escape. Keep recall useful while
# making the tool output bounded and non-recursive.
_MESSAGE_CONTENT_MAX_CHARS = 6_000
_SNIPPET_MAX_CHARS = 1_200
_TOOL_CALL_ARGUMENTS_MAX_CHARS = 1_200
_TOOL_CALL_METADATA_MAX_CHARS = 256
_TOOL_CALLS_MAX_ITEMS = 6
_RESPONSE_FIELD_MAX_CHARS = 8_000
_RESPONSE_MAX_CHARS = 120_000
_TOOL_CALL_METADATA_FIELDS = ("id", "call_id", "type", "name", "status", "index")
_TOOL_CALL_ARGUMENT_FIELDS = ("arguments", "input")


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Convert a Unix timestamp (float/int) or ISO string to a human-readable date.

    Returns "unknown" for None, str(ts) if conversion fails.
    """
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        if isinstance(ts, str):
            if ts.replace(".", "").replace("-", "").isdigit():
                from datetime import datetime
                dt = datetime.fromtimestamp(float(ts))
                return dt.strftime("%B %d, %Y at %I:%M %p")
            return ts
    except (ValueError, OSError, OverflowError) as e:
        logging.debug("Failed to format timestamp %s: %s", ts, e, exc_info=True)
    except Exception as e:
        logging.debug("Unexpected error formatting timestamp %s: %s", ts, e, exc_info=True)
    return str(ts)


def _resolve_to_parent(db, session_id: str) -> str:
    """Walk parent_session_id chain to the lineage root. Falls back to input on errors."""
    if not session_id:
        return session_id
    visited = set()
    cur = session_id
    while cur and cur not in visited:
        visited.add(cur)
        try:
            s = db.get_session(cur)
            if not s:
                break
            parent = s.get("parent_session_id")
            if not parent:
                break
            cur = parent
        except Exception as e:
            logging.debug("Error resolving parent for %s: %s", cur, e, exc_info=True)
            break
    return cur


def _order_for_recall(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable-sort FTS rows so interactive sessions rank above automation.

    Within each class (interactive vs demoted) the original BM25 ``rank``
    order is preserved — Python's sort is stable, and rows arrive already
    ranked by relevance. This only changes cross-class ordering: a cron hit
    never displaces an interactive hit during lineage dedup, so the user's
    own conversations surface first even when cron rows out-rank them under
    bare BM25 (#19434). Demoted rows still appear when they're the only
    matches.
    """
    return sorted(
        raw_results,
        key=lambda r: 1 if (r.get("source") or "") in _DEMOTED_SESSION_SOURCES else 0,
    )


def _text_char_len(content: Any) -> int:
    """Best-effort character length for persisted message content."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content, ensure_ascii=False))
    except Exception:
        return len(str(content))


def _is_compaction_summary_content(content: Any) -> bool:
    # Import lazily so loading the core tool schema does not pull in the
    # compressor and its provider dependencies. The invocation path still uses
    # the compressor's one canonical detector, including multimodal content and
    # every exact historical prefix.
    from agent.context_compressor import is_context_summary_content

    return is_context_summary_content(content)


def _truncate_string(value: str, limit: int) -> tuple[str, bool, int]:
    original_len = len(value)
    if original_len <= limit:
        return value, False, original_len
    omitted = original_len - limit
    return (
        value[:limit].rstrip()
        + f"\n[session_search truncated {omitted:,} chars from this field; scroll/read a narrower window if needed]",
        True,
        original_len,
    )


def _truncate_string_to_budget(value: str, limit: int) -> str:
    """Truncate a string to an exact final size for response budgeting."""
    if len(value) <= limit:
        return value
    suffix = "...[session_search response budget]"
    if limit <= len(suffix):
        return value[:limit]
    return value[: limit - len(suffix)].rstrip() + suffix


def _limit_response_strings(value: Any, limit: int) -> Any:
    """Return a copy with every string key and value capped to ``limit`` chars."""
    if isinstance(value, str):
        return _truncate_string_to_budget(value, limit)
    if isinstance(value, list):
        return [_limit_response_strings(item, limit) for item in value]
    if isinstance(value, dict):
        return {
            _truncate_string_to_budget(key, limit) if isinstance(key, str) else key:
            _limit_response_strings(item, limit)
            for key, item in value.items()
        }
    return value


def _bound_serialized_response(serialized: str) -> str:
    """Enforce per-field and total ceilings on the final JSON response."""
    original_response_chars = len(serialized)
    if original_response_chars <= _RESPONSE_FIELD_MAX_CHARS:
        return serialized

    try:
        payload = json.loads(serialized)
    except (TypeError, ValueError):
        return json.dumps(
            {
                "success": False,
                "response_truncated": True,
                "original_response_chars": original_response_chars,
                "error": "session_search produced an oversized non-JSON response",
            },
            ensure_ascii=False,
        )

    if not isinstance(payload, dict):
        payload = {"success": True, "result": payload}

    field_bounded = _limit_response_strings(payload, _RESPONSE_FIELD_MAX_CHARS)
    if field_bounded != payload:
        payload = field_bounded
        payload["response_truncated"] = True
        payload["response_fields_truncated"] = True
        payload["original_response_chars"] = original_response_chars
        serialized = json.dumps(payload, ensure_ascii=False)

    if len(serialized) <= _RESPONSE_MAX_CHARS:
        return serialized

    payload["response_truncated"] = True
    payload["original_response_chars"] = original_response_chars

    # Preserve the response structure while finding the largest uniform value
    # cap that fits the final serialization inside the hard ceiling.
    low, high = 32, max(32, _RESPONSE_FIELD_MAX_CHARS)
    best: Optional[str] = None
    while low <= high:
        candidate_limit = (low + high) // 2
        candidate = _limit_response_strings(payload, candidate_limit)
        candidate_json = json.dumps(candidate, ensure_ascii=False)
        if len(candidate_json) <= _RESPONSE_MAX_CHARS:
            best = candidate_json
            low = candidate_limit + 1
        else:
            high = candidate_limit - 1
    if best is not None:
        return best

    # Fixed structural overhead can only exceed the budget for pathological
    # persisted shapes. Fail closed rather than reinjecting an unbounded payload.
    return json.dumps(
        {
            "success": False,
            "mode": payload.get("mode"),
            "response_truncated": True,
            "original_response_chars": original_response_chars,
            "error": "session_recall_response_too_large",
            "message": "Session recall exceeded the response budget; request a narrower window.",
        },
        ensure_ascii=False,
    )


def _shape_content_for_recall(content: Any) -> tuple[Any, Dict[str, Any]]:
    """Return context-safe message content plus metadata about any elision."""
    original_chars = _text_char_len(content)
    if _is_compaction_summary_content(content):
        return (
            "[Context compaction summary omitted by session_search to prevent stale-task/context bloat. "
            f"Original summary was {original_chars:,} chars. Search or scroll the original source messages instead.]",
            {
                "content_omitted": "context_compaction_summary",
                "original_content_chars": original_chars,
            },
        )

    if isinstance(content, str):
        shaped, truncated, original_len = _truncate_string(content, _MESSAGE_CONTENT_MAX_CHARS)
        meta: Dict[str, Any] = {}
        if truncated:
            meta["content_truncated"] = True
            meta["original_content_chars"] = original_len
        return shaped, meta

    if original_chars > _MESSAGE_CONTENT_MAX_CHARS:
        try:
            serialized = json.dumps(content, ensure_ascii=False)
        except Exception:
            serialized = str(content)
        shaped, _, original_len = _truncate_string(serialized, _MESSAGE_CONTENT_MAX_CHARS)
        return shaped, {
            "content_truncated": True,
            "original_content_chars": original_len,
        }

    return content, {}


def _shape_snippet_for_recall(
    snippet: Any,
    *,
    source_content: Any = None,
) -> tuple[str, Dict[str, Any]]:
    """Return a bounded discovery snippet plus metadata."""
    if not isinstance(snippet, str):
        snippet = "" if snippet is None else str(snippet)

    original_chars = len(snippet)
    if _is_compaction_summary_content(source_content):
        return (
            "[Context compaction summary snippet omitted by session_search.]",
            {
                "snippet_omitted": "context_compaction_summary",
                "original_snippet_chars": original_chars,
            },
        )

    shaped, truncated, original_len = _truncate_string(snippet, _SNIPPET_MAX_CHARS)
    if truncated:
        return shaped, {
            "snippet_truncated": True,
            "original_snippet_chars": original_len,
        }
    return shaped, {}


def _shape_tool_call_arguments(arguments: Any) -> tuple[Any, bool]:
    if isinstance(arguments, str):
        shaped, truncated, _ = _truncate_string(arguments, _TOOL_CALL_ARGUMENTS_MAX_CHARS)
        return shaped, truncated
    if _text_char_len(arguments) > _TOOL_CALL_ARGUMENTS_MAX_CHARS:
        try:
            serialized = json.dumps(arguments, ensure_ascii=False)
        except Exception:
            serialized = str(arguments)
        shaped, _, _ = _truncate_string(serialized, _TOOL_CALL_ARGUMENTS_MAX_CHARS)
        return shaped, True
    return arguments, False


def _shape_tool_call_metadata(value: Any) -> tuple[Any, bool]:
    """Bound a retained provider-neutral tool-call metadata value."""
    if value is None or isinstance(value, bool):
        return value, False
    if isinstance(value, (int, float)):
        rendered = str(value)
        if len(rendered) <= _TOOL_CALL_METADATA_MAX_CHARS:
            return value, False
        value = rendered
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    shaped, truncated, _ = _truncate_string(value, _TOOL_CALL_METADATA_MAX_CHARS)
    return shaped, truncated


def _shape_tool_calls_for_recall(tool_calls: Any) -> tuple[Any, Dict[str, Any]]:
    """Return bounded provider-neutral tool-call metadata for recall output.

    Persisted call objects can include arbitrarily large provider bookkeeping.
    Keep only fields useful for understanding the historical call, and bound
    every retained value before it re-enters the active model context.
    """
    if not tool_calls:
        return tool_calls, {}

    if not isinstance(tool_calls, list):
        calls = [tool_calls]
        original_count = 1
    else:
        calls = tool_calls
        original_count = len(tool_calls)

    allowed_fields = {
        *_TOOL_CALL_METADATA_FIELDS,
        *_TOOL_CALL_ARGUMENT_FIELDS,
        "function",
    }
    shaped_calls = []
    truncated_args = False
    truncated_fields = False
    omitted_fields = False
    for call in calls[:_TOOL_CALLS_MAX_ITEMS]:
        if not isinstance(call, dict):
            shaped_call, did_truncate = _shape_tool_call_arguments(call)
            shaped_calls.append(shaped_call)
            truncated_args = truncated_args or did_truncate
            continue

        shaped: Dict[str, Any] = {}
        for field in _TOOL_CALL_METADATA_FIELDS:
            if field not in call:
                continue
            shaped_value, did_truncate = _shape_tool_call_metadata(call[field])
            shaped[field] = shaped_value
            truncated_fields = truncated_fields or did_truncate

        for field in _TOOL_CALL_ARGUMENT_FIELDS:
            if field not in call:
                continue
            shaped_value, did_truncate = _shape_tool_call_arguments(call[field])
            shaped[field] = shaped_value
            truncated_args = truncated_args or did_truncate

        fn = call.get("function")
        if isinstance(fn, dict):
            fn_shaped: Dict[str, Any] = {}
            if "name" in fn:
                shaped_name, did_truncate = _shape_tool_call_metadata(fn["name"])
                fn_shaped["name"] = shaped_name
                truncated_fields = truncated_fields or did_truncate
            if "arguments" in fn:
                shaped_args, did_truncate = _shape_tool_call_arguments(fn["arguments"])
                fn_shaped["arguments"] = shaped_args
                truncated_args = truncated_args or did_truncate
            shaped["function"] = fn_shaped
            omitted_fields = omitted_fields or bool(set(fn) - {"name", "arguments"})
        elif "function" in call:
            shaped_function, did_truncate = _shape_tool_call_metadata(fn)
            shaped["function"] = shaped_function
            truncated_fields = truncated_fields or did_truncate

        omitted_fields = omitted_fields or bool(set(call) - allowed_fields)
        shaped_calls.append(shaped)

    meta: Dict[str, Any] = {}
    if original_count > len(shaped_calls):
        meta["tool_calls_truncated"] = True
        meta["original_tool_call_count"] = original_count
    if truncated_args:
        meta["tool_call_arguments_truncated"] = True
    if truncated_fields:
        meta["tool_call_fields_truncated"] = True
    if omitted_fields:
        meta["tool_call_fields_omitted"] = True

    return shaped_calls, meta


def _shape_message(m: Dict[str, Any], anchor_id: Optional[int] = None) -> Dict[str, Any]:
    """Slim a message row for the tool response. Keeps content even if empty."""
    shaped_content, content_meta = _shape_content_for_recall(m.get("content"))
    entry = {
        "id": m.get("id"),
        "role": m.get("role"),
        "content": shaped_content,
        "timestamp": m.get("timestamp"),
    }
    entry.update(content_meta)
    if m.get("tool_name"):
        tool_name, truncated = _shape_tool_call_metadata(m.get("tool_name"))
        entry["tool_name"] = tool_name
        if truncated:
            entry["tool_name_truncated"] = True
    if m.get("tool_calls"):
        shaped_tool_calls, tool_meta = _shape_tool_calls_for_recall(m.get("tool_calls"))
        entry["tool_calls"] = shaped_tool_calls
        entry.update(tool_meta)
    if m.get("tool_call_id"):
        tool_call_id, truncated = _shape_tool_call_metadata(m.get("tool_call_id"))
        entry["tool_call_id"] = tool_call_id
        if truncated:
            entry["tool_call_id_truncated"] = True
    if anchor_id is not None and m.get("id") == anchor_id:
        entry["anchor"] = True
    # Strip None values to keep payload tight, but always keep content
    # (absent content is meaningful — tool-call-only assistant turns).
    return {k: v for k, v in entry.items() if v is not None or k in ("content",)}


def _resolve_profile_db(profile: str):
    """Open another profile's ``state.db`` read-only, or None for the current one.

    The desktop's ``@session:<profile>/<id>`` links always carry the source
    profile, so a linked session from profile B can be read while the agent
    runs in profile A. ``read_only=True`` (mode=ro) takes no write lock — safe
    to point at a live profile's DB, including our own. Returns None when no
    profile is given (use the caller's default db).
    """
    if profile is None or not str(profile).strip():
        return None

    from hermes_cli import profiles as profiles_mod
    from hermes_state import SessionDB

    canon = profiles_mod.normalize_profile_name(profile)
    profiles_mod.validate_profile_name(canon)
    if not profiles_mod.profile_exists(canon):
        raise ValueError(f"profile '{canon}' does not exist")

    return SessionDB(db_path=profiles_mod.get_profile_dir(canon) / "state.db", read_only=True)


def _locate_session_db(session_id: str):
    """Scan every profile's ``state.db`` (read-only) for a session id.

    Returns ``(db, profile_name)`` for the first profile that owns the id, or
    ``(None, None)``. Session ids are globally unique (timestamp + random hex),
    so the first hit is authoritative. This is the safety net for linked-session
    reads where the model dropped the owning profile from the link and passed a
    bare id — we find it wherever it actually lives instead of failing.
    """
    from pathlib import Path

    try:
        from hermes_cli import profiles as profiles_mod
        from hermes_state import SessionDB
    except Exception:
        return None, None

    targets = [("default", profiles_mod.get_profile_dir("default"))]
    try:
        targets += [(info.name, info.path) for info in profiles_mod.list_profiles()]
    except Exception:
        logging.debug("list_profiles failed during session locate", exc_info=True)

    seen: set = set()
    for name, home in targets:
        db_path = Path(home) / "state.db"
        key = str(db_path)
        if key in seen or not db_path.exists():
            continue
        seen.add(key)
        try:
            pdb = SessionDB(db_path=db_path, read_only=True)
        except Exception:
            continue
        try:
            if pdb.get_session(session_id):
                return pdb, name
        except Exception:
            logging.debug("get_session probe failed for %s in %s", session_id, name, exc_info=True)
        pdb.close()

    return None, None


def _read_session(db, session_id: str, head: int = 20, tail: int = 10) -> str:
    """Read shape: dump a whole session by id (head + tail when large).

    Serves the linked-session case — the user dropped an @session reference and
    the agent wants the transcript. Bounded payload: small sessions return in
    full, large ones return the first ``head`` and last ``tail`` messages with a
    pointer to scroll the middle.
    """
    try:
        meta = db.get_session(session_id) or {}
    except Exception as e:
        logging.debug("get_session failed for %s: %s", session_id, e, exc_info=True)
        meta = {}
    if not meta:
        return tool_error(f"session_id not found: {session_id}", success=False)

    try:
        total = max(0, int(meta.get("message_count") or 0))
    except (TypeError, ValueError):
        total = 0

    try:
        if total > head + tail:
            rows = db.get_messages(session_id, limit=head)
            rows += db.get_messages(session_id, limit=tail, offset=max(total - tail, 0))
            truncated = True
        else:
            rows = db.get_messages(session_id)
            total = len(rows)
            truncated = total > head + tail
            if truncated:
                rows = rows[:head] + rows[-tail:]
    except Exception as e:
        logging.error("get_messages failed for %s: %s", session_id, e, exc_info=True)
        return tool_error(f"failed to load session: {e}", success=False)

    window = [_shape_message(m) for m in rows]

    response = {
        "success": True,
        "mode": "read",
        "session_id": session_id,
        "session_meta": {
            "when": _format_timestamp(meta.get("started_at")),
            "source": meta.get("source"),
            "model": meta.get("model"),
            "title": meta.get("title"),
        },
        "message_count": total,
        "truncated": truncated,
        "messages": window,
    }
    if truncated:
        response["message"] = (
            f"Session has {total} messages; showing first {head} + last {tail}. "
            "Pass around_message_id (any id above) to scroll the middle."
        )
    return json.dumps(response, ensure_ascii=False)


def _list_recent_sessions(db, limit: int, current_session_id: Optional[str] = None) -> str:
    """Return metadata for the most recent sessions (no LLM calls, no FTS5)."""
    try:
        sessions = db.list_sessions_rich(
            limit=limit + 5,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            order_by_last_active=True,
        )  # fetch extra so we can skip current

        current_root = _resolve_to_parent(db, current_session_id) if current_session_id else None

        results = []
        for s in sessions:
            sid = s.get("id", "")
            if current_root and (sid == current_root or sid == current_session_id):
                continue
            # Skip child / delegation sessions
            if s.get("parent_session_id"):
                continue
            results.append({
                "session_id": sid,
                "title": s.get("title") or None,
                "source": s.get("source", ""),
                "started_at": s.get("started_at", ""),
                "last_active": s.get("last_active", ""),
                "message_count": s.get("message_count", 0),
                "preview": s.get("preview", ""),
            })
            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "mode": "browse",
            "results": results,
            "count": len(results),
            "message": f"Showing {len(results)} most recent sessions. Pass a query= to search, or session_id+around_message_id to scroll.",
        }, ensure_ascii=False)
    except Exception as e:
        logging.error("Error listing recent sessions: %s", e, exc_info=True)
        return tool_error(f"Failed to list recent sessions: {e}", success=False)


def _scroll(
    db,
    session_id: str,
    around_message_id: int,
    window: int = 5,
    current_session_id: Optional[str] = None,
) -> str:
    """Scroll shape: return a window of messages centered on an anchor.

    No FTS5, no bookends — just the slice. The discovery shape's lineage
    fixup is preserved: if the anchor doesn't live in the named session
    but does live in a child session in the same lineage, rebind silently.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        return tool_error("scroll requires session_id", success=False)
    session_id = session_id.strip()

    try:
        around_message_id = int(around_message_id)
    except (TypeError, ValueError):
        return tool_error("scroll requires integer around_message_id", success=False)

    # Window clamp [1, 20]
    if not isinstance(window, int):
        try:
            window = int(window)
        except (TypeError, ValueError):
            window = 5
    window = max(1, min(window, 20))

    # Reject scrolling inside the active session lineage — those messages are
    # already in context.
    if current_session_id:
        a_root = _resolve_to_parent(db, session_id)
        c_root = _resolve_to_parent(db, current_session_id)
        if a_root and c_root and a_root == c_root:
            return tool_error(
                "scroll rejected: anchor lives in the current session lineage (already in your active context)",
                success=False,
            )

    # Session existence check
    try:
        session_meta = db.get_session(session_id) or {}
    except Exception as e:
        logging.debug("get_session failed for %s: %s", session_id, e, exc_info=True)
        session_meta = {}
    if not session_meta:
        return tool_error(f"session_id not found: {session_id}", success=False)

    # Fetch the window
    try:
        view = db.get_messages_around(session_id, around_message_id, window=window)
    except Exception as e:
        logging.error("get_messages_around failed: %s", e, exc_info=True)
        return tool_error(f"failed to load messages: {e}", success=False)

    messages = view.get("window") or []

    # Lineage rebind: caller may have paired a parent session_id with a
    # message id that lives in a descendant (compaction / delegation creates
    # child sessions). Locate the real owning session and refetch.
    rebind_warning = None
    if not messages:
        owning = None
        try:
            conn = getattr(db, "_conn", None)
            if conn is not None:
                row = conn.execute(
                    "SELECT session_id FROM messages WHERE id = ?",
                    (around_message_id,),
                ).fetchone()
                owning = row[0] if row else None
        except Exception as e:
            logging.debug("owning-session lookup failed: %s", e, exc_info=True)
            owning = None
        if owning and owning != session_id:
            a_root = _resolve_to_parent(db, session_id)
            o_root = _resolve_to_parent(db, owning)
            if a_root and o_root and a_root == o_root:
                try:
                    rebind_view = db.get_messages_around(owning, around_message_id, window=window)
                    messages = rebind_view.get("window") or []
                    if messages:
                        view = rebind_view
                        rebind_warning = (
                            f"around_message_id {around_message_id} lives in {owning} "
                            f"(child of {session_id}); rebound transparently"
                        )
                        try:
                            session_meta = db.get_session(owning) or session_meta
                        except Exception:
                            pass
                        session_id = owning
                except Exception as e:
                    logging.debug("rebind get_messages_around failed: %s", e, exc_info=True)

    if not messages:
        return tool_error(
            f"around_message_id {around_message_id} not in session_id {session_id}",
            success=False,
        )

    response = {
        "success": True,
        "mode": "scroll",
        "session_id": session_id,
        "around_message_id": around_message_id,
        "session_meta": {
            "when": _format_timestamp(session_meta.get("started_at")),
            "source": session_meta.get("source"),
            "model": session_meta.get("model"),
            "title": session_meta.get("title"),
        },
        "window": window,
        "messages": [_shape_message(m, anchor_id=around_message_id) for m in messages],
        "messages_before": view.get("messages_before", 0),
        "messages_after": view.get("messages_after", 0),
    }
    if rebind_warning:
        response["warning"] = rebind_warning
    return json.dumps(response, ensure_ascii=False)


def _normalize_title_query(query: str) -> str:
    """Strip common quoting the model may include around a remembered title."""
    return query.strip().strip("`'\"")


def _title_match_result(
    db,
    query: str,
    current_lineage_root: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return a discovery-shaped result when the query matches a session title."""
    title_query = _normalize_title_query(query)
    if not title_query:
        return None

    try:
        session_id = db.resolve_session_by_title(title_query)
    except Exception:
        logging.debug("resolve_session_by_title failed for %r", title_query, exc_info=True)
        return None
    if not session_id:
        return None

    lineage_root = _resolve_to_parent(db, session_id)
    if current_lineage_root and lineage_root == current_lineage_root:
        return None

    try:
        session_meta = db.get_session(lineage_root) or db.get_session(session_id) or {}
    except Exception:
        logging.debug("get_session failed for title match %s", session_id, exc_info=True)
        session_meta = {}
    if session_meta.get("source") in _HIDDEN_SESSION_SOURCES:
        return None

    try:
        messages = db.get_messages(session_id)
    except Exception:
        logging.debug("get_messages failed for title match %s", session_id, exc_info=True)
        messages = []

    anchor_id = messages[0].get("id") if messages else None
    if anchor_id is not None:
        try:
            view = db.get_anchored_view(session_id, anchor_id, window=5, bookend=3)
        except Exception:
            logging.debug("get_anchored_view failed for title match %s/%s", session_id, anchor_id, exc_info=True)
            view = {}
    else:
        view = {}

    entry = {
        "session_id": session_id,
        "when": _format_timestamp(session_meta.get("started_at")),
        "source": session_meta.get("source", "unknown"),
        "model": session_meta.get("model") or "unknown",
        "title": session_meta.get("title") or title_query,
        "matched_role": "session_title",
        "match_message_id": anchor_id,
        "snippet": f"Session title matched: {session_meta.get('title') or title_query}",
        "bookend_start": [_shape_message(m) for m in (view.get("bookend_start") or messages[:3])],
        "messages": [_shape_message(m, anchor_id=anchor_id) for m in (view.get("window") or messages[:5])],
        "bookend_end": [_shape_message(m) for m in (view.get("bookend_end") or messages[-3:])],
        "messages_before": view.get("messages_before", 0),
        "messages_after": view.get("messages_after", max(len(messages) - 5, 0)),
        "_lineage_root": lineage_root,
    }
    if lineage_root and lineage_root != session_id:
        entry["parent_session_id"] = lineage_root
    return entry


def _discover(
    db,
    query: str,
    role_filter: Optional[List[str]],
    limit: int,
    sort: Optional[str],
    current_session_id: Optional[str] = None,
) -> str:
    """Discovery shape: FTS5 + anchored window + bookends per hit. Single call."""
    role_list = role_filter if role_filter else ["user", "assistant"]
    current_lineage_root = _resolve_to_parent(db, current_session_id) if current_session_id else None
    title_result = _title_match_result(db, query, current_lineage_root)

    try:
        raw_results = db.search_messages(
            query=query,
            role_filter=role_list,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            limit=_DISCOVER_SCAN_LIMIT,  # widen so dedup-by-lineage can find
            # distinct sessions AND so interactive matches buried under a wall
            # of cron rows are still in hand for the demotion pass below.
            offset=0,
            sort=sort,
        )
    except Exception as e:
        logging.error("FTS5 search failed: %s", e, exc_info=True)
        return tool_error(f"Search failed: {e}", success=False)

    # Demote automation (cron) rows below interactive ones before dedup, so a
    # high-volume cron corpus can't starve the user's own sessions out of the
    # top `limit` results (#19434). Stable — preserves BM25/recency order
    # within each class.
    raw_results = _order_for_recall(raw_results)

    if not raw_results and not title_result:
        return json.dumps({
            "success": True,
            "mode": "discover",
            "query": query,
            "results": [],
            "count": 0,
            "message": "No matching sessions found.",
        }, ensure_ascii=False)

    # Dedupe by lineage. Keep the raw owning session_id on the surviving
    # row — only that pairs validly with the FTS5 match id for the anchored
    # window. parent_session_id is exposed separately when different.
    seen_sessions = {}
    results = []

    if title_result:
        title_lineage = title_result.pop("_lineage_root", None)
        if title_lineage:
            seen_sessions[title_lineage] = {"_title_only": True}
        results.append(title_result)

    for r in raw_results:
        if len(seen_sessions) >= limit:
            break
        raw_sid = r["session_id"]
        resolved_sid = _resolve_to_parent(db, raw_sid)
        # Skip the current session lineage
        if current_lineage_root and resolved_sid == current_lineage_root:
            continue
        if current_session_id and raw_sid == current_session_id:
            continue
        if resolved_sid not in seen_sessions:
            row = dict(r)
            row["_lineage_root"] = resolved_sid
            seen_sessions[resolved_sid] = row
        if len(seen_sessions) >= limit:
            break

    for lineage_root, match_info in seen_sessions.items():
        if match_info.get("_title_only"):
            continue
        hit_sid = match_info.get("session_id") or lineage_root
        msg_id = match_info.get("id")
        try:
            view = db.get_anchored_view(hit_sid, msg_id, window=5, bookend=3)
        except Exception as e:
            logging.warning("get_anchored_view failed for %s/%s: %s", hit_sid, msg_id, e, exc_info=True)
            continue

        try:
            session_meta = db.get_session(lineage_root) or {}
        except Exception:
            session_meta = {}

        anchor_content = next(
            (
                message.get("content")
                for message in (view.get("window") or [])
                if message.get("id") == msg_id
            ),
            None,
        )
        shaped_snippet, snippet_meta = _shape_snippet_for_recall(
            match_info.get("snippet"),
            source_content=anchor_content,
        )
        entry = {
            "session_id": hit_sid,
            "when": _format_timestamp(
                session_meta.get("started_at") or match_info.get("session_started")
            ),
            "source": session_meta.get("source") or match_info.get("source", "unknown"),
            "model": session_meta.get("model") or match_info.get("model") or "unknown",
            "title": session_meta.get("title") or None,
            "matched_role": match_info.get("role"),
            "match_message_id": msg_id,
            "snippet": shaped_snippet,
            "bookend_start": [_shape_message(m) for m in (view.get("bookend_start") or [])],
            "messages": [_shape_message(m, anchor_id=msg_id) for m in (view.get("window") or [])],
            "bookend_end": [_shape_message(m) for m in (view.get("bookend_end") or [])],
            "messages_before": view.get("messages_before", 0),
            "messages_after": view.get("messages_after", 0),
        }
        entry.update(snippet_meta)
        if lineage_root and lineage_root != hit_sid:
            entry["parent_session_id"] = lineage_root
        results.append(entry)

    return json.dumps({
        "success": True,
        "mode": "discover",
        "query": query,
        "results": results,
        "count": len(results),
        "sessions_searched": len(seen_sessions),
    }, ensure_ascii=False)


def _session_search_impl(
    query: str = "",
    role_filter: Optional[str] = None,
    limit: int = 3,
    db=None,
    current_session_id: Optional[str] = None,
    # Scroll shape
    session_id: Optional[str] = None,
    around_message_id: Optional[int] = None,
    window: int = 5,
    # Discovery shape
    sort: Optional[str] = None,
    # Cross-profile (any shape)
    profile: Optional[str] = None,
) -> str:
    """Single-shape tool. Mode inferred from which args are set.

    Discovery: pass ``query``.
    Scroll:    pass ``session_id`` + ``around_message_id``.
    Read:      pass ``session_id`` (no anchor) — dumps the whole session.
    Browse:    pass nothing.

    Pass ``profile`` to read another profile's sessions (e.g. resolving an
    ``@session:<profile>/<id>`` link). Scroll wins over read/discovery when an
    anchor is set — the agent has asked for a specific slice.
    """
    if db is None:
        try:
            from hermes_state import SessionDB
            db = SessionDB()
        except Exception:
            logging.debug("SessionDB unavailable for session_search", exc_info=True)
            from hermes_state import format_session_db_unavailable
            return tool_error(format_session_db_unavailable(), success=False)

    # Normalise a raw `@session:<profile>/<id>` link value passed as session_id.
    # Session ids never contain "/", so a slash unambiguously means profile/id —
    # always strip the prefix off the id, and adopt the embedded profile only
    # when one wasn't passed explicitly. Handles every permutation the model
    # might send (full value as id, with or without a separate profile=).
    if isinstance(session_id, str) and "/" in session_id:
        emb_profile, _, emb_id = session_id.partition("/")
        if emb_id:
            session_id = emb_id
            if emb_profile and (profile is None or not str(profile).strip()):
                profile = emb_profile

    # Cross-profile read: swap in the named profile's DB (read-only) for every
    # shape below. The current-session-lineage guards no longer apply across
    # profiles, but they key off ids that won't collide, so they stay inert.
    if profile is not None and str(profile).strip():
        try:
            profile_db = _resolve_profile_db(profile)
        except Exception as e:
            return tool_error(f"profile '{profile}': {e}", success=False)
        if profile_db is not None:
            db = profile_db
            current_session_id = None

    # Scroll shape takes precedence — explicit anchor beats any query.
    if (isinstance(session_id, str) and session_id.strip()) and around_message_id is not None:
        return _scroll(
            db=db,
            session_id=session_id,
            around_message_id=around_message_id,
            window=window,
            current_session_id=current_session_id,
        )

    # Read shape: a session_id with no anchor → dump the whole session.
    if isinstance(session_id, str) and session_id.strip():
        sid = session_id.strip()
        result = _read_session(db, sid)
        if json.loads(result).get("success"):
            return result

        # Miss in the target profile — the model may have dropped the owning
        # profile from the link. Scan every profile and read it from wherever
        # it lives, tagging the profile it was found in.
        located, owner = _locate_session_db(sid)
        if located is not None:
            try:
                found = json.loads(_read_session(located, sid))
            finally:
                located.close()
            if found.get("success"):
                found["profile"] = owner
                return json.dumps(found, ensure_ascii=False)
        return result

    # Limit clamp [1, 10]
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 3
    limit = max(1, min(limit, 10))

    # Browse shape: no query → recent sessions.
    if not query or not isinstance(query, str) or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id)

    # Parse role_filter
    role_list: Optional[List[str]] = None
    if isinstance(role_filter, str) and role_filter.strip():
        role_list = [r.strip() for r in role_filter.split(",") if r.strip()]

    # Normalise sort
    sort_norm: Optional[str] = None
    if isinstance(sort, str):
        candidate = sort.strip().lower()
        if candidate in ("newest", "oldest"):
            sort_norm = candidate

    return _discover(
        db=db,
        query=query.strip(),
        role_filter=role_list,
        limit=limit,
        sort=sort_norm,
        current_session_id=current_session_id,
    )


def session_search(
    query: str = "",
    role_filter: Optional[str] = None,
    limit: int = 3,
    db=None,
    current_session_id: Optional[str] = None,
    session_id: Optional[str] = None,
    around_message_id: Optional[int] = None,
    window: int = 5,
    sort: Optional[str] = None,
    profile: Optional[str] = None,
) -> str:
    """Run session recall and enforce the final serialized response budget."""
    return _bound_serialized_response(
        _session_search_impl(
            query=query,
            role_filter=role_filter,
            limit=limit,
            db=db,
            current_session_id=current_session_id,
            session_id=session_id,
            around_message_id=around_message_id,
            window=window,
            sort=sort,
            profile=profile,
        )
    )


def check_session_search_requirements() -> bool:
    """Requires the SQLite state database."""
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search past sessions stored in the local session DB, or scroll inside one. "
        "FTS5-backed retrieval over the SQLite message store. No LLM calls — every "
        "shape returns source DB messages, but recall payloads are bounded before "
        "model injection: old compaction-summary bodies may be omitted, and large "
        "message/tool-call payloads may be truncated with metadata.\n\n"
        "SOURCE-FIRST LIMIT\n\n"
        "  This tool searches Hermes conversation history only. It is not evidence "
        "about the current contents of external sources. If the user provided a "
        "direct source such as a URL, phone number/contact, app/thread, file path, "
        "account, website, or live system, inspect that original source before or "
        "instead of session_search when accessible. Use session_search as secondary "
        "context for what was previously said, not as primary proof of what the "
        "source currently contains. If the original source is inaccessible, say so "
        "and why before falling back to session history. Do not conclude 'not found' "
        "or 'no prior correspondence' from session_search alone when a direct source "
        "was provided.\n\n"
        "FOUR CALLING SHAPES\n\n"
        "  1) DISCOVERY — pass `query`:\n"
        "     session_search(query=\"auth refactor\", limit=3)\n"
        "     Runs FTS5, dedupes hits by session lineage, returns the top N sessions. "
        "Each result carries:\n"
        "       - session_id, title, when, source\n"
        "       - snippet: FTS5-highlighted match excerpt\n"
        "       - bookend_start: first 3 user+assistant messages of the session "
        "(the goal / kickoff)\n"
        "       - messages: ±5 messages around the FTS5 match, with the anchor message "
        "flagged (the hit in context)\n"
        "       - bookend_end: last 3 user+assistant messages of the session "
        "(the resolution / decisions)\n"
        "       - match_message_id, messages_before, messages_after\n"
        "     Bookends + window together let you reconstruct goal → match → resolution "
        "without paying for the whole transcript.\n\n"
        "  2) SCROLL — pass `session_id` + `around_message_id`:\n"
        "     session_search(session_id=\"...\", around_message_id=12345, window=10)\n"
        "     Returns a window of ±`window` messages centered on the anchor. No FTS5, "
        "no bookends — just the slice. Use after a discovery call when you need more "
        "context than the ±5 default window.\n"
        "       - To scroll FORWARD: pass messages[-1].id back as around_message_id.\n"
        "       - To scroll BACKWARD: pass messages[0].id back as around_message_id.\n"
        "       - The boundary message appears in both windows — orientation marker.\n"
        "       - When messages_before or messages_after is < window, you're at the "
        "start or end of the session.\n\n"
        "  3) READ — pass `session_id` only (no around_message_id):\n"
        "     session_search(session_id=\"...\", profile=\"work\")\n"
        "     Dumps the whole session by id (first 20 + last 10 messages when "
        "large). This is how you resolve an `@session:<profile>/<id>` link the "
        "user dropped into the chat: split the value on `/` into profile + id "
        "and call session_search(session_id=id, profile=profile).\n\n"
        "  4) BROWSE — no args:\n"
        "     session_search()\n"
        "     Returns recent sessions chronologically: titles, previews, timestamps. "
        "Use when the user asks \"what was I working on\" without naming a topic.\n\n"
        "FTS5 SYNTAX\n\n"
        "  AND is the default — multi-word queries require all terms. Use OR explicitly "
        "for broader recall (`alpha OR beta OR gamma`), quoted phrases for exact match "
        "(`\"docker networking\"`), boolean (`python NOT java`), or prefix wildcards "
        "(`deploy*`).\n\n"
        "WHEN TO USE\n\n"
        "  Reach for this on questions about Hermes conversation history itself, such "
        "as \"what did we do about X\", \"where did we leave Y\", or \"find the "
        "session where Z\". If the user provided a direct source identifier, inspect "
        "that source first when accessible; session_search can then supply historical "
        "context. The session DB carries what was said when; external tools show "
        "current source/world state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query (discovery shape). Keywords, phrases, or boolean "
                    "expressions to find in past sessions. Omit to browse recent "
                    "sessions. Ignored when session_id + around_message_id are set "
                    "(scroll shape)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Discovery shape only. Max sessions to return (default 3, max 10). "
                    "Bump to 5–10 when the topic likely spans several sessions and you "
                    "want to pick the right one to scroll into."
                ),
                "default": 3,
            },
            "sort": {
                "type": "string",
                "enum": ["newest", "oldest"],
                "description": (
                    "Discovery shape only. Temporal bias on top of FTS5 ranking. Omit "
                    "to keep relevance-only ordering (suitable for exploratory recall — "
                    "\"what do we know about X\"). Set 'newest' for recency-shaped "
                    "questions (\"where did we leave X\"). Set 'oldest' for "
                    "origin-shaped questions (\"how did X start\"). Ignored in scroll "
                    "and browse shapes."
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Scroll shape. Session to read inside. Use the session_id returned "
                    "from a prior discovery call. Must be paired with "
                    "around_message_id."
                ),
            },
            "around_message_id": {
                "type": "integer",
                "description": (
                    "Scroll shape. Message id to center the window on. From a discovery "
                    "result use match_message_id, or any id seen in a prior window. To "
                    "scroll forward pass the last window message's id; to scroll "
                    "backward pass the first."
                ),
            },
            "window": {
                "type": "integer",
                "description": (
                    "Scroll shape only. Messages to return on each side of the anchor "
                    "(anchor itself always included). Clamped to [1, 20]. Default 5."
                ),
                "default": 5,
            },
            "role_filter": {
                "type": "string",
                "description": (
                    "Optional. Comma-separated roles to include. Discovery defaults to "
                    "'user,assistant' (tool output is usually noise). Pass "
                    "'user,assistant,tool' to include tool output (debugging tool "
                    "behaviour) or 'tool' to search tool output only."
                ),
            },
            "profile": {
                "type": "string",
                "description": (
                    "Optional. Read sessions from another Hermes profile's database "
                    "(read-only). Use when resolving an `@session:<profile>/<id>` link: "
                    "pass the profile segment here with session_id as the id segment. "
                    "Omit to use the current profile."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        session_id=args.get("session_id"),
        around_message_id=args.get("around_message_id"),
        window=args.get("window", 5),
        sort=args.get("sort"),
        profile=args.get("profile"),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)
