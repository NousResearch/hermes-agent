"""skill_view repeat-view dedup registry: per-task cache of (skill name, file_path) ->
(skill file mtime+size, writer session). A repeat view of an UNCHANGED file returns a short
stub — the earlier tool result already carries the content verbatim. Cleared via
``reset_skill_view_dedup()`` on context compression AND on a committed proactive tool-result
prune, because both replace the original content with a one-line marker.

The cache is process-global (one long-lived process serves many sessions) and bucketed by
task_id, so each entry is ALSO attributed to the session that served it: when a session
reads a bucket that holds entries written by a different session (a fresh session reusing
a task_id — TUI resume/reconnect, a background-review fork, or any task_id collision),
ONLY those foreign entries are dropped. Other sessions' buckets (other task_ids — the
gateway serves many sessions per process, task_id == session_id) are left alone, so a
live session never wipes another session's legitimate cross-turn dedup. Without this, a
fresh session that reuses a task_id could match a stale entry from a previous session and
get a false "unchanged" stub for a skill it has never actually loaded into its own context
— the model then has to read the file from disk to recover the content.

``session_id`` of None or "" is "no session" (the fork / pre-session path, and
``agent.session_id or ""`` at the tool layer): it neither drops entries nor re-attributes
them, so it can neither clobber a real session's entries nor be clobbered by one.
"""

import json
import os
import threading
from typing import Dict

# value: (src, mtime_ns, size, writer_session_id) — the trailing element attributes the
# entry to the session that served it (None/"" = no session).
_skill_view_tracker: Dict[str, Dict[tuple, tuple]] = {}
_skill_view_tracker_lock = threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200

_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (Re-issued after context compression, this "
    "returns the full content again.)")


def _skill_view_fingerprint(payload: dict) -> tuple | None:
    """Stat the skill file a successful skill_view served, for change detection."""
    if not (src := payload.get("_source_path")):
        return None
    try:
        st = os.stat(src)
        return (src, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _record_skill_view(task_id, name, file_path, payload: dict, session_id: str | None = None) -> None:
    """Record a served skill_view so an identical repeat can be deduped.

    The entry is attributed to the serving session (``session_id``; None/"" = no session)
    so a different session reading the same task_id's bucket can drop the foreign entries
    (see ``_drop_foreign_entries``) instead of the whole cache being dropped.
    """
    # Never dedup setup-needed views: readiness depends on config/env state that
    # changes without the file changing; the model must see the refreshed status.
    if (not task_id or payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"):
        return
    if (fp := _skill_view_fingerprint(payload)) is None:
        return
    key = (str(payload.get("name") or name), file_path or "")
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.setdefault(str(task_id), {})
        cache[key] = fp + (session_id,)
        while len(cache) > _SKILL_VIEW_DEDUP_CAP:  # FIFO eviction
            del cache[next(iter(cache))]


def _drop_foreign_entries(task_id: str, session_id: str | None) -> None:
    """Drop the entries of this task's bucket written by a different session (caller holds
    the lock).

    The cache is a per-conversation optimization, but the process is long-lived and serves
    many sessions: a fresh session that reuses a task_id (TUI resume/reconnect, a
    background-review fork, or any task_id collision) must NOT match a stale entry from a
    previous session, or its first view of a skill would return a false "unchanged" stub.
    The drop is scoped to the affected bucket's foreign entries — other sessions' buckets
    (other task_ids) keep their entries, so their cross-turn dedup survives a concurrent
    session. ``session_id`` of None or "" is "no session" and a no-op: it neither drops
    foreign entries nor re-attributes them ("" is not a session id — the tool layer passes
    ``agent.session_id or ""``).
    """
    if not session_id:
        return
    cache = _skill_view_tracker.get(task_id)
    if not cache:
        return
    for key in [k for k, v in cache.items() if v[3] != session_id]:
        del cache[key]


def _check_skill_view_dedup(task_id, name, file_path, session_id: str | None = None) -> str | None:
    """Dedup stub when this exact skill file was already served to this task and
    is unchanged on disk; None otherwise.

    ``session_id`` scopes the cache to the serving conversation: entries of this task's
    bucket written by a different session are dropped first, so a fresh session (or a
    task_id reused across sessions) never gets a false "unchanged" stub — while other
    sessions' buckets are left alone. ``session_id`` of None or "" is a no-op.
    """
    if not task_id:
        return None
    n = str(name)
    with _skill_view_tracker_lock:
        _drop_foreign_entries(str(task_id), session_id)
        if not (cache := _skill_view_tracker.get(str(task_id))):
            return None
        # Record key is the RESOLVED name; match raw and resolved forms so
        # 'category/skill' and bare-name views coalesce.
        for key, (src, mtime_ns, size, _writer) in list(cache.items()):
            rec_name, rec_fp = key
            if rec_fp != (file_path or "") or (
                    rec_name != n and not n.endswith("/" + rec_name)
                    and not rec_name.endswith("/" + n) and n.split(":")[-1] != rec_name):
                continue
            try:
                st = os.stat(src)
                changed = (st.st_mtime_ns, st.st_size) != (mtime_ns, size)
            except OSError:
                changed = True
            if changed:
                cache.pop(key, None)
                return None
            return json.dumps({
                "success": True, "status": "unchanged", "name": rec_name,
                "file": file_path or "SKILL.md", "dedup": True, "content_returned": False,
                "message": _SKILL_VIEW_DEDUP_MESSAGE}, ensure_ascii=False)
    return None


def reset_skill_view_dedup(task_id: str | None = None) -> None:
    """Clear the dedup cache (all tasks when task_id is None); called on context compression."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
        else:
            _skill_view_tracker.pop(str(task_id), None)
