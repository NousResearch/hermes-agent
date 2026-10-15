"""skill_view repeat-view dedup registry: per-task cache of (skill name, file_path) ->
(skill file mtime+size). A repeat view of an UNCHANGED file returns a short stub — the earlier
tool result already carries the content verbatim. Released via ``reset_skill_view_dedup()`` on
context compression AND on a committed proactive tool-result prune, because both can replace the
original content with a one-line marker; entries whose full result survives the rewrite are kept.
"""

import json
import os
import threading

_skill_view_tracker: dict[str, dict[tuple, tuple]] = {}
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


def _record_skill_view(task_id, name, file_path, payload: dict) -> None:
    """Record a served skill_view so an identical repeat can be deduped."""
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
        cache[key] = fp
        while len(cache) > _SKILL_VIEW_DEDUP_CAP:  # FIFO eviction
            del cache[next(iter(cache))]


def _check_skill_view_dedup(task_id, name, file_path) -> str | None:
    """Dedup stub when this exact skill file was already served to this task and
    is unchanged on disk; None otherwise."""
    if not task_id:
        return None
    n = str(name)
    with _skill_view_tracker_lock:
        if not (cache := _skill_view_tracker.get(str(task_id))):
            return None
        # Record key is the RESOLVED name; match raw and resolved forms so
        # 'category/skill' and bare-name views coalesce.
        for key, (src, mtime_ns, size) in list(cache.items()):
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


def _surviving_skill_view_keys(messages: list) -> set[tuple[str, str]]:
    """Dedup keys whose full skill_view result is still verbatim in *messages*."""
    keys: set[tuple[str, str]] = set()
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if msg.get("role") != "tool" or not isinstance(content, str) or '"_source_path"' not in content:
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("success") and payload.get("content") is not None:
            keys.add((str(payload.get("name") or ""), str(payload.get("file") or "")))
    return keys


def reset_skill_view_dedup(task_id: str | None = None, *, surviving_messages: list | None = None) -> None:
    """Clear the dedup cache (all tasks when task_id is None) after a transcript rewrite.
    With *surviving_messages*, keep the task's entries whose full result is still in them."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
            return
        cache = _skill_view_tracker.pop(str(task_id), None)
        if cache and surviving_messages:
            surviving = _surviving_skill_view_keys(surviving_messages)
            if kept := {key: fp for key, fp in cache.items() if key in surviving}:
                _skill_view_tracker[str(task_id)] = kept
