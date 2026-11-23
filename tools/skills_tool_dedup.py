"""skill_view repeat-view dedup registry: per-task cache of (resolved skill name, file_path) ->
digest of the skill_view result served. A repeat view whose result is byte-identical returns a short
stub — the earlier tool result already carries the content verbatim. Content digests need no file
timestamps, so a same-size rewrite in one coarse mtime tick (or on a host whose clock disagrees with
the filesystem's) is never mistaken for "unchanged". Cleared via ``reset_skill_view_dedup()`` on
context compression AND on a committed proactive tool-result prune, because both replace the
original content with a one-line marker. Each entry also keeps the tool call that served it, so a
view the model received as anything but that whole result (a spill preview, a guardrail reference
stub) is released by that call (``release_replaced_skill_view``); a notice appended after the whole
result keeps it.
"""

import hashlib
import json
import threading

_skill_view_tracker: dict[str, dict[tuple, tuple[str, str, int]]] = {}
_skill_view_tracker_lock = threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200

_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (Re-issued after context compression, this "
    "returns the full content again.)")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_skill_view(task_id, name, file_path, payload: dict, result: str) -> str | None:
    """The "unchanged" stub when this task was already served exactly ``result`` for this skill
    file; otherwise record ``result``'s digest and return None (serve the full result)."""
    # Never dedup setup-needed views: readiness depends on config/env state that
    # changes without the file changing; the model must see the refreshed status.
    if (not task_id or payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"):
        return None
    from tools.approval_context import _approval_tool_call_id

    resolved = str(payload.get("name") or name)
    key = (resolved, file_path or "")
    digest = _digest(result)
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.setdefault(str(task_id), {})
        if (entry := cache.get(key)) and entry[0] == digest:
            return json.dumps({
                "success": True, "status": "unchanged", "name": resolved,
                "file": file_path or "SKILL.md", "dedup": True, "content_returned": False,
                "message": _SKILL_VIEW_DEDUP_MESSAGE}, ensure_ascii=False)
        cache.pop(key, None)  # re-insert at the end: FIFO eviction keeps the recently served
        cache[key] = (digest, _approval_tool_call_id.get(), len(result))
        while len(cache) > _SKILL_VIEW_DEDUP_CAP:  # FIFO eviction
            del cache[next(iter(cache))]
    return None


def release_replaced_skill_view(task_id, tool_call_id, delivered) -> None:
    """Forget the view ``tool_call_id`` recorded unless ``delivered`` starts with that whole result: a
    spill preview or a reference stub in its place leaves the model without the content a later stub
    would point at, while a guardrail notice appended after it does not."""
    if not tool_call_id:
        return
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.get(str(task_id)) or {}
        key = next((key for key, (_, call_id, _) in cache.items() if call_id == tool_call_id), None)
        if key is None:
            return
        digest, _, length = cache[key]
        if not isinstance(delivered, str) or _digest(delivered[:length]) != digest:
            del cache[key]


def release_replaced_skill_views(task_id, batch: list[dict], contents_before: list) -> None:
    """``release_replaced_skill_view`` for each row of a batch whose content the turn budget replaced."""
    for message, before in zip(batch, contents_before):
        if message.get("content") is not before:
            release_replaced_skill_view(task_id, message.get("tool_call_id"), message.get("content"))


def reset_skill_view_dedup(task_id: str | None = None) -> None:
    """Clear the dedup cache (all tasks when task_id is None); called on context compression."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
        else:
            _skill_view_tracker.pop(str(task_id), None)
