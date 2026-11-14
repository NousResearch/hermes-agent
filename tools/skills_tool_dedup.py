"""skill_view repeat-view dedup registry: per-task cache of (resolved skill name, file_path) ->
digest of the skill_view result served (or the body of an inlined invocation). A byte-identical
repeat returns a short stub — the earlier result or invocation carries the content verbatim.
Content digests need no file
timestamps, so a same-size rewrite in one coarse mtime tick (or on a host whose clock disagrees with
the filesystem's) is never mistaken for "unchanged". Cleared via ``reset_skill_view_dedup()`` on
context compression AND on a committed proactive tool-result prune, because both replace the
original content with a one-line marker.
"""

import hashlib
import json
import threading

from hermes_constants import hermes_home_key

_skill_view_tracker: dict[str, dict[tuple, tuple]] = {}
# Builders can know only a route/job ID, so wait for the actual turn to bind these.
# Retain bounded blocks for concurrent turns that inline identical skills.
_pending_inline_blocks: dict[tuple[str, str], tuple[tuple, tuple]] = {}
_skill_view_tracker_lock = threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200
_INLINE_BLOCK_CHAR_CAP = 4 * 1024 * 1024

_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (Re-issued after context compression, this "
    "returns the full content again.)")

_SKILL_INLINE_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded in the earlier invocation message "
    "in this conversation — refer to that message; it is still current and complete. "
    "After context compression, this returns the full content again.")


def _inline_fingerprint(payload: dict, content: str) -> tuple:
    return ("inline", payload.get("skill_dir") or payload.get("path"),
            hashlib.sha256(content.encode("utf-8")).hexdigest())


def _remember(cache: dict, key: tuple, fingerprint: tuple) -> None:
    cache.pop(key, None)
    cache[key] = fingerprint
    while len(cache) > _SKILL_VIEW_DEDUP_CAP:
        del cache[next(iter(cache))]


def register_inline_skill_block(block: str, payload: dict, content: str) -> None:
    """Keep a bounded record of the exact rendered block and its served body digest.

    Do not hash the JSON envelope or re-read disk: preprocessing and concurrent edits
    can make either differ from the body actually included in the invocation.
    """
    name = payload.get("name")
    if (not name or not payload.get("success") or payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"):
        return
    key = (str(name), "")
    pending_key = (hermes_home_key(), block)
    with _skill_view_tracker_lock:
        _remember(_pending_inline_blocks, pending_key, (key, _inline_fingerprint(payload, content)))
        while len(_pending_inline_blocks) > 1 and sum(len(k[1]) for k in _pending_inline_blocks) > _INLINE_BLOCK_CHAR_CAP:
            del _pending_inline_blocks[next(iter(_pending_inline_blocks))]


def bind_inline_skill_views(task_id: str | None, message: str) -> None:
    """Bind only blocks present in THIS turn, under the ID the tool executor will use."""
    from tools.skill_provenance import is_background_review

    if not task_id or is_background_review():
        return
    home = hermes_home_key()
    with _skill_view_tracker_lock:
        for pending_key, (key, fingerprint) in list(_pending_inline_blocks.items()):
            if pending_key[0] == home and pending_key[1] in message:
                cache = _skill_view_tracker.setdefault(str(task_id), {})
                _remember(cache, key, fingerprint)


def dedup_skill_view(task_id, name, file_path, payload: dict, result: str) -> str | None:
    """Stub a byte-identical result or an identical inlined SKILL.md body in this task;
    otherwise record ``result``'s digest and return None (serve the full result)."""
    # Never dedup setup-needed views: readiness depends on config/env state that
    # changes without the file changing; the model must see the refreshed status.
    if (not task_id or payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"):
        return None
    resolved = str(payload.get("name") or name)
    key = (resolved, file_path or "")
    fingerprint = ("view", hashlib.sha256(result.encode("utf-8")).hexdigest())
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.setdefault(str(task_id), {})
        previous = cache.get(key)
        inline_match = not file_path and previous == _inline_fingerprint(payload, str(payload.get("content") or ""))
        if inline_match or previous == fingerprint:
            return json.dumps({
                "success": True, "status": "unchanged", "name": resolved,
                "file": file_path or "SKILL.md", "dedup": True, "content_returned": False,
                "message": _SKILL_INLINE_DEDUP_MESSAGE if inline_match else _SKILL_VIEW_DEDUP_MESSAGE}, ensure_ascii=False)
        _remember(cache, key, fingerprint)
    return None


def reset_skill_view_dedup(task_id: str | None = None) -> None:
    """Clear the dedup cache (all tasks when task_id is None); called on context compression."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
            _pending_inline_blocks.clear()
        else:
            _skill_view_tracker.pop(str(task_id), None)
