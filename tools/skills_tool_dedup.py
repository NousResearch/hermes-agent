"""skill_view repeat-view dedup registry: per-task cache of (skill name, file_path) ->
(skill file mtime+size). A repeat view of an UNCHANGED file returns a short stub — the earlier
tool result already carries the content verbatim. Cleared on context compression via
``reset_skill_view_dedup()`` because the original content is summarized away.

Ghost tracking (#32114): proactive prune passes can demote a skill_view result WITHOUT a
compaction boundary, so ``reset_skill_view_dedup()`` never runs and the cache keeps serving
"unchanged" stubs for a body that now exists only as a prune marker. Compression records such
skills via ``_mark_ghosted_skill_views()``; the dedup check then treats them as never-viewed
(self-healing on the next skill_view call) until a fresh full view re-records them.
"""

import json
import os
import threading
from typing import Dict

_skill_view_tracker: Dict[str, Dict[tuple, tuple]] = {}
_skill_view_tracker_lock = threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200
# Demoted-skill ghosts (lower-cased, insertion-ordered for FIFO bounding).
_ghosted_skill_names: Dict[str, None] = {}
_GHOST_CAP = 200

_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (Re-issued after context compression, this "
    "returns the full content again.)")


def _mark_ghosted_skill_views(names) -> None:
    """Record skills whose full body was demoted out of the transcript (#32114)."""
    cleaned = [str(n or "").strip().lower() for n in (names or [])]
    cleaned = [n for n in cleaned if n]
    if not cleaned:
        return
    with _skill_view_tracker_lock:
        for n in cleaned:
            _ghosted_skill_names[n] = None
        while len(_ghosted_skill_names) > _GHOST_CAP:
            _ghosted_skill_names.pop(next(iter(_ghosted_skill_names)))


def _is_ghosted_skill_view(name: str) -> bool:
    """True when *name* (or a category/plugin variant of it) is currently ghosted."""
    n = str(name or "").strip().lower()
    if not n:
        return False
    with _skill_view_tracker_lock:
        if n in _ghosted_skill_names:
            return True
        base = n.split(":")[-1]
        for ghost in _ghosted_skill_names:
            if base == ghost or base.endswith("/" + ghost) or ghost.endswith("/" + base):
                return True
    return False



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
        # A fresh full view re-materializes the body in the transcript: unghost it (#32114).
        _ghosted_skill_names.pop(str(payload.get("name") or name).strip().lower(), None)
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
    # #32114 self-heal: a ghosted skill's body was demoted out of the transcript
    # by a no-boundary prune, so an "unchanged" stub would point at a marker.
    # Treat it as never-viewed and drop any stale cache entry for it.
    if _is_ghosted_skill_view(n):
        with _skill_view_tracker_lock:
            cache = _skill_view_tracker.get(str(task_id)) or {}
            for key in [k for k in cache if k[0] == n or k[0].split(":")[-1] == n.split(":")[-1]]:
                cache.pop(key, None)
        return None
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


def reset_skill_view_dedup(task_id: str | None = None) -> None:
    """Clear the dedup cache (all tasks when task_id is None); called on context compression."""
    with _skill_view_tracker_lock:
        if task_id is None:
            # Full boundary: cache and ghost state reset together.
            _ghosted_skill_names.clear()
            _skill_view_tracker.clear()
        else:
            # Task-scoped reset (a sibling in-process task crossed a boundary): keep ghost
            # flags. They only force one extra full re-read (harmless); dropping them while
            # another task's cache still holds entries would resurrect the #32114 deadlock
            # for that task.
            _skill_view_tracker.pop(str(task_id), None)
