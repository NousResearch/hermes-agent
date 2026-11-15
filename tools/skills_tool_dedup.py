"""skill_view repeat-view dedup registry: per-task cache of (resolved skill name, file_path) ->
digest of the skill_view result served. A repeat view whose result is byte-identical returns a short
stub — the earlier tool result already carries the content verbatim. Content digests need no file
timestamps, so a same-size rewrite in one coarse mtime tick (or on a host whose clock disagrees with
the filesystem's) is never mistaken for "unchanged". Cleared via ``reset_skill_view_dedup()`` on
context compression AND on a committed proactive tool-result prune, because both replace the
original content with a one-line marker.
"""

import hashlib
import json
import threading

_existing_skill_tracker = globals().get("_skill_view_tracker")
_skill_view_tracker: dict[str, dict[tuple, str]] = (
    _existing_skill_tracker if isinstance(_existing_skill_tracker, dict) else {}
)
_existing_skill_epoch_history = globals().get("_skill_view_epoch_history")
_skill_view_epoch_history: dict[str, dict[tuple, tuple[str, int, str]]] = (
    _existing_skill_epoch_history if isinstance(_existing_skill_epoch_history, dict) else {}
)
_skill_view_tracker_lock = globals().get("_skill_view_tracker_lock") or threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200
_SKILL_VIEW_CROSS_EPOCH_FULL_LOAD_LIMIT = 2

_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (Re-issued after context compression, this "
    "returns the full content again.)")


def _build_cross_epoch_skill_digest(payload: dict, resolved: str, file_path: str, epoch_loads: int) -> str:
    """Build a compact (~1.5 KB) structural digest of a skill after it has already been
    loaded in full at least ``_SKILL_VIEW_CROSS_EPOCH_FULL_LOAD_LIMIT`` times across compaction epochs."""
    raw_content = str(payload.get("content") or "")
    headings: list[str] = []
    key_lines: list[str] = []
    for line in raw_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            headings.append(stripped[:120])
        elif any(tok in stripped for tok in ("`", "python", "pytest", "gcloud", "roq", "SKILL.md", "references/")) and len(key_lines) < 12:
            key_lines.append(stripped[:140])
    summary_lines = []
    if headings:
        summary_lines.append("Headings: " + " | ".join(headings[:20]))
    if key_lines:
        summary_lines.append("Key commands/anchors:\n" + "\n".join(f"- {kl}" for kl in key_lines))
    else:
        summary_lines.append(raw_content[:1200])
    digest_body = "\n\n".join(summary_lines)[:1600]
    return json.dumps(
        {
            "success": True,
            "status": "cross_epoch_digest",
            "name": resolved,
            "file": file_path or "SKILL.md",
            "dedup": True,
            "content_returned": False,
            "epoch_load_count": epoch_loads,
            "linked_files": payload.get("linked_files") or {},
            "description": payload.get("description") or "",
            "content_digest": digest_body,
            "message": (
                f"Skill '{resolved}' ({file_path or 'SKILL.md'}) is unchanged and has already been loaded "
                f"in full {epoch_loads - 1} times across compaction epochs in this session. Returning "
                "compact structural digest to preserve context window and KV-cache."
            ),
        },
        ensure_ascii=False,
    )


def dedup_skill_view(task_id, name, file_path, payload: dict, result: str) -> str | None:
    """The "unchanged" stub when this task was already served exactly ``result`` for this skill
    file; otherwise record ``result``'s digest and return None (serve the full result)."""
    # Never dedup setup-needed views: readiness depends on config/env state that
    # changes without the file changing; the model must see the refreshed status.
    if (not task_id or payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"):
        return None
    tid = str(task_id)
    resolved = str(payload.get("name") or name)
    key = (resolved, file_path or "")
    digest = hashlib.sha256(result.encode("utf-8")).hexdigest()
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.setdefault(tid, {})
        if cache.get(key) == digest:
            return json.dumps({
                "success": True, "status": "unchanged", "name": resolved,
                "file": file_path or "SKILL.md", "dedup": True, "content_returned": False,
                "message": _SKILL_VIEW_DEDUP_MESSAGE}, ensure_ascii=False)
        epoch_map = _skill_view_epoch_history.setdefault(tid, {})
        prev_epoch = epoch_map.get(key)
        if prev_epoch is not None and prev_epoch[0] == digest:
            full_loads = prev_epoch[1] + 1
        else:
            full_loads = 1
        cache.pop(key, None)  # re-insert at the end: FIFO eviction keeps the recently served
        cache[key] = digest
        while len(cache) > _SKILL_VIEW_DEDUP_CAP:  # FIFO eviction
            del cache[next(iter(cache))]
        if full_loads > _SKILL_VIEW_CROSS_EPOCH_FULL_LOAD_LIMIT:
            compact_json = _build_cross_epoch_skill_digest(payload, resolved, file_path or "", full_loads)
            epoch_map[key] = (digest, full_loads, compact_json)
            return compact_json
        epoch_map.pop(key, None)
        epoch_map[key] = (digest, full_loads, "")
        while len(epoch_map) > _SKILL_VIEW_DEDUP_CAP:
            del epoch_map[next(iter(epoch_map))]
    return None


_JSON_DECODER = json.JSONDecoder()


def parse_unpruned_skill_view_payload(raw: str | None) -> tuple[dict, str] | None:
    """Parse the leading JSON object from a ``skill_view`` tool result string.

    Uses ``json.JSONDecoder().raw_decode`` rather than ``json.loads`` so trailing
    blocks appended by post-tool hooks (``[skill-error-router]...``) or tool-loop
    guardrails (``[Tool loop warning: ...]``) do not raise ``JSONDecodeError: Extra data``.
    Returns ``(payload, json_prefix)`` for full unpruned skill/reference bodies, or ``None``.
    """
    if not isinstance(raw, str):
        return None
    stripped = raw.lstrip()
    if not stripped.startswith("{") or '"success": true' not in stripped[:120]:
        return None
    try:
        payload, end = _JSON_DECODER.raw_decode(stripped)
    except Exception:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("success") is not True
        or payload.get("dedup")
        or payload.get("setup_needed")
        or payload.get("readiness_status") == "setup_needed"
        or not isinstance(payload.get("content"), str)
        or not ("skill_dir" in payload or "linked_files" in payload or "file_type" in payload)
    ):
        return None
    return payload, stripped[:end]


def seed_skill_view_dedup_from_messages(task_id: str | None, messages: list | None) -> None:
    """Populate the per-task dedup cache from unpruned ``skill_view`` tool results in *messages*.

    Called after compaction or proactive prune so skills whose full JSON bodies survived in the
    protected tail stay deduplicated, while skills pruned to ``[SKILL_PRUNED: ...]`` markers are
    absent and serve full content on reload.
    """
    if not task_id or not isinstance(messages, list):
        return
    entries: list[tuple[tuple[str, str], str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "tool":
            parsed = parse_unpruned_skill_view_payload(msg.get("content"))
            if parsed is None:
                continue
            payload, json_prefix = parsed
            resolved = str(payload.get("name") or "")
            if not resolved:
                continue
            key = (resolved, str(payload.get("file") or ""))
            digest = hashlib.sha256(json_prefix.encode("utf-8")).hexdigest()
            entries.append((key, digest))
        elif role == "user":
            raw_user = msg.get("content")
            if not isinstance(raw_user, str) or "[IMPORTANT: The " not in raw_user[:64]:
                continue
            try:
                from agent.skill_commands import skills_loaded_by_scaffold
                from agent.skill_utils import normalize_skill_lookup_name
                from tools.skills_tool import skill_view
                for sname in skills_loaded_by_scaffold(raw_user):
                    norm_name = normalize_skill_lookup_name(sname)
                    sv_raw = skill_view(norm_name, task_id=task_id, preprocess=True)
                    parsed_sv = parse_unpruned_skill_view_payload(sv_raw)
                    if parsed_sv is None:
                        continue
                    sv_payload, sv_prefix = parsed_sv
                    sv_body = str(sv_payload.get("content") or "").strip()
                    if not sv_body or sv_body not in raw_user:
                        continue
                    resolved = str(sv_payload.get("name") or norm_name)
                    if resolved:
                        entries.append(((resolved, ""), hashlib.sha256(sv_prefix.encode("utf-8")).hexdigest()))
            except Exception:
                continue
    if not entries:
        return
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.setdefault(str(task_id), {})
        for key, digest in entries:
            cache.pop(key, None)
            cache[key] = digest
            while len(cache) > _SKILL_VIEW_DEDUP_CAP:
                del cache[next(iter(cache))]


def reset_skill_view_dedup(
    task_id: str | None = None, *, surviving_messages: list | None = None
) -> None:
    """Clear the dedup cache (all tasks when task_id is None); called on context compression.
    When *surviving_messages* is provided, re-seeds unpruned ``skill_view`` results that survived
    in the kept tail so redundant reloads of still-live skills return the unchanged stub."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
            _skill_view_epoch_history.clear()
        else:
            existing = _skill_view_tracker.get(str(task_id))
            if isinstance(existing, dict):
                existing.clear()
    if task_id is not None and surviving_messages:
        seed_skill_view_dedup_from_messages(task_id, surviving_messages)


def migrate_skill_view_state(old_task_id: str | None, new_task_id: str | None) -> None:
    """Carry skill_view dedup and cross-epoch load history across session rotation.
    Shares the underlying dicts across ``old_tid`` and ``new_tid`` so mid-turn calls
    bound to ``effective_task_id`` and subsequent turns bound to ``new_tid`` stay in sync."""
    if not old_task_id or not new_task_id:
        return
    old_tid = str(old_task_id)
    new_tid = str(new_task_id)
    if old_tid == new_tid:
        return
    with _skill_view_tracker_lock:
        old_cache = _skill_view_tracker.get(old_tid)
        dst_cache = _skill_view_tracker.setdefault(new_tid, {})
        if old_cache and old_cache is not dst_cache:
            for k, v in old_cache.items():
                dst_cache.setdefault(k, v)
            for k, v in list(_skill_view_tracker.items()):
                if v is old_cache:
                    _skill_view_tracker[k] = dst_cache
        else:
            _skill_view_tracker[old_tid] = dst_cache
        old_epochs = _skill_view_epoch_history.get(old_tid)
        dst_epochs = _skill_view_epoch_history.setdefault(new_tid, {})
        if old_epochs and old_epochs is not dst_epochs:
            for k, (digest, loads, compact_json) in old_epochs.items():
                prev = dst_epochs.get(k)
                if prev is None or (prev[0] == digest and loads > prev[1]):
                    dst_epochs[k] = (digest, loads, compact_json)
            for k, v in list(_skill_view_epoch_history.items()):
                if v is old_epochs:
                    _skill_view_epoch_history[k] = dst_epochs
        else:
            _skill_view_epoch_history[old_tid] = dst_epochs


def clear_skill_view_state(task_id: str | None = None) -> None:
    """Release both current-epoch and cross-epoch skill_view state when a session/task closes."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
            _skill_view_epoch_history.clear()
        else:
            tid = str(task_id)
            _skill_view_tracker.pop(tid, None)
            _skill_view_epoch_history.pop(tid, None)
