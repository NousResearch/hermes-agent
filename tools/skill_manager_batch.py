"""Atomic multi-op batch path for ``skill_manage``. Origin state
(``skill_manage``/``_find_skill``/``_skill_gate_bypass``) is reached lazily
through ``tools.skill_manager_tool`` so that module owns it."""

import json
import logging
import posixpath
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger("tools.skill_manager_tool")

_BATCH_OP_ACTIONS = {"create", "patch", "write_file", "remove_file"}
_BATCH_MAX_OPS = 20
_ACTION_KEY_MAP = {
    "create": "create",
    "patch": "patch",
    "rewrite": "patch",
    "write_file": "write_file",
    "remove_file": "remove_file",
    "delete": "delete",
}
_ACTION_KEY_FIELDS = {
    "create": {"content", "category"},
    "patch": {"old_string", "new_string", "replace_all", "file_path"},
    "rewrite": {"content"},
    "write_file": {"file_path", "content"},
    "remove_file": {"file_path"},
    "delete": {"absorbed_into"},
}
_CROSS_ACTION_HINTS = {
    "file_content": "write_file.content",
    "content": "rewrite.content (or create.content/write_file.content)",
}


def _normalize_batch_operations(operations, tool_error):
    """Translate the advertised action-keyed shape to the legacy flat handler shape."""
    normalized = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict) or "action" in op:
            normalized.append(op)
            continue
        action_keys = [key for key in _ACTION_KEY_MAP if key in op]
        if len(action_keys) != 1:
            return None, tool_error(
                f"operations[{i}] must contain exactly one action key: "
                f"{', '.join(_ACTION_KEY_MAP)}.", success=False)
        key = action_keys[0]
        payload = op[key]
        if not isinstance(payload, dict):
            return None, tool_error(f"operations[{i}].{key} must be an object.", success=False)
        wrong_fields = set(payload) - _ACTION_KEY_FIELDS[key]
        if wrong_fields:
            field = sorted(wrong_fields)[0]
            hint = _CROSS_ACTION_HINTS.get(field)
            correction = f" Use {hint} instead." if hint else ""
            return None, tool_error(
                f"operations[{i}].{key} does not accept '{field}'.{correction}", success=False)
        unexpected = set(op) - {"name", key}
        if unexpected:
            return None, tool_error(
                f"operations[{i}] has fields outside its '{key}' object: "
                f"{', '.join(sorted(unexpected))}.", success=False)
        flat = {"name": op.get("name"), "action": _ACTION_KEY_MAP[key], **payload}
        if key == "write_file" and "content" in flat:
            flat["file_content"] = flat.pop("content")
        normalized.append(flat)
    return normalized, None


def _validate_batch_ops(operations, default_name, tool_error):
    """Shape checks with no side effects. Returns (names, None) or (None, error_json)."""
    from tools.skill_manager_guards import _background_review_preflight
    def fail(i, msg):
        return None, tool_error(f"operations[{i}]{msg}", success=False)
    names = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict) or not op.get("action"):
            return fail(i, " needs an 'action'.")
        act = op["action"]
        if act not in _BATCH_OP_ACTIONS:
            return fail(i, f": unknown action '{act}'. Batchable: "
                           f"{', '.join(sorted(_BATCH_OP_ACTIONS))}; delete must be sole.")
        nm = op.get("name") or default_name
        if not nm:
            return fail(i, " needs a 'name' (the skill it targets).")
        names.append(nm)
        if act == "create" and nm in names[:-1]:
            return fail(i, f": create for '{nm}' must precede that skill's other ops.")
        if (preflight := _background_review_preflight(act, nm)) is not None:
            return None, json.dumps(preflight, ensure_ascii=False)
    # Clobber guard: a DESTRUCTIVE op (create/write_file/remove_file/full rewrite) on
    # a file an earlier op touched would SILENTLY discard its work — reject it.
    # Additive patches are always legal. Paths are normalized against spelling variants.
    touched_files = set()
    for i, op in enumerate(operations):
        act, nm = op["action"], names[i]
        # create and full-rewrite patch (content) always hit SKILL.md.
        full_rewrite = act == "patch" and bool(op.get("content"))
        fp = (op.get("file_path") or "").strip()
        target = ("SKILL.md" if (act == "create" or full_rewrite or not fp)
                  else posixpath.normpath(fp.lstrip("/")))
        key = (nm, target)
        if (act in ("create", "write_file", "remove_file") or full_rewrite) and key in touched_files:
            return fail(i, f": {act} on '{target}' of skill '{nm}' — an earlier op in this "
                           f"batch already touched that file, and this op would silently discard its work. "
                           f"One destructive op (write_file/remove_file/full rewrite) per file per batch; put "
                           f"it first, or fold the change in. Patch chains are fine.")
        touched_files.add(key)
    return names, None


def _snapshot_skills(names, snap_root, find_skill):
    """Copy every touched skill aside. Returns (snapshots, None) or (None, error_text)."""
    snapshots = {}  # skill name -> (pre_dir or None, snapshot_dir or None)
    for nm in dict.fromkeys(names):  # ordered unique
        pre = find_skill(nm)
        pre_dir = Path(pre["path"]) if pre else None
        snap = snap_root / nm if pre_dir is not None and pre_dir.is_dir() else None
        if snap is not None:
            try:
                shutil.copytree(pre_dir, snap)
            except Exception as exc:  # noqa: BLE001 — no snapshot, no atomicity
                return None, f"Could not snapshot '{nm}' for atomic batch: {exc}"
        snapshots[nm] = (pre_dir, snap)
    return snapshots, None


def _restore_snapshot(pre_dir, snap, post_dir) -> None:
    post_exists = post_dir is not None and post_dir.is_dir()
    if snap is None:
        if post_exists:  # Batch created this skill: remove the partial result.
            shutil.rmtree(post_dir)
        return
    if not post_exists:
        shutil.copytree(snap, pre_dir)
        return
    # Move the broken state aside and delete it only after the snapshot is
    # back, so a failed copytree (disk full, locked file) can't mean total loss.
    aside = post_dir.with_name(post_dir.name + ".rollback-broken")
    shutil.rmtree(aside, ignore_errors=True)
    post_dir.rename(aside)
    try:
        shutil.copytree(snap, pre_dir)
    except Exception:
        # Restore failed: put the half-applied state back rather than nothing.
        shutil.rmtree(pre_dir, ignore_errors=True)
        aside.rename(pre_dir)
        raise
    shutil.rmtree(aside, ignore_errors=True)


def _rollback(snapshots, find_skill):
    """Restore every snapshot. Returns (note, failed)."""
    notes = []
    for nm, (pre_dir, snap) in snapshots.items():
        try:
            post = find_skill(nm)
            _restore_snapshot(pre_dir, snap, Path(post["path"]) if post else None)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"ROLLBACK FAILED for '{nm}' ({exc})"
                         + (f"; snapshot preserved at '{snap}'" if snap is not None else ""))
    return ("; ".join(notes) if notes else "all touched skills rolled back"), bool(notes)


def _skill_manage_batch(operations, default_name: str = None, task_id: str = None,
                        session_id: str = None) -> str:
    """Apply operations atomically: every touched skill is snapshotted first and any
    failure rolls ALL of them back (batch-created skills are removed). ``delete`` is
    only legal as the SOLE op (its recoverable-archive path doesn't compose with
    rollback) and routes to the single-op handler. ``default_name`` is the legacy
    top-level ``name`` fallback (staged replay)."""
    from tools import skill_manager_tool as _smt
    from tools.registry import tool_error
    if not isinstance(operations, list) or not operations:
        return tool_error("operations must be a non-empty array.", success=False)
    if len(operations) > _BATCH_MAX_OPS:
        return tool_error(f"operations is capped at {_BATCH_MAX_OPS} ops per call.", success=False)
    operations, normalize_error = _normalize_batch_operations(operations, tool_error)
    if normalize_error is not None:
        return normalize_error
    if any(isinstance(op, dict) and op.get("action") == "delete" for op in operations):
        if len(operations) != 1:
            return tool_error("delete must be the SOLE op in its call — it doesn't "
                              "compose with other ops' rollback.", success=False)
        nm = operations[0].get("name") or default_name
        if not nm:
            return tool_error("operations[0] (delete) needs a 'name'.", success=False)
        return _smt.skill_manage(action="delete", name=nm, task_id=task_id, session_id=session_id,
                                 absorbed_into=operations[0].get("absorbed_into"))
    names, err = _validate_batch_ops(operations, default_name, tool_error)
    if err is not None:
        return err
    if not _smt._skill_gate_bypass.get():
        # Approval gate for the WHOLE batch as one pending write.
        def _staging(wa):
            acts = ", ".join(op["action"] for op in operations)
            gist = f"batch({len(operations)} ops: {acts}) on {', '.join(sorted(set(names)))}"
            return {"action": "batch", "operations": operations}, gist
        staged = _smt._run_write_gate(_staging)
        if staged is not None:
            return staged
    snap_root = Path(tempfile.mkdtemp(prefix="skill_batch_"))
    snapshots, snap_err = _snapshot_skills(names, snap_root, _smt._find_skill)
    if snap_err is not None:
        shutil.rmtree(snap_root, ignore_errors=True)
        return tool_error(snap_err, success=False)
    # Single-op path with the gate bypassed (the batch already cleared/staged it).
    results = []
    rollback_failed = False
    token = _smt._skill_gate_bypass.set(True)
    try:
        for i, op in enumerate(operations):
            raw = _smt._skill_manage_from({**op, "name": names[i], "operations": None},
                                          task_id=task_id, session_id=session_id)
            try:
                parsed = json.loads(raw)
            except Exception:  # noqa: BLE001
                parsed = {"success": False, "error": "unparseable op result"}
            if not parsed.get("success"):
                note, rollback_failed = _rollback(snapshots, _smt._find_skill)
                fail = {  # key order is wire-visible
                    "success": False,
                    "error": (f"operations[{i}] ({op['action']} on '{names[i]}') failed: "
                              f"{parsed.get('error', 'unknown error')} — batch aborted, {note}."),
                    "failed_index": i, "completed_before_failure": i}
                # Carry the failing op's teaching payload (patch's file_preview /
                # fuzzy-match hints) through — without it the model recovers blind.
                for k, v in parsed.items():
                    if k not in ("success", "error") and v is not None:
                        fail.setdefault(k, v)
                return json.dumps(fail, ensure_ascii=False)
            results.append({"name": names[i], "action": op["action"],
                            "file_path": op.get("file_path"), "success": True})
    finally:
        _smt._skill_gate_bypass.reset(token)
        if rollback_failed:
            # Keep the snapshots so the operator can still recover by hand.
            logger.warning("skill_manage batch rollback failed, snapshots kept at %s", snap_root)
        else:
            shutil.rmtree(snap_root, ignore_errors=True)
    # utf-8-sig + errors="replace": SKILL.md files are user-authored and sometimes carry a Notepad BOM or
    # stray non-UTF-8 bytes. Pinning UTF-8 with replacement keeps skill_view deterministic across platforms
    # — falling back to the machine locale (cp1252/GBK) would make the same skill render differently per
    # host (see PR #51701).
    return json.dumps(
        {"success": True, "operations_applied": len(results), "results": results},
        ensure_ascii=False)
