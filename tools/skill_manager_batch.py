"""Atomic multi-op batch path for ``skill_manage``. Origin state
(``skill_manage``/``_find_skill``/``_skill_gate_bypass``) is reached lazily
through ``tools.skill_manager_tool`` so that module owns it."""

from contextlib import contextmanager, suppress
import json
import logging
import os
import posixpath
import shutil
import stat
import tempfile
from pathlib import Path

from tools import skill_manager_guards as _guards

logger = logging.getLogger("tools.skill_manager_tool")

_BATCH_OP_ACTIONS = {"create", "patch", "write_file", "remove_file"}
_BATCH_MAX_OPS = 20

# --- Per-op argument shape (checked before any effect) ---------------------------------
# action -> (arg, is_missing, error) checks run before the handler.
_MISSING, _IS_NONE = (lambda v: not v), (lambda v: v is None)
_REQUIRED_ARGS = {
    "create": [("content", _MISSING,
                "content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).")],
    "edit": [("content", _MISSING,
              "content is required for a full rewrite. Provide the full updated SKILL.md text.")],
    "write_file": [
        ("file_path", _MISSING, "file_path is required for 'write_file'. Example: 'references/api-guide.md'"),
        ("file_content", _IS_NONE, "file_content is required for 'write_file'.")],
    "remove_file": [("file_path", _MISSING, "file_path is required for 'remove_file'.")]}
# A bare "required" error is a dead end: the model retries blindly and often escapes to
# action='write_file', clobbering the whole file.
_PATCH_NEEDS_OLD_STRING = (
    "old_string is required for 'patch' and must be the EXACT text currently in the file. "
    "Read the target file first (read_file on the skill's SKILL.md, or the file named by "
    "file_path) and copy the snippet verbatim, then retry 'patch'. Do NOT fall back to "
    "action='write_file' — that rewrites the entire file and destroys unrelated content.")
_PATCH_NEEDS_NEW_STRING = "new_string is required for 'patch'. Use an empty string to delete matched text."
_PATCH_EITHER_OR = ("Pass EITHER content (full SKILL.md rewrite) OR old_string/new_string "
                    "(targeted replacement), not both.")
# Text-slot keys a model confuses: key -> the action that reads it. A 27B model that just
# used write_file's file_content re-emits it on create/patch and then replays the identical
# payload when the error only says the right key is "required" — the hint has to name where
# the text actually landed so the retry can move it.
_TEXT_SLOT_OWNER = {"content": "create (and a full-rewrite patch)",
                    "new_string": "a targeted patch (with old_string)",
                    "file_content": "write_file"}
# action -> (text slots it reads, where misfiled text belongs)
_TEXT_SLOT_FOR = {
    "create": (("content",), "'content'"),
    "edit": (("content",), "'content'"),
    "patch": (("content", "new_string"), "old_string/new_string (targeted) or 'content' (full rewrite, last resort)"),
    "write_file": (("file_content",), "'file_content'")}


def _misplaced_text_hint(action: str, args: dict) -> str:
    """Sentence naming the text-slot key(s) this op carries that ``action`` never reads, or ''."""
    if action not in _TEXT_SLOT_FOR:
        return ""  # delete/remove_file/unknown: no text slot, so no destination to point at
    reads, destination = _TEXT_SLOT_FOR[action]
    stray = [k for k in _TEXT_SLOT_OWNER if k not in reads and args.get(k) is not None]
    if not stray:
        return ""
    carried = " and ".join(f"'{k}' (that key is for {_TEXT_SLOT_OWNER[k]})" for k in stray)
    return f" Note: this op carries {carried} — move that text to {destination}."


def _op_shape_error(action: str, args: dict):
    """Argument-shape error text for one op, or None. Only shape misses carry the misplaced-text
    hint: a patch whose real problem is an unmatched old_string must not be steered to a full
    rewrite. Pure function so the batch can reject a misfiled op BEFORE any sibling is applied."""
    for arg, missing, message in _REQUIRED_ARGS.get(action, ()):
        if missing(args.get(arg)):
            return message + _misplaced_text_hint(action, args)
    if action == "patch":
        # Every patch shape miss is decided here, not in the handler, so a batch never applies
        # op[0] only to roll it back over op[1]'s missing new_string or content+old_string mix.
        if args.get("content") and (args.get("old_string") or args.get("new_string") is not None):
            return _PATCH_EITHER_OR
        if not args.get("old_string") and not args.get("content"):
            return _PATCH_NEEDS_OLD_STRING + _misplaced_text_hint(action, args)
        if not args.get("content") and args.get("new_string") is None:
            return _PATCH_NEEDS_NEW_STRING
    return None


def _validate_batch_ops(operations, default_name, tool_error):
    """Shape checks with no side effects. Returns (names, None) or (None, error_json)."""
    from tools.skill_manager_guards import _background_review_preflight
    from tools.skill_manager_tool import _validate_category
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
        # Reject a misfiled op here, before any sibling is applied: a runtime failure on
        # op[1] would first apply op[0] and then roll the whole batch back.
        if (shape_err := _op_shape_error(act, op)) is not None:
            return fail(i, f" ({act} on '{nm}'): {shape_err}")
        # create's category is resolved to a target dir before the snapshot: reject a bad one here
        # so it returns a JSON error (not a TypeError) and never leaks the snapshot tempdir.
        if act == "create" and (cat_err := _validate_category(op.get("category"))) is not None:
            return fail(i, f" ({act} on '{nm}'): {cat_err}")
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


def _snapshot_skills(names, snap_root, find_skill, create_targets):
    """Copy every touched skill aside. Returns (snapshots, None) or (None, error_text).

    ``create_targets`` maps a name with no skill yet to the dir its ``create`` op will use.
    An EMPTY pre-existing dir there has no SKILL.md to snapshot, yet create adopts it (see
    ``_create_skill``): record that it pre-dated the batch so rollback never rmtree()s it."""
    snapshots = {}  # skill name -> (pre_dir or None, snapshot_dir or None, dir_pre_existed)
    for nm in dict.fromkeys(names):  # ordered unique
        pre = find_skill(nm)
        pre_dir = Path(pre["path"]) if pre else None
        snap = snap_root / nm if pre_dir is not None and pre_dir.is_dir() else None
        if snap is not None:
            try:
                shutil.copytree(pre_dir, snap)
            except Exception as exc:
                return None, f"Could not snapshot '{nm}' for atomic batch: {exc}"
        target = create_targets.get(nm) if pre is None else None
        snapshots[nm] = (pre_dir, snap, target is not None and target.is_dir())
    return snapshots, None


def _restore_snapshot(pre_dir, snap, post_dir, dir_pre_existed=False, written=()) -> None:
    post_exists = post_dir is not None and post_dir.is_dir()
    if snap is None:
        if not post_exists:
            return
        if not dir_pre_existed:  # Batch created this skill: remove the partial result.
            shutil.rmtree(post_dir)
            return
        # The dir predates the batch (adopted empty leftover): unlink exactly the files the
        # batch wrote there, then rmdir() the now-empty dirs up to and including the skill dir.
        # rmdir() fails on anything left, so a file that landed out-of-band survives.
        for rel in written:  # only what the batch recorded writing; anything else there is not its own
            target = post_dir / rel
            target.unlink(missing_ok=True)  # any other failure propagates: the rollback did not finish
            for parent in target.parents:
                if parent == post_dir or not parent.is_relative_to(post_dir):
                    break
                with suppress(OSError):
                    parent.rmdir()
        with suppress(OSError):
            post_dir.rmdir()
        return
    if not post_exists:
        shutil.copytree(snap, pre_dir)
        return
    # Move the broken state aside and delete it only after the snapshot is
    # back, so a failed copytree (disk full, locked file) can't mean total loss.
    # The aside folder is created fresh for this rollback (never a fixed name), so the
    # only thing deleted is what this rollback moved there.
    holder = Path(tempfile.mkdtemp(prefix=f".{post_dir.name}.rollback-", dir=post_dir.parent))
    aside = holder / post_dir.name
    try:
        post_dir.rename(aside)
    except BaseException:
        with suppress(OSError):  # nothing was moved: the holder is still empty
            holder.rmdir()
        raise
    try:
        shutil.copytree(snap, pre_dir)
    except Exception:
        # Restore failed: put the half-applied state back rather than nothing.
        shutil.rmtree(pre_dir, ignore_errors=True)
        aside.rename(pre_dir)
        with suppress(OSError):
            holder.rmdir()
        raise
    _remove_aside(holder)


def _remove_aside(holder) -> None:
    """Delete a rollback's aside holder, read-only folders inside it included (the moved-aside tree is the
    batch's broken copy of the skill). A holder that still cannot be removed fails the rollback, so the
    batch reports it and keeps its snapshots instead of claiming a clean rollback."""
    for folder, _dirs, _files in os.walk(holder):
        with suppress(OSError):
            os.chmod(folder, os.stat(folder).st_mode | stat.S_IRWXU)
    try:
        shutil.rmtree(holder)
    except OSError as exc:
        raise OSError(f"skill restored, but the aside folder '{holder}' could not be removed: {exc}") from exc


def _snapshots_by_folder(snapshots):
    """``{resolved folder: (folder, snapshot)}`` for every skill found when the batch started: the
    folders the batch owns from the start. Two names finding one folder share one entry."""
    by_folder = {}
    for pre_dir, snap, _existed in snapshots.values():
        if pre_dir is not None and snap is not None:
            by_folder.setdefault(Path(pre_dir).resolve(), (Path(pre_dir), snap))
    return by_folder


def _unowned_target(find_skill, name, action, owned):
    """``(reason, folder)``: why an op must not run ("" if it may) and the resolved folder it acts in
    (None for a ``create`` or a name that finds nothing). The batch can only undo what happens in
    folders it owns (``owned``: those snapshotted at the start and those its creates made), so an op
    may act only there. A name that finds another folder by now (a removed local SKILL.md revealing
    a same-named skill in another root, an ambiguous name made unique) is refused."""
    if action == "create" or (found := find_skill(name)) is None:
        return "", None  # nothing found: the op fails with its usual error, nothing to protect
    folder = Path(found["path"]).resolve()
    if folder in owned:
        return "", folder
    return (f"'{name}' now resolves to {found['path']}, a folder this batch did not snapshot or "
            f"create; an earlier op changed which skill the name finds"), None


def _rollback(by_folder, journal):
    """Undo only what this batch did, folder by folder, from its journal (``_guards._BatchJournal``:
    what the batch's own execution wrote and created, recorded where it happened). A folder a create
    made is removed, or for an adopted empty leftover dir emptied of the files written there. A
    snapshotted folder the batch wrote is restored from its snapshot; one it never wrote is left alone
    (restoring it could only undo someone else's write). A folder the batch neither snapshotted nor
    created (a hook writing another existing skill) is not the batch's to undo. Names are never resolved
    again: by now they can find another root's skill. Returns (note, failed)."""
    notes = []
    undo = [(f, None, f, adopted) for f, adopted in journal.created.items() if f not in by_folder]
    undo += [(f, snap, pre_dir, True) for f, (pre_dir, snap) in by_folder.items() if f in journal.writes]
    for folder, snap, pre_dir, adopted in undo:
        written = sorted(posixpath.normpath(rel) for rel in journal.writes.get(folder, ()))
        try:
            _restore_snapshot(pre_dir, snap, pre_dir, adopted, written)
        except Exception as exc:
            notes.append(f"ROLLBACK FAILED for '{pre_dir}' ({exc})"
                         + (f"; snapshot preserved at '{snap}'" if snap is not None else ""))
    return ("; ".join(notes) if notes else "all touched skills rolled back"), bool(notes)


@contextmanager
def _write_journal():
    """The batch's journal (``_guards._batch_journal``), set for everything the batch runs. A batch run
    from inside another batch's lifecycle hook hands its journal on, so the enclosing batch can undo
    that work too."""
    outer, journal = _guards._batch_journal.get(), _guards._BatchJournal()
    token = _guards._batch_journal.set(journal)
    try:
        yield journal
    finally:
        _guards._batch_journal.reset(token)
        if outer is not None:
            journal.merge_into(outer)


def _run_op(smt, op, name, folder, refusal, create_target, task_id, session_id):
    """Run one op through the single-op handler, bound to the folder the batch checked (a create to the
    folder planned), and return its parsed result. ``refusal``: why the op must not run at all."""
    if refusal:
        return {"success": False, "error": refusal}
    bound = Path(create_target).resolve() if op["action"] == "create" else (folder or _guards._BOUND_NOTHING)
    bind = _guards._batch_bound_folder.set((name, bound))
    try:
        raw = smt._skill_manage_from({**op, "name": name, "operations": None}, task_id=task_id, session_id=session_id)
    finally:
        _guards._batch_bound_folder.reset(bind)
    try:
        return json.loads(raw)
    except (TypeError, ValueError):  # not a JSON string
        return {"success": False, "error": "unparseable op result"}


_ADVISORY_KEYS = ("lint_warnings", "lint_hint")


def _skill_manage_batch(operations, default_name: str | None = None, task_id: str | None = None,
                        session_id: str | None = None) -> str:
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
    # Every target's lock is held from the snapshot through commit or rollback; the per-op
    # skill_manage() calls re-enter them. Without the outer fence a concurrent writer landing
    # between the snapshot and a rollback would be silently reverted.
    with _smt._skill_mutation_locks(names), _write_journal() as journal:
        snap_root = Path(tempfile.mkdtemp(prefix="skill_batch_"))
        create_targets = {names[i]: _smt._resolve_skill_dir(names[i], op.get("category"))
                          for i, op in enumerate(operations) if op.get("action") == "create"}
        snapshots, snap_err = _snapshot_skills(names, snap_root, _smt._find_skill, create_targets)
        if snap_err is not None:
            shutil.rmtree(snap_root, ignore_errors=True)
            return tool_error(snap_err, success=False)
        by_folder = _snapshots_by_folder(snapshots)
        # Single-op path with the gate bypassed (the batch already cleared/staged it).
        results = []
        # The snapshots are deleted only once the skills are known consistent: after every op succeeded,
        # or after a rollback that completed. Any other exit (a failed rollback, an interrupt anywhere,
        # including during a rollback) keeps them for recovery by hand.
        consistent = False
        token = _smt._skill_gate_bypass.set(True)
        try:
            for i, op in enumerate(operations):
                owned = set(by_folder) | set(journal.created)  # the folders the batch can undo
                moved, folder = _unowned_target(_smt._find_skill, names[i], op["action"], owned)
                parsed = _run_op(_smt, op, names[i], folder, moved, create_targets.get(names[i]),
                                 task_id, session_id)
                if not parsed.get("success"):
                    note, rollback_failed = _rollback(by_folder, journal)
                    consistent = not rollback_failed
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
                entry = {"name": names[i], "action": op["action"],
                         "file_path": op.get("file_path"), "success": True}
                # Advisory payloads (linter findings, org-sharing note) ride on the op result; the
                # compact success row otherwise hides them and the model never sees a finding.
                entry.update({k: parsed[k] for k in _ADVISORY_KEYS if parsed.get(k) is not None})
                results.append(entry)
            consistent = True
        except Exception:  # an op raised: undo what the batch did, then let the error through
            _note, rollback_failed = _rollback(by_folder, journal)
            consistent = not rollback_failed
            raise
        finally:
            _smt._skill_gate_bypass.reset(token)
            if consistent:
                shutil.rmtree(snap_root, ignore_errors=True)
            else:  # keep the snapshots so the operator can still recover by hand
                logger.warning("skill_manage batch did not finish cleanly (rollback failed or "
                               "interrupted), snapshots kept at %s", snap_root)
    # utf-8-sig + errors="replace": SKILL.md files are user-authored and sometimes carry a Notepad BOM or
    # stray non-UTF-8 bytes. Pinning UTF-8 with replacement keeps skill_view deterministic across platforms
    # — falling back to the machine locale (cp1252/GBK) would make the same skill render differently per
    # host (see PR #51701).
    return json.dumps(
        {"success": True, "operations_applied": len(results), "results": results},
        ensure_ascii=False)
