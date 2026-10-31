"""Atomic multi-op batch path for ``skill_manage``. Origin state
(``skill_manage``/``_find_skill``/``_skill_gate_bypass``) is reached lazily
through ``tools.skill_manager_tool`` so that module owns it."""

from contextlib import suppress
import json
import logging
import os
import posixpath
import shutil
import tempfile
from pathlib import Path

from utils import file_matches_identity, record_publications, unlink_if_unchanged

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
        if op.get("file_path") is not None and not isinstance(op["file_path"], str):
            return fail(i, f" ({act} on '{nm}'): file_path must be a string, got {type(op['file_path']).__name__}.")
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


def _published_under(log, directory) -> dict:
    """``{relative posix path: FileIdentity}`` of what the batch's publishes put under ``directory``.

    Matched on the real parent directory + file name (the file itself is never followed), because the
    publishing code and the batch may spell the same skill dir differently (symlinked skills root)."""
    root = os.path.realpath(directory)
    files = {}
    for path, identity in log.items():
        located = Path(os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path)))
        try:
            files[located.relative_to(root).as_posix()] = identity
        except ValueError:
            continue
    return files


def _undo_created(post_dir, files) -> tuple[list[str], list[str]]:
    """Undo a skill the batch created (or adopted as an empty leftover dir) -> ``(problems, kept)``.

    Only files the batch itself published are removed, and only while each is still byte-for-byte the file
    that publish left (``files`` holds the identity taken AT each publish, never re-read afterwards):
    another writer's SKILL.md, notes or edits are NEVER deleted (no ``rmtree``), because the per-skill lock
    binds only cooperative callers. Directories go only while empty. ``problems`` are batch files that could
    not be removed (rollback incomplete); ``kept`` is other people's data left in place (reported, not an
    error). Residual risk, see ``utils._remove_if``: no POSIX compare-and-delete exists, so a writer that
    replaces the batch's file inside the verify-to-unlink window of that one removal can still lose its data.
    """
    problems, kept = [], []
    for rel, identity in files.items():
        target = post_dir / rel
        removal = unlink_if_unchanged(target, identity)
        if removal.status == "failed":
            problems.append(f"could not remove '{target}' ({removal.error})"
                            + (f"; left as '{removal.preserved_at}'" if removal.preserved_at else ""))
        elif removal.status == "changed":
            kept.append(f"'{target}' (changed by another writer)"
                        + (f", kept as '{removal.preserved_at}'" if removal.preserved_at else ""))
    inner = {parent for rel in files for parent in (post_dir / rel).parents
             if parent != post_dir and parent.is_relative_to(post_dir)}
    for directory in sorted(inner, key=lambda d: len(d.parts), reverse=True):
        with suppress(OSError):
            directory.rmdir()  # rmdir, never rmtree: only an empty dir goes
    with suppress(OSError):
        post_dir.rmdir()
    if post_dir.exists():
        with suppress(OSError):
            accounted = {Path(rel).parts[0] for rel in files}  # already reported above, not "someone else's"
            leftovers = sorted(e.name for e in post_dir.iterdir() if e.name not in accounted)
            if leftovers:
                kept.append(f"'{post_dir}' (holds files this batch did not write: {', '.join(leftovers)})")
    return problems, kept


def _accounted_for(snap, displaced, published, path) -> bool:
    """True when ``path`` (a file in ``displaced``, the skill as the batch left it) is provably nobody else's:
    byte-identical to the snapshot's file, or exactly the file a batch publish left. A symlink, an unreadable
    file or anything else is NOT accounted for - it is kept rather than assumed to be ours."""
    rel = Path(path).relative_to(displaced).as_posix()
    try:
        counterpart = snap / rel
        if (not path.is_symlink() and counterpart.is_file() and not counterpart.is_symlink()
                and counterpart.read_bytes() == path.read_bytes()):
            return True
        identity = published.get(rel)
        return identity is not None and not path.is_symlink() and file_matches_identity(path, identity)
    except OSError:
        return False


def _drop_displaced(snap, displaced, published) -> list[str]:
    """Empty the displaced copy of a restored skill WITHOUT ``rmtree`` -> relative paths still left in it.

    A tree delete would take with it whatever a writer that ignores the skill lock saved into the tree after
    it was inspected. Instead each file is unlinked only while it is (re-checked right before the unlink)
    byte-identical to the snapshot or exactly a file the batch published, and each directory is removed with
    ``rmdir`` - which refuses a directory that is not empty. Anything else, including a file that appeared
    after the check, is therefore left where it is. Residual, not closed (no POSIX compare-and-delete): a
    process that already holds one of OUR files open can still write into it between that re-check and its
    ``unlink``, and those bytes go with the file."""
    for dirpath, dirs, names in os.walk(displaced, topdown=False):
        for filename in names:
            path = Path(dirpath) / filename
            if _accounted_for(snap, displaced, published, path):
                with suppress(OSError):
                    os.unlink(path)
        for directory in dirs:  # a symlink to a directory is listed here too; rmdir refuses it, so it stays
            with suppress(OSError):
                os.rmdir(Path(dirpath) / directory)
    with suppress(OSError):
        os.rmdir(displaced)
    left = []
    if displaced.exists():
        for dirpath, dirs, names in os.walk(displaced):
            linked = [d for d in dirs if (Path(dirpath) / d).is_symlink()]
            left += [(Path(dirpath) / n).relative_to(displaced).as_posix() for n in (*names, *linked)]
        left = sorted(left) or ["(empty directories)"]
    return left


class _RestoreFailed(Exception):
    """The snapshot could not be put back; the message says where the state the batch left is."""


def _restore_snapshot(pre_dir, snap, post_dir, published=None) -> list[str]:
    """Put a pre-existing skill back from its snapshot -> notes about data it preserved instead of dropping.

    Nothing here ever deletes a name another writer may own. The state the batch left is renamed aside under
    a UNIQUE name (an existing ``*.rollback-broken`` of someone else is never touched). The snapshot is copied
    to a private staging directory and published with ``os.rename``, which cannot replace a directory that
    holds anything: if a writer that ignores the skill lock created the skill dir meanwhile, the publish fails,
    that directory is left exactly as it is and the error names where the batch's state was kept. The displaced
    copy is then dropped with ``_drop_displaced`` (files proven ours, ``rmdir`` only); files nobody in this
    batch wrote are moved out to a fresh directory and named in the returned note, never deleted.
    """
    aside = None
    if post_dir is not None and post_dir.is_dir():
        aside = post_dir.with_name(f"{post_dir.name}.rollback-broken-{os.urandom(4).hex()}")
        post_dir.rename(aside)
    staging = pre_dir.with_name(f".{pre_dir.name}.restore-{os.urandom(4).hex()}")
    try:
        shutil.copytree(snap, staging)
        os.rename(staging, pre_dir)  # replaces at most an EMPTY directory; refuses anything that holds data
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)  # our own private copy under a name nobody else knows
        if aside is not None:
            try:
                os.rename(aside, pre_dir)  # put the half-applied state back rather than nothing
            except OSError as back:
                raise _RestoreFailed(f"{exc}; the skill name is now in use by something this batch did not write "
                                     f"({back}), left untouched; batch state preserved at '{aside}'") from exc
        raise
    if aside is None:
        return []
    unknown = _drop_displaced(snap, aside, published or {})
    if not unknown:
        return []
    keep = Path(tempfile.mkdtemp(prefix="skill_batch_preserved_"))
    try:
        os.rename(aside, keep / post_dir.name)  # same-filesystem rename: never a copy-then-delete
        where = keep / post_dir.name
    except OSError:  # could not move it out: leave it where it is, still not deleted
        with suppress(OSError):
            keep.rmdir()
        where = aside
    return [f"'{post_dir}' was restored from its snapshot, but files nobody in this batch wrote were preserved at "
            f"'{where}': {', '.join(unknown)}"]


def _rollback(snapshots, find_skill, log, create_targets, attempted):
    """Undo every skill an executed op touched; ``attempted`` names those (an op that never ran has nothing to
    undo, and its skill is never restored over whatever is there now). ``log`` holds what the batch published
    (``utils.record_publications``). Returns ``(note, failed)``: ``failed`` only when something the batch wrote
    could not be undone; other writers' files that were kept are reported in the note, not as a failure."""
    problems, kept = [], []
    for nm, (pre_dir, snap, dir_pre_existed) in snapshots.items():
        if nm not in attempted:
            continue
        try:
            if snap is None:
                post_dir = create_targets.get(nm)
                if post_dir is not None and post_dir.is_dir():
                    skill_problems, skill_kept = _undo_created(post_dir, _published_under(log, post_dir))
                    problems += [f"ROLLBACK INCOMPLETE for '{nm}': {text}" for text in skill_problems]
                    kept += skill_kept
                continue
            post = find_skill(nm)
            post_dir = Path(post["path"]) if post else None
            kept += _restore_snapshot(pre_dir, snap, post_dir, _published_under(log, post_dir) if post_dir else {})
        except Exception as exc:
            problems.append(f"ROLLBACK FAILED for '{nm}' ({exc})"
                            + (f"; snapshot preserved at '{snap}'" if snap is not None else ""))
    notes = list(problems)
    if kept:
        notes.append(("other writers' data preserved untouched: " if problems else
                      "rolled back everything this batch wrote; preserved other writers' data untouched: ")
                     + "; ".join(kept))
    return (("; ".join(notes) if notes else "all touched skills rolled back"), bool(problems))


_ADVISORY_KEYS = ("lint_warnings", "lint_hint")


def _skill_manage_batch(operations, default_name: str | None = None, task_id: str | None = None,
                        session_id: str | None = None) -> str:
    """Apply operations as a batch: every touched skill is snapshotted first and any
    failure rolls back every skill an executed op touched (batch-created skills are removed).
    The rollback is best effort against writers that do not take the skill lock: it deletes
    only files it can prove it published (identity taken at the publish), keeps and names
    anything else, and reports what it kept or could not undo instead of claiming a clean
    rollback. Not guaranteed: a restored pre-existing skill is put back to its snapshot, so
    an edit or addition an independent writer made to such a skill is reverted in place (the
    displaced copy is preserved and named, see ``_restore_snapshot``); and POSIX has no compare-and-delete,
    so one removal keeps a tiny verify-to-unlink window (``utils._remove_if``). ``delete`` is
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
    with _smt._skill_mutation_locks(names):
        snap_root = Path(tempfile.mkdtemp(prefix="skill_batch_"))
        create_targets = {names[i]: _smt._resolve_skill_dir(names[i], op.get("category"))
                          for i, op in enumerate(operations) if op.get("action") == "create"}
        snapshots, snap_err = _snapshot_skills(names, snap_root, _smt._find_skill, create_targets)
        if snap_err is not None:
            shutil.rmtree(snap_root, ignore_errors=True)
            return tool_error(snap_err, success=False)
        # Single-op path with the gate bypassed (the batch already cleared/staged it).
        results = []
        rollback_failed = False
        token = _smt._skill_gate_bypass.set(True)
        try:
            with record_publications() as log:  # {Path: FileIdentity} taken AT each publish, never re-read
                for i, op in enumerate(operations):
                    try:
                        raw = _smt._skill_manage_from({**op, "name": names[i], "operations": None},
                                                      task_id=task_id, session_id=session_id)
                    except Exception as exc:  # a raising op is a failed op: roll back, never strand partial state
                        logger.warning("skill_manage batch op %d (%s on '%s') raised", i, op["action"], names[i],
                                       exc_info=True)
                        raw = json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = {"success": False, "error": "unparseable op result"}
                    if not parsed.get("success"):
                        note, rollback_failed = _rollback(snapshots, _smt._find_skill, log, create_targets,
                                                          set(names[:i + 1]))
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
        finally:
            _smt._skill_gate_bypass.reset(token)
            if rollback_failed and any(snap is not None for _, snap, _ in snapshots.values()):
                # Keep the snapshots so the operator can still recover by hand.
                logger.warning("skill_manage batch rollback failed, snapshots kept at %s", snap_root)
            else:  # nothing to recover from the snapshot area (an incomplete undo of NEW skills is in the error)
                shutil.rmtree(snap_root, ignore_errors=True)
    # utf-8-sig + errors="replace": SKILL.md files are user-authored and sometimes carry a Notepad BOM or
    # stray non-UTF-8 bytes. Pinning UTF-8 with replacement keeps skill_view deterministic across platforms
    # — falling back to the machine locale (cp1252/GBK) would make the same skill render differently per
    # host (see PR #51701).
    return json.dumps(
        {"success": True, "operations_applied": len(results), "results": results},
        ensure_ascii=False)
