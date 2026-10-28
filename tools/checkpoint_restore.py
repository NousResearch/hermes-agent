"""Restore, diff and rollback planning plus the agent-write ledger for checkpoints.

The restore side of ``tools.checkpoint_manager``: validating a commit hash and
a restore path, classifying a safe restore against the agent-write ledger
(``record_agent_write`` / ``_safe_restore_plan``), diffing a checkpoint against
the working tree, and restoring files without clobbering user edits.

The ledger records, per project, the sha256 of every file Hermes last wrote, so
a safe restore can tell "Hermes wrote this" apart from "the user hand-edited it
afterwards".

Seam-sensitive names (``_run_git``, ``_resolve_checkpoint_base``,
``_staging_args``, ``_checkpoint_base_rel_path``) stay defined in
``checkpoint_manager``; the delegates at the bottom resolve them at call
time, so ``monkeypatch.setattr(checkpoint_manager, ...)`` is still observed
here.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Optional

from tools.checkpoint_store import (
    _GIT_TIMEOUT,
    _LEDGERS_DIRNAME,
    _gitlink_paths,
    _index_path,
    _normalize_path,
    _project_hash,
    _ref_name,
    _store_path,
)

logger = logging.getLogger(__name__)


# Agent-write ledger cap: newest entries retained per project.
_LEDGER_MAX_ENTRIES = 2000

# Valid git commit hash pattern: 4–40 hex chars (short or full SHA-1/SHA-256).
_COMMIT_HASH_RE = re.compile(r'^[0-9a-fA-F]{4,64}$')


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_commit_hash(commit_hash: str) -> Optional[str]:
    """Validate a commit hash to prevent git argument injection.

    Returns an error string if invalid, None if valid.
    Values starting with '-' would be interpreted as git flags
    (e.g., '--patch', '-p') instead of revision specifiers.
    """
    if not commit_hash or not commit_hash.strip():
        return "Empty commit hash"
    if commit_hash.startswith("-"):
        return f"Invalid commit hash (must not start with '-'): {commit_hash!r}"
    if not _COMMIT_HASH_RE.match(commit_hash):
        return f"Invalid commit hash (expected 4-64 hex characters): {commit_hash!r}"
    return None


def _validate_file_path(file_path: str, working_dir: str) -> Optional[str]:
    """Validate a file path to prevent path traversal outside the working directory.

    Returns an error string if invalid, None if valid.
    """
    if not file_path or not file_path.strip():
        return "Empty file path"
    if os.path.isabs(file_path):
        return f"File path must be relative, got absolute path: {file_path!r}"
    abs_workdir = _normalize_path(working_dir)
    resolved = (abs_workdir / file_path).resolve()
    try:
        resolved.relative_to(abs_workdir)
    except ValueError:
        return f"File path escapes the working directory via traversal: {file_path!r}"
    return None


# ---------------------------------------------------------------------------
# Agent-write ledger
# ---------------------------------------------------------------------------

def _ledger_path(store: Path, dir_hash: str) -> Path:
    return store / _LEDGERS_DIRNAME / f"{dir_hash}.json"


def _hash_file(path: Path) -> Optional[str]:
    """Streaming sha256 of a file's bytes. None if unreadable/missing."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _load_ledger(store: Path, dir_hash: str) -> dict[str, dict]:
    """Load the agent-write ledger: {relpath: {"sha256": ..., "ts": ...}}.

    The ledger records the content hash of every file the last successful
    ``write_file`` / ``patch`` produced, so restores can tell "Hermes wrote
    this" apart from "the user hand-edited this afterwards".
    """
    try:
        raw = _ledger_path(store, dir_hash).read_text(encoding="utf-8-sig")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_ledger(store: Path, dir_hash: str, ledger: dict[str, dict]) -> None:
    """Persist the agent-write ledger, capped to the newest entries."""
    try:
        if len(ledger) > _LEDGER_MAX_ENTRIES:
            newest = sorted(
                ledger.items(),
                key=lambda kv: kv[1].get("ts", 0) if isinstance(kv[1], dict) else 0,
                reverse=True,
            )[:_LEDGER_MAX_ENTRIES]
            ledger = dict(newest)
        path = _ledger_path(store, dir_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ledger), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.debug("Failed to save agent-write ledger for %s", dir_hash, exc_info=True)


# ---------------------------------------------------------------------------
# RestoreMixin
# ---------------------------------------------------------------------------

class RestoreMixin:
    """Restore/diff/rollback planning and agent-write ledger methods.

    Mixed into ``CheckpointManager``; the methods call ``_take``, ``_prune``
    and ``_exceeds_size_cap`` on the composed manager.
    """

    def record_agent_write(self, file_path: str) -> None:
        """Record the content hash of a file Hermes just successfully wrote.

        Feeds the agent-write ledger used by :meth:`restore` in safe mode:
        at restore time, a file whose current content no longer matches the
        recorded hash was hand-edited by the user after Hermes last touched
        it, and is skipped instead of clobbered.

        Never raises — the ledger is best-effort bookkeeping.
        """
        if not self.enabled:
            return
        try:
            from tools.checkpoint_pruning import store_lock

            path = _normalize_path(file_path)
            digest = _hash_file(path)
            if digest is None:
                return
            with store_lock(_resolve_checkpoint_base()):
                store = _store_path()
                dir_hash = self._ledger_key(str(path))
                ledger = _load_ledger(store, dir_hash)
                ledger[str(path)] = {"sha256": digest, "ts": time.time()}
                _save_ledger(store, dir_hash, ledger)
        except Exception as exc:
            logger.debug("record_agent_write failed for %s: %s", file_path, exc)

    def _safe_restore_plan(self, working_dir: str, commit_hash: str) -> dict:
        """Classify files changed since ``commit_hash`` for a safe restore.

        Returns ``{"success", "restore": [rel...], "skipped": [rel...],
        "error"?}`` where ``restore`` lists files whose current content
        still matches what Hermes last wrote (per the agent-write ledger)
        and ``skipped`` lists files the user hand-edited after Hermes'
        last write or that Hermes never wrote at all.
        """
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))
        store = _store_path()
        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        # Stage the current tree so the name-only diff sees new files too.
        # Exclude the checkpoint base so we don't stage the store's own paths.
        base_rel = _checkpoint_base_rel_path(working_dir)
        _run_git(_staging_args(working_dir), store, abs_dir,
                 timeout=_GIT_TIMEOUT * 2, index_file=index_file)
        ok, names_out, err = _run_git(
            ["diff", "--name-only", "-z", commit_hash, "--cached"],
            store, abs_dir, index_file=index_file,
        )
        # Reset the index back to the project ref so it doesn't drift.
        _run_git(["read-tree", _ref_name(dir_hash)], store, abs_dir,
                 index_file=index_file, allowed_returncodes={128})
        if not ok:
            return {"success": False, "error": f"Could not compute changed files: {err}"}

        # Read the same marker-walked project key as record_agent_write.
        ledger = _load_ledger(store, self._ledger_key(abs_dir))
        if not ledger:
            # No agent-write ledger yet (pre-existing store, or Hermes has
            # not written any files here since the ledger was introduced).
            # Signal callers to fall back to a full restore rather than
            # skipping every file.
            return {"success": True, "restore": [], "skipped": [],
                    "ledger_empty": True}
        restore: list[str] = []
        skipped: list[str] = []
        for rel in filter(None, names_out.split("\x00")):
            # Filter out paths under the checkpoint base. Old checkpoints
            # (taken before the exclusion fix) may contain store paths,
            # which would otherwise be classified as restore/skipped.
            if base_rel is not None and (rel.startswith(base_rel + "/") or rel == base_rel):
                continue
            abs_path = Path(abs_dir) / rel
            entry = ledger.get(str(abs_path))
            recorded = entry.get("sha256") if isinstance(entry, dict) else None
            if recorded is None:
                # Hermes never wrote this file (or the ledger predates it) —
                # do not touch it in safe mode.
                skipped.append(rel)
                continue
            current = _hash_file(abs_path)
            if current is None:
                # File deleted since Hermes wrote it: restoring it back is
                # safe — its last content was Hermes-authored.
                restore.append(rel)
            elif current == recorded:
                restore.append(rel)
            else:
                skipped.append(rel)
        return {"success": True, "restore": restore, "skipped": skipped}

    def diff(self, working_dir: str, commit_hash: str) -> dict:
        """Show diff between a checkpoint and the current working tree."""
        from tools.checkpoint_pruning import PruneError, store_lock

        try:
            with store_lock(_resolve_checkpoint_base()):
                return self._diff(working_dir, commit_hash)
        except (PruneError, OSError) as exc:
            return {"success": False, "error": str(exc)}

    def _diff(self, working_dir: str, commit_hash: str) -> dict:
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))
        store = _store_path()

        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        ok, _, _err = _run_git(
            ["cat-file", "-t", commit_hash], store, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found"}

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        # Stage current state into the per-project index to compare.
        # Exclude the checkpoint base so we don't stage the store's own paths.
        base_rel = _checkpoint_base_rel_path(working_dir)
        _run_git(_staging_args(working_dir), store, abs_dir,
                 timeout=_GIT_TIMEOUT * 2, index_file=index_file)

        # Exclude the base from the diff output via pathspec so git never
        # emits store paths (even from old checkpoints that contain them).
        exclude = ["--", ".", f":(exclude,literal){base_rel}"] if base_rel is not None else []

        ok_stat, stat_out, _ = _run_git(
            ["diff", "--stat", commit_hash, "--cached", *exclude],
            store, abs_dir, index_file=index_file,
        )
        ok_diff, diff_out, _ = _run_git(
            ["diff", commit_hash, "--cached", "--no-color", *exclude],
            store, abs_dir, index_file=index_file,
        )

        # Reset staged tree back to the project's last checkpoint so the
        # index doesn't drift out of sync with the ref.
        ref = _ref_name(dir_hash)
        _run_git(["read-tree", ref], store, abs_dir,
                 index_file=index_file,
                 allowed_returncodes={128})

        if not ok_stat and not ok_diff:
            return {"success": False, "error": "Could not generate diff"}

        return {
            "success": True,
            "stat": stat_out if ok_stat else "",
            "diff": diff_out if ok_diff else "",
        }

    def session_diff(self, working_dir: str) -> dict:
        """Show the cumulative diff of everything changed in this directory.

        This powers ``/diff session``.  It answers "what has Hermes changed
        here?" by diffing the *earliest retained checkpoint* — the snapshot
        taken before the first recorded edit — against the current working
        tree.  Because checkpoints are captured just before each file-mutating
        tool call, that baseline is the pre-edit state, so the diff covers the
        first edit and everything after it.

        Note: checkpoints are a persistent per-project ref, so the earliest
        *retained* checkpoint may predate the current session (or, after
        pruning, postdate its true start).  It is an approximation of "what
        Hermes changed", not an exact per-session ledger.

        Returns the same shape as :meth:`diff` (``{"success", "stat",
        "diff"}``).  When no checkpoints exist yet — nothing has been edited —
        the call still *succeeds* with empty output and ``"empty": True`` so
        callers can show a friendly "no changes" message rather than an error.
        """
        checkpoints = self.list_checkpoints(working_dir)
        if not checkpoints:
            return {"success": True, "stat": "", "diff": "", "empty": True}

        baseline = checkpoints[-1].get("hash") or ""
        result = self.diff(working_dir, baseline)
        if result.get("success"):
            result.setdefault("baseline", baseline)
            if not result.get("stat") and not result.get("diff"):
                result["empty"] = True
        return result

    def restore(
        self,
        working_dir: str,
        commit_hash: str,
        file_path: str | None = None,
        safe: bool = False,
    ) -> dict:
        """Restore files to a checkpoint state.

        With ``safe=True`` (full-directory restores only), files the user
        hand-edited after Hermes' last write — per the agent-write ledger —
        are left untouched, and only Hermes-authored changes are reverted.
        The result gains ``skipped_user_edits`` listing the preserved paths,
        ``skipped_oversize`` listing paths kept because the size cap excluded
        them from every checkpoint, and — only when a delete failed —
        ``failed_deletes`` listing paths that could not be removed.
        """
        from tools.checkpoint_pruning import PruneError, store_lock

        try:
            with store_lock(_resolve_checkpoint_base()):
                return self._restore(working_dir, commit_hash, file_path, safe)
        except (PruneError, OSError) as exc:
            return {"success": False, "error": str(exc)}

    def _restore(self, working_dir: str, commit_hash: str, file_path: str | None, safe: bool) -> dict:
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))

        if file_path:
            path_err = _validate_file_path(file_path, abs_dir)
            if path_err:
                return {"success": False, "error": path_err}

        store = _store_path()

        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], store, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found",
                    "debug": err or None}

        ok, tree_out, err = _run_git(
            ["ls-tree", "-r", "-z", commit_hash], store, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Could not inspect checkpoint: {err}"}
        nested_repos = _gitlink_paths(tree_out)
        selected_paths: Optional[list[str]] = None
        if nested_repos:
            blocked_repos = nested_repos
            if file_path:
                # Select by the path as recorded in the checkpoint, not by
                # where it points now: a captured file later replaced by a
                # symlink into a nested repo is still recoverable. Resolved
                # confinement is enforced separately by _validate_file_path.
                root_path = Path(abs_dir)
                requested_path = Path(os.path.normpath(root_path / file_path))
                try:
                    requested_rel = requested_path.relative_to(root_path)
                except ValueError:
                    requested_rel = Path(file_path)
                requested_parts = tuple(requested_rel.parts)

                def _scope_intersects_nested_repo(repo_path: str) -> bool:
                    repo_parts = tuple(PurePosixPath(repo_path).parts)
                    return (
                        requested_parts[:len(repo_parts)] == repo_parts
                        or repo_parts[:len(requested_parts)] == requested_parts
                    )

                blocked_repos = [
                    repo for repo in nested_repos
                    if _scope_intersects_nested_repo(repo)
                ]

                # Literal paths below a gitlink have no tree entry to match.
                # Keep the explicit refusal above, but let Git resolve all
                # other selections exactly as checkout does (glob, icase,
                # top, exclude, literal, etc.). ls-tree lacks that pathspec
                # support. A disposable index avoids touching the project's
                # index, refs or files before we know rollback is possible.
                if not blocked_repos:
                    with tempfile.TemporaryDirectory(prefix="restore-inspect-", dir=store) as scratch:
                        inspect_index = Path(scratch) / "index"
                        ok, _, err = _run_git(
                            ["read-tree", commit_hash], store, abs_dir,
                            index_file=inspect_index,
                        )
                        if ok:
                            ok, selected, err = _run_git(
                                ["ls-files", "--stage", "--error-unmatch", "-z", "--", file_path],
                                store, abs_dir, index_file=inspect_index,
                            )
                        if not ok:
                            return {"success": False, "error": f"Could not inspect restore selection: {err}"}
                    blocked_repos = [
                        record.split("\t", 1)[1]
                        for record in selected.split("\x00")
                        if record.startswith("160000 ") and "\t" in record
                    ]
                    selected_paths = [
                        record.split("\t", 1)[1]
                        for record in selected.split("\x00")
                        if "\t" in record
                    ]
                    if not selected_paths:
                        # An exclusion-only spec can match nothing. Checkout
                        # with no paths would switch the store's HEAD instead
                        # of restoring files, so refuse before the snapshot.
                        return {
                            "success": False,
                            "error": f"Restore selection matched no files in checkpoint: {file_path}",
                        }

            if blocked_repos:
                paths = ", ".join(blocked_repos)
                return {
                    "success": False,
                    "error": (
                        "Checkpoint contains nested git repositories that were not captured "
                        f"({paths}); rollback was not performed"
                    ),
                    "nested_repositories": blocked_repos,
                }

        skipped_user_edits: list[str] = []
        kept_oversize: list[str] = []
        failed_deletes: list[str] = []
        restore_paths: Optional[list[str]] = None
        if safe and not file_path:
            plan = self._safe_restore_plan(abs_dir, commit_hash)
            if not plan.get("success"):
                return {"success": False, "error": plan.get("error", "Safe-restore plan failed")}
            if plan.get("ledger_empty"):
                # No agent-write history to compare against — fall back to
                # the classic full restore rather than restoring nothing.
                restore_paths = None
            else:
                restore_paths = plan["restore"]
                skipped_user_edits = plan["skipped"]
                if not restore_paths:
                    return {
                        "success": True,
                        "restored_to": commit_hash[:8],
                        "reason": "nothing to restore (all changed files were user-edited)",
                        "directory": abs_dir,
                        "restored_files": [],
                        "skipped_user_edits": skipped_user_edits,
                        "skipped_oversize": [],
                    }

        # Take a pre-rollback snapshot so you can undo the undo.
        self._take(abs_dir, f"pre-rollback snapshot (restoring to {commit_hash[:8]})", prune=False)

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        if restore_paths is not None:
            # Split into files present in the checkpoint (checkout) and
            # Hermes-created files absent from it (delete to restore state).
            checkout_targets: list[str] = []
            delete_targets: list[str] = []
            for rel in restore_paths:
                ok_in_commit, _, _ = _run_git(
                    ["cat-file", "-e", f"{commit_hash}:{rel}"],
                    store, abs_dir, allowed_returncodes={1, 128},
                )
                if ok_in_commit:
                    checkout_targets.append(rel)
                elif self._exceeds_size_cap(Path(abs_dir) / rel):
                    # Absent from the checkpoint because ``max_file_size_mb``
                    # kept it out (_drop_oversize_from_index), not because
                    # Hermes created it. Deleting it would not restore a prior
                    # state — no checkpoint holds one — it would destroy the
                    # only copy. The ledger records a content hash, not whether
                    # a write created or modified the file, so an oversize path
                    # cannot be proven agent-created; leaving it costs a stale
                    # file, deleting it costs the file.
                    kept_oversize.append(rel)
                else:
                    delete_targets.append(rel)
            for rel in delete_targets:
                try:
                    target = Path(abs_dir) / rel
                    if target.is_file() or target.is_symlink():
                        target.unlink()
                except OSError as exc:
                    logger.warning(
                        "Safe restore: could not remove %s: %s", rel, exc,
                    )
                    failed_deletes.append(rel)
            if not checkout_targets:
                ok, _stdout, err = True, "", ""
            else:
                ok, _stdout, err = _run_git(
                    ["checkout", commit_hash, "--", *checkout_targets],
                    store, abs_dir, timeout=_GIT_TIMEOUT * 2,
                    index_file=index_file,
                )
        elif selected_paths is not None:
            # Check out exactly the entries inspected above. Re-evaluating
            # file_path here could select something else: _take() has just
            # restaged the project index, and attribute pathspecs read
            # .gitattributes from it. A NUL-separated literal pathspec file
            # keeps long selections off the command line.
            with tempfile.TemporaryDirectory(prefix="restore-select-", dir=store) as scratch:
                spec_file = Path(scratch) / "pathspec"
                spec_file.write_bytes(b"".join(os.fsencode(p) + b"\0" for p in selected_paths))
                ok, _stdout, err = _run_git(
                    ["checkout", commit_hash, f"--pathspec-from-file={spec_file}",
                     "--pathspec-file-nul"],
                    store, abs_dir, timeout=_GIT_TIMEOUT * 2,
                    index_file=index_file,
                    extra_env={"GIT_LITERAL_PATHSPECS": "1"},
                )
        else:
            ok, _stdout, err = _run_git(
                ["checkout", commit_hash, "--", file_path if file_path else "."],
                store, abs_dir, timeout=_GIT_TIMEOUT * 2,
                index_file=index_file,
            )

        if not ok:
            return {"success": False, "error": f"Restore failed: {err}",
                    "debug": err or None}

        ok2, reason_out, _ = _run_git(
            ["log", "--format=%s", "-1", commit_hash], store, abs_dir,
        )
        reason = reason_out if ok2 else "unknown"

        result = {
            "success": True,
            "restored_to": commit_hash[:8],
            "reason": reason,
            "directory": abs_dir,
        }
        if file_path:
            result["file"] = file_path
        if restore_paths is not None:
            # Only what was actually acted on. A kept oversize path was not
            # restored (and a failed unlink left the file in place), and
            # reporting either as restored is how the data loss above stayed
            # silent: the user was told "Restored" for a file that had just
            # been unlinked.
            not_restored = set(kept_oversize) | set(failed_deletes)
            result["restored_files"] = [
                rel for rel in restore_paths if rel not in not_restored
            ]
            result["skipped_user_edits"] = skipped_user_edits
            result["skipped_oversize"] = kept_oversize
            if failed_deletes:
                result["failed_deletes"] = failed_deletes
        # The selected tree was needed until checkout completed. Only now may
        # the safety snapshot's count/size budget make that tree unreachable.
        self._prune(store, abs_dir, _ref_name(dir_hash))
        return result

    def _ledger_key(self, path: str) -> str:
        """Agent-write ledger key: hash of the marker-walked project dir, for writer and reader alike."""
        return _project_hash(self.get_working_dir_for_path(path))


# --- facade delegates -------------------------------------------------------
# Tests monkeypatch these attributes on ``tools.checkpoint_manager``.  The
# delegates resolve them at call time (function-local import, so the module
# cycle facade -> restore -> facade only closes when a call happens), which is
# what keeps ``monkeypatch.setattr(checkpoint_manager, ...)`` visible here.

def _resolve_checkpoint_base(*args, **kwargs):
    from tools import checkpoint_manager

    return checkpoint_manager._resolve_checkpoint_base(*args, **kwargs)


def _run_git(*args, **kwargs):
    from tools import checkpoint_manager

    return checkpoint_manager._run_git(*args, **kwargs)


def _staging_args(*args, **kwargs):
    from tools import checkpoint_manager

    return checkpoint_manager._staging_args(*args, **kwargs)


def _checkpoint_base_rel_path(*args, **kwargs):
    from tools import checkpoint_manager

    return checkpoint_manager._checkpoint_base_rel_path(*args, **kwargs)
