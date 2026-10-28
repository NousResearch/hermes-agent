"""Snapshot staging, index maintenance and the post-commit budget trim for checkpoints.

The take side of ``tools.checkpoint_manager``: the file-count guard that keeps a
take off enormous trees, the pathspec-aware staging call that keeps the
checkpoint store out of its own snapshots, the per-project index maintenance
that drops paths the previous tip re-seeded (the store itself, oversize files),
and the count/size budget trim applied after a commit lands.

Seam-sensitive names (``_run_git``, ``_staging_args``,
``_checkpoint_base_rel_path``) stay defined in ``checkpoint_manager``; the
delegates at the bottom resolve them at call time, so
``monkeypatch.setattr(checkpoint_manager, ...)`` is still observed here.
"""

import logging
from pathlib import Path

from tools.checkpoint_store import (
    _GIT_TIMEOUT,
    _REFS_PREFIX,
    _gitlink_paths,
    _index_path,
    _init_store,
    _normalize_path,
    _project_hash,
    _ref_name,
    _store_path,
    _touch_project,
)

logger = logging.getLogger(__name__)


# Max files to snapshot — skip huge directories to avoid slowdowns.
_MAX_FILES = 50_000


def _uncaptured_note(nested_repos: list[str]) -> str:
    """Checkpoint-message suffix naming nested repositories whose files were
    not captured, so every /rollback listing discloses it."""
    shown = ", ".join(nested_repos[:3])
    more = f" (+{len(nested_repos) - 3})" if len(nested_repos) > 3 else ""
    return f" [nested git repos not captured: {shown}{more}]"


def _dir_file_count(path: str) -> int:
    """Quick file count estimate (stops early if over _MAX_FILES)."""
    count = 0
    try:
        for _ in Path(path).rglob("*"):
            count += 1
            if count > _MAX_FILES:
                return count
    except (PermissionError, OSError):
        pass
    return count


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive size in bytes.  Returns 0 on error."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


class StagingMixin:
    """Take-path staging, index maintenance and the post-commit budget trim.

    Mixed into ``CheckpointManager``.  :meth:`_take` is the snapshot path itself;
    ``RestoreMixin`` calls back into it for the pre-rollback safety snapshot and
    consults :meth:`_exceeds_size_cap` and :meth:`_prune` on the composed class.
    """

    def _take(self, working_dir: str, reason: str, *, prune: bool = True) -> bool:
        """Take a snapshot.  Returns True on success."""
        store = _store_path()

        # A workdir that *is* the checkpoint base holds nothing but the store
        # and its bookkeeping; every path would be excluded, so there is no
        # tree to write.  Refuse rather than commit an empty snapshot.
        base_rel = _checkpoint_base_rel_path(working_dir)
        if base_rel == ".":
            logger.debug("Checkpoint skipped: %s is the checkpoint base", working_dir)
            return False

        err = _init_store(store, working_dir)
        if err:
            logger.debug("Checkpoint store init failed: %s", err)
            return False

        _touch_project(store, working_dir)

        # Quick size guard — don't try to snapshot enormous directories
        if _dir_file_count(working_dir) > _MAX_FILES:
            logger.debug("Checkpoint skipped: >%d files in %s", _MAX_FILES, working_dir)
            return False

        dir_hash = _project_hash(working_dir)
        index_file = _index_path(store, dir_hash)
        ref = _ref_name(dir_hash)

        # Seed the per-project index from the last checkpoint, if any, so the
        # diff/commit machinery sees only changes since then.  On first call,
        # clear the index so ``git add -A`` produces a clean tree.
        if index_file.exists():
            # Reset index to current ref tip to avoid accumulating stale paths.
            ok_ref, ref_commit, _ = _run_git(
                ["rev-parse", "--verify", ref + "^{commit}"],
                store, working_dir,
                allowed_returncodes={128},
            )
            if ok_ref and ref_commit:
                _run_git(
                    ["read-tree", ref_commit],
                    store, working_dir,
                    index_file=index_file,
                    allowed_returncodes={128},
                )
            else:
                try:
                    index_file.unlink()
                except OSError:
                    pass
        else:
            # First snapshot for this project.
            index_file.parent.mkdir(parents=True, exist_ok=True)

        # Stage with per-project index.  Include a per-stage file-size filter
        # via ``core.bigFileThreshold`` is not what we want — instead, we
        # rely on the exclude file for broad patterns and post-stage prune
        # any path whose size exceeds max_file_size_mb.  The checkpoint base
        # itself is excluded by pathspec: staging it would store the store's
        # own object database in the snapshot it is taking.
        ok, _, err = _run_git(
            _staging_args(working_dir), store, working_dir,
            timeout=_GIT_TIMEOUT * 2, index_file=index_file,
        )
        if not ok:
            logger.debug("Checkpoint git-add failed: %s", err)
            return False

        # The index was seeded from the previous tip, so a store that already
        # carries a recursive snapshot still has those paths on it.  Drop them
        # before the tree is written; older commits stay untouched.
        if base_rel is not None:
            self._drop_index_prefix(store, working_dir, index_file, base_rel)

        if self.max_file_size_mb > 0:
            self._drop_oversize_from_index(store, working_dir, index_file)

        # Compare against the current ref tip (not HEAD — HEAD points to a
        # branch that doesn't exist on a bare store, so ``diff --cached``
        # against HEAD would always show "new file" for every staged path).
        ok_ref, ref_commit, _ = _run_git(
            ["rev-parse", "--verify", ref + "^{commit}"],
            store, working_dir,
            allowed_returncodes={128},
        )
        has_ref = ok_ref and bool(ref_commit)

        if has_ref:
            ok_diff, _, _ = _run_git(
                ["diff-index", "--cached", "--quiet", ref_commit],
                store, working_dir,
                allowed_returncodes={1},
                index_file=index_file,
            )
            if ok_diff:
                logger.debug("Checkpoint skipped: no changes in %s", working_dir)
                return False
        else:
            # No ref yet — skip only if the index is empty.
            ok_ls, ls_out, _ = _run_git(
                ["ls-files", "--cached"],
                store, working_dir,
                index_file=index_file,
            )
            if ok_ls and not ls_out.strip():
                logger.debug("Checkpoint skipped: empty tree in %s", working_dir)
                return False

        # Write tree from per-project index.
        ok_tree, tree_sha, err = _run_git(
            ["write-tree"], store, working_dir,
            index_file=index_file,
        )
        if not ok_tree or not tree_sha:
            logger.debug("Checkpoint write-tree failed: %s", err)
            return False

        # A nested repository is stored as a gitlink, not its files: say so in
        # the checkpoint itself rather than only when a rollback is refused.
        ok_ls, tree_out, _ = _run_git(["ls-tree", "-r", "-z", tree_sha], store, working_dir)
        nested_repos = _gitlink_paths(tree_out) if ok_ls else []
        if nested_repos:
            logger.info("Checkpoint of %s does not capture nested git repositories: %s",
                        working_dir, ", ".join(nested_repos))
            # Listings read the subject (%s), which ends at the first blank
            # line, so keep a multiline reason (a terminal command) on one
            # line or the note falls outside every listing.
            reason = " ".join(
                line.strip() for line in reason.splitlines() if line.strip()
            ) + _uncaptured_note(nested_repos)

        # Build commit (parent = current ref tip, if any).
        commit_args = ["commit-tree", tree_sha, "-m", reason, "--no-gpg-sign"]
        if has_ref:
            commit_args = ["commit-tree", tree_sha, "-p", ref_commit, "-m", reason, "--no-gpg-sign"]
        ok_commit, new_sha, err = _run_git(
            commit_args, store, working_dir,
            index_file=index_file,
        )
        if not ok_commit or not new_sha:
            logger.debug("Checkpoint commit-tree failed: %s", err)
            return False

        # Update the per-project ref.
        update_args = ["update-ref", ref, new_sha]
        if has_ref:
            update_args = ["update-ref", ref, new_sha, ref_commit]
        ok_update, _, err = _run_git(
            update_args, store, working_dir,
        )
        if not ok_update:
            logger.debug("Checkpoint update-ref failed: %s", err)
            return False

        logger.debug("Checkpoint taken in %s: %s (%s)", working_dir, reason, new_sha[:8])

        # Count and size budgets share one failure boundary.
        if prune:
            self._prune(store, working_dir, ref)

        return True

    def _exceeds_size_cap(self, path: Path) -> bool:
        """Whether *path* is larger than ``max_file_size_mb``.

        The same test :meth:`_drop_oversize_from_index` applies when building a
        checkpoint, so "excluded from the checkpoint" and "refused deletion at
        restore" agree on one definition. A cap of 0 disables it, and an
        unstattable path is not claimed to be oversize.
        """
        cap = self.max_file_size_mb * 1024 * 1024
        if cap <= 0:
            return False
        try:
            return path.stat().st_size > cap
        except OSError:
            return False

    def _drop_index_prefix(
        self, store: Path, working_dir: str, index_file: Path, rel_prefix: str,
    ) -> None:
        """Unstage every index entry under ``rel_prefix``.

        ``_take`` seeds the per-project index from the previous tip
        (``read-tree``), so a store that already holds a recursive snapshot
        keeps those paths staged even though the staging pathspec excludes
        them.  Purging them here is what makes the exclusion apply to existing
        stores instead of only to stores initialised after the fix.  The new
        commit drops them; older commits are neither rewritten nor deleted.
        """
        ok, stdout, _ = _run_git(
            ["ls-files", "--cached", "-z", "--", f":(literal){rel_prefix}"],
            store, working_dir, index_file=index_file,
        )
        if not ok or not stdout:
            return
        # NUL separators preserve whitespace within each literal filename.
        paths = [p for p in stdout.split("\x00") if p]
        if not paths:
            return
        logger.debug(
            "Checkpoint: dropping %d previously staged path(s) under %s from index",
            len(paths), rel_prefix,
        )
        BATCH = 200
        for i in range(0, len(paths), BATCH):
            chunk = [f":(literal){p}" for p in paths[i:i + BATCH]]
            # ``-f`` only overrides git's refusal to drop an entry whose
            # staged content differs from both the worktree and the old tip
            # (the store's own bookkeeping files churn between takes); with
            # ``--cached`` no working-tree file is touched.
            _run_git(
                ["rm", "-f", "--cached", "--quiet", "--"] + chunk,
                store, working_dir, index_file=index_file,
                allowed_returncodes={128},
            )

    def _drop_oversize_from_index(
        self, store: Path, working_dir: str, index_file: Path,
    ) -> None:
        """Remove any staged file larger than ``max_file_size_mb`` from the index.

        Lets the agent keep snapshotting source code while refusing to
        swallow generated assets (datasets, model weights, logs, videos).
        """
        if self.max_file_size_mb <= 0:
            return
        ok, stdout, _ = _run_git(
            ["ls-files", "--cached", "-z"],
            store, working_dir, index_file=index_file,
        )
        if not ok or not stdout:
            return
        # NUL separators preserve whitespace within each literal filename.
        paths = [p for p in stdout.split("\x00") if p]
        abs_workdir = _normalize_path(working_dir)
        # Same predicate safe restore consults, called rather than restated:
        # a threshold that drifted between the two would make a file both
        # absent from the checkpoint and not recognised as capped at restore,
        # which is precisely the deletion this change exists to prevent.
        oversize = [
            rel for rel in paths if self._exceeds_size_cap(abs_workdir / rel)
        ]
        if not oversize:
            return
        logger.debug(
            "Checkpoint: dropping %d oversize file(s) (>%d MB) from index",
            len(oversize), self.max_file_size_mb,
        )
        # Use --pathspec-from-file for safety with many paths.
        # Chunk into manageable batches.
        BATCH = 200
        for i in range(0, len(oversize), BATCH):
            chunk = oversize[i:i + BATCH]
            _run_git(
                ["rm", "--cached", "--quiet", "--"] + chunk,
                store, working_dir, index_file=index_file,
                allowed_returncodes={128},
            )

    def _prune(self, store: Path, working_dir: str, ref: str) -> None:
        """Checkpoint-take path: snapshot-count budget plus one size round, gc deferred to the
        periodic prune — a repack here held the tool call for the whole gc on a large store."""
        from tools.checkpoint_pruning import Pruner, PruneError

        pruner = Pruner(_run_git, store, working_dir, _GIT_TIMEOUT, _dir_size_bytes, _REFS_PREFIX)
        try:
            pruner.trim(ref, self.max_snapshots)
            if pruner.drop_one_round(self.max_total_size_mb * 1024 * 1024):
                logger.info("Checkpoint store exceeded %d MB — dropped the oldest snapshot per project; "
                            "space is reclaimed by the next prune", self.max_total_size_mb)
        except (PruneError, OSError) as exc:
            logger.warning("Checkpoint pruning stopped: %s", exc)


# --- facade delegates -------------------------------------------------------
# Tests monkeypatch these attributes on ``tools.checkpoint_manager``.  The
# delegates resolve them at call time (function-local import, so the module
# cycle facade -> staging -> facade only closes when a call happens), which is
# what keeps ``monkeypatch.setattr(checkpoint_manager, ...)`` visible here.


def _run_git(*args, **kwargs):
    from tools import checkpoint_manager

    return checkpoint_manager._run_git(*args, **kwargs)


def _staging_args(*args, **kwargs):
    from tools import checkpoint_manager

    return checkpoint_manager._staging_args(*args, **kwargs)


def _checkpoint_base_rel_path(*args, **kwargs):
    from tools import checkpoint_manager

    return checkpoint_manager._checkpoint_base_rel_path(*args, **kwargs)
