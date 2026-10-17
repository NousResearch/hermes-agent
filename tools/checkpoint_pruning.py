"""Checked history rewrites shared by snapshot limits and store maintenance."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable


class PruneError(RuntimeError):
    """Pruning stopped without permission to discard another snapshot."""


GC_PENDING_NAME = ".gc-pending"


def store_lock_path(base: Path) -> Path:
    base = base.resolve()
    return base.with_name(f".{base.name}.lock")


@contextmanager
def store_lock(base: Path):
    """Serialize whole operations, including GC and clear, across processes."""
    from hermes_cli.runtime_state import _lock

    base.parent.mkdir(parents=True, exist_ok=True)
    # Outside base so clear_all and legacy migration cannot replace its inode.
    fd = os.open(store_lock_path(base), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not _lock(fd, wait=False):
            raise PruneError(f"checkpoint store is busy: {base}")
        yield
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Pruner:
    run_git: Callable
    store: Path
    cwd: str
    timeout: int
    dir_size: Callable[[Path], int]
    refs_prefix: str

    def _git(self, args: list[str], **kwargs) -> str:
        ok, output, error = self.run_git(args, self.store, self.cwd, **kwargs)
        if not ok:
            raise PruneError(f"checkpoint {args[0]} failed: {error}")
        return output

    def reclaim(self) -> None:
        """Full ``git gc`` — tens of seconds on a GB store, so only the periodic prune calls it.
        A ref rewrite that ran without it leaves a marker so the next prune reclaims."""
        self._git(["reflog", "expire", "--expire=now", "--all"])
        self._git(["gc", "--prune=now", "--quiet"], timeout=self.timeout * 3)
        # Bare stores must remain usable when gc packs all refs.
        for name in ("refs/heads", "branches"):
            (self.store / name).mkdir(parents=True, exist_ok=True)
        try:
            (self.store / GC_PENDING_NAME).unlink()
        except OSError:
            pass

    def gc_pending(self) -> bool:
        return (self.store / GC_PENDING_NAME).exists()

    def _mark_gc_pending(self) -> None:
        try:
            (self.store / GC_PENDING_NAME).touch()
        except OSError:
            pass

    def _commits(self, ref: str) -> list[str]:
        commits = self._git(["rev-list", "--reverse", ref]).splitlines()
        if not commits:
            raise PruneError(f"checkpoint ref has no readable history: {ref}")
        return commits

    def _rev_list(self, ref: str, max_count: int | None = None) -> list[str] | None:
        """Raw rev-list returning oldest-first commits, or None when the walk fails."""
        args = ["rev-list"]
        if max_count is not None:
            args += ["--max-count", str(max_count)]
        args += ["--reverse", ref]
        ok, output, _ = self.run_git(args, self.store, self.cwd)
        if not ok:
            return None
        return output.splitlines()

    def repair_unwalkable(self, ref: str) -> bool:
        """Rebuild a ref whose history walk dies on a missing parent object onto a fresh
        root of its readable commits, so ``git gc`` can traverse the store again (a
        missing mid-chain object is unreachable garbage after the rewrite). CAS-guarded
        by _rewrite; the reclaimed space is collected by the next gc. Returns whether a
        repair was applied."""
        tip = self._rev_list(ref, 1)
        if not tip:
            return False  # nothing readable at the tip; rewriting cannot help
        if self._rev_list(ref) is not None:
            return False  # walkable: the failure came from elsewhere, not this ref
        hi = 1
        while self._rev_list(ref, hi * 2) is not None:
            hi *= 2
        lo = hi
        hi *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._rev_list(ref, mid) is not None:
                lo = mid
            else:
                hi = mid
        readable = self._rev_list(ref, lo)
        if not readable:
            return False
        self._rewrite(ref, readable, readable, gc=False)
        return True

    def _rewrite(self, ref: str, original: list[str], keep: list[str], *, gc: bool) -> None:
        parent = None
        for sha in keep:
            metadata = self._git(["log", "-1", "--format=%T%x00%aI%x00%cI%x00%s", sha])
            fields = metadata.split("\x00", 3)
            if len(fields) != 4:
                raise PruneError(f"invalid checkpoint metadata: {sha}")
            tree, author_date, committer_date, message = fields
            args = ["commit-tree", tree, "--no-gpg-sign", "-m", message]
            if parent is not None:
                args += ["-p", parent]
            parent = self._git(args, extra_env={"GIT_AUTHOR_DATE": author_date, "GIT_COMMITTER_DATE": committer_date})
            if not parent:
                raise PruneError("commit-tree returned no checkpoint identity")
        if parent is None:
            raise PruneError("cannot remove a project's last snapshot")
        # Never overwrite a snapshot published since we read this history.
        self._git(["update-ref", ref, parent, original[-1]])
        if gc:
            self.reclaim()
        else:
            self._mark_gc_pending()

    def trim(self, ref: str, keep_count: int) -> None:
        """Snapshot-count budget on the checkpoint-take path: rewrites the ref, defers the gc."""
        commits = self._commits(ref)
        keep_count = max(1, keep_count)
        if len(commits) > keep_count:
            self._rewrite(ref, commits, commits[-keep_count:], gc=False)

    def drop_one_round(self, cap_bytes: int) -> bool:
        """Over the cap on the take path: drop the oldest snapshot of every project once and
        defer the gc. One round per checkpoint converges over turns; the full loop measured
        against an unchanging pack once flattened every project to a single snapshot."""
        if cap_bytes <= 0 or self.dir_size(self.store) <= cap_bytes:
            return False
        changed = False
        for ref in self._git(["for-each-ref", "--format=%(refname)", self.refs_prefix]).splitlines():
            try:
                commits = self._commits(ref)
            except PruneError:
                if not self.repair_unwalkable(ref):
                    raise
                changed = True
                continue
            if len(commits) > 1:
                self._rewrite(ref, commits, commits[1:], gc=False)
                changed = True
        return changed

    def enforce_size(self, cap_bytes: int) -> bool:
        """Periodic prune: gc per round (the measurement only moves after a repack) until the
        store fits. Returns whether it did; a failed Git step raises immediately."""
        if cap_bytes <= 0 or self.dir_size(self.store) <= cap_bytes:
            return True
        try:
            self.reclaim()
        except PruneError:
            pass  # an unwalkable ref blocks repack; the rounds below repair or isolate it
        if self.dir_size(self.store) <= cap_bytes:
            return True
        refs = self._git(["for-each-ref", "--format=%(refname)", self.refs_prefix]).splitlines()
        # Keep the existing round-robin policy, remeasure after EACH rewrite.
        # Twenty rounds bound maintenance when many large histories remain.
        for _ in range(20):
            changed = False
            for ref in refs:
                try:
                    commits = self._commits(ref)
                except PruneError:
                    if not self.repair_unwalkable(ref):
                        raise
                    changed = True
                    continue
                try:
                    if len(commits) <= 1:
                        continue
                    self._rewrite(ref, commits, commits[1:], gc=True)
                except PruneError:
                    continue  # gc is still blocked by another ref; retry next round
                changed = True
                if self.dir_size(self.store) <= cap_bytes:
                    return True
            if not changed:
                break
        return False
