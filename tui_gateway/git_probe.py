"""Git working-tree probing for the gateway: run git, resolve repo roots, fold linked worktrees.
Probing runs where the gateway runs (covers remote backends). Roots go through a thread-safe
single-flight cache so concurrent identical probes share one ``git`` spawn: positives live for the
process, negatives (not a repo / deleted dir) for ``_NEG_TTL`` — hundreds of non-git session cwds
would otherwise re-spawn ``git`` on every sidebar open, while the TTL keeps ``git init`` re-probable."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from hermes_cli._subprocess_compat import bounded_git_probe

_GIT_TIMEOUT = 1.5
_WARM_WORKERS = 8
_NEG_TTL = 30.0  # "not a git repo" TTL: a fresh `git init` shows within seconds


def run_git(cwd: str, *args: str) -> str:
    """``git -C <cwd> <args>`` → stripped stdout, or ``""`` on any failure. ``bounded_git_probe``
    bounds post-kill cleanup on Windows (a killed git's suspended descendant held the pipes).

    Uses the shared :func:`bounded_git_probe` so the post-kill cleanup is bounded on Windows — a plain
    ``subprocess.run(timeout=...)`` here deadlocked Desktop session readiness when a killed git left a
    suspended descendant holding the pipe handles (issue #68609).
    """
    # A missing dir can only fail at the price of a fork; deleted worktrees dominate a long
    # session history's cwds, so the stat pays off.
    if not cwd or not os.path.isdir(cwd):
        return ""
    return bounded_git_probe(["git", "-C", cwd, *args], timeout=_GIT_TIMEOUT)


def branch(cwd: str) -> str:
    return run_git(cwd, "branch", "--show-current") or run_git(cwd, "rev-parse", "--short", "HEAD")


class _RootCache:
    """Thread-safe, single-flight cache of git-root probes: positives live for
    the process, negatives for ``_NEG_TTL``; followers wait on the leader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roots: dict[str, str] = {}
        self._neg: dict[str, float] = {}  # key -> monotonic expiry
        self._inflight: dict[str, threading.Event] = {}

    def invalidate(self) -> None:
        with self._lock:
            self._roots.clear()
            self._neg.clear()
            self._inflight.clear()

    def resolve(self, key: str, probe) -> str:
        while True:
            with self._lock:
                if hit := self._roots.get(key):
                    return hit
                expiry = self._neg.get(key)
                if expiry is not None:
                    if expiry > time.monotonic():
                        return ""  # recent "not a repo": trust it briefly
                    del self._neg[key]  # TTL elapsed: re-probe (may be a repo now)
                gate = self._inflight.get(key)
                leader = gate is None
                if leader:
                    gate = self._inflight[key] = threading.Event()
            if not leader:  # another thread is probing this key — wait, then re-read
                gate.wait(timeout=_GIT_TIMEOUT + 0.5)
                continue
            value = ""
            try:
                value = probe()
            finally:
                with self._lock:
                    if value:
                        self._roots[key] = value
                    else:
                        self._neg[key] = time.monotonic() + _NEG_TTL
                    self._inflight.pop(key, None)
                gate.set()
            return value


_cache = _RootCache()


def invalidate() -> None:
    """Drop cached roots after a known mutation (e.g. a worktree was added)."""
    _cache.invalidate()


def repo_root(cwd: str) -> str:
    """Top-level git repo root for ``cwd`` (``""`` when not a repo)."""
    return _cache.resolve(cwd, lambda: run_git(cwd, "rev-parse", "--show-toplevel")) if cwd else ""


def common_repo_root(cwd: str) -> str:
    """The MAIN (common) repo root for ``cwd``, folding linked worktrees: ``--show-toplevel`` is a
    linked worktree's OWN root; the parent of the shared ``--git-common-dir`` is the one true root
    (fallback: toplevel). Normalized to git's forward-slash spelling so it compares equal to
    :func:`repo_root` (native ``\\`` on Windows made the main checkout look like a worktree)."""
    # Checking the (warmed, negative-cached) toplevel first spares every non-repo cwd a second
    # `git` spawn the parallel warm can't absorb.
    if not cwd or not repo_root(cwd):
        return ""

    def _probe() -> str:
        gitdir = run_git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if gitdir:
            gitdir = os.path.realpath(gitdir)
            if os.path.basename(gitdir) == ".git":
                return os.path.dirname(gitdir).replace(os.sep, "/")
        return repo_root(cwd)

    return _cache.resolve(f"common:{cwd}", _probe)


def resolve(cwd: str) -> dict | None:
    """Inject-able resolver for ``project_tree.build_tree``: ``{repo_root: <common root>,
    worktree_root: <this checkout>}`` or None outside a repo (equal roots = main checkout)."""
    worktree_root = repo_root(cwd)
    if not worktree_root:
        return None
    return {"repo_root": common_repo_root(cwd) or worktree_root, "worktree_root": worktree_root}


def warm_roots(cwds: Iterable[str], max_workers: int = _WARM_WORKERS) -> None:
    """Pre-resolve many cwds' roots in parallel (bounded) so a cold first paint
    doesn't serialize one git spawn per session cwd; results land in the cache."""
    pending = sorted({(cwd or "").strip() for cwd in cwds} - {""})
    if len(pending) == 1:
        resolve(pending[0])
    elif pending:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
            list(pool.map(resolve, pending))
