"""Git working-tree probing for the gateway: run git, resolve repo roots, fold
linked worktrees under their common root.

Probing runs where the gateway runs, so it resolves repos for both local and
remote backends (unlike the desktop's electron probe, which only sees the local
fs). Resolved roots are cached with a thread-safe, single-flight cache: the
gateway's long handlers run on worker threads, so concurrent identical probes
(e.g. two overlapping project-tree builds) share one `git` invocation instead of
racing an unguarded dict.

Positive results are cached for the process lifetime; negative results (a cwd
that isn't a git repo, or a deleted/nonexistent dir) are cached only for a short
TTL (`_NEG_TTL`). Caching negatives matters a lot for the desktop Projects tree:
``project_tree.build_tree`` resolves a cwd once *per session* (not per distinct
cwd), so a power user with hundreds of sessions in non-git/deleted dirs would
otherwise re-spawn ``git`` hundreds of times on *every* sidebar open — the cause
of the multi-second "Projects" load. The TTL keeps a not-yet-repo cwd
re-probable (we `git init` a new project's folder on its first worktree, and a
frozen "" would mislabel its main lane by the dir basename) — it just stops the
same "not a repo" answer from being re-derived dozens of times within one build
and across rapid re-opens. `invalidate()` drops everything after a known
mutation.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

_GIT_TIMEOUT = 1.5
_WARM_WORKERS = 8

# How long a "not a git repo" answer stays cached before it's re-probed. Short
# enough that a freshly `git init`-ed / newly-created folder shows correctly
# within a few seconds; long enough to collapse the hundreds of redundant probes
# a single project-tree build (and rapid re-opens) would otherwise fire.
_NEG_TTL = 30.0


def run_git(cwd: str, *args: str) -> str:
    """``git -C <cwd> <args>`` → stripped stdout, or ``""`` on any failure."""
    if not cwd:
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def branch(cwd: str) -> str:
    return run_git(cwd, "branch", "--show-current") or run_git(cwd, "rev-parse", "--short", "HEAD")


class _RootCache:
    """Thread-safe, single-flight cache of git-root probes. Positive results are
    cached for the process lifetime; negative ("not a repo") results are cached
    only for ``_NEG_TTL`` seconds so a not-yet-repo cwd stays re-probable.
    Followers wait on the leader's probe instead of duplicating it."""

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
                hit = self._roots.get(key)
                if hit:
                    return hit
                expiry = self._neg.get(key)
                if expiry is not None:
                    if expiry > time.monotonic():
                        # Recently probed as "not a repo" — trust it briefly
                        # instead of re-spawning git for the same dead/non-repo
                        # cwd on every session in the tree build.
                        return ""
                    # TTL elapsed: drop it and re-probe (it may be a repo now).
                    del self._neg[key]
                gate = self._inflight.get(key)
                if gate is None:
                    gate = threading.Event()
                    self._inflight[key] = gate
                    leader = True
                else:
                    leader = False

            if not leader:
                # Another thread is probing this key — wait, then re-read.
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
    if not cwd:
        return ""
    return _cache.resolve(cwd, lambda: run_git(cwd, "rev-parse", "--show-toplevel"))


def common_repo_root(cwd: str) -> str:
    """The MAIN (common) repo root for ``cwd``, folding linked worktrees.

    ``--show-toplevel`` returns a linked worktree's OWN root, so grouping by it
    splits every worktree into a separate "repo". The common ``.git`` dir
    (``--git-common-dir``) is shared by a repo and all its worktrees, so its
    parent is the one true repo root; fall back to the toplevel root otherwise.
    """
    if not cwd:
        return ""

    def _probe() -> str:
        gitdir = run_git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if gitdir:
            gitdir = os.path.realpath(gitdir)
            if os.path.basename(gitdir) == ".git":
                return os.path.dirname(gitdir)
        return repo_root(cwd)

    return _cache.resolve(f"common:{cwd}", _probe)


def resolve(cwd: str) -> dict | None:
    """Inject-able resolver for ``project_tree.build_tree``.

    Returns ``{"repo_root": <common root>, "worktree_root": <this checkout>}``
    or ``None`` when ``cwd`` is not in a git repo. ``build_tree`` treats
    ``worktree_root == repo_root`` as the main checkout.
    """
    worktree_root = repo_root(cwd)
    if not worktree_root:
        return None
    return {"repo_root": common_repo_root(cwd) or worktree_root, "worktree_root": worktree_root}


def warm_roots(cwds: Iterable[str], max_workers: int = _WARM_WORKERS) -> None:
    """Pre-resolve many cwds' roots in parallel (bounded) so a cold first paint
    doesn't serialize one git subprocess per session cwd. Single-flight dedupes
    overlap; results land in the shared cache for the sequential consumers."""
    pending = sorted({(cwd or "").strip() for cwd in cwds} - {""})
    if not pending:
        return
    if len(pending) == 1:
        resolve(pending[0])
        return
    with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
        list(pool.map(resolve, pending))
