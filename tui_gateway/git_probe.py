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
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from hermes_cli._subprocess_compat import bounded_git_probe

_GIT_TIMEOUT = 1.5
_WARM_WORKERS = 8

# How long a "not a git repo" answer stays cached before it's re-probed. Short
# enough that a freshly `git init`-ed / newly-created folder shows correctly
# within a few seconds; long enough to collapse the hundreds of redundant probes
# a single project-tree build (and rapid re-opens) would otherwise fire.
_NEG_TTL = 30.0


def run_git(cwd: str, *args: str) -> str:
    """``git -C <cwd> <args>`` → stripped stdout, or ``""`` on any failure.

    Uses the shared :func:`bounded_git_probe` so the post-kill cleanup is bounded
    on Windows — a plain ``subprocess.run(timeout=...)`` here deadlocked Desktop
    session readiness when a killed git left a suspended descendant holding the
    pipe handles (issue #68609).
    """
    if not cwd or not os.path.isdir(cwd):
        # `git -C` on a directory that no longer exists can only fail, and it
        # fails at the price of a fork. Deleted worktrees dominate the cwds a
        # long-lived session history hands us, so the stat pays for itself many
        # times over on every project-tree build.
        return ""
    return bounded_git_probe(["git", "-C", cwd, *args], timeout=_GIT_TIMEOUT)


def branch(cwd: str) -> str:
    return run_git(cwd, "branch", "--show-current") or run_git(cwd, "rev-parse", "--short", "HEAD")


def capture_run_binding(
    cwd: str,
    *,
    session_key: str = "",
    ui_session_id: str = "",
    profile: str = "",
    owner_generation: str = "",
    transport_generation: str = "",
):
    """Capture immutable checkout and live-session identity for one turn.

    The gateway is the authority for the session fields.  Git is probed here
    at the moment of capture and again immediately before delegation; no
    model-supplied path, branch, or SHA is accepted.
    """
    from gateway.session_context import RunBinding

    try:
        canonical_cwd = os.path.realpath(os.path.abspath(os.path.expanduser(cwd)))
    except (TypeError, ValueError):
        canonical_cwd = ""
    if not canonical_cwd or not os.path.isdir(canonical_cwd):
        raise ValueError("Unable to bind this conversation: working directory is unavailable.")

    worktree = run_git(canonical_cwd, "rev-parse", "--show-toplevel")
    head = run_git(canonical_cwd, "rev-parse", "HEAD")
    common_dir = run_git(
        canonical_cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    if not worktree or not head or not common_dir:
        raise ValueError(
            "Unable to bind this conversation: the selected working directory is not a complete Git worktree."
        )
    worktree = os.path.realpath(worktree)
    common_dir = os.path.realpath(common_dir)
    repo = os.path.dirname(common_dir) if os.path.basename(common_dir) == ".git" else common_repo_root(canonical_cwd)
    repo = os.path.realpath(repo or worktree)
    ref = run_git(canonical_cwd, "symbolic-ref", "--quiet", "HEAD") or "HEAD"
    current_branch = run_git(canonical_cwd, "branch", "--show-current")
    return RunBinding(
        cwd=canonical_cwd,
        repo_root=repo,
        worktree_root=worktree,
        git_common_dir=common_dir,
        branch=current_branch or "HEAD",
        ref=ref,
        head=head.strip(),
        session_key=str(session_key or ""),
        ui_session_id=str(ui_session_id or ""),
        profile=str(profile or ""),
        owner_generation=str(owner_generation or ""),
        transport_generation=str(transport_generation or ""),
    )


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

    The returned path is normalized to git's forward-slash spelling so it can be
    compared against :func:`repo_root` (which returns raw ``--show-toplevel``
    output). ``os.path.realpath`` rewrites separators to the platform's native
    ``\\`` on Windows, so without this the SAME directory came back spelled two
    ways and the repo's own checkout compared unequal to its common root — the
    main checkout was then misread as a linked worktree and the desktop sidebar
    rendered it twice (a dir-labeled lane plus a branch-labeled ``main`` lane).
    """
    if not cwd:
        return ""

    # No work tree, nothing to fold. Reading the (warmed, negative-cached)
    # toplevel first spares every non-repo cwd a second `git` spawn — one the
    # parallel warm can never absorb, since `resolve()` only reaches here for
    # cwds that ARE repos.
    if not repo_root(cwd):
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
