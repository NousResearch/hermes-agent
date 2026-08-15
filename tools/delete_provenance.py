"""Provenance gate for deleting artifacts the session did not create (#84718).

Issue #84718 traces a session that, after two compactions, worked toward
deleting a checkout route that pre-dated the task entirely. The investigation
todo that established "this route lives in git history, untouched by this
session" had already completed; compaction re-injected the *action* item
without that finding, and nothing checked provenance before acting.

This module answers the one question that would have stopped it: does the path
about to be deleted pre-date this session's work? A path with commit history
was created by somebody else, at some other time, for some other reason — so
the first delete attempt is refused with the commit quoted, and an identical
retry proceeds. That is a guidance gate, not an execution barrier: the model is
never wedged, it just cannot delete pre-existing work without the provenance
having crossed its context first.

Fails open by design. No git, no repo, no history, a slow subprocess, or an
unreadable path all mean "proceed" — a provenance check must never be the
reason a legitimate delete fails.

Set ``HERMES_DELETE_PROVENANCE_GATE=0`` to disable.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Optional, Set, Tuple

# Acknowledged (repo, path, sha) triples. Process-scoped and deliberately not
# persisted: the acknowledgement is the model having *seen* the provenance in
# this context, which is exactly what compaction destroyed in #84718. A fresh
# process re-asks, which is the desired behavior.
_acknowledged: Set[Tuple[str, str, str]] = set()
_lock = threading.Lock()

# git must never stall a tool call.
_GIT_TIMEOUT_SECONDS = 5
# Bound the acknowledgement set so a long-lived process can't grow it without
# limit. Far above any realistic per-session delete count.
_MAX_ACKNOWLEDGED = 512


def _gate_enabled() -> bool:
    raw = (os.environ.get("HERMES_DELETE_PROVENANCE_GATE") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _run_git(args: list[str], cwd: str) -> Optional[str]:
    """Run a git command, returning stripped stdout or None on any failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def reset_acknowledgements() -> None:
    """Clear acknowledged deletions (tests, and session boundaries)."""
    with _lock:
        _acknowledged.clear()


def check_delete_provenance(path: "str | os.PathLike[str]") -> Optional[str]:
    """Return a confirmation message when *path* pre-dates this session.

    Returns ``None`` when the delete may proceed: the gate is disabled, the
    path is untracked or has no commit history (so this session, or nobody,
    created it), git is unavailable, or the same commit was already surfaced
    for the same path and the model is retrying deliberately.
    """
    if not _gate_enabled():
        return None
    try:
        target = Path(path).resolve()
    except (OSError, ValueError):
        return None

    try:
        search_dir = str(target.parent if target.parent.exists() else Path.cwd())
    except OSError:
        return None
    repo_root = _run_git(["rev-parse", "--show-toplevel"], search_dir)
    if not repo_root:
        return None

    log = _run_git(
        [
            "log",
            "-1",
            "--format=%h|%ad|%an|%s",
            "--date=short",
            "--",
            str(target),
        ],
        repo_root,
    )
    if not log:
        # Untracked, or tracked with no commit touching it — nothing pre-dates
        # this session, so there is no provenance to confirm.
        return None

    parts = log.split("|", 3)
    sha = parts[0] if parts else ""
    date = parts[1] if len(parts) > 1 else "?"
    author = parts[2] if len(parts) > 2 else "?"
    subject = parts[3] if len(parts) > 3 else ""
    if not sha:
        return None
    if len(subject) > 120:
        subject = subject[:117].rstrip() + "..."

    try:
        display = str(target.relative_to(repo_root))
    except ValueError:
        display = str(target)

    key = (repo_root, display, sha)
    with _lock:
        if key in _acknowledged:
            return None
        if len(_acknowledged) >= _MAX_ACKNOWLEDGED:
            _acknowledged.clear()
        _acknowledged.add(key)

    return (
        f"{display}: refusing to delete a pre-existing artifact without "
        f"confirmation. It was last changed in commit {sha} ({date}, {author})"
        f"{': ' + subject if subject else ''} — it pre-dates this session and "
        "was not created by this task. If deleting it is genuinely what the "
        "latest user message asks for, re-issue the identical patch to "
        "confirm. If the plan item that called for this delete came from an "
        "earlier context window, re-read the request before repeating it."
    )
