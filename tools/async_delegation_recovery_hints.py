"""Forensic hints attached to an abandoned-delegation recovery event.

When the owner process dies (OOM-kill, crash) before a background child records a
terminal result, the recovery event used to carry only "outcome unknown" plus the
transcript paths — the parent then had to open files to find out where the child
got to. These helpers put the last lines of each live transcript and a one-line git
snapshot of the owner's working directory INTO the event so the parent can decide
whether to continue or re-dispatch without forensics. Everything here is best-effort
and bounded: a missing file, a non-git cwd or a slow git never fails recovery.
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, Optional

TAIL_LINES = 20
TAIL_CHARS = 2_000
_GIT_TIMEOUT_S = 5


def transcript_tail(path: str, *, lines: int = TAIL_LINES, chars: int = TAIL_CHARS) -> Optional[str]:
    """Last ``lines`` lines (at most ``chars`` characters) of a live transcript; None when unreadable."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - chars * 4))  # a few bytes per char of headroom; bounded read
            raw = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    tail = "\n".join(raw.splitlines()[-lines:]).strip()
    if len(tail) > chars:
        tail = "…" + tail[-chars:]
    return tail or None


def transcript_tails(paths: Dict[str, str]) -> Dict[str, str]:
    return {index: tail for index, path in (paths or {}).items() if (tail := transcript_tail(path))}


def git_state_hint(cwd: Optional[str]) -> Optional[str]:
    """``branch @ sha (N uncommitted file(s))`` for the owner's cwd; None when not a git checkout."""
    if not cwd or not os.path.isdir(cwd):
        return None

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace",
                                 stdin=subprocess.DEVNULL, timeout=_GIT_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    head = run("log", "-1", "--format=%h %s")
    if head is None:
        return None
    branch = (run("rev-parse", "--abbrev-ref", "HEAD") or "?").strip()
    status = run("status", "--porcelain")
    dirty = len(status.splitlines()) if status is not None else "?"
    return f"{branch} @ {head.strip()} ({dirty} uncommitted file(s)) in {cwd}"
