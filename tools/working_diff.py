"""Working-tree git diff collection shared by the CLI and gateway ``/diff``.

Surface-agnostic so the CLI (colored terminal) and gateway (fenced, truncated
messages) render the same data. Modes: ``working`` (unstaged + untracked),
``staged`` (``git diff --cached``), ``all`` (everything since HEAD plus untracked).
Untracked files are folded in via ``git diff --no-index /dev/null <file>`` so
brand-new files show as additions instead of being invisible.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from contextlib import suppress
from typing import Dict, List, Optional

from hermes_cli._subprocess_compat import harden_git_argv, noninteractive_git_env

_GIT_TIMEOUT = 15
_MAX_UNTRACKED_FILES = 50  # sanity cap so a node_modules explosion can't hang us

_MODE_ARGS = {
    "working": ["diff"],
    "staged": ["diff", "--cached"],
    "all": ["diff", "HEAD"],
}
VALID_MODES = tuple(_MODE_ARGS)


@functools.lru_cache(maxsize=1)
def _git_command() -> Optional[List[str]]:
    """Resolve the git invocation: pm's pinned Git first, then system git.

    pm's git package is the canonical Windows git (Git for Windows,
    pinned in pm/lock.json) — it wins over PATH so a stale or broken
    system git never breaks diff collection. On POSIX pm deliberately
    gaps git (system git by choice), and when pm can't provide it for any
    other reason we fall back to bare ``git`` on PATH. None when git is
    nowhere — the caller reports it unavailable.
    """
    try:
        import pm

        runner = pm.ensure("git")
        for candidate in ("git.exe", "git"):
            resolved = shutil.which(candidate, path=runner.env.get("PATH"))
            if resolved:
                return [resolved]
    except Exception:
        pass
    return ["git"] if shutil.which("git") else None


def _run(args: List[str], cwd: str, timeout: int = _GIT_TIMEOUT):
    """Run git, returning (returncode, stdout). Never raises on git failure.

    Hardened against a malicious repo's ``.git/config`` (GHSA-7x36-8jrh-v4pw):
    ``noninteractive_git_env`` disables fsmonitor/hooks/pager/editor/credential
    sinks, and ``harden_git_argv`` appends ``--no-ext-diff --no-textconv`` to
    the diff-rendering subcommands so attribute-scoped diff/textconv drivers
    can't execute either.
    """
    command = _git_command()
    if command is None:
        return 127, ""
    proc = subprocess.run(
        [*command, "-c", "core.quotePath=false", *harden_git_argv(args)],
        cwd=cwd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, env=noninteractive_git_env(),
    )
    return proc.returncode, proc.stdout


def _untracked_files(cwd: str) -> List[str]:
    code, out = _run(["ls-files", "--others", "--exclude-standard"], cwd)
    return [line for line in out.splitlines() if line.strip()] if code == 0 else []


def _untracked_diff(cwd: str, files: List[str]) -> str:
    """Render untracked files as new-file diffs via ``git diff --no-index``."""
    chunks: List[str] = []
    for rel in files[:_MAX_UNTRACKED_FILES]:
        with suppress(subprocess.TimeoutExpired, OSError):
            # --no-index exits 1 when the files differ — that's the success
            # path here, so ignore the return code and keep the output.
            _, out = _run(
                ["diff", "--no-ext-diff", "--no-index", "--", os.devnull, rel], cwd,
            )
            if out.strip():
                chunks.append(out.rstrip("\n"))
    if len(files) > _MAX_UNTRACKED_FILES:
        chunks.append(f"... ({len(files) - _MAX_UNTRACKED_FILES} more untracked files not shown)")
    return "\n".join(chunks)


def collect_working_diff(cwd: str, mode: str = "working",
                         paths: List[str] | None = None) -> Dict:
    """Collect a git diff of the working directory.

    Returns ``{"success", "stat", "diff", "untracked", "empty"}`` on success or
    ``{"success": False, "error": ...}`` when git is unavailable / not a repo.
    ``paths`` optionally restricts the diff to specific pathspecs (passed
    through to git verbatim, so quoted paths with spaces survive).
    """
    if mode not in VALID_MODES:
        return {"success": False,
                "error": f"Unknown mode '{mode}'. Use: {', '.join(VALID_MODES)}"}

    if _git_command() is None:
        return {"success": False, "error": "git is not installed or not on PATH."}
    try:
        code, _ = _run(["rev-parse", "--is-inside-work-tree"], cwd, timeout=5)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"success": False, "error": f"git failed: {e}"}
    if code != 0:
        return {"success": False, "error": "Not a git repository."}

    # --no-ext-diff: a user-configured external differ (diff.external in
    # gitconfig, e.g. difftastic) replaces the unified-diff format that the
    # CLI/gateway renderers and truncation logic parse. Force the internal
    # diff engine so the collected output shape is stable for all users.
    # (_run's harden_git_argv also enforces this; explicit here for clarity.)
    base_args = [a for a in _MODE_ARGS[mode] if a != "diff"]
    base_args = ["diff", "--no-ext-diff", *base_args]
    pathspec = ["--", *paths] if paths else []
    try:
        _, stat_out = _run([*base_args, "--stat", *pathspec], cwd)
        _, diff_out = _run([*base_args, *pathspec], cwd, timeout=_GIT_TIMEOUT * 2)
        untracked = _untracked_files(cwd) if mode in ("working", "all") and not paths else []
        untracked_diff = _untracked_diff(cwd, untracked) if untracked else ""
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "git diff timed out."}
    except OSError as e:
        return {"success": False, "error": f"git failed: {e}"}

    stat, diff = stat_out.strip(), diff_out.strip()
    if untracked_diff:
        diff = f"{diff}\n{untracked_diff}".strip()
    result = {"success": True, "stat": stat, "diff": diff, "untracked": untracked}
    if not stat and not diff and not untracked:
        result["empty"] = True
    return result
