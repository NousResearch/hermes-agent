"""Best-effort workspace metadata for the TUI status HUD.

This module is deliberately read-only. Git probes run on the same machine as the
Hermes gateway, which keeps the HUD correct for SSH/remote backends as well as a
local TUI. Every value returned to the client is normalized or bounded before it
leaves this module; raw remotes, command output, and PR URLs never cross the RPC
boundary.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from urllib.parse import unquote, urlsplit

from . import git_probe

_CACHE_TTL = 12.0
_GH_TIMEOUT = 2.0
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_URL_RE = re.compile(r"(?:https?|ssh)://[^\s]+|git@github\.com:[^\s]+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"\b(?:token|password|passwd|secret|api[_-]?key|authorization)\s*[:=]\s*[^\s]+",
    re.IGNORECASE,
)

_EMPTY = {
    "git_root": None,
    "github": None,
    "branch": None,
    "dirty": None,
    "upstream": None,
    "pull_request": None,
}


def _copy_payload(value: dict) -> dict:
    """Copy the small nested payload so callers cannot mutate the cache."""
    return {
        **value,
        "github": dict(value["github"]) if isinstance(value.get("github"), dict) else None,
        "upstream": dict(value["upstream"]) if isinstance(value.get("upstream"), dict) else None,
        "pull_request": (
            dict(value["pull_request"])
            if isinstance(value.get("pull_request"), dict)
            else None
        ),
    }


def empty_workspace() -> dict:
    """Return an unavailable workspace payload without exposing probe errors."""
    return _copy_payload(_EMPTY)


def _repo_from_path(path: str) -> dict | None:
    raw = path.strip().strip("/")
    if raw.lower().endswith(".git"):
        raw = raw[:-4].rstrip("/")
    try:
        raw = unquote(raw)
    except Exception:
        return None
    parts = raw.split("/")
    if len(parts) != 2:
        return None
    owner, repo = parts
    if not _NAME_RE.fullmatch(owner) or not _NAME_RE.fullmatch(repo):
        return None
    return {"owner": owner, "repo": repo, "full_name": f"{owner}/{repo}"}


def normalize_github_origin(raw: str | None) -> dict | None:
    """Normalize a GitHub remote to ``owner/repo`` without returning the URL."""
    value = (raw or "").strip()
    if not value:
        return None

    if value.lower().startswith("git@github.com:"):
        return _repo_from_path(value.split(":", 1)[1])

    candidate = value
    if "://" not in candidate and (
        candidate.lower().startswith("github.com/")
        or candidate.lower().startswith("www.github.com/")
    ):
        candidate = f"https://{candidate}"

    try:
        parsed = urlsplit(candidate)
    except Exception:
        return None
    if (parsed.hostname or "").lower() not in _GITHUB_HOSTS:
        return None
    return _repo_from_path(parsed.path)


def _safe_title(value: str, limit: int = 80) -> str:
    text = _URL_RE.sub("[url]", value)
    text = _SECRET_RE.sub("[redacted]", text)
    text = _CONTROL_RE.sub(" ", text)
    text = " ".join(text.split()).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def parse_pull_request_payload(value: object) -> dict | None:
    """Validate and sanitize the small subset returned by ``gh pr view``."""
    if not isinstance(value, dict):
        return None
    number = value.get("number")
    if isinstance(number, bool):
        return None
    try:
        number = int(number)
    except (TypeError, ValueError):
        return None
    if number < 1:
        return None

    title = value.get("title")
    title = _safe_title(title if isinstance(title, str) else "")
    state = value.get("state")
    state = state.strip().lower() if isinstance(state, str) else ""
    if state not in {"open", "closed", "merged"}:
        state = ""
    head_ref_name = value.get("headRefName")
    head_ref_name = (
        _safe_title(head_ref_name, limit=200) if isinstance(head_ref_name, str) else ""
    )
    return {
        "number": number,
        "title": title,
        "state": state,
        "head_ref_name": head_ref_name,
    }


def _upstream_counts(raw: str) -> dict | None:
    parts = raw.split()
    if len(parts) != 2:
        return None
    try:
        ahead = int(parts[0])
        behind = int(parts[1])
    except (TypeError, ValueError):
        return None
    if behind < 0 or ahead < 0:
        return None
    return {"ahead": ahead, "behind": behind}


def _dirty_from_status(raw: str) -> bool | None:
    """Return Git's clean/dirty state, or None when the status probe failed."""
    lines = raw.splitlines()
    if not lines or not lines[0].startswith("## "):
        return None
    return any(not line.startswith("## ") for line in lines[1:])


def _run_gh(root: str, repo: dict) -> dict | None:
    """Ask gh for the current PR, keeping all command output private."""
    env = os.environ.copy()
    env["GH_PROMPT_DISABLED"] = "1"
    env["GH_NO_UPDATE_NOTIFIER"] = "1"
    try:
        completed = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                "--repo",
                repo["full_name"],
                "--json",
                "number,title,state,headRefName",
            ],
            cwd=root,
            capture_output=True,
            check=False,
            env=env,
            text=True,
            timeout=_GH_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return parse_pull_request_payload(json.loads(completed.stdout))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _probe(cwd: str) -> dict:
    worktree_root = git_probe.repo_root(cwd)
    if not worktree_root:
        return empty_workspace()

    worktree_root = os.path.realpath(worktree_root)
    canonical_root = os.path.realpath(
        git_probe.common_repo_root(cwd) or worktree_root
    )
    branch_name = git_probe.run_git(cwd, "branch", "--show-current")
    branch = branch_name or git_probe.branch(cwd) or None
    origin = normalize_github_origin(git_probe.run_git(cwd, "config", "--get", "remote.origin.url"))
    status = git_probe.run_git(
        cwd,
        "status",
        "--porcelain=v1",
        "--branch",
        "--untracked-files=normal",
    )
    dirty = _dirty_from_status(status)
    upstream = _upstream_counts(
        git_probe.run_git(cwd, "rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    )

    pull_request = _run_gh(worktree_root, origin) if origin and branch_name else None
    if pull_request and pull_request.get("head_ref_name") and pull_request["head_ref_name"] != branch_name:
        pull_request = None
    if pull_request:
        pull_request.pop("head_ref_name", None)

    return {
        "git_root": canonical_root,
        "github": origin,
        "branch": branch,
        "dirty": dirty,
        "upstream": upstream,
        "pull_request": pull_request,
    }


_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}
_inflight: dict[str, threading.Event] = {}


def resolve(cwd: str | None) -> dict:
    """Return cached workspace metadata; all probe failures degrade to empty."""
    key = (cwd or "").strip()
    if not key:
        return empty_workspace()

    while True:
        now = time.monotonic()
        with _cache_lock:
            hit = _cache.get(key)
            if hit and now - hit[0] < _CACHE_TTL:
                return _copy_payload(hit[1])
            gate = _inflight.get(key)
            if gate is None:
                gate = threading.Event()
                _inflight[key] = gate
                leader = True
            else:
                leader = False

        if not leader:
            gate.wait(timeout=_GH_TIMEOUT + 1.0)
            continue

        try:
            value = _probe(key)
        except Exception:
            value = empty_workspace()
        with _cache_lock:
            _cache[key] = (time.monotonic(), value)
            _inflight.pop(key, None)
        gate.set()
        return _copy_payload(value)
