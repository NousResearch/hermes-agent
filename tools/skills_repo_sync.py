"""Git-repo-backed skills: the configured repo IS the agent's skills home.

When ``skills.external_repo.enabled`` is true, Hermes uses a git repository
as the live home for agent-created skills instead of (only) the local
``~/.hermes/skills/`` dir.  This is the multi-install answer to keeping a
server, a laptop, and the desktop app on the exact same skills: one repo is
the source of truth, every install works directly on it, and changes flow
both ways.

Config::

    skills:
      external_repo:
        enabled: true
        url: "https://github.com/owner/skills.git"
        branch: "main"     # optional, default = remote's default branch
        path: ""           # optional subdir of the repo holding SKILL.md files

Lifecycle
---------
- **Startup** (``sync_external_repo``): clone on first run, then fetch +
  rebase onto the remote on every later start, into
  ``$HERMES_HOME/skills/.repo-sync/<slug>/``.  The checkout is exposed like
  an ``external_dirs`` entry, so every skill reader sees the repo's skills.
- **Write-back** (``maybe_push_external_repo``): when the agent creates or
  edits a skill that lives in the checkout, the change is committed and
  pushed back to the repo.  Debounced from the ``skill_manage`` hook so a
  burst of edits collapses to one push.  Before pushing, local commits are
  reconciled with the remote (fetch + rebase) so concurrent edits from
  another install never wedge the checkout; on a genuine conflict the
  rebase is aborted and the change stays local for the next try.
  Best-effort and never raises — a push failure (offline, auth, conflict)
  only logs a debug line, and the change stays in the local checkout,
  ready for the next successful push.

Failure model (must never break the agent)
------------------------------------------
The repo is a convenience, not a blocking dependency.  If git is missing,
the network is down, a branch is missing, or a push conflicts, we log a
debug line and continue — the agent still works on the checkout it has.
This module never raises out of its public entrypoints.

Local skills dir
----------------
``~/.hermes/skills/`` remains the only place bundled skills live and the
fallback when ``external_repo`` is disabled or not yet cloned.  When the
feature is on, *new* skills are created in the repo checkout and edits to
repo skills happen in place there — see ``tools/skill_manager_tool.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Subdirectory of HERMES_HOME/skills/ holding repo checkouts.  Hidden so the
# regular skills scanners (EXCLUDED_SKILL_DIRS) never mistake a checkout for
# a locally installed skill.
CHECKOUT_ROOT_NAME = ".repo-sync"

# How long a single git invocation may take before we give up.
_GIT_TIMEOUT = 60

_URL_SLUG_STRIP = re.compile(r"^[a-zA-Z0-9_.-]+$")


def get_external_repo_config() -> Optional[Dict[str, Any]]:
    """Read ``skills.external_repo`` from config.yaml, or None when unset."""
    try:
        from agent.skill_utils import _load_raw_config
    except ImportError:
        return None
    parsed = _load_raw_config()
    if not parsed:
        return None
    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return None
    repo_cfg = skills_cfg.get("external_repo")
    if not isinstance(repo_cfg, dict):
        return None
    return repo_cfg


def external_repo_enabled() -> bool:
    """True when the feature is configured on with a usable url."""
    repo_cfg = get_external_repo_config()
    if not repo_cfg or not repo_cfg.get("enabled"):
        return False
    return bool(str(repo_cfg.get("url", "")).strip())


def _repo_slug(url: str) -> str:
    """Stable filesystem-safe name for a repo URL.

    ``https://github.com/owner/skills.git`` and
    ``git@github.com:owner/skills.git`` both slug to ``owner-skills`` so
    two repos with the same short name but different owners never collide;
    anything that would not survive as a directory name falls back to a
    short hash of the URL so the slug is always unique per URL.
    """
    bare = url.rstrip("/")
    if bare.endswith(".git"):
        bare = bare[:-4]
    segments = [s for s in re.split(r"[/:]", bare) if s]
    tail = segments[-1] if segments else ""
    owner = segments[-2] if len(segments) >= 2 else ""
    candidate = f"{owner}-{tail}" if owner else tail
    if candidate and _URL_SLUG_STRIP.match(candidate) and candidate not in (".", ".."):
        return candidate
    return "repo-" + hashlib.md5(url.encode("utf-8")).hexdigest()[:8]


def get_checkout_dir(url: str) -> Path:
    """Absolute path of the checkout for *url* (may not exist yet)."""
    return get_hermes_home() / "skills" / CHECKOUT_ROOT_NAME / _repo_slug(url)


def get_repo_skills_dir(url: str, subdir: str = "") -> Optional[Path]:
    """Resolve the directory inside the checkout that holds the skills.

    Returns the checkout root when *subdir* is empty, the subdir when it is
    set and exists, and ``None`` when the checkout (or requested subdir) is
    not present on disk.  Read-only: never clones, never touches the network
    — this is what the hot-path skill scanners call.
    """
    checkout = get_checkout_dir(url)
    if not checkout.is_dir():
        return None
    if not subdir:
        return checkout
    target = checkout / subdir
    return target if target.is_dir() else None


def get_repo_write_dir() -> Optional[Path]:
    """Where new agent skills should be created when the feature is on.

    Returns the configured repo's skills subdir under the checkout, or None
    when the feature is disabled or the url is missing.  Unlike
    ``get_repo_skills_dir`` this does NOT require the dir to exist yet —
    creation will make it — but it does require the checkout to have been
    cloned (startup sync), otherwise the repo would be rewritten from a
    stale clone on the next push.
    """
    repo_cfg = get_external_repo_config()
    if not repo_cfg or not repo_cfg.get("enabled"):
        return None
    url = str(repo_cfg.get("url", "")).strip()
    if not url:
        return None
    checkout = get_checkout_dir(url)
    if not (checkout / ".git").exists():
        return None
    subdir = str(repo_cfg.get("path", "")).strip()
    base = checkout / subdir if subdir else checkout
    base.mkdir(parents=True, exist_ok=True)
    return base


def _run_git(args: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run a git subcommand; return stdout (stripped) or None on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            cwd=str(cwd) if cwd else None,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("git %s failed: %s", args[0], e)
        return None
    if proc.returncode != 0:
        logger.debug("git %s exited %d: %s", args[0], proc.returncode,
                     proc.stderr.strip()[:200])
        return None
    return proc.stdout.strip()


def _reconcile_checkout(checkout: Path) -> bool:
    """Fetch and fast-forward/rebase the checkout onto the remote.

    Handles every checkout shape without ever raising:

    * clean clone: rebase local commits onto the remote (or fast-forward);
    * unborn HEAD + empty remote (cloned before the first skill existed):
      nothing to reconcile, returns True;
    * unborn HEAD + remote that grew meanwhile: materialize the branch at
      the remote HEAD;
    * local commits + remote moved: rebase replays them on top;
    * genuine conflict: abort and return False, local work stays put.
    """
    if _run_git(["fetch"], cwd=checkout) is None:
        return False
    # Branch exists and tracks an existing remote ref: plain rebase.
    if _run_git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=checkout) is not None:
        upstream = _run_git(
            ["rev-parse", "--verify", "--quiet", "@{u}"], cwd=checkout
        )
        if upstream:
            if _run_git(["rebase"], cwd=checkout) is None:
                _run_git(["rebase", "--abort"], cwd=checkout)
                return False
        return True
    # Unborn HEAD: the checkout was cloned while the remote was still empty.
    remote_head = _run_git(
        ["rev-parse", "--verify", "--quiet", "origin/HEAD"], cwd=checkout
    )
    if remote_head:
        return _run_git(["reset", "--hard", "origin/HEAD"], cwd=checkout) is not None
    return True  # empty remote, empty checkout — nothing to sync yet


def sync_external_repo(quiet: bool = True) -> Optional[Path]:
    """Clone or update the configured external skills repo at startup.

    Returns the resolved skills directory inside the checkout when the sync
    succeeded (or the checkout was already current), ``None`` when the
    feature is disabled, misconfigured, or the sync failed.  On success the
    in-process external-dirs cache is invalidated so the fresh checkout is
    picked up by skill scanners in the current process.
    """
    repo_cfg = get_external_repo_config()
    if not repo_cfg or not repo_cfg.get("enabled"):
        return None
    url = str(repo_cfg.get("url", "")).strip()
    if not url:
        logger.debug("skills.external_repo enabled but no url; skipping")
        return None
    branch = str(repo_cfg.get("branch", "")).strip()
    subdir = str(repo_cfg.get("path", "")).strip()

    checkout = get_checkout_dir(url)
    try:
        checkout.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("Could not create checkout dir %s: %s", checkout, e)
        return None

    # Clone on first run; reconcile with the remote after.  Full clone (not
    # shallow): skill repos are tiny and a truncated history makes later
    # fetches/rebase-pushes fragile.
    cloned = not (checkout / ".git").exists()
    if cloned:
        clone_args = ["clone"]
        if branch:
            clone_args += ["--branch", branch]
        clone_args += [url, str(checkout)]
        ok = _run_git(clone_args) is not None
    else:
        ok = _reconcile_checkout(checkout)

    if not ok:
        if not quiet:
            logger.warning("skills.external_repo sync failed for %s", url)
        return None

    try:
        from agent.skill_utils import _external_dirs_cache_clear

        _external_dirs_cache_clear()
    except ImportError:
        pass

    if not quiet:
        state = "cloned" if cloned else "updated"
        logger.info("skills.external_repo %s from %s (%s)", state, url, branch)
    return get_repo_skills_dir(url, subdir)


def maybe_push_external_repo(message: str = "hermes skill sync") -> bool:
    """Best-effort commit+push of the configured repo checkout.

    Returns True when a push actually happened, False when the feature is
    off, nothing changed, or the push failed.  Never raises.  Called from
    the debounced ``skill_manage`` write hook so a burst of edits collapses
    to one push.
    """
    repo_cfg = get_external_repo_config()
    if not repo_cfg or not repo_cfg.get("enabled"):
        return False
    url = str(repo_cfg.get("url", "")).strip()
    if not url:
        return False
    checkout = get_checkout_dir(url)
    if not (checkout / ".git").exists():
        return False

    # 1. Commit any pending worktree changes first (works offline).
    if _run_git(["add", "-A"], cwd=checkout) is None:
        return False
    status = _run_git(["status", "--porcelain"], cwd=checkout)
    if status is None:
        return False
    if status.strip():
        # Commit with a pinned one-shot identity. The checkout is a fresh clone
        # with no user.name/email configured (git clone does not inherit the
        # source's identity), so a bare `git commit` would fail with "Please
        # tell me who you are". -c scopes the identity to this single command
        # instead of mutating the checkout's config.
        if (
            _run_git(
                [
                    "-c",
                    "user.name=Hermes Agent",
                    "-c",
                    "user.email=hermes@nousresearch.com",
                    "commit",
                    "-m",
                    f"hermes: {message[:200]}",
                ],
                cwd=checkout,
            )
            is None
        ):
            return False

    # 2. Reconcile with the remote: another install may have pushed while we
    #    were offline, and a bare `git push` would then fail (non-fast-
    #    forward) — worse, the next pull would refuse too, wedging the
    #    checkout forever.  Reconcile replays our local commits on top of
    #    the remote (or materializes the branch on a first-ever push to an
    #    empty repo); on a genuine conflict we abort and leave the local
    #    work for the next try.
    if not _reconcile_checkout(checkout):
        logger.debug("skills.external_repo reconcile failed for %s", url)
        return False

    # 3. Nothing to send? Stop here (no-op pushes are pointless work).
    ahead = _run_git(
        ["rev-list", "--count", "HEAD", "--not", "--remotes"], cwd=checkout
    )
    if ahead is None or ahead == "0":
        return False

    if _run_git(["push"], cwd=checkout) is None:
        logger.debug("skills.external_repo push failed for %s", url)
        return False
    logger.info("skills.external_repo pushed: %s", message)
    return True