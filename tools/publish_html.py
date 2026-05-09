#!/usr/bin/env python3
"""
publish_html — publish an HTML page to the hermes-pages Cloudflare Pages site.

Workflow:
  1. Hermes calls publish_html(slug, html_content) with a human-readable slug
     and the page content.
  2. Tool clones rousegordon-ops/hermes-pages on first use to
     /opt/data/hermes-pages-repo, pulls thereafter.
  3. Generates a 12-char hash for unguessable URLs (URL-only auth model:
     no Cloudflare Access, security through obscurity).
  4. Writes <hash>-<slug>.html, commits, pushes to origin/main.
  5. Cloudflare Pages auto-deploys on push.
  6. Returns the public URL as a JSON-encoded result.

Configuration via env vars:
  HERMES_PAGES_REPO_URL     — git URL of the pages repo
                              (default: https://github.com/rousegordon-ops/hermes-pages.git)
  HERMES_PAGES_REPO_DIR     — local clone path
                              (default: /opt/data/hermes-pages-repo)
  HERMES_PAGES_BASE_URL     — public base URL where files are served
                              (default: https://hermes-pages.rouse-gordon.workers.dev)
  GITHUB_TOKEN              — push credential (same token the source-watcher uses)
"""

import json
import logging
import os
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------- Configuration ----------

DEFAULT_REPO_URL = "https://github.com/rousegordon-ops/hermes-pages.git"
DEFAULT_REPO_DIR = "/opt/data/hermes-pages-repo"
DEFAULT_BASE_URL = "https://hermes-pages.rouse-gordon.workers.dev"

HASH_LEN = 12
SLUG_MAX_LEN = 60
HASH_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

# Single-process lock so concurrent agent calls don't fight over the local
# clone. cross-container concurrency is left to git itself (push retries).
_publish_lock = threading.Lock()


# ---------- Helpers ----------

def _slugify(text: str) -> str:
    """Normalize a slug: lowercase, hyphenate non-alnum runs, trim. Empty in → 'page'."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        text = "page"
    return text[:SLUG_MAX_LEN].rstrip("-") or "page"


def _generate_hash() -> str:
    """12 chars from [a-z0-9] — ~62 bits of entropy, URL-safe."""
    return "".join(secrets.choice(HASH_ALPHABET) for _ in range(HASH_LEN))


def _git(*args: str, cwd: str, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run git with a timeout. Captures both streams. Raises CalledProcessError on check=True failure."""
    return subprocess.run(
        ["git", "-C", cwd, *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def _authed_repo_url(base_url: str, token: Optional[str]) -> str:
    """Inject GITHUB_TOKEN into HTTPS git URL for push auth, mirroring the source-watcher pattern."""
    if not token or not base_url.startswith("https://"):
        return base_url
    return base_url.replace("https://", f"https://x-access-token:{token}@", 1)


def _ensure_repo(repo_url: str, repo_dir: str) -> Optional[str]:
    """Clone if missing, pull if present. Returns None on success, error string on failure."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    authed_url = _authed_repo_url(repo_url, token)
    repo_path = Path(repo_dir)
    if not (repo_path / ".git").is_dir():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "50", authed_url, str(repo_path)],
                capture_output=True, text=True, check=True, timeout=120,
            )
        except subprocess.CalledProcessError as err:
            return f"git clone failed: {err.stderr.strip() or err.stdout.strip() or 'unknown error'}"
        except subprocess.TimeoutExpired:
            return "git clone timed out after 120s"
        return None
    try:
        _git("remote", "set-url", "origin", authed_url, cwd=repo_dir, check=False)
        _git("fetch", "origin", "main", cwd=repo_dir, timeout=60)
        _git("reset", "--hard", "origin/main", cwd=repo_dir, timeout=30)
    except subprocess.CalledProcessError as err:
        return f"git refresh failed: {err.stderr.strip() or 'unknown error'}"
    except subprocess.TimeoutExpired:
        return "git refresh timed out"
    return None


def _commit_and_push(repo_dir: str, filename: str, slug: str) -> Optional[str]:
    """Stage the new file, commit, push with one rebase retry on non-fast-forward."""
    try:
        _git("add", filename, cwd=repo_dir)
        _git(
            "-c", "user.email=hermes@hermes-agent.local",
            "-c", "user.name=Hermes",
            "commit", "-m", f"publish: {slug}",
            cwd=repo_dir,
        )
    except subprocess.CalledProcessError as err:
        return f"git commit failed: {err.stderr.strip() or 'unknown error'}"
    for attempt in (1, 2):
        try:
            _git("push", "origin", "main", cwd=repo_dir, timeout=60)
            return None
        except subprocess.CalledProcessError as err:
            stderr = (err.stderr or "").strip()
            if attempt == 1 and ("non-fast-forward" in stderr or "rejected" in stderr):
                try:
                    _git("fetch", "origin", "main", cwd=repo_dir)
                    _git("rebase", "origin/main", cwd=repo_dir)
                    continue
                except subprocess.CalledProcessError as rebase_err:
                    _git("rebase", "--abort", cwd=repo_dir, check=False)
                    return f"rebase after non-fast-forward failed: {rebase_err.stderr.strip()}"
            return f"git push failed: {stderr or 'unknown error'}"
    return "git push exhausted retries"


def _tool_result(success: bool, **fields) -> str:
    payload = {"success": success}
    payload.update(fields)
    return json.dumps(payload)


# ---------- Public handler ----------

def publish_html(slug: str, html_content: str) -> str:
    """Publish an HTML page and return its public URL.

    Args:
        slug: Human-readable name for the page (e.g. "daily-summary"). Used in
              the URL after the hash. Will be normalized to lowercase + hyphens.
        html_content: Full HTML source to publish.

    Returns:
        JSON string. On success: {"success": true, "url": "...", "filename": "..."}
        On failure: {"success": false, "error": "..."}
    """
    if not isinstance(html_content, str) or not html_content.strip():
        return _tool_result(False, error="html_content is required and must be a non-empty string")

    repo_url = os.environ.get("HERMES_PAGES_REPO_URL", DEFAULT_REPO_URL).strip() or DEFAULT_REPO_URL
    repo_dir = os.environ.get("HERMES_PAGES_REPO_DIR", DEFAULT_REPO_DIR).strip() or DEFAULT_REPO_DIR
    base_url = (os.environ.get("HERMES_PAGES_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL).rstrip("/")

    if not os.environ.get("GITHUB_TOKEN", "").strip():
        return _tool_result(False, error="GITHUB_TOKEN is not set; cannot push to hermes-pages")

    norm_slug = _slugify(slug)
    file_hash = _generate_hash()
    filename = f"{file_hash}-{norm_slug}.html"

    with _publish_lock:
        err = _ensure_repo(repo_url, repo_dir)
        if err:
            logger.warning("publish_html: %s", err)
            return _tool_result(False, error=err)
        try:
            (Path(repo_dir) / filename).write_text(html_content, encoding="utf-8")
        except OSError as exc:
            return _tool_result(False, error=f"failed to write {filename}: {exc}")
        err = _commit_and_push(repo_dir, filename, norm_slug)
        if err:
            logger.warning("publish_html: %s", err)
            return _tool_result(False, error=err)

    url = f"{base_url}/{filename}"
    logger.info("publish_html: published %s", url)
    return _tool_result(True, url=url, filename=filename, slug=norm_slug)


# ---------- Registration ----------

PUBLISH_HTML_SCHEMA = {
    "name": "publish_html",
    "description": (
        "Publish an HTML page to the hermes-pages site and return its public URL. "
        "Use for sharing rendered reports, dashboards, or generated HTML with the user. "
        "URLs include a 12-char unguessable hash so the page is private through obscurity. "
        "The user can open the returned URL in any browser. Cloudflare Pages auto-deploys "
        "on push, typically live within 30 seconds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Human-readable name for the page, used in the URL after the hash "
                    "(e.g. 'daily-summary', 'q3-report'). Will be normalized to lowercase "
                    "with hyphens. Helps you identify what each URL is when you see it later."
                ),
            },
            "html_content": {
                "type": "string",
                "description": (
                    "Complete HTML source to publish. Should be a full HTML document "
                    "(<!DOCTYPE html><html>...</html>) — what the user sees in their browser."
                ),
            },
        },
        "required": ["slug", "html_content"],
    },
}


def _check_publish_html() -> tuple[bool, str]:
    """Toolset availability check: needs GITHUB_TOKEN to push."""
    if not os.environ.get("GITHUB_TOKEN", "").strip():
        return False, "GITHUB_TOKEN env var is required to push to hermes-pages"
    return True, ""


# --- Registry ---
from tools.registry import registry  # noqa: E402

registry.register(
    name="publish_html",
    toolset="publish",
    schema=PUBLISH_HTML_SCHEMA,
    handler=lambda args, **kw: publish_html(
        slug=args.get("slug", ""),
        html_content=args.get("html_content", ""),
    ),
    check_fn=_check_publish_html,
    requires_env=["GITHUB_TOKEN"],
    emoji="📄",
    description=(
        "Publish HTML to the hermes-pages Cloudflare site. Returns a hash-prefixed URL "
        "for unguessable sharing."
    ),
)
