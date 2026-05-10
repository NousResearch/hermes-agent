#!/usr/bin/env python3
"""
publish_html — publish an HTML page to the hermes-pages Cloudflare Pages site.

Workflow:
  1. Hermes calls publish_html(slug, html_content) with a human-readable slug
     and the page content.
  2. Tool maintains a local files directory at /opt/data/hermes-pages-files/.
  3. Generates a 12-char hash for unguessable URLs (URL-only auth model).
  4. Writes <hash>-<slug>.html into the files directory.
  5. Shells out to wrangler to deploy the directory to Cloudflare Pages
     (direct upload — no GitHub in the loop).
  6. On first use, installs wrangler to /opt/data/.npm-global so it
     persists across container rebuilds.
  7. Returns the public URL as a JSON-encoded result.

Configuration via env vars:
  CLOUDFLARE_API_TOKEN      — token with Cloudflare Pages: Edit on the account
  CLOUDFLARE_ACCOUNT_ID     — 32-char hex account ID
  HERMES_PAGES_FILES_DIR    — local mirror of the Pages site
                              (default: /opt/data/hermes-pages-files)
  HERMES_PAGES_PROJECT      — Cloudflare Pages project name
                              (default: hermes-pages)
  HERMES_PAGES_BASE_URL     — public base URL where files are served
                              (default: https://hermes-pages.rouse-gordon.workers.dev)
  HERMES_PAGES_BRANCH       — production branch in Cloudflare
                              (default: main)
"""

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------- Configuration ----------

DEFAULT_FILES_DIR = "/opt/data/hermes-pages-files"
DEFAULT_PROJECT = "hermes-pages"
DEFAULT_BASE_URL = "https://hermes-pages.rouse-gordon.workers.dev"
DEFAULT_BRANCH = "main"

NPM_PREFIX = "/opt/data/.npm-global"
WRANGLER_BIN = f"{NPM_PREFIX}/bin/wrangler"

HASH_LEN = 12
SLUG_MAX_LEN = 60
HASH_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

# Single-process lock so concurrent agent calls don't fight over wrangler.
_publish_lock = threading.Lock()


# ---------- Helpers ----------

def _slugify(text: str) -> str:
    """Normalize a slug: lowercase, hyphenate non-alnum runs, trim. Empty in -> 'page'."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        text = "page"
    return text[:SLUG_MAX_LEN].rstrip("-") or "page"


def _generate_hash() -> str:
    """12 chars from [a-z0-9] - ~62 bits of entropy, URL-safe."""
    return "".join(secrets.choice(HASH_ALPHABET) for _ in range(HASH_LEN))


def _ensure_wrangler_installed() -> Optional[str]:
    """Install wrangler to /opt/data/.npm-global on first use. Returns error string on failure, None on success."""
    if Path(WRANGLER_BIN).is_file() and os.access(WRANGLER_BIN, os.X_OK):
        return None
    if not shutil.which("npm"):
        return "npm is not available in the container; cannot install wrangler"
    try:
        Path(NPM_PREFIX).mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["NPM_CONFIG_PREFIX"] = NPM_PREFIX
        logger.info("publish_html: installing wrangler to %s (one-time, ~30s)", NPM_PREFIX)
        result = subprocess.run(
            ["npm", "install", "-g", "--prefix", NPM_PREFIX, "wrangler"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-500:]
            return f"npm install wrangler failed (exit {result.returncode}): {tail.strip()}"
    except subprocess.TimeoutExpired:
        return "npm install wrangler timed out after 300s"
    except OSError as exc:
        return f"npm install wrangler failed: {exc}"
    if not (Path(WRANGLER_BIN).is_file() and os.access(WRANGLER_BIN, os.X_OK)):
        return f"wrangler installed but not found at {WRANGLER_BIN}"
    return None


def _ensure_files_dir(files_dir: str) -> Optional[str]:
    """Create the local files mirror if missing. Returns error or None."""
    p = Path(files_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"failed to create files dir {files_dir}: {exc}"
    # Cloudflare Pages requires at least one file at the root for a deploy
    # to succeed. If the directory is empty (fresh install), drop a tiny
    # index so the first deploy has content.
    if not any(p.iterdir()):
        try:
            (p / "index.html").write_text(
                "<!doctype html><meta charset=utf-8><title>hermes-pages</title>"
                "<p>hermes-pages root.</p>",
                encoding="utf-8",
            )
        except OSError as exc:
            return f"failed to seed index.html in {files_dir}: {exc}"
    return None


def _wrangler_deploy(files_dir: str, project: str, branch: str) -> Optional[str]:
    """Deploy the local files directory to Cloudflare Pages via wrangler. Returns error or None."""
    env = os.environ.copy()
    # wrangler reads these from env; both are required for non-interactive deploys
    env["CLOUDFLARE_API_TOKEN"] = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    env["CLOUDFLARE_ACCOUNT_ID"] = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    env["NPM_CONFIG_PREFIX"] = NPM_PREFIX
    cmd = [
        WRANGLER_BIN,
        "pages", "deploy", files_dir,
        "--project-name", project,
        "--branch", branch,
        "--commit-dirty=true",
    ]
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return "wrangler pages deploy timed out after 180s"
    except OSError as exc:
        return f"wrangler invocation failed: {exc}"
    if result.returncode != 0:
        tail = ((result.stderr or "") + "\n" + (result.stdout or ""))[-700:]
        return f"wrangler pages deploy failed (exit {result.returncode}): {tail.strip()}"
    return None


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

    files_dir = (os.environ.get("HERMES_PAGES_FILES_DIR", DEFAULT_FILES_DIR).strip()
                 or DEFAULT_FILES_DIR)
    project = (os.environ.get("HERMES_PAGES_PROJECT", DEFAULT_PROJECT).strip()
               or DEFAULT_PROJECT)
    base_url = ((os.environ.get("HERMES_PAGES_BASE_URL", DEFAULT_BASE_URL).strip()
                 or DEFAULT_BASE_URL).rstrip("/"))
    branch = (os.environ.get("HERMES_PAGES_BRANCH", DEFAULT_BRANCH).strip()
              or DEFAULT_BRANCH)

    if not os.environ.get("CLOUDFLARE_API_TOKEN", "").strip():
        return _tool_result(False, error="CLOUDFLARE_API_TOKEN is not set; cannot deploy to Cloudflare Pages")
    if not os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip():
        return _tool_result(False, error="CLOUDFLARE_ACCOUNT_ID is not set; cannot deploy to Cloudflare Pages")

    norm_slug = _slugify(slug)
    file_hash = _generate_hash()
    filename = f"{file_hash}-{norm_slug}.html"

    with _publish_lock:
        err = _ensure_wrangler_installed()
        if err:
            logger.warning("publish_html: %s", err)
            return _tool_result(False, error=err)
        err = _ensure_files_dir(files_dir)
        if err:
            logger.warning("publish_html: %s", err)
            return _tool_result(False, error=err)
        try:
            (Path(files_dir) / filename).write_text(html_content, encoding="utf-8")
        except OSError as exc:
            return _tool_result(False, error=f"failed to write {filename}: {exc}")
        err = _wrangler_deploy(files_dir, project, branch)
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
    """Toolset availability check: needs Cloudflare credentials to deploy."""
    if not os.environ.get("CLOUDFLARE_API_TOKEN", "").strip():
        return False, "CLOUDFLARE_API_TOKEN env var is required to deploy to Cloudflare Pages"
    if not os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip():
        return False, "CLOUDFLARE_ACCOUNT_ID env var is required to deploy to Cloudflare Pages"
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
    requires_env=["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
    emoji="📄",
    description=(
        "Publish HTML to the hermes-pages Cloudflare site via direct upload. "
        "Returns a hash-prefixed URL for unguessable sharing."
    ),
)
