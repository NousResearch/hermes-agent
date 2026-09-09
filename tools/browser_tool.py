#!/usr/bin/env python3
"""Browser automation tools driven by the agent-browser CLI.

Backends — local headless Chromium, Browser Use / Browserbase / Firecrawl cloud
(auto-detected from config + credentials), a user-supplied CDP endpoint, or Camofox —
share one agent-facing behaviour: per-task sessions, accessibility-tree snapshots with
``@eN`` refs, automatic cleanup. Settings live under ``browser.*`` in config.yaml.
Sibling ``browser_tool_*`` modules hold extracted clusters.
"""

import atexit
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from agent.redact import redact_cdp_url
from hermes_constants import get_hermes_home
from utils import env_int
from hermes_cli.config import DEFAULT_CONFIG, cfg_get


# Env keys re-added to the agent-browser subprocess AFTER credential stripping.
# agent-browser is a Node process loading npm deps: a compromised transitive
# dependency could read every Hermes secret from process.env.
# Strip by default, then re-add only the browser-backend keys the worker legitimately needs. See #29157.
_BROWSER_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID", "BROWSER_USE_API_KEY",
    "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "FIRECRAWL_BROWSER_TTL",
)


def _build_browser_env() -> dict:
    """Credential-scrubbed env for an agent-browser subprocess (deferred import: test
    harnesses stub the ``tools`` package)."""
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    env.update({k: os.environ[k] for k in _BROWSER_PASSTHROUGH_KEYS if k in os.environ})
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
# Browser-provider ABC + registry; per-vendor providers live under
# ``plugins/browser/<vendor>/``. The dispatcher consults the registry. See #25214.
from agent.browser_provider import BrowserProvider
try:
    from agent.browser_registry import registry_generation as _browser_registry_generation
except ImportError:
    # Isolated compat tests install a minimal ``agent.browser_registry`` stub
    # with only ``get_provider``; no mutable registry → constant generation.
    def _browser_registry_generation(*, scope=None):
        return (0, 0)
# Optional backends: Camofox (CAMOFOX_URL routes everything through its REST API)
# and the Browser Use CLI.
try:
    from tools.browser_camofox import is_camofox_mode as _is_camofox_mode
except ImportError:
    _is_camofox_mode = lambda: False  # noqa: E731
try:
    from tools.browser_use_cli import is_browser_use_cli_mode as _is_browser_use_cli_mode
except ImportError:
    _is_browser_use_cli_mode = lambda: False  # noqa: E731

logger = logging.getLogger(__name__)

# PATH fallbacks for minimal-PATH environments (systemd services): Termux,
# macOS Homebrew, and the usual system dirs — needed for agent-browser/npx/node.
_SANE_PATH_DIRS = (
    "/data/data/com.termux/files/usr/bin", "/data/data/com.termux/files/usr/sbin",
    "/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/sbin", "/usr/local/bin",
    "/usr/sbin", "/usr/bin", "/sbin", "/bin",
)
_SANE_PATH = os.pathsep.join(_SANE_PATH_DIRS)

from tools import browser_tool_install as _install

_last_screenshot_cleanup_by_dir: dict[str, float] = {}  # throttles full directory scans

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEFAULT_COMMAND_TIMEOUT = 30  # seconds

# Floors for ``open``: cold daemon + first Chromium launch can exceed the
# generic command_timeout on slow or library-starved Linux hosts.
MIN_OPEN_TIMEOUT = 60
MIN_FIRST_OPEN_TIMEOUT = 120

# Snapshot truncation budget — aligned with web_tools.DEFAULT_EXTRACT_CHAR_LIMIT
# so the model gets the same per-page budget from both paths. Configurable via
# ``browser.snapshot_threshold``.
DEFAULT_SNAPSHOT_THRESHOLD = 15000
MIN_SNAPSHOT_THRESHOLD = 1000
# Ceiling on the stored full-snapshot file (mirrors web_tools.MAX_STORED_TEXT_CHARS):
# the stored copy exists for read_file paging and must not be unbounded.
MAX_STORED_SNAPSHOT_CHARS = 2_000_000
_EMPTY_OK_COMMANDS: frozenset = frozenset({"close", "record"})  # legitimately empty stdout

# Sentinel _find_agent_browser returns/caches to mean "resolve via npx" rather
# than a concrete path (also compared in hermes_cli/tools_config.py and doctor.py).
NPX_AGENT_BROWSER_SENTINEL = "npx agent-browser"
# Pinned to match scripts/install.sh / install.ps1's managed install so a bare-npx
# resolution gets the same version instead of floating latest. Update together.
AGENT_BROWSER_NPX_SPEC = "agent-browser@^0.26.0"

# Process caches (``_cached_X`` + ``_X_resolved`` pairs) for config-derived lookups;
# reset by ``cleanup_all_browsers``. Written/read by the sibling modules via ``browser_tool_origin``.
_cached_command_timeout: Optional[int] = None
# Flip the resolved flag BEFORE nulling the cache so a concurrent reader never sees ``resolved=True`` with
# ``cache=None`` (#14331).
_command_timeout_resolved = False
_cached_snapshot_threshold: Optional[int] = None
_snapshot_threshold_resolved = False
_cached_cloud_provider: Optional[BrowserProvider] = None
_cloud_provider_resolved = False
_cached_cloud_provider_scope: Optional[str] = None
_cached_cloud_providers: Dict[tuple[str, tuple[int, int]], Optional[BrowserProvider]] = {}
_cloud_provider_cache_lock = threading.RLock()
_allow_private_urls_resolved = False
_cached_allow_private_urls: Optional[bool] = None
_cached_agent_browser: Optional[str] = None
_agent_browser_resolved = False
_cached_browser_engine: Optional[str] = None  # agent-browser v0.25.3+ ``--engine lightpanda``
_browser_engine_resolved = False
_auto_local_for_private_urls_resolved = False
_cached_auto_local_for_private_urls: bool = True
_cached_headed_mode: Optional[bool] = None
_headed_mode_resolved = False
_cached_chromium_installed: Optional[bool] = None
_chromium_autoinstall_attempted = False  # one-shot: a failed 170MB download must not retry per call

# Mask secrets in logged CDP URLs; agent.redact.redact_cdp_url is the single policy.
_sanitize_url_for_logs = redact_cdp_url


def _browser_cfg(key: str, default, parse, log_label: str):
    """``parse(browser.<key>)`` from the RAW profile config (loader warnings must not
    leak into tool JSON); ``default`` when absent, not a mapping, or on any error."""
    try:
        from hermes_cli.config import read_raw_config
        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict) and key in browser_cfg:
            return parse(browser_cfg[key])
    except Exception as e:
        logger.debug("Could not read %s: %s", log_label, e)
    return default


def _cached_browser_cfg(cache_name: str, flag_name: str, key: str, default, parse, log_label: str):
    """Process-cached ``_browser_cfg`` read (cleared by ``cleanup_all_browsers``). The value is
    stored BEFORE the resolved flag flips so a concurrent reader never sees ``resolved=True``
    with a ``None`` cache."""
    g = globals()
    if g[flag_name] and g[cache_name] is not None:
        return g[cache_name]
    result = _browser_cfg(key, default, parse, log_label)
    g[cache_name] = result
    g[flag_name] = True
    return result


def _get_command_timeout() -> int:
    """``browser.command_timeout`` (floored at 5s; default 30s)."""
    return _cached_browser_cfg(
        "_cached_command_timeout", "_command_timeout_resolved",
        "command_timeout", DEFAULT_COMMAND_TIMEOUT,
        lambda v: DEFAULT_COMMAND_TIMEOUT if v is None else max(int(v), 5),
        "command_timeout from config",
    )


def _safe_command_timeout() -> int:
    """``_get_command_timeout`` guaranteed non-None (cache reset mid-flight); ``is not
    None`` rather than ``or`` so a configured ``0`` is preserved."""
    val = _get_command_timeout()
    return val if val is not None else DEFAULT_COMMAND_TIMEOUT


def get_browser_snapshot_threshold() -> int:
    """``browser.snapshot_threshold`` (floored at MIN_SNAPSHOT_THRESHOLD)."""
    return _cached_browser_cfg(
        "_cached_snapshot_threshold", "_snapshot_threshold_resolved",
        "snapshot_threshold", DEFAULT_SNAPSHOT_THRESHOLD,
        lambda v: DEFAULT_SNAPSHOT_THRESHOLD if v is None else max(int(v), MIN_SNAPSHOT_THRESHOLD),
        "browser.snapshot_threshold",
    )


def _get_open_command_timeout(*, first_open: bool = False) -> int:
    """Timeout for agent-browser ``open`` (navigation / daemon cold start)."""
    return max(_safe_command_timeout(), MIN_FIRST_OPEN_TIMEOUT if first_open else MIN_OPEN_TIMEOUT)


from tools import browser_tool_session as _session


def _get_vision_model() -> Optional[str]:
    """Model for browser_vision (screenshot analysis — multimodal)."""
    return os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None


from tools import browser_tool_cdp as _cdp

from tools import browser_tool_cloud as _cloud

from tools import browser_tool_lightpanda_fallback as _lp


# Single shared real-profile copy-browser session: concurrent tasks reuse it
# instead of each launching a rival Chromium on the same copied user-data-dir.
_REAL_PROFILE_SESSION = "hermes-real-profile"
_real_profile_cdp_lock = threading.Lock()
_real_profile_cdp_cache: dict = {}
_real_profile_chrome_procs: list = []  # Popen handles of directly-launched real browsers



_PRIVATE_HOST_SUFFIXES = (".localhost", ".local", ".lan", ".internal")


def _url_is_private(url: str) -> bool:
    """True when the URL's host is (or resolves to) a private/LAN/loopback/CGNAT address.
    Routing oracle only: DNS failures are NOT private (the configured backend surfaces the
    error); obvious names short-circuit the DNS hop."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    def private(host: str) -> Optional[bool]:  # None when ``host`` is not an IP literal
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return None
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip in ipaddress.ip_network("100.64.0.0/10")

    try:
        hostname = (urlparse(url).hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        if (literal := private(hostname)) is not None:
            return literal
        if hostname == "localhost" or hostname.endswith(_PRIVATE_HOST_SUFFIXES):
            return True
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return False
        return any(private(sockaddr[0]) for *_, sockaddr in addr_info)
    except Exception as exc:
        logger.debug("URL-privacy check failed for %s: %s", url, exc)
        return False


def _navigation_session_key(task_id: str, url: str) -> str:
    """Session key that should handle ``url`` for ``task_id``: ``f"{task_id}::local"`` (hybrid
    local sidecar while the cloud session keeps serving public URLs) only when ALL hold —
    cloud provider configured, ``browser.auto_local_for_private_urls`` on, private URL, no
    CDP override (it owns the whole session), Camofox off (already local-only)."""
    if task_id is None:
        task_id = "default"
    hybrid = (
        not _cdp._get_cdp_override_raw()
        and not _is_camofox_mode()
        and _cloud._get_cloud_provider() is not None
        and _cloud._auto_local_for_private_urls()
        and _url_is_private(url)
    )
    return f"{task_id}{_LOCAL_SUFFIX}" if hybrid else task_id


def _is_local_sidecar_key(session_key: str) -> bool:
    return session_key.endswith(_LOCAL_SUFFIX)


def _bare_task_id_for_session_key(session_key: str) -> str:
    return session_key[: -len(_LOCAL_SUFFIX)] if _is_local_sidecar_key(session_key) else session_key


def _session_info_owned_by_task(session_info: Dict[str, Any], task_id: str, session_key: str) -> bool:
    """Ownership check; entries without metadata (older in-memory / hot-reload) pass,
    any explicit mismatch fails before a non-nav tool can act on the wrong session."""
    owner = session_info.get("owner_task_id")
    key = session_info.get("session_key")
    return (owner is None or owner == task_id) and (key is None or key == session_key)


def _last_session_key(task_id: str) -> str:
    """Session key a non-nav tool must use: the one that served the task's last navigation.
    If it was cleaned up or ownership no longer matches, fail closed by dropping the stale
    binding rather than recreating or mutating the wrong browser."""
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
    logger.debug("browser session ownership: dropping stale/mismatched last-active binding %s -> %s",
                 task_id, recorded_key)
    return task_id


def _socket_safe_tmpdir() -> str:
    """Short temp dir for Unix sockets: macOS ``TMPDIR`` + ``agent-browser-hermes_…``
    exceeds the 104-byte AF_UNIX limit (silent screenshot failures), so use /tmp there."""
    return "/tmp" if sys.platform == "darwin" else tempfile.gettempdir()


# Active sessions keyed by "session key": the bare task_id, or f"{task_id}::local"
# for a hybrid-routing local sidecar (opaque to _run_browser_command / cleanup_browser).
# Values: session_name (always), bb_session_id + cdp_url (cloud).
_active_sessions: Dict[str, Dict[str, Any]] = {}
_recording_sessions: set = set()  # session_keys with active recordings
# Most recent session_key per task_id (set by browser_navigate, read by every non-nav
# tool) so click/snapshot land in the session that served the last navigation.
_last_active_session_key: Dict[str, str] = {}
_LOCAL_SUFFIX = "::local"
_cleanup_done = False

# Inactivity timeout: config.yaml is authoritative; BROWSER_INACTIVITY_TIMEOUT
# remains a legacy env fallback for unmigrated deployments.
DEFAULT_SESSION_INACTIVITY_TIMEOUT = int(DEFAULT_CONFIG.get("browser", {}).get("inactivity_timeout", 120))


def _get_session_inactivity_timeout() -> int:
    env_default = env_int("BROWSER_INACTIVITY_TIMEOUT", DEFAULT_SESSION_INACTIVITY_TIMEOUT)
    return _browser_cfg(
        "inactivity_timeout", env_default,
        lambda v: env_default if v is None else max(int(v), 30),  # 30s floor: no instant reaping
        "inactivity_timeout from config",
    )


BROWSER_SESSION_INACTIVITY_TIMEOUT = _get_session_inactivity_timeout()
# Orphan reaper cadence: a startup-only reap can never recover from a leak that
# appears after boot in a long-lived process.
BROWSER_ORPHAN_REAP_INTERVAL = 300  # seconds
# Idle ceiling for a daemon whose owner is alive but which fell out of in-memory
# tracking (owner-alive alone would make it immortal); large multiple so a busy
# session is never touched.
BROWSER_ORPHAN_GRACE_SECONDS = max(3600, BROWSER_SESSION_INACTIVITY_TIMEOUT * 20)

_session_last_activity: Dict[str, float] = {}
# Owner Hermes home per session: the janitor is one process-global thread, so each
# teardown must re-enter the OWNING profile's scope (copy_context at spawn would
# pin the first profile's secrets onto every other profile's teardown).
# See #86402.
_session_owner_homes: Dict[str, str] = {}
# Consecutive janitor failures per session; force-reaped after MAX_INACTIVITY_CLEANUP_FAILURES.
# See #100738.
_cleanup_failures: Dict[str, int] = {}
MAX_INACTIVITY_CLEANUP_FAILURES = 3

# Session keys flagged suspect after a command timeout (written lock-free by
# mark_suspect; consumed by ensure_healthy() at next use, which recycles).
# See #72205.
_suspect_browser_sessions: Dict[str, str] = {}


class _BrowserSessionBackend:
    """``agent.deadline.SuspectableBackend`` adapter for one cached session key: the
    timeout path calls ``mark_suspect`` inline; ``ensure_healthy`` runs at the top of
    ``_get_session_info`` — the choke point every command passes through."""

    __slots__ = ("_session_key",)

    def __init__(self, session_key: str) -> None:
        self._session_key = session_key

    def mark_suspect(self, reason: str) -> None:
        """MUST stay cheap and lock-free (runs inline on the timed-out caller's thread)."""
        _suspect_browser_sessions[self._session_key] = reason

    def ensure_healthy(self) -> bool:
        """Recycle the session when a prior timeout marked it suspect; False after teardown.
        The flag is popped BEFORE teardown: ``close`` re-enters ``_get_session_info``
        and must not recurse into another recycle."""
        reason = _suspect_browser_sessions.pop(self._session_key, None)
        if reason is None:
            return True
        logger.info("Recycling suspect browser session %s before reuse (%s)", self._session_key, reason)
        try:
            _lifecycle._cleanup_single_browser_session(self._session_key)
        except Exception:
            logger.warning("Teardown of suspect browser session %s failed; a fresh "
                           "session will be created anyway", self._session_key, exc_info=True)
        return False


_browser_session_backend = _BrowserSessionBackend

_cleanup_thread = None
_cleanup_running = False
_cleanup_lock = threading.Lock()  # protects _session_last_activity AND _active_sessions

from tools import browser_tool_lifecycle as _lifecycle

# atexit only — NO SIGINT/SIGTERM handlers calling sys.exit(): a SystemExit raised
# inside a prompt_toolkit key-binding callback corrupts the coroutine state and
# makes the process unkillable.
atexit.register(_lifecycle._emergency_cleanup_all_sessions)
atexit.register(_lifecycle._stop_browser_cleanup_thread)

# ----------------------------------------------------------------------------
# Tool Schemas
# ----------------------------------------------------------------------------
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
                    "type": "string", "description": "The text to type into the field"
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
                    "type": "string", "enum": ["up", "down"], "description": "Direction to scroll"
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "browser_back",
        "description": "Navigate back to the previous page in browser history. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object", "properties": {}, "required": []
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
            "type": "object", "properties": {}, "required": []
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

from tools import browser_tool_snapshot as _snapshot

# ----------------------------------------------------------------------------
# Browser Tool Functions
# ----------------------------------------------------------------------------

def _err(error: str, **extra) -> dict:
    return {"success": False, "error": error, **extra}


def _dumps(payload: Dict[str, Any], **kw) -> str:
    return json.dumps(payload, ensure_ascii=False, **kw)


def _secret_url_error(url: str) -> Optional[dict]:
    """Refuse URLs embedding an API key/token (raw and URL-decoded, catching ``%2D``
    tricks) — a prompt injection could otherwise exfiltrate secrets via the URL."""
    import urllib.parse
    from agent.redact import _PREFIX_RE

    if _PREFIX_RE.search(url) or _PREFIX_RE.search(urllib.parse.unquote(url)):
        return _err("Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs.")
    return None


def _url_policy_error(url: str, *, auto_local: bool = False) -> Optional[dict]:
    """Backend-aware URL checks on an already-normalized URL; None if allowed. Ordered floors:
    (1) credential-like query params refused for cloud backends (third-party readers);
    (2) cloud metadata / IMDS refused UNCONDITIONALLY (a local Chromium on a cloud VM still
    reaches the host IMDS); (3) private addresses refused unless local, sidecar-routed, or
    ``browser.allow_private_urls``; (4) website policy allow/deny lists."""
    local = _cloud._is_local_backend()
    sensitive_query_key = _sensitive_query_param_name(url)
    if sensitive_query_key and not local and not auto_local:
        return _err(
            "Blocked: URL contains a credential-like query parameter "
            f"({sensitive_query_key}). Cloud browser backends are third-party "
            "readers; use a local browser/CDP session or remove the sensitive "
            "query parameter before navigating.")
    # Always-blocked floor: cloud metadata / IMDS endpoints are denied regardless of backend, hybrid
    # routing, or allow_private_urls. There's no legitimate agent use case for navigating to 169.254.169.254
    # / metadata.google.internal / ECS task metadata via a browser, and routing those to a local Chromium
    # sidecar on an EC2/GCP/Azure host exfiltrates IAM credentials (#16234). The floor is UNCONDITIONAL — it
    # must fire for every backend, including the pure-local headless Chromium and off-host CDP cases (a
    # local Chromium on a cloud VM still reaches the host IMDS).
    if _is_always_blocked_url(url):
        return _err("Blocked: URL targets a cloud metadata endpoint")
    if not local and not auto_local and not _cloud._allow_private_urls() and not _is_safe_url(url):
        return _err("Blocked: URL targets a private or internal address")
    blocked = check_website_access(url)
    if blocked:
        return _err(blocked["message"],
                    blocked_by_policy={"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]})
    return None


def _secret_url_error_normalized(url: str) -> tuple[str, Optional[dict]]:
    """Secret check on the raw URL, then again on the normalized one; returns ``(url, error)``."""
    err = _secret_url_error(url)
    if err is None:
        url = _normalize_url_for_request(url)
        err = _secret_url_error(url)
    return url, err


def evaluate_url_safety(url: str) -> Optional[dict]:
    """Run URL safety checks; None if safe, else an error dict"""
    url, err = _secret_url_error_normalized(url)
    return err or _url_policy_error(url)


_BOT_DETECTION_TITLE_PATTERNS = (
    "access denied", "access to this page has been denied", "blocked", "bot detected", "verification required",
    "please verify", "are you a robot", "captcha", "cloudflare", "ddos protection", "checking your browser",
    "just a moment", "attention required",
)


def _post_redirect_block(nav_session_key: str, url: str, final_url: str, auto_local_this_nav: bool) -> Optional[str]:
    """Post-redirect SSRF check; blocked JSON payload or None. The page is moved to about:blank
    first so later snapshots can't read the internal content. The metadata floor fires for
    every backend; the private-address check is skipped for local, the sidecar, and
    ``browser.allow_private_urls``."""
    if not final_url or final_url == url:
        return None
    if _is_always_blocked_url(final_url):
        what = "a cloud metadata endpoint"
    elif (
        not _cloud._is_local_backend()
        and not auto_local_this_nav
        and not _cloud._allow_private_urls()
        and not _is_safe_url(final_url)
    ):
        what = "a private/internal address"
    else:
        return None
    _session._run_browser_command(nav_session_key, "open", ["about:blank"], timeout=10)
    return json.dumps(_err(f"Blocked: redirect landed on {what}"))


def _snapshot_fields(snap_result: Dict[str, Any]) -> Dict[str, Any]:
    """``snapshot`` + ``element_count`` fields from a successful snapshot result; oversized
    snapshots truncate at line boundaries with the full tree stored for read_file paging."""
    data = snap_result.get("data", {})
    snapshot_text = data.get("snapshot", "")
    refs = data.get("refs", {})
    threshold = get_browser_snapshot_threshold()
    if len(snapshot_text) > threshold:
        snapshot_text = _snapshot._truncate_snapshot(snapshot_text, max_chars=threshold)
    return {"snapshot": _snapshot._redact_browser_output(snapshot_text), "element_count": len(refs) if refs else 0}


def _merge_fallback_warning(response: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Copy a secondary result's fallback warning only if the response has none yet."""
    if result.get("fallback_warning") and not response.get("fallback_warning"):
        _lp._copy_fallback_warning(response, result)


def _attach_auto_snapshot(response: Dict[str, Any], nav_session_key: str) -> None:
    """Add a compact snapshot to a navigate response so the model can act without browser_snapshot."""
    try:
        snap_result = _session._run_browser_command(nav_session_key, "snapshot", ["-c"])
        if snap_result.get("success"):
            response.update(_snapshot_fields(snap_result))
            _merge_fallback_warning(response, snap_result)
    except Exception as e:
        logger.debug("Auto-snapshot after navigate failed: %s", e)


def browser_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to ``url``; JSON with title, compact snapshot and, on first nav, stealth features.
    Hybrid routing decides BEFORE the safety checks whether this URL goes to a local sidecar
    (the cloud provider never sees it then, so the private-address checks are relaxed)."""
    url, safety_error = _secret_url_error_normalized(url)
    if safety_error is not None:
        return json.dumps(safety_error)

    effective_task_id = task_id or "default"
    nav_session_key = _navigation_session_key(effective_task_id, url)
    auto_local_this_nav = _is_local_sidecar_key(nav_session_key)

    safety_error = _url_policy_error(url, auto_local=auto_local_this_nav)
    if safety_error is not None:
        return json.dumps(safety_error)

    if _is_camofox_mode():
        return _camofox("camofox_navigate", url, task_id)

    if auto_local_this_nav:
        logger.info("browser_navigate: auto-routing %s to local Chromium sidecar (cloud provider %s stays on "
                    "cloud for public URLs; set browser.auto_local_for_private_urls: false to disable)",
                    url, type(_cloud._get_cloud_provider()).__name__ if _cloud._get_cloud_provider() else "none")

    session_info = _session._get_session_info(nav_session_key)
    is_first_nav = session_info.get("_first_nav", True)
    if is_first_nav:
        session_info["_first_nav"] = False
        _maybe_start_recording(nav_session_key)

    result = _session._run_browser_command(nav_session_key, "open", [url],
                                  timeout=_get_open_command_timeout(first_open=is_first_nav))
    if not result.get("success"):
        return _dumps(_err(result.get("error", "Navigation failed")))

    data = result.get("data", {})
    title = data.get("title", "")
    final_url = data.get("url", url)
    blocked = _post_redirect_block(nav_session_key, url, final_url, auto_local_this_nav)
    if blocked is not None:
        return blocked

    response = {"success": True, "url": final_url, "title": title}
    features = session_info.get("features") or {}
    if features.get("real_profile"):  # auditability: this ran on the user's real-profile copy-browser
        response["used_real_profile"] = True
    # Only a successful, non-blocked navigation becomes the task owner: failed opens
    # and blocked redirects must not retarget follow-up clicks to an irrelevant session.
    _last_active_session_key[effective_task_id] = nav_session_key
    _lp._copy_fallback_warning(response, result)
    _add_navigate_warnings(response, title, session_info if is_first_nav else None)
    _attach_auto_snapshot(response, nav_session_key)
    return _dumps(response)


def _add_navigate_warnings(response: Dict[str, Any], title: str, first_nav_session: Optional[Dict[str, Any]]) -> None:
    """Bot-detection hint from the page title; on first navigation, the session's stealth features."""
    title_lower = title.lower()
    if any(pattern in title_lower for pattern in _BOT_DETECTION_TITLE_PATTERNS):
        response["bot_detection_warning"] = (
            f"Page title '{title}' suggests bot detection. The site may have blocked this request. "
            "Options: 1) Try adding delays between actions, 2) Access different pages first, "
            "3) Enable advanced stealth (BROWSERBASE_ADVANCED_STEALTH=true, requires Scale plan), "
            "4) Some sites have very aggressive bot detection that may be unavoidable."
        )
    if first_nav_session is not None and "features" in first_nav_session:
        features = first_nav_session["features"]
        if not features.get("proxies"):
            response["stealth_warning"] = (
                "Running WITHOUT residential proxies. Bot detection may be more aggressive. "
                "Consider upgrading Browserbase plan for proxy support."
            )
        response["stealth_features"] = [k for k, v in features.items() if v]


def browser_snapshot(
    full: bool = False, task_id: Optional[str] = None, user_task: Optional[str] = None
) -> str:
    """Text snapshot of the page's accessibility tree (compact unless ``full``).
    ``user_task`` is deprecated and unused (oversized snapshots always truncate-and-store)."""
    if _is_camofox_mode():
        return _camofox("camofox_snapshot", full, task_id)
    effective_task_id = _last_session_key(task_id or "default")
    result = _session._run_browser_command(effective_task_id, "snapshot", [] if full else ["-c"])
    if not result.get("success"):
        return _failed_response(result, "Failed to get snapshot")

    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    response = {"success": True, **_snapshot_fields(result)}
    _lp._copy_fallback_warning(response, result)

    # Merge supervisor state (pending dialogs + frame tree) when a CDP supervisor is
    # attached. See website/docs/developer-guide/browser-supervisor.md.
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        _supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
        if _supervisor is not None:
            _sv_snap = _supervisor.snapshot()
            if _sv_snap.active:
                response.update(_snapshot._redact_browser_output(_sv_snap.to_dict()))
    except Exception as _sv_exc:
        logger.debug("supervisor snapshot merge failed: %s", _sv_exc)

    return _dumps(response)


def _json_with_fallback(response: Dict[str, Any], result: Dict[str, Any]) -> str:
    """``json.dumps`` of ``response`` with the Lightpanda fallback metadata copied from ``result``."""
    return _dumps(_lp._copy_fallback_warning(response, result))


def _failed_response(result: Dict[str, Any], default_error: str) -> str:
    return _json_with_fallback(_err(result.get("error", default_error)), result)


def _tool_response(result: Dict[str, Any], ok: Dict[str, Any], default_error: str) -> str:
    """``{"success": True, **ok}`` or ``{"success": False, "error": result.error or default}``, plus fallback metadata."""
    if not result.get("success"):
        return _failed_response(result, default_error)
    return _json_with_fallback({"success": True, **ok}, result)


def _camofox(func_name: str, *args):
    """Call ``tools.browser_camofox.<func_name>(*args)`` (Camofox mode delegation)."""
    import importlib
    return getattr(importlib.import_module("tools.browser_camofox"), func_name)(*args)


def _guarded_action(task_id: Optional[str], action: str, command: str, args: list, ok: Dict[str, Any], err: str) -> str:
    """Input action on the task's current page, refused when the SSRF guard flags the page."""
    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, action)
    if blocked is not None:
        return blocked
    return _tool_response(_session._run_browser_command(effective_task_id, command, args), ok, err)


def _at_ref(ref: str) -> str:
    return ref if ref.startswith("@") else f"@{ref}"


def browser_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click the element ``ref`` (e.g. "@e5")."""
    if _is_camofox_mode():
        return _camofox("camofox_click", ref, task_id)
    ref = _at_ref(ref)
    return _guarded_action(task_id, "click", "click", [ref], {"clicked": ref}, f"Failed to click {ref}")


def browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type ``text`` into the element ``ref`` (fill: clears, then types)."""
    if _is_camofox_mode():
        return _camofox("camofox_type", ref, text, task_id)
    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "type")
    if blocked is not None:
        return blocked
    ref = _at_ref(ref)
    result = _session._run_browser_command(effective_task_id, "fill", [ref, text])
    from agent.display import redact_browser_typed_text_for_display, redact_tool_args_for_display
    # Typed text goes through the secret-pattern redactor so API keys / tokens don't
    # leak into tool progress or chat history (the raw value already went to the browser).
    display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]
    if result.get("success"):
        response = {"success": True, "typed": display_text, "element": ref}
    else:
        response = _err(result.get("error", f"Failed to type into {ref}"))
    return _dumps(redact_browser_typed_text_for_display(_lp._copy_fallback_warning(response, result), text))


def browser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page ``direction`` ("up"/"down") by about half a viewport."""
    if direction not in {"up", "down"}:
        return _dumps(_err(f"Invalid direction '{direction}'. Use 'up' or 'down'."))
    _SCROLL_PIXELS = 500  # ~half a viewport in one call instead of 5x subprocess calls
    if _is_camofox_mode():  # Camofox REST API has no pixel argument; use repeated calls
        return [_camofox("camofox_scroll", direction, task_id) for _ in range(5)][-1]
    effective_task_id = _last_session_key(task_id or "default")
    result = _session._run_browser_command(effective_task_id, "scroll", [direction, str(_SCROLL_PIXELS)])
    return _tool_response(result, {"scrolled": direction}, f"Failed to scroll {direction}")


def browser_back(task_id: Optional[str] = None) -> str:
    """Navigate back in browser history."""
    if _is_camofox_mode():
        return _camofox("camofox_back", task_id)
    effective_task_id = _last_session_key(task_id or "default")
    result = _session._run_browser_command(effective_task_id, "back", [])
    if result.get("success"):
        # History can land on a private/internal/metadata address the navigate
        # preflight never saw (earlier redirect chain, manipulated client-side history).
        blocked = _blocked_private_page(effective_task_id, "Browser history navigation (back) landed on this address.")
        if blocked is not None:
            return blocked
    return _tool_response(result, {"url": result.get("data", {}).get("url", "")}, "Failed to go back")


def browser_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key (e.g. "Enter", "Tab")."""
    if _is_camofox_mode():
        return _camofox("camofox_press", key, task_id)
    return _guarded_action(task_id, "press", "press", [key], {"pressed": key}, f"Failed to press {key}")


def _blocked_private_page_json(blocked_url: str, why: str) -> str:
    """Refusal payload for a page whose URL targets a private/internal address."""
    return _dumps(_err(f"Blocked: page URL targets a private or internal address ({blocked_url}). {why}"))


def _blocked_private_page(effective_task_id: str, why: str) -> Optional[str]:
    """Blocked payload when the SSRF guard is active and the current page is private, else
    None. Fail-open on probe failure (see ``_current_page_private_url``)."""
    if not _eval_policy._eval_ssrf_guard_active(effective_task_id):
        return None
    blocked_url = _eval_policy._current_page_private_url(effective_task_id)
    return _blocked_private_page_json(blocked_url, why) if blocked_url else None


def _blocked_private_page_action(effective_task_id: str, action: str) -> Optional[str]:
    """Blocked payload when an unsafe cloud page would receive input."""
    return _blocked_private_page(effective_task_id, f"Refusing to {action} on this page in this browser mode.")


_EVAL_NAVIGATED_WHY = "This may have been caused by a JavaScript navigation via browser_console."


def _blocked_private_page_content(effective_task_id: str) -> Optional[str]:
    """Content-returning tools (snapshot/vision/eval/get_images): after an eval that may
    have moved ``location.href`` to a private address, returning content would expose it."""
    return _blocked_private_page(effective_task_id, _EVAL_NAVIGATED_WHY)


def browser_console(clear: bool = False, expression: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Console messages + uncaught JS errors (optionally ``clear``ing the buffers),
    or — when ``expression`` is given — evaluate JS in the page like the DevTools console."""
    if expression is not None:
        policy_error = _eval_policy._enforce_browser_eval_policy(expression)
        if policy_error:
            return _dumps(_err(policy_error))
        return _browser_eval(expression, task_id)

    if _is_camofox_mode():
        return _camofox("camofox_console", clear, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    clear_args = ["--clear"] if clear else []
    console_result = _session._run_browser_command(effective_task_id, "console", clear_args)
    errors_result = _session._run_browser_command(effective_task_id, "errors", clear_args)

    messages = [
        {"type": msg.get("type", "log"), "text": _snapshot._redact_browser_output(msg.get("text", "")), "source": "console"}
        for msg in console_result.get("data", {}).get("messages", [])
    ] if console_result.get("success") else []
    errors = [
        {"message": _snapshot._redact_browser_output(err.get("message", "")), "source": "exception"}
        for err in errors_result.get("data", {}).get("errors", [])
    ] if errors_result.get("success") else []
    response = {
        "success": True, "console_messages": messages, "js_errors": errors,
        "total_messages": len(messages), "total_errors": len(errors),
    }
    _lp._copy_fallback_warning(response, console_result)
    _merge_fallback_warning(response, errors_result)
    return _dumps(response)


from tools import browser_tool_eval_policy as _eval_policy


def _parse_eval_value(raw_result: Any) -> Any:
    """Eval returns the JS value as a string; parse valid JSON so the model gets structured data."""
    if isinstance(raw_result, str):
        try:
            return json.loads(raw_result)
        except (json.JSONDecodeError, ValueError):
            pass  # keep as string
    return raw_result


def _eval_ok_response(parsed: Any, **extra) -> Dict[str, Any]:
    return {"success": True, "result": _snapshot._redact_browser_output(parsed), "result_type": type(parsed).__name__, **extra}


def _eval_result_or_blocked(effective_task_id: str, parsed: Any, result: Dict[str, Any], **extra) -> str:
    """Eval tool JSON, unless the post-eval page-URL recheck finds an eval navigated the
    page to a private address — then the result is withheld."""
    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked
    return _dumps(_lp._copy_fallback_warning(_eval_ok_response(parsed, **extra), result), default=str)


def _eval_supervisor_fast_path(effective_task_id: str, expression: str) -> Optional[str]:
    """``Runtime.evaluate`` on the CDP supervisor's persistent WebSocket (no subprocess cost).
    Tool JSON when the supervisor gave a definitive answer (value, blocked page, or a real
    JS-side exception — NOT retried via subprocess, that would just reproduce it slower);
    None to fall through to the subprocess path."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
        if supervisor is None:
            return None
        sup_result = supervisor.evaluate_runtime(expression)
        if sup_result.get("ok"):
            return _eval_result_or_blocked(
                effective_task_id, _parse_eval_value(sup_result.get("result")), {}, method="cdp_supervisor")
        err = sup_result.get("error") or "evaluate_runtime failed"
        if "supervisor" not in err.lower():
            return _dumps(_err(err))
        logger.debug("browser_eval: supervisor path unavailable (%s), falling back to subprocess", err)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_eval: supervisor path errored (%s), falling back", exc)
    return None


def _eval_failure_response(result: Dict[str, Any]) -> str:
    """Tool JSON for a failed ``agent-browser eval``, with actionable rewrites of known errors."""
    err = result.get("error", "eval failed")
    if any(hint in err.lower() for hint in ("unknown command", "not supported", "not found", "no such command")):
        err = f"JavaScript evaluation is not supported by this browser backend. {err}"
    elif "reference chain is too long" in err.lower():
        # A live DOM node / NodeList / Window can't be JSON-serialized by CDP. The
        # supervisor path retries with returnByValue=false; the CLI can't.
        err = (
            "Expression returned a live DOM node / NodeList / Window, "
            "which can't be serialized. Extract a primitive value "
            "(e.g. .innerText, .href, .src, .value) or use "
            "JSON.stringify() / a snapshot tool instead."
        )
    return json.dumps(_lp._copy_fallback_warning(_err(err), result))


def _browser_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JS in the page context. Private-network guard in two halves: the literal
    pre-scan closes direct fetches (they never update ``location.href``); the post-eval
    page-URL recheck closes navigate-then-read."""
    effective_task_id = _last_session_key(task_id or "default")

    if _eval_policy._eval_ssrf_guard_active(effective_task_id):
        blocked_literal = _eval_policy._expression_targets_private_url(expression)
        if blocked_literal:
            return _dumps(_err(
                "Blocked: JavaScript expression targets a private or "
                f"internal address ({blocked_literal}). Reading internal "
                "endpoints via browser_console is not permitted in this "
                "browser mode."
            ))

    # Camofox keeps its own raw-task_id-keyed session map, so pass the raw id.
    if _is_camofox_mode():
        return _camofox_eval(expression, task_id)

    fast = _eval_supervisor_fast_path(effective_task_id, expression)
    if fast is not None:
        return fast

    result = _session._run_browser_command(effective_task_id, "eval", [expression])
    if not result.get("success"):
        return _eval_failure_response(result)
    return _eval_result_or_blocked(effective_task_id, _parse_eval_value(result.get("data", {}).get("result")), result)


def _camofox_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JS via Camofox's /tabs/{tab_id}/evaluate endpoint (if available)."""
    from tools.browser_camofox import _ensure_tab, _post
    try:
        tab_info = _ensure_tab(task_id or "default")
        tab_id = tab_info.get("tab_id") or tab_info.get("id")
        user_id = tab_info["user_id"]
        resp = _post(f"/tabs/{tab_id}/evaluate", body={"expression": expression, "userId": user_id})
        parsed = _parse_eval_value(resp.get("result") if isinstance(resp, dict) else resp)

        if _eval_policy._eval_ssrf_guard_active(task_id or "default"):
            _blocked_url = _eval_policy._camofox_current_page_private_url(tab_id, user_id)
            if _blocked_url:
                return _blocked_private_page_json(_blocked_url, _EVAL_NAVIGATED_WHY)

        return _dumps(_eval_ok_response(parsed), default=str)
    except Exception as e:
        if any(code in str(e) for code in ("404", "405", "501")):  # server without eval support
            return json.dumps(_err("JavaScript evaluation is not supported by this Camofox server. "
                                   "Use browser_snapshot or browser_vision to inspect page state."))
        return tool_error(str(e), success=False)


def _maybe_start_recording(task_id: str):
    """Start recording if browser.record_sessions is enabled in config."""
    with _cleanup_lock:
        if task_id in _recording_sessions:
            return
    try:
        from hermes_cli.config import read_raw_config
        hermes_home = get_hermes_home()
        if not cfg_get(read_raw_config(), "browser", "record_sessions", default=False):
            return
        recordings_dir = hermes_home / "browser_recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        _lifecycle._cleanup_old_recordings(max_age_hours=72)
        recording_path = recordings_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}_{task_id[:16]}.webm"
        result = _session._run_browser_command(task_id, "record", ["start", str(recording_path)])
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
        result = _session._run_browser_command(task_id, "record", ["stop"])
        if result.get("success"):
            logger.info("Saved browser recording for session %s: %s", task_id, result.get("data", {}).get("path", ""))
    except Exception as e:
        logger.debug("Could not stop recording for %s: %s", task_id, e)
    finally:
        with _cleanup_lock:
            _recording_sessions.discard(task_id)


_GET_IMAGES_JS = """JSON.stringify(
        [...document.images].map(img => ({
            src: img.src, alt: img.alt || '', width: img.naturalWidth, height: img.naturalHeight
        })).filter(img => img.src && !img.src.startsWith('data:'))
    )"""


def browser_get_images(task_id: Optional[str] = None) -> str:
    """List the page's images (src, alt, natural size), excluding data: URIs."""
    if _is_camofox_mode():
        return _camofox("camofox_get_images", task_id)

    effective_task_id = _last_session_key(task_id or "default")
    result = _session._run_browser_command(effective_task_id, "eval", [_GET_IMAGES_JS])
    if not result.get("success"):
        return _failed_response(result, "Failed to get images")

    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    raw_result = result.get("data", {}).get("result", "[]")
    try:
        images = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        return _json_with_fallback({"success": True, "images": _snapshot._redact_browser_output(images), "count": len(images)}, result)
    except json.JSONDecodeError:
        return _json_with_fallback({"success": True, "images": [], "count": 0, "warning": "Could not parse image data"}, result)


_LP_VISION_FALLBACK_REASON = "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture."


from tools import browser_tool_vision as _vision


def _capture_vision_screenshot(effective_task_id: str, annotate: bool, screenshot_path: Path, lp_prerouted: bool):
    """Take (or adopt the pre-routed) screenshot; returns ``(result, path, error_json_or_None)``."""
    if lp_prerouted and screenshot_path.exists():
        result = _lp._annotate_lightpanda_fallback(
            {"success": True, "data": {"path": str(screenshot_path)}}, _LP_VISION_FALLBACK_REASON)
    else:
        screenshot_args = (["--annotate"] if annotate else []) + ["--full", str(screenshot_path)]
        # A failed Lightpanda pre-route forces Chrome so _run_browser_command
        # doesn't trigger a redundant LP fallback.
        result = _session._run_browser_command(effective_task_id, "screenshot", screenshot_args,
                                      _engine_override="auto" if lp_prerouted else None)
    if not result.get("success"):
        return result, screenshot_path, _json_with_fallback(_err(
            f"Failed to take screenshot ({_vision._vision_mode_label()} mode): {result.get('error', 'Unknown error')}"
        ), result)
    if result.get("data", {}).get("path"):
        screenshot_path = Path(result["data"]["path"])
    if not screenshot_path.exists():
        return result, screenshot_path, _dumps(_err(
            f"Screenshot file was not created at {screenshot_path} ({_vision._vision_mode_label()} mode). "
            f"This may indicate a socket path issue (macOS /var/folders/), "
            f"a missing Chromium install ('agent-browser install'), "
            f"or a stale daemon process."
        ))
    return result, screenshot_path, None


def browser_vision(question: str, annotate: bool = False, task_id: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    """Screenshot the current page for visual inspection. Native-vision models get the image
    attached to the conversation; otherwise the auxiliary vision model returns a text
    analysis. The file is kept and its path returned (MEDIA:<path>)."""
    if _is_camofox_mode():
        return _camofox("camofox_vision", question, annotate, task_id)

    import uuid as uuid_mod
    from hermes_constants import get_hermes_dir
    screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
    screenshot_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    _lp_prerouted, _lp_fallback_warning, screenshot_path = _vision._lightpanda_vision_preroute(
        effective_task_id, annotate, screenshot_path)
    result: Dict[str, Any] = {}
    try:
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        _lifecycle._cleanup_old_screenshots(screenshots_dir, max_age_hours=24)
        result, screenshot_path, error = _capture_vision_screenshot(
            effective_task_id, annotate, screenshot_path, _lp_prerouted)
        if error is not None:
            return error
        # Native image routing: attach the screenshot directly instead of describing it
        # through an aux vision LLM (no information loss).
        from tools.vision_tools import _should_use_native_vision_fast_path
        if _should_use_native_vision_fast_path():
            return _vision._native_vision_result(screenshot_path, question, annotate, result, _lp_fallback_warning)

        analysis = _vision._analyze_screenshot_with_aux_llm(screenshot_path, question)
        response_data = {"success": True, "analysis": analysis or "Vision analysis returned no content.",
                         "screenshot_path": str(screenshot_path)}
        _lp._copy_fallback_warning(response_data, result)
        if annotate and result.get("data", {}).get("annotations"):
            response_data["annotations"] = result["data"]["annotations"]
        return _dumps(response_data)
    except Exception as e:
        # Keep a captured screenshot — the failure is in the analysis, not the capture,
        # and deleting it loses evidence. The 24-hour cleanup bounds disk growth.
        logger.warning("browser_vision failed: %s", e, exc_info=True)
        error_info = _err(f"Error during vision analysis: {str(e)}")
        if screenshot_path.exists():
            error_info["screenshot_path"] = str(screenshot_path)
            error_info["note"] = "Screenshot was captured but vision analysis failed. You can still share it via MEDIA:<path>."
        _lp._copy_fallback_warning(error_info, result)
        return _dumps(error_info)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error
from tools.browser_extension_router import extension_controller_available, routed_browser_handler

_BROWSER_SCHEMA_MAP = {s["name"]: s for s in BROWSER_TOOL_SCHEMAS}


def check_browser_routed_requirements(action: str = "browser_snapshot") -> bool:
    """Availability gate for tools that can use either browser backend."""
    return _install.check_browser_requirements() or extension_controller_available(action)


def _fallback_call(fn_name: str, arg_defaults: Dict[str, Any], extra_kw: tuple = ()):
    """Adapter from the registry's ``(args, kw)`` to ``<fn_name>(**schema_args, task_id=...)``;
    the function is looked up in module globals at call time so monkeypatching works."""
    def call(args, kw):
        params = {a: args.get(a, d) for a, d in arg_defaults.items()}
        params["task_id"] = kw.get("task_id")
        for k in extra_kw:
            params[k] = kw.get(k)
        return globals()[fn_name](**params)
    return call


# (tool name, emoji, availability gate, schema-arg defaults[, extra kw names]); the tool
# function is the module global of the same name. Routed-through-extension tools (gate None)
# use the per-action gate; get_images/console/vision keep the plain requirement checks.
_BROWSER_TOOL_TABLE = (
    ("browser_navigate", "🌐", None, {"url": ""}),
    ("browser_snapshot", "📸", None, {"full": False}, ("user_task",)),
    ("browser_click", "👆", None, {"ref": ""}),
    ("browser_type", "⌨️", None, {"ref": "", "text": ""}),
    ("browser_scroll", "📜", None, {"direction": "down"}),
    ("browser_back", "◀️", None, {}),
    ("browser_press", "⌨️", None, {"key": ""}),
    ("browser_get_images", "🖼️", _install.check_browser_requirements, {}),
    ("browser_vision", "👁️", _install.check_browser_vision_requirements, {"question": "", "annotate": False}),
    ("browser_console", "🖥️", _install.check_browser_requirements, {"clear": False, "expression": None}),
)


def _routed_check_fn(name: str):
    """Per-action availability gate (a named function, as the registry expects)."""
    def check() -> bool:
        return check_browser_routed_requirements(name)
    check.__name__ = check.__qualname__ = f"check_{name}_requirements"
    return check


def _routed_handler(name: str, fallback):
    def handler(args, **kw):
        return routed_browser_handler(name, args, fallback=lambda: fallback(args, kw),
                                      task_id=kw.get("task_id"), session_id=kw.get("session_id"))
    return handler


for _name, _emoji, _check_fn, _defaults, *_extra in _BROWSER_TOOL_TABLE:
    if _check_fn is None:  # also binds the legacy check_browser_<x>_requirements globals (tests + callers)
        _check_fn = globals()[f"check_{_name}_requirements"] = _routed_check_fn(_name)
    registry.register(name=_name, toolset="browser", schema=_BROWSER_SCHEMA_MAP[_name],
                      handler=_routed_handler(_name, _fallback_call(_name, _defaults, *_extra)),
                      check_fn=_check_fn, emoji=_emoji)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import List  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
import contextlib  # noqa: F401,E402
from datetime import datetime  # noqa: F401,E402
import functools  # noqa: F401,E402
import re  # noqa: F401,E402
import shutil  # noqa: F401,E402
import signal  # noqa: F401,E402
from datetime import timezone  # noqa: F401,E402

SNAPSHOT_SUMMARIZE_THRESHOLD = DEFAULT_SNAPSHOT_THRESHOLD


_PLUGIN_COMPAT_LAZY = {
    'BrowserUseProvider': ('plugins.browser.browser_use.provider', 'BrowserUseBrowserProvider'),
    'BrowserbaseProvider': ('plugins.browser.browserbase.provider', 'BrowserbaseBrowserProvider'),
    'CloudBrowserProvider': ('agent.browser_provider', 'BrowserProvider'),
    'FirecrawlProvider': ('plugins.browser.firecrawl.provider', 'FirecrawlBrowserProvider'),
    'agent_browser_runnable': ('hermes_constants', 'agent_browser_runnable'),
    'check_browser_requirements': ('tools.browser_tool_install', 'check_browser_requirements'),
    'check_browser_vision_requirements': ('tools.browser_tool_install', 'check_browser_vision_requirements'),
    'cleanup_all_browsers': ('tools.browser_tool_lifecycle', 'cleanup_all_browsers'),
    'cleanup_browser': ('tools.browser_tool_lifecycle', 'cleanup_browser'),
    'get_hermes_home_override': ('hermes_constants', 'get_hermes_home_override'),
    'hermes_home_key': ('hermes_constants', 'hermes_home_key'),
    'is_truthy_value': ('utils', 'is_truthy_value'),
    'lightpanda_engine_status': ('tools.browser_tool_lightpanda_fallback', 'lightpanda_engine_status'),
    'node_tool_runnable': ('hermes_constants', 'node_tool_runnable'),
    'normalize_browser_cloud_provider': ('tools.tool_backend_helpers', 'normalize_browser_cloud_provider'),
    'reset_hermes_home_override': ('hermes_constants', 'reset_hermes_home_override'),
    'set_hermes_home_override': ('hermes_constants', 'set_hermes_home_override'),
    'warm_agent_browser_npx_cache': ('tools.browser_tool_install', 'warm_agent_browser_npx_cache'),
    'windows_hide_flags': ('hermes_cli._subprocess_compat', 'windows_hide_flags'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
