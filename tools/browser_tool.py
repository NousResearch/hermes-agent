#!/usr/bin/env python3
"""Browser automation tools driven by the agent-browser CLI.

Backends — local headless Chromium, Browser Use / Browserbase / Firecrawl cloud
(auto-detected from config + credentials), a user-supplied CDP endpoint, or Camofox —
share one agent-facing behaviour: per-task sessions, accessibility-tree snapshots with
``@eN`` refs, automatic cleanup. Settings live under ``browser.*`` in config.yaml.
Sibling ``browser_tool_*`` modules hold extracted clusters.
"""

import atexit
import functools
import hashlib
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
from hermes_constants import get_hermes_home, hermes_home_key
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
    harnesses stub the ``tools`` package). The passthrough keys are re-added from the active
    profile's secret scope, never ``os.environ``: under multiplex that holds the LAUNCH profile's
    Browserbase/Firecrawl keys, and a served profile's browser must run on its own (or none)."""
    from agent.secret_scope import UnscopedSecretError, get_secret
    from tools.environments.local import served_profile_child_env

    from agent.proxy_bypass import add_loopback_no_proxy

    env = served_profile_child_env(inherit_credentials=False)
    for key in _BROWSER_PASSTHROUGH_KEYS:
        try:
            value = get_secret(key)
        except UnscopedSecretError:
            value = None  # multiplex, no scope bound: no key rather than a sibling profile's
        if value is not None:
            env[key] = value
    # The Browser Use harness dials the resolved local CDP URL over ``websockets``; without a
    # loopback NO_PROXY a macOS system proxy captures that dial (#110565).
    return add_loopback_no_proxy(env)


try:
    from tools.website_policy import check_website_access
except Exception:
    check_website_access = lambda url: None  # noqa: E731 — fail-open if policy module unavailable

try:
    from tools.url_safety import (
        is_safe_url as _is_safe_url,
        is_always_blocked_url as _is_always_blocked_url,
        normalize_url_for_request as _normalize_url_for_request,
    )
except Exception:
    _is_safe_url = lambda url: False  # noqa: E731 — fail-closed: block all if safety module unavailable
    _is_always_blocked_url = lambda url: True  # noqa: E731 — fail-closed on the floor too
    _normalize_url_for_request = lambda url: url  # noqa: E731 — best-effort fallback
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

# Default max chars for snapshot content before truncation. Aligned with
# web_tools.DEFAULT_EXTRACT_CHAR_LIMIT (15000) — the snapshot and
# web_extract paths share the same truncate-and-store pattern, so the model
# gets the same per-page budget from both. Configurable via
# ``browser.snapshot_threshold`` in config.yaml.
DEFAULT_SNAPSHOT_THRESHOLD = 15000
MIN_SNAPSHOT_THRESHOLD = 1000

# Backwards-compatible import surface. Runtime call sites use
# ``get_browser_snapshot_threshold()`` so config overrides take effect.
SNAPSHOT_SUMMARIZE_THRESHOLD = DEFAULT_SNAPSHOT_THRESHOLD

# Hard ceiling on the full-snapshot file written to cache/web when a snapshot
# is truncated. Mirrors web_tools.MAX_STORED_TEXT_CHARS —
# the model only ever sees the truncated view; the stored copy exists for
# read_file paging and must not write unbounded bytes to disk.
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
# The config-derived ones are keyed by profile home (``hermes_home_key()``): the multiplexed
# gateway serves every profile from one process, so a single slot would hand the launch
# profile's browser settings to every other profile.
_cached_command_timeout: Optional[Dict[str, int]] = None
# Flip the resolved flag BEFORE nulling the cache so a concurrent reader never sees ``resolved=True`` with
# ``cache=None`` (#14331).
_command_timeout_resolved = False
_cached_snapshot_threshold: Optional[int] = None
_snapshot_threshold_resolved = False


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
    """Process-cached ``_browser_cfg`` read, one slot per profile home (cleared by
    ``cleanup_all_browsers``). The value is stored BEFORE the resolved flag flips so a
    concurrent reader never sees ``resolved=True`` with an empty cache."""
    g = globals()
    home = hermes_home_key()
    cache = g[cache_name]
    if cache is None:
        cache = g[cache_name] = {}
    if g[flag_name] and cache.get(home) is not None:
        return cache[home]
    result = _browser_cfg(key, default, parse, log_label)
    cache[home] = result
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
    """Return the configured maximum browser snapshot size in characters.

    Reads the raw profile-aware config so tool JSON output is not affected by
    config-loader warnings. The value is cached for the browser lifecycle and
    reset by :func:`cleanup_all_browsers`.
    """
    global _cached_snapshot_threshold, _snapshot_threshold_resolved
    if _snapshot_threshold_resolved and _cached_snapshot_threshold is not None:
        return _cached_snapshot_threshold

    result = DEFAULT_SNAPSHOT_THRESHOLD
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg_get(cfg, "browser", "snapshot_threshold")
        if val is not None:
            result = max(int(val), MIN_SNAPSHOT_THRESHOLD)
    except Exception as exc:
        logger.debug("Could not read browser.snapshot_threshold: %s", exc)

    # Preserve the same race-safety invariant as the command-timeout cache.
    _cached_snapshot_threshold = result
    _snapshot_threshold_resolved = True
    return result


def _get_open_command_timeout(*, first_open: bool = False) -> int:
    """Timeout for agent-browser ``open`` (navigation / daemon cold start)."""
    return max(_safe_command_timeout(), MIN_FIRST_OPEN_TIMEOUT if first_open else MIN_OPEN_TIMEOUT)


from tools import browser_tool_session as _session


def _get_vision_model() -> Optional[str]:
    """Model for browser_vision (screenshot analysis — multimodal)."""
    return os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None


def _resolve_cdp_override(cdp_url: str) -> str:
    """Normalize a user-supplied CDP endpoint into a concrete connectable URL.


# Single shared real-profile copy-browser session: concurrent tasks reuse it
# instead of each launching a rival Chromium on the same copied user-data-dir.
_REAL_PROFILE_SESSION = "hermes-real-profile"
_real_profile_cdp_lock = threading.Lock()
_real_profile_cdp_cache: dict = {}
_real_profile_chrome_procs: list = []  # Popen handles of directly-launched real browsers



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


def _ensure_cdp_supervisor(task_id: str) -> None:
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

    Swallows all errors — failing to attach the supervisor must not break
    the browser session itself.  The agent simply won't see
    ``pending_dialogs`` / ``frame_tree`` fields in snapshots.
    """
    cdp_url = _get_cdp_override()
    if not cdp_url:
        # Fallback: active session may carry a per-session CDP URL from a
        # cloud provider (Browserbase sets this).
        with _cleanup_lock:
            session_info = _active_sessions.get(task_id, {})
        maybe = str(session_info.get("cdp_url") or "")
        if maybe:
            cdp_url = _resolve_cdp_override(maybe)
    if not cdp_url:
        return
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        policy, timeout_s = _get_dialog_policy_config()
        SUPERVISOR_REGISTRY.get_or_start(
            task_id=task_id,
            cdp_url=cdp_url,
            dialog_policy=policy,
            dialog_timeout_s=timeout_s,
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
    if _is_termux_environment():
        return "npm install -g agent-browser && agent-browser install"
    return "npm install -g agent-browser && agent-browser install --with-deps"


# Sentinel _find_agent_browser returns/caches to mean "resolve via npx" rather
# than a concrete executable path. A named constant + predicate keep the six
# comparison sites (four here, plus hermes_cli/tools_config.py and
# hermes_cli/doctor.py) from drifting if the sentinel's exact spelling ever
# changes.
NPX_AGENT_BROWSER_SENTINEL = "npx agent-browser"

# Pinned to match scripts/install.sh / scripts/install.ps1's
# "agent-browser@^0.26.0" managed install so a git-clone install resolving
# agent-browser via bare npx gets the same version as a managed install,
# instead of floating latest with no integrity check. Update both together.
AGENT_BROWSER_NPX_SPEC = "agent-browser@^0.26.0"


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
    """Copy browser fallback metadata from an internal result into a tool response."""
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
        # --ignore-scripts: AGENT_BROWSER_NPX_SPEC is a floating ^0.26.0 range,
        # not an exact pin — a compromised future 0.26.x patch must not get to
        # run its own install-time lifecycle scripts on this machine.
        cmd_prefix = [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", AGENT_BROWSER_NPX_SPEC]
    else:
        cmd_prefix = [browser_cmd]
    base_args = cmd_prefix + ["--engine", "chrome", "--session", tmp_session, "--json"]

    task_socket_dir = os.path.join(_socket_safe_tmpdir(), f"agent-browser-{tmp_session}")
    os.makedirs(task_socket_dir, mode=0o700, exist_ok=True)
    browser_env = _build_browser_env()
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


def _use_real_profile() -> bool:
    """Return whether the user consented to real-profile local browsing.

    Reads ``browser.use_real_profile`` (default False) on EVERY call — it is a
    consent switch, so flipping it off must take effect without a restart, and
    in a multiplexed gateway each profile's config must decide for itself.
    The read is one YAML load per local session creation (not per command),
    so there is no hot-path cost to keeping it uncached.
    """
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            return bool(browser_cfg.get("use_real_profile", False))
    except Exception as e:
        logger.debug("Could not read use_real_profile from config: %s", e)
    return False


# Prefix for profile-scoped real-profile copy-browser sessions. Each Hermes
# home gets a distinct agent-browser session and cache entry.
_REAL_PROFILE_SESSION = "hermes-real-profile"
_real_profile_cdp_lock = threading.Lock()
_real_profile_cdp_cache: dict = {}


def _real_profile_scope() -> tuple[str, str]:
    home_key = hermes_home_key()
    digest = hashlib.sha256(home_key.encode("utf-8")).hexdigest()[:12]
    return home_key, f"{_REAL_PROFILE_SESSION}-{digest}"


def _agent_browser_argv(browser_cmd: str) -> list:
    """Command prefix to invoke agent-browser (binary or npx sentinel)."""
    if _is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _resolve_npx_bin() or "npx"
        return [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", AGENT_BROWSER_NPX_SPEC]
    return [browser_cmd]


def _cdp_http_ready(http_cdp: str) -> bool:
    """True when an ``http://host:port`` CDP discovery root answers."""
    try:
        from hermes_cli.browser_connect import is_browser_debug_ready

        return is_browser_debug_ready(http_cdp, timeout=1.0)
    except Exception:
        return False


def _agent_browser_get_cdp(session_name: str) -> Optional[str]:
    """Return the HTTP CDP endpoint of an agent-browser session, or None.

    agent-browser prints ``ws://127.0.0.1:<port>/devtools/browser/<id>`` for
    ``get cdp-url``; the browser-use harness wants the HTTP discovery root
    (``http://127.0.0.1:<port>``), so we convert. None when the session isn't
    running or the port can't be parsed.
    """
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError:
        return None
    try:
        proc = subprocess.run(
            [*_agent_browser_argv(browser_cmd), "--session", session_name, "get", "cdp-url"],
            capture_output=True, text=True, timeout=15, env=_build_browser_env(),
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("real-profile get cdp-url failed: %s", e)
        return None
    out = (proc.stdout or "").strip()
    m = re.search(r"ws://127\.0\.0\.1:(\d+)/", out)
    if not m:
        return None
    return f"http://127.0.0.1:{m.group(1)}"


def _cdp_on_data_dir(http_cdp: str, data_dir: str) -> bool:
    """True when the CDP endpoint's browser is running on ``data_dir``.

    agent-browser's launched Chrome writes ``DevToolsActivePort`` into its
    user-data-dir with the live debug port on the first line. Matching that
    port against the CDP endpoint's port confirms the running browser is our
    profile copy — not a throwaway temp dir a raced/stale launch fell back to.
    """
    m = re.search(r":(\d+)", http_cdp or "")
    if not m:
        return False
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            port_line = fh.readline().strip()
        return port_line == m.group(1)
    except OSError:
        return False


def _agent_browser_close_session(session_name: str) -> None:
    """Best-effort close of an agent-browser session (stale/wrong-dir cleanup)."""
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError:
        return
    try:
        subprocess.run(
            [*_agent_browser_argv(browser_cmd), "--session", session_name, "close"],
            capture_output=True, text=True, timeout=15, env=_build_browser_env(),
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("real-profile session close failed: %s", e)


def _real_profile_cdp() -> tuple:
    """Resolve ``(cdp_url, error)`` for consented real-profile browsing.

    Snapshots the user's default-Chromium profile into a hermes-owned copy
    (auth/login state only), then has agent-browser launch the matching installed
    browser binary on that copy and returns its HTTP CDP endpoint. Matching the
    binary matters on platforms where browser credentials are app-bound. The
    copy is a non-default dir, so it sidesteps the Chrome >=136 default-profile
    remote-debugging block and never contends with the user's live profile.

    A single shared agent-browser session is reused across calls (its CDP URL
    is cached and re-validated). Returns ``(None, message)`` fail-closed when
    the default browser is non-Chromium or the snapshot/launch fails;
    ``(None, None)`` when consent is off.
    """
    home_key, session_name = _real_profile_scope()
    if not _use_real_profile():
        # Consent is off. If a snapshot store from a previous consented run is
        # still on disk, it holds copies of the user's cookies/logins — delete
        # it so revoking consent actually removes the credential copies. Cheap
        # (one isdir check) and idempotent.
        with _real_profile_cdp_lock:
            snapshot_root = get_hermes_home() / "browser-profile"
            if home_key in _real_profile_cdp_cache or os.path.lexists(snapshot_root):
                _agent_browser_close_session(session_name)
            _real_profile_cdp_cache.pop(home_key, None)
            try:
                from hermes_cli.browser_connect import cleanup_real_profile_snapshots

                cleanup_error = cleanup_real_profile_snapshots()
            except Exception as e:
                cleanup_error = f"real-profile cleanup failed: {e}"
            if cleanup_error:
                logger.warning("cleanup-on-consent-off failed: %s", cleanup_error)
                return None, cleanup_error
        return None, None

    # Lightpanda cannot load a Chromium profile — agent-browser rejects
    # ``--profile`` outright under that engine ("Profiles are not supported
    # with Lightpanda"). Detect it here, BEFORE default-browser detection, so
    # even a host with no Chromium default reports the actionable conflict (the
    # engine setting) rather than a generic launch failure.
    if _using_lightpanda_engine():
        return None, (
            "browser.use_real_profile is on, but browser.engine is set to "
            "'lightpanda', which cannot load a real Chromium profile. Set "
            "browser.engine to 'auto' or 'chrome' to use real-profile browsing, "
            "or turn the toggle off."
        )

    from hermes_cli.browser_connect import (
        UNSUPPORTED_CHANNEL,
        chromium_executable,
        detect_default_chromium,
        real_profile_copy_dir,
        snapshot_real_profile,
    )

    with _real_profile_cdp_lock:
        browser = detect_default_chromium()
        if browser is None:
            return None, (
                "browser.use_real_profile is on, but your default browser is not a "
                "supported Chromium browser (Chrome, Edge, Brave, Chromium). "
                "Real-profile browsing requires a Chromium default; set one or turn "
                "the toggle off."
            )
        if browser == UNSUPPORTED_CHANNEL:
            # A recognized pre-release channel (Beta/Dev/Canary) is the OS
            # default. Its profile lives in a channel-specific directory we
            # don't resolve, and normalizing it to the stable family would
            # drive a DIFFERENT profile/account — a wrong-principal bug. Fail
            # closed rather than guess (#95549 invariant).
            return None, (
                "browser.use_real_profile is on, but your default browser is a "
                "pre-release Chromium channel (Beta / Dev / Canary), which "
                "real-profile browsing does not support. Set your default to a "
                "stable Chrome / Edge / Brave / Chromium, or turn the toggle off."
            )

        # Reuse BEFORE writing anything. A shared copy-browser may already be up
        # from a previous hermes process; if it is driving OUR copy dir, hand it
        # back untouched. CRITICAL: the snapshot overlay (which truncates and
        # rewrites Cookies / Login Data) must NOT run while that browser holds
        # the user-data-dir open — doing so corrupts the live databases (torn
        # reads, locked transactions, phantom logouts). So resolve the copy dir
        # as a PATH only (no copy), probe reuse, and return early on a hit. The
        # snapshot/overlay happens solely on the relaunch path below, when no
        # live browser owns the dir.
        copy_dir = real_profile_copy_dir(browser)

        cached = _real_profile_cdp_cache.get(home_key)
        if isinstance(cached, dict):
            cached_cdp = cached.get("cdp")
            cached_dir = cached.get("copy_dir")
            if (
                cached_cdp
                and cached_dir == os.path.normcase(os.path.normpath(copy_dir))
                and _cdp_http_ready(cached_cdp)
                and _cdp_on_data_dir(cached_cdp, copy_dir)
            ):
                return cached_cdp, None
            if cached_cdp:
                _agent_browser_close_session(session_name)
        _real_profile_cdp_cache.pop(home_key, None)

        existing = _agent_browser_get_cdp(session_name)
        if existing and _cdp_http_ready(existing) and _cdp_on_data_dir(existing, copy_dir):
            _real_profile_cdp_cache[home_key] = {
                "cdp": existing,
                "copy_dir": os.path.normcase(os.path.normpath(copy_dir)),
            }
            return existing, None
        if existing:
            # Stale/wrong-dir session (throwaway-temp fallback, or an old copy):
            # close it so nothing holds the dir open before we overlay + relaunch.
            _agent_browser_close_session(session_name)

        # No live browser owns the dir now — safe to (re)snapshot + overlay.
        snap_dir, err = snapshot_real_profile(browser)
        if err or not snap_dir:
            from hermes_cli.browser_connect import _PROFILE_LOCKED_PREFIX

            if err and err.startswith(_PROFILE_LOCKED_PREFIX):
                # The user's browser is holding the profile. Surface the guidance
                # verbatim (it already tells the agent whether closing is armed)
                # plus the exact approved-close command. The agent must ASK the
                # user before running it — it quits their browser.
                body = err[len(_PROFILE_LOCKED_PREFIX):]
                return None, (
                    body + " To close it (only after the user approves — it "
                    "quits their browser and loses unsaved tabs), run: "
                    "`hermes browser close-profile`, then retry."
                )
            return None, f"browser.use_real_profile is on, but {err}"
        copy_dir = snap_dir

        # Launch the matching installed browser on the profile COPY through
        # agent-browser. A different packaged Chromium may be unable to decrypt
        # app-bound credentials copied from Chrome, Edge, or Brave.
        try:
            browser_cmd = _find_agent_browser()
        except FileNotFoundError as e:
            return None, (
                "browser.use_real_profile is on, but the local browser engine "
                f"(agent-browser) is not installed: {e}"
            )
        executable = chromium_executable(browser)
        if not executable:
            return None, (
                f"browser.use_real_profile is on, but the installed {browser} "
                "executable could not be resolved. Hermes will not open copied "
                "credentials with a different Chromium binary."
            )
        argv = [
            *_agent_browser_argv(browser_cmd),
            "--session", session_name,
            "--profile", copy_dir,
            "--executable-path", executable,
        ]
        # Do NOT pass agent-browser's ``--headless``: it maps to Chrome's legacy
        # headless mode, which uses a SEPARATE cookie store and loads none of the
        # copied profile's cookies (verified: --headless → 0 cookies, default →
        # full jar). agent-browser's default already runs windowless on a
        # server (no DISPLAY) while reading the real cookie store, which is
        # exactly what real-profile browsing needs. Headed mode is a superset
        # (visible window) and equally fine, so no flag either way.
        argv += ["open", "about:blank"]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=_get_open_command_timeout(first_open=True),
                env=_build_browser_env(),
            )
        except subprocess.TimeoutExpired:
            return None, (
                "browser.use_real_profile is on, but the real-profile browser "
                "took too long to start. Retry, or turn the toggle off."
            )
        except (subprocess.SubprocessError, OSError) as e:
            return None, f"browser.use_real_profile is on, but the launch failed: {e}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = tail[-1] if tail else f"exit {proc.returncode}"
            return None, (
                f"browser.use_real_profile is on, but the real-profile browser "
                f"failed to start: {reason}"
            )

        cdp = _agent_browser_get_cdp(session_name)
        if not cdp:
            return None, (
                "browser.use_real_profile is on, but the real-profile browser "
                "started without exposing a devtools endpoint. Retry, or turn "
                "the toggle off."
            )
        _real_profile_cdp_cache[home_key] = {
            "cdp": cdp,
            "copy_dir": os.path.normcase(os.path.normpath(copy_dir)),
        }
        logger.info("real-profile browser ready for %s at %s (%s)", browser, cdp, copy_dir)
        return cdp, None


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
    continues to serve public URLs. With ``browser.use_real_profile`` consent,
    local sessions (bare or sidecar) attach to the user's real-profile
    copy-browser via ``_create_local_session``; forcing a local session from
    the model side is the Browser Use lane's ``local`` argument, not a
    built-in-tools argument.
    """
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

# Session keys flagged suspect after a command timeout (#72205 / #85125 3b).
# Written by _BrowserSessionBackend.mark_suspect (cheap, lock-free — a single
# GIL-atomic dict write per the agent.deadline.SuspectableBackend contract);
# consumed by ensure_healthy() at next use, which recycles the session.
_suspect_browser_sessions: Dict[str, str] = {}


class _BrowserSessionBackend:
    """``agent.deadline.SuspectableBackend`` adapter for one cached session key.

    The browser "backend" is the module-level ``_active_sessions[key]`` cache
    entry plus its agent-browser daemon, so the adapter is a thin stateless
    view keyed by session key rather than a long-lived object.  Browser
    commands run through raw ``subprocess`` waits (not ``run_bounded_*``), so
    the timeout path calls ``mark_suspect`` inline; ``ensure_healthy`` runs at
    the top of ``_get_session_info`` — the single choke point every browser
    command passes through before reusing a cached session.
    """

    __slots__ = ("_session_key",)

    def __init__(self, session_key: str) -> None:
        self._session_key = session_key

    def mark_suspect(self, reason: str) -> None:
        """Flag the cached session as possibly poisoned.

        MUST stay cheap, non-blocking, and lock-free (SuspectableBackend
        adopter contract): it runs inline on the timed-out caller's thread.
        All expensive recycle work is deferred to ``ensure_healthy``.
        """
        _suspect_browser_sessions[self._session_key] = reason

    def ensure_healthy(self) -> bool:
        """Recycle the session when a prior timeout marked it suspect.

        Returns ``True`` when the cached session is safe to reuse, ``False``
        after tearing down a suspect session (caller creates a fresh one).
        The flag is popped *before* teardown: ``_cleanup_single_browser_session``
        issues an agent-browser ``close`` through ``_run_browser_command`` /
        ``_get_session_info``, and clearing first keeps that re-entrant call
        from recursing back into another recycle.
        """
        reason = _suspect_browser_sessions.pop(self._session_key, None)
        if reason is None:
            return True
        logger.info(
            "Recycling suspect browser session %s before reuse (%s)",
            self._session_key, reason,
        )
        try:
            _cleanup_single_browser_session(self._session_key)
        except Exception:
            logger.warning(
                "Teardown of suspect browser session %s failed; a fresh "
                "session will be created anyway", self._session_key,
                exc_info=True,
            )
        return False


def _browser_session_backend(session_key: str) -> _BrowserSessionBackend:
    """Return the SuspectableBackend adapter for ``session_key``."""
    return _BrowserSessionBackend(session_key)

# Background cleanup thread state
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
        "description": "Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer a lightweight retrieval tool when one is available (faster, cheaper). For plain-text endpoints — URLs ending in .md, .txt, .json, .yaml, .yml, .csv, .xml, raw.githubusercontent.com, or any documented API endpoint — prefer an available text-extraction or terminal-fetch tool; the browser stack is overkill and much slower for these. Use browser tools when you need to interact with a page (click, fill forms, dynamic content). Returns a compact page snapshot with interactive elements and ref IDs — no need to call browser_snapshot separately after navigating.",
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

# ============================================================================
# Utility Functions
# ============================================================================

def _create_local_session(task_id: str, allow_real_profile: bool = True) -> Dict[str, str]:
    import uuid

    # Real-profile consent: instead of an agent-browser-managed throwaway
    # Chromium, attach this local session (via CDP) to the user's default
    # browser running on a hermes-owned SNAPSHOT of their real profile —
    # live logins/cookies included. Fail closed on resolver/launch errors:
    # a consented user must never be silently downgraded to a throwaway.
    #
    # ``allow_real_profile=False`` is passed by the hybrid private-URL sidecar
    # (``::local`` key): that path exists to keep a LAN/loopback host OFF the
    # cloud backend, and routing the user's full authenticated cookie jar to an
    # arbitrary internal host the model chose to visit is a strictly larger
    # exposure than the routing rule was protecting against — and one the user
    # never consented to for that URL. The sidecar always gets a throwaway
    # profile. (Also keeps a real-profile resolve failure from breaking
    # private-URL routing, which has nothing to do with the real profile.)
    if allow_real_profile:
        cdp_url, err = _real_profile_cdp()
        if err:
            raise RuntimeError(err)
        if cdp_url:
            session_name = f"rp_{uuid.uuid4().hex[:10]}"
            logger.info(
                "Created real-profile local session %s for task %s", session_name, task_id
            )
            return {
                "session_name": session_name,
                "bb_session_id": None,
                "cdp_url": _resolve_cdp_override(cdp_url),
                "features": {"local": True, "real_profile": True},
            }

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


def _get_session_info(task_id: Optional[str] = None) -> Dict[str, Any]:
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

    # Start the cleanup thread if not running (handles inactivity timeouts)
    _start_browser_cleanup_thread()

    # Update activity timestamp for this session
    _update_session_activity(task_id)

    with _cleanup_lock:
        # Check if we already have a session for this task
        existing_session = _active_sessions.get(task_id)

    # Suspect-session recycle (#72205 / #85125 3b): a previous command
    # timeout marked this cached session suspect via the SuspectableBackend
    # adapter.  ensure_healthy() tears it down here, at next use, and we fall
    # through to create a fresh session — the expensive recycle lives on this
    # path, not on the timeout path (mark must stay cheap).
    if existing_session is not None and not _browser_session_backend(task_id).ensure_healthy():
        # Teardown removes the activity entry; the replacement must be
        # tracked by the inactivity reaper like an initial session.
        _update_session_activity(task_id)
        with _cleanup_lock:
            replacement = _active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            # Another thread already recycled and re-created it.
            return replacement
        existing_session = None

    if existing_session is not None:
        if not _session_has_expired(existing_session):
            return existing_session

        logger.info(
            "Replacing expired cloud browser session for task %s",
            task_id,
        )
        _cleanup_single_browser_session(task_id)
        # Cleanup removes the activity entry. The replacement session must be
        # tracked by the inactivity reaper just like an initial session.
        _update_session_activity(task_id)

        # Guard against a concurrent replacement: another thread may have
        # already cleaned up the expired session and created a fresh one
        # while we were waiting.  If so, return the live replacement instead
        # of falling through to create yet another session.
        with _cleanup_lock:
            replacement = _active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            return replacement

    # Hybrid routing: session keys ending with ``::local`` force a local
    # Chromium regardless of the globally-configured cloud provider.  Public
    # URLs in the same conversation continue to use the cloud session under
    # the bare task_id key.
    force_local = _is_local_sidecar_key(task_id)

    # Create session outside the lock (network call in cloud mode)
    cdp_override = _get_cdp_override()
    if cdp_override and not force_local:
        session_info = _create_cdp_session(task_id, cdp_override)
    elif force_local:
        # Hybrid private-URL sidecar: NEVER the real profile (see
        # _create_local_session — presenting real cookies to an arbitrary LAN
        # host the model routed here is unconsented exposure).
        session_info = _create_local_session(task_id, allow_real_profile=False)
    else:
        provider = _get_cloud_provider()
        if provider is None:
            session_info = _create_local_session(task_id)
        else:
            try:
                session_info = provider.create_session(task_id)
                # Validate cloud provider returned a usable session
                if not session_info or not isinstance(session_info, dict):
                    raise ValueError(f"Cloud provider returned invalid session: {session_info!r}")
                if session_info.get("cdp_url"):
                    # Some cloud providers (including Browser-Use v3) return an HTTP
                    # CDP discovery URL instead of a raw websocket endpoint.
                    session_info = dict(session_info)
                    session_info["cdp_url"] = _resolve_cdp_override(str(session_info["cdp_url"]))
            except Exception as e:
                provider_name = type(provider).__name__
                logger.warning(
                    "Cloud provider %s failed (%s); attempting fallback to local "
                    "Chromium for task %s",
                    provider_name, e, task_id,
                    exc_info=True,
                )
                try:
                    session_info = _create_local_session(
                        task_id, allow_real_profile=False
                    )
                except Exception as local_error:
                    raise RuntimeError(
                        f"Cloud provider {provider_name} failed ({e}) and local "
                        f"fallback also failed ({local_error})"
                    ) from e
                # Mark session as degraded for observability
                if isinstance(session_info, dict):
                    session_info = dict(session_info)
                    session_info["fallback_from_cloud"] = True
                    session_info["fallback_reason"] = str(e)
                    session_info["fallback_provider"] = provider_name

    with _cleanup_lock:
        # Double-check: another thread may have created a session while we
        # were doing the network call. Use the existing one to avoid leaking
        # orphan cloud sessions.
        if task_id in _active_sessions:
            return _active_sessions[task_id]
        session_info = dict(session_info)
        session_info.setdefault("session_key", task_id)
        session_info.setdefault("owner_task_id", _bare_task_id_for_session_key(task_id))
        _active_sessions[task_id] = session_info
        # A brand-new session is healthy by definition — drop any stale
        # suspect flag left by a wedged-path eviction of its predecessor.
        _suspect_browser_sessions.pop(task_id, None)

    # Lazy-start the CDP supervisor now that the session exists (if the
    # backend surfaces a CDP URL via override or session_info["cdp_url"]).
    # Idempotent; swallows errors. See _ensure_cdp_supervisor for details.
    # Skip for local sidecars — they have no CDP URL.
    if not force_local:
        _ensure_cdp_supervisor(task_id)

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


def _find_agent_browser(*, validate: bool = True) -> str:
    """
    Find the agent-browser CLI executable.

    Checks in order: current PATH, Homebrew/common bin dirs, Hermes-managed
    node, local node_modules/.bin/, npx fallback.

    Returns:
        Path to agent-browser executable

    Raises:
        FileNotFoundError: If agent-browser is not installed
    """
    global _cached_agent_browser, _agent_browser_resolved
    if _agent_browser_resolved:
        if _cached_agent_browser is None:
            raise FileNotFoundError(
                "agent-browser CLI not found (cached). Install it with: "
                f"{_browser_install_hint()}\n"
                "Or ensure npx is available in your PATH."
            )
        return _cached_agent_browser

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
    if which_result and (
        agent_browser_runnable(which_result) if validate else _agent_browser_candidate_present(which_result)
    ):
        if not validate:
            return which_result
        _cached_agent_browser = which_result
        _agent_browser_resolved = True
        return which_result

    # Build an extended search PATH including Hermes-managed Node, macOS
    # versioned Homebrew installs, and fallback system dirs like Termux.
    extended_path = _merge_browser_path("")
    if extended_path:
        which_result = shutil.which("agent-browser", path=extended_path)
        if which_result and (
            agent_browser_runnable(which_result) if validate else _agent_browser_candidate_present(which_result)
        ):
            if not validate:
                return which_result
            _cached_agent_browser = which_result
            _agent_browser_resolved = True
            return which_result

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
        if local_which and (
            agent_browser_runnable(local_which) if validate else _agent_browser_candidate_present(local_which)
        ):
            if not validate:
                return local_which
            _cached_agent_browser = local_which
            _agent_browser_resolved = True
            return _cached_agent_browser

    # Check common npx locations (also search the extended fallback PATH)
    npx_path = _resolve_npx_bin()
    if npx_path:
        if not validate:
            return NPX_AGENT_BROWSER_SENTINEL
        _cached_agent_browser = NPX_AGENT_BROWSER_SENTINEL
        _agent_browser_resolved = True
        return _cached_agent_browser

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

    Delegates to :func:`agent.deadline.kill_process_tree` (#85125 4d): same
    ``taskkill /T /F`` on Windows and killpg-when-group-leader on POSIX, plus
    a psutil descendant sweep that also reaches descendants that ``setsid``'d
    into their own session (agent-browser's detached daemon grandchild).
    SIGKILL-only instead of the old zero-grace SIGTERM→SIGKILL pair — the
    grace period was already zero, so the observable effect is identical.
    Any delegation failure falls back to the original local implementation
    (:func:`_legacy_kill_process_tree`); never raises either way.
    """
    try:
        from agent.deadline import kill_process_tree as _deadline_kill_tree

        _deadline_kill_tree(proc.pid)
    except Exception:
        _legacy_kill_process_tree(proc)


def _legacy_kill_process_tree(proc: "subprocess.Popen") -> None:
    """Pre-#85125 local tree-kill — fallback when agent.deadline is unavailable."""
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
        # --ignore-scripts: AGENT_BROWSER_NPX_SPEC is a floating ^0.26.0
        # range, not an exact pin — a compromised future 0.26.x patch must
        # not get to run its own install-time lifecycle scripts here.
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


def _discard_timed_out_browser_session(
    task_id: str,
    session_info: Dict[str, Any],
    task_socket_dir: str,
) -> None:
    """Drop a stuck client generation without losing cloud cleanup state."""
    with _cleanup_lock:
        if _active_sessions.get(task_id) is not session_info:
            return
        _stop_cdp_supervisor(task_id)
        if session_info.get("bb_session_id") or session_info.get("cdp_url"):
            import uuid
            replacement = dict(session_info)
            replacement["session_name"] = f"h_{uuid.uuid4().hex[:10]}"
            replacement.pop("_first_nav", None)
            _active_sessions[task_id] = replacement
        else:
            _active_sessions.pop(task_id, None)
            _session_last_activity.pop(task_id, None)

        bare_task_id = _bare_task_id_for_session_key(task_id)
        if _last_active_session_key.get(bare_task_id) == task_id:
            _last_active_session_key.pop(bare_task_id, None)

    session_name = str(session_info.get("session_name") or "")
    if session_name:
        pid_file = os.path.join(task_socket_dir, f"{session_name}.pid")
        if os.path.isfile(pid_file):
            try:
                daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
                if not _verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name):
                    return
                # Tree-kill (#68139 / #85125 4c): the daemon spawns Chromium
                # children; terminating only the daemon PID leaks the whole
                # Chromium tree.  agent.deadline.kill_process_tree escalates
                # SIGTERM → SIGKILL across the tree.
                from agent import deadline as _deadline

                _deadline.kill_process_tree(daemon_pid)
            except (ProcessLookupError, ValueError, PermissionError, OSError):
                logger.debug("Could not kill timed-out browser daemon for %s", session_name)
                return
    shutil.rmtree(task_socket_dir, ignore_errors=True)


def _read_browser_daemon_pid(task_socket_dir: str, session_name: str) -> Optional[int]:
    """Read the agent-browser daemon PID for a session (best-effort)."""
    pid_file = os.path.join(task_socket_dir, f"{session_name}.pid")
    try:
        return int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _browser_daemon_responsive(task_socket_dir: str, probe_timeout_s: float = 1.0) -> bool:
    """Cheap liveness probe: can we connect to the daemon's control socket?

    The agent-browser daemon listens on a unix socket inside the session's
    socket dir.  A successful connect proves the daemon's accept loop is
    alive (the timed-out command was wedged on the page/CDP side, not the
    daemon).  A refused / missing / timed-out connect means the daemon is
    wedged or dead.  Windows agent-browser uses named pipes, not unix
    sockets — no probe is possible there, so we conservatively report
    unresponsive (tree-kill + respawn is the safe recovery).
    """
    if os.name == "nt":
        return False
    import socket as socket_mod

    if not hasattr(socket_mod, "AF_UNIX"):
        return False
    try:
        entries = os.listdir(task_socket_dir)
    except OSError:
        return False
    sock_paths = [
        os.path.join(task_socket_dir, e) for e in entries if e.endswith(".sock")
    ]
    for sock_path in sock_paths:
        try:
            with socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM) as s:
                s.settimeout(probe_timeout_s)
                s.connect(sock_path)
                return True
        except OSError:
            continue
    return False


def _handle_browser_command_timeout(
    task_id: str,
    session_info: Dict[str, Any],
    task_socket_dir: str,
) -> None:
    """Recover session state after a browser command timeout (#72205, #68139).

    The wedged-vs-alive rule:

    * **Cloud / CDP sessions** — there is no local daemon to probe or kill.
      Replace the stuck client generation immediately (fresh ``session_name``,
      same ``bb_session_id`` so cloud cleanup still works) — #72206's
      original behavior, preserved verbatim.
    * **Local daemon alive** (PID readable, process alive, identity-verified
      as ours, control socket accepts a connection): the *command* wedged —
      page hang, stuck navigation — but the daemon itself is fine.  Killing
      it would be overkill and slow.  Mark the session suspect only; the
      next use recycles it through ``ensure_healthy`` → clean agent-browser
      ``close`` → fresh session.
    * **Local daemon wedged or dead** (no PID, dead PID, failed identity
      check, or unresponsive socket): the daemon cannot service a clean
      close, and its Chromium children would leak.  Tree-kill the daemon's
      process tree via ``agent.deadline.kill_process_tree`` and evict the
      cache entry now; the next browser call respawns from scratch.

    Both local branches ``mark_suspect`` first — cheap, lock-free — so the
    poisoned-cache invariant holds even if the eviction below races another
    thread's replacement (``_discard_timed_out_browser_session`` no-ops on a
    concurrent replacement; the flag then triggers one harmless no-op
    teardown at next use).
    """
    if session_info.get("bb_session_id") or session_info.get("cdp_url"):
        _discard_timed_out_browser_session(task_id, session_info, task_socket_dir)
        return

    _browser_session_backend(task_id).mark_suspect(
        "browser command timed out; session may be poisoned"
    )

    session_name = str(session_info.get("session_name") or "")
    daemon_pid = _read_browser_daemon_pid(task_socket_dir, session_name) if session_name else None
    daemon_alive = (
        daemon_pid is not None
        and _pid_exists(daemon_pid)
        and _verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name)
        and _browser_daemon_responsive(task_socket_dir)
    )
    if daemon_alive:
        logger.warning(
            "browser daemon for %s is alive after command timeout; session "
            "marked suspect and will be recycled at next use", task_id,
        )
        return

    logger.warning(
        "browser daemon for %s is wedged or dead after command timeout; "
        "tree-killing and evicting the session", task_id,
    )
    _discard_timed_out_browser_session(task_id, session_info, task_socket_dir)
    # The poisoned entry is gone (evicted, or superseded by a concurrent
    # replacement discard refused to touch) — either way the cache no longer
    # holds the timed-out session, so drop the flag: it must not poison a
    # session created later under the same key.
    _suspect_browser_sessions.pop(task_id, None)


def _pid_exists(pid: int) -> bool:
    """Best-effort 'is this PID alive' check (signal 0 / psutil on Windows)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import psutil

            return psutil.pid_exists(pid)
        except Exception:
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — psutil.pid_exists above handles Windows
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _run_browser_command(
    task_id: str,
    command: str,
    args: List[str] = None,
    timeout: Optional[int] = None,
    _engine_override: Optional[str] = None,
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
    if is_interrupted():
        return {"success": False, "error": "Interrupted"}

    # Get session info (creates Browserbase session with proxies if needed)
    try:
        session_info = _get_session_info(task_id)
    except Exception as e:
        logger.warning("Failed to create browser session for task=%s: %s", task_id, e)
        return {"success": False, "error": f"Failed to create browser session: {str(e)}"}
    # Cleanup stops the supervisor before closing the backend; keep it stopped.
    if command != "close" and session_info.get("cdp_url"):
        _ensure_cdp_supervisor(task_id)

    # Build the command with the appropriate backend flag.
    # Cloud mode: --cdp <websocket_url> connects to Browserbase.
    # Local mode: --session <name> launches a local headless Chromium.
    # The rest of the command (--json, command, args) is identical.
    if session_info.get("cdp_url"):
        # Cloud mode — connect to remote Browserbase browser via CDP
        # IMPORTANT: Do NOT use --session with --cdp. In agent-browser >=0.13,
        # --session creates a local browser instance and silently ignores --cdp.
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
        cmd_prefix = [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", AGENT_BROWSER_NPX_SPEC]
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
        logger.debug("browser cmd=%s task=%s socket_dir=%s (%d chars)",
                     command, task_id, task_socket_dir, len(task_socket_dir))

        browser_env = _build_browser_env()

        # Ensure subprocesses inherit the same browser-specific PATH fallbacks
        # used during CLI discovery.
        browser_env["PATH"] = _merge_browser_path(browser_env.get("PATH", ""))
        browser_env["AGENT_BROWSER_SOCKET_DIR"] = task_socket_dir

        # Tell the agent-browser daemon to self-terminate after being idle
        # for our configured inactivity timeout.  This is the daemon-side
        # counterpart to our Python-side _cleanup_inactive_browser_sessions
        # — the daemon kills itself and its Chrome children when no CLI
        # commands arrive within the window.  Added in agent-browser 0.24.
        if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in browser_env:
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
            _handle_browser_command_timeout(task_id, session_info, task_socket_dir)
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
        from tools.spill_safety import ensure_spill_dir, write_text_exclusive

        cache_dir = get_hermes_dir("cache/web", "web_cache")
        ensure_spill_dir(cache_dir, private=False)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"browser-snapshot-{digest}.txt"
        # Deterministic filename in a well-known dir: refuse symlinks via
        # lstat-unlink + exclusive create. Re-snapshotting the same page
        # state legitimately overwrites (same content-hash name). Not
        # private: cache/web is bind-mounted into remote backends whose
        # container UID must be able to read it.
        write_text_exclusive(path, content, private=False, overwrite=True)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to store full browser snapshot: %s", exc)
        return None


def _truncate_snapshot(snapshot_text: str, max_chars: Optional[int] = None) -> str:
    """Structure-aware truncation for snapshots.

    Cuts at line boundaries so that accessibility tree elements are never
    split mid-line. The full snapshot is saved to cache/web (same pattern as
    web_extract's truncate-and-store) and the appended note tells the agent
    exactly where the complete text lives and how to page through it with
    read_file — element refs beyond the cut are in the file, not lost.

    Args:
        snapshot_text: The snapshot text to truncate
        max_chars: Maximum characters to keep. Defaults to the configured
            ``browser.snapshot_threshold`` (see
            :func:`get_browser_snapshot_threshold`).

    Returns:
        Truncated text with a stored-full-text pointer if truncated
    """
    if max_chars is None:
        max_chars = get_browser_snapshot_threshold()
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
    (1) cloud metadata / IMDS refused UNCONDITIONALLY (a local Chromium on a cloud VM still
    reaches the host IMDS); (2) private addresses refused unless local, sidecar-routed, or
    ``browser.allow_private_urls``; (3) website policy allow/deny lists.

    Credential-NAMED query params (``?token=``, ``?signature=``) are deliberately NOT a floor:
    magic links, OAuth callbacks and signed CDN assets are how the agent signs in and browses, and
    a cloud browser already sees every cookie and typed password of the session — refusing the
    URL protects nothing. Hermes' own secrets leaking into a URL are caught by ``_secret_url_error``."""
    local = _cloud._is_local_backend()
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

    if result.get("success"):
        data = result.get("data", {})
        title = data.get("title", "")
        final_url = data.get("url", url)

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
            return json.dumps({
                "success": False,
                "error": "Blocked: redirect landed on a private/internal address",
            })

        response = {
            "success": True,
            "url": final_url,
            "title": title
        }
        # Auditability: stamp navigations that ran on the user's real-profile
        # copy-browser so usage is visible in the tool result.
        try:
            if (session_info.get("features") or {}).get("real_profile"):
                response["used_real_profile"] = True
        except Exception:
            pass
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
            if snap_result.get("success"):
                snap_data = snap_result.get("data", {})
                snapshot_text = snap_data.get("snapshot", "")
                refs = snap_data.get("refs", {})
                threshold = get_browser_snapshot_threshold()
                if len(snapshot_text) > threshold:
                    snapshot_text = _truncate_snapshot(snapshot_text, max_chars=threshold)
                response["snapshot"] = _redact_browser_output(snapshot_text)
                response["element_count"] = len(refs) if refs else 0
                if snap_result.get("fallback_warning") and not response.get("fallback_warning"):
                    _copy_fallback_warning(response, snap_result)
        except Exception as e:
            logger.debug("Auto-snapshot after navigate failed: %s", e)

        return json.dumps(response, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", "Navigation failed")
        }, ensure_ascii=False)


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
        user_task: Deprecated — accepted for call-site compatibility, unused.
            Oversized snapshots always truncate-and-store (no LLM pass).

    Returns:
        JSON string with page snapshot
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_snapshot
        return camofox_snapshot(full, task_id)

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

        # Oversized snapshots truncate at line boundaries; the full
        # accessibility tree is stored to cache/web and the appended note
        # tells the agent how to page through it with read_file (same
        # pattern as web_extract — no LLM summarization). Threshold is
        # configurable via browser.snapshot_threshold.
        threshold = get_browser_snapshot_threshold()
        if len(snapshot_text) > threshold:
            snapshot_text = _truncate_snapshot(snapshot_text, max_chars=threshold)

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

        # NOTE: the full-resolution base64 encode is deliberately deferred.
        # The native fast path below sizes its own history-reuse embed via
        # _resize_image_for_vision (stat-based quick estimate — no full-res
        # encode when oversized), and only the aux-LLM fallback path needs
        # the one-shot full-res data URL.

        # Fast path: when native image routing is in effect for the active main
        # model, attach the screenshot directly instead of describing it through
        # an auxiliary vision LLM. The model inspects the pixels on its next
        # turn — no aux call, no information loss. Consistent with vision_analyze.
        from tools.vision_tools import (
            _EMBED_MAX_DIMENSION,
            _EMBED_TARGET_BYTES,
            _build_native_vision_tool_result,
            _resize_image_for_vision,
            _should_use_native_vision_fast_path,
        )

        if _should_use_native_vision_fast_path():
            # History-reuse cap (#92699): this embed is baked into the tool
            # result and re-sent on every later turn, exactly like
            # vision_analyze's native path — apply the same proactive resize
            # so full-res screenshots can't enter immutable history uncapped.
            # The helper's internal stat/dimension quick-estimate skips the
            # resize (and encodes directly) when the screenshot is already
            # under both caps, so no full-res base64 is built just to be
            # thrown away. Fail-open: without Pillow it falls back to the
            # raw bytes and the compressor's keep-newest pass still retires
            # stale embeds.
            data_url = _resize_image_for_vision(
                screenshot_path,
                mime_type="image/png",
                max_base64_bytes=_EMBED_TARGET_BYTES,
                max_dimension=_EMBED_MAX_DIMENSION,
                force_jpeg=True,
            )
            native_result = _build_native_vision_tool_result(
                image_url=str(screenshot_path),
                question=question,
                image_data_url=data_url,
                image_size_bytes=screenshot_path.stat().st_size,
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

        # Aux-LLM path: one-shot analysis, not baked into history — encode at
        # full resolution here (the pre-existing 5 MB oversize guard below
        # still applies).
        _screenshot_bytes = screenshot_path.read_bytes()
        _screenshot_b64 = base64.b64encode(_screenshot_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{_screenshot_b64}"

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
        return _dumps(response_data)
    except Exception as e:
        # Keep a captured screenshot — the failure is in the analysis, not the capture,
        # and deleting it loses evidence. The 24-hour cleanup bounds disk growth.
        logger.warning("browser_vision failed: %s", e, exc_info=True)
        error_info = _err(f"Error during vision analysis: {str(e)}")
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

def cleanup_browser(task_id: Optional[str] = None) -> None:
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
    if task_id is None:
        task_id = "default"

    # Expand to the full set of session keys to reap. For a bare task_id
    # that includes the cloud/primary key + the local sidecar if one exists.
    if _is_local_sidecar_key(task_id):
        session_keys = [task_id]
        bare_task_id = task_id[: -len(_LOCAL_SUFFIX)]
    else:
        session_keys = [task_id]
        sidecar_key = f"{task_id}{_LOCAL_SUFFIX}"
        with _cleanup_lock:
            if sidecar_key in _active_sessions:
                session_keys.append(sidecar_key)
        bare_task_id = task_id

    for session_key in session_keys:
        _cleanup_single_browser_session(session_key)

    # Drop stale last-active ownership. Cleaning a bare task drops its binding;
    # cleaning a sidecar drops the binding only if that sidecar was still the
    # recorded owner. This prevents a later click/snapshot from resurrecting a
    # cleaned sidecar on about:blank while preserving a primary-session binding.
    if _is_local_sidecar_key(task_id):
        if _last_active_session_key.get(bare_task_id) == task_id:
            _last_active_session_key.pop(bare_task_id, None)
    else:
        _last_active_session_key.pop(bare_task_id, None)


def _cleanup_single_browser_session(task_id: str) -> None:
    """Internal: reap a single browser session by its exact session key."""
    # Stop the CDP supervisor for this task FIRST so we close our WebSocket
    # before the backend tears down the underlying CDP endpoint.
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
        bb_session_id = session_info.get("bb_session_id", "unknown")
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
                _run_browser_command(task_id, "close", [], timeout=10)
                logger.debug(
                    "agent-browser close command completed for task %s",
                    task_id,
                )
            except Exception as e:
                logger.warning("agent-browser close failed for task %s: %s", task_id, e)

        # Now remove from tracking under lock
        with _cleanup_lock:
            _active_sessions.pop(task_id, None)
            _session_last_activity.pop(task_id, None)

        # Cloud mode: close the cloud browser session via provider API.
        # Local sidecars have bb_session_id=None so this no-ops for them.
        if bb_session_id:
            provider = _get_cloud_provider()
            if provider is not None:
                try:
                    provider.close_session(bb_session_id)
                except Exception as e:
                    logger.warning("Could not close cloud browser session: %s", e)

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
    else:
        logger.debug("No active session found for task_id: %s", task_id)


def cleanup_all_browsers() -> None:
    """
    Clean up all active browser sessions.

    Useful for cleanup on shutdown.
    """
    with _cleanup_lock:
        task_ids = list(_active_sessions.keys())
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
    global _cached_command_timeout, _command_timeout_resolved
    global _cached_snapshot_threshold, _snapshot_threshold_resolved
    global _cached_chromium_installed
    global _cached_browser_engine, _browser_engine_resolved
    _cached_agent_browser = None
    _agent_browser_resolved = False
    _discover_homebrew_node_dirs.cache_clear()
    # Flip the resolved flag BEFORE nulling the cache so a concurrent
    # reader never sees ``resolved=True`` with ``cache=None`` (#14331).
    _command_timeout_resolved = False
    _cached_command_timeout = None
    _snapshot_threshold_resolved = False
    _cached_snapshot_threshold = None
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
        proc = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=600,
            env=_build_browser_env(),
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
from tools.browser_extension_router import (
    extension_controller_available,
    routed_browser_handler,
)

_BROWSER_SCHEMA_MAP = {s["name"]: s for s in BROWSER_TOOL_SCHEMAS}


def _browser_router_kw(kw: dict) -> dict:
    """Identity kwargs forwarded to the extension router wrapper."""
    return {
        "task_id": kw.get("task_id"),
        "session_id": kw.get("session_id"),
    }


def check_browser_routed_requirements(action: str = "browser_snapshot") -> bool:
    """Availability gate for tools that can use either browser backend."""
    return check_browser_requirements() or extension_controller_available(action)


def check_browser_navigate_requirements() -> bool:
    return check_browser_routed_requirements("browser_navigate")


def check_browser_snapshot_requirements() -> bool:
    return check_browser_routed_requirements("browser_snapshot")


def check_browser_click_requirements() -> bool:
    return check_browser_routed_requirements("browser_click")


def check_browser_type_requirements() -> bool:
    return check_browser_routed_requirements("browser_type")


def check_browser_scroll_requirements() -> bool:
    return check_browser_routed_requirements("browser_scroll")


def check_browser_back_requirements() -> bool:
    return check_browser_routed_requirements("browser_back")


def check_browser_press_requirements() -> bool:
    return check_browser_routed_requirements("browser_press")


registry.register(
    name="browser_navigate",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_navigate"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_navigate",
        args,
        fallback=lambda: browser_navigate(url=args.get("url", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_navigate_requirements,
    emoji="🌐",
)
registry.register(
    name="browser_snapshot",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_snapshot"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_snapshot",
        args,
        fallback=lambda: browser_snapshot(
            full=args.get("full", False), task_id=kw.get("task_id"), user_task=kw.get("user_task")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_snapshot_requirements,
    emoji="📸",
)
registry.register(
    name="browser_click",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_click"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_click",
        args,
        fallback=lambda: browser_click(ref=args.get("ref", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_click_requirements,
    emoji="👆",
)
registry.register(
    name="browser_type",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_type"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_type",
        args,
        fallback=lambda: browser_type(ref=args.get("ref", ""), text=args.get("text", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_type_requirements,
    emoji="⌨️",
)
registry.register(
    name="browser_scroll",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_scroll"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_scroll",
        args,
        fallback=lambda: browser_scroll(direction=args.get("direction", "down"), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_scroll_requirements,
    emoji="📜",
)
registry.register(
    name="browser_back",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_back"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_back",
        args,
        fallback=lambda: browser_back(task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_back_requirements,
    emoji="◀️",
)
registry.register(
    name="browser_press",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_press"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_press",
        args,
        fallback=lambda: browser_press(key=args.get("key", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_press_requirements,
    emoji="⌨️",
)

registry.register(
    name="browser_get_images",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_get_images"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_get_images",
        args,
        fallback=lambda: browser_get_images(task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_requirements,
    emoji="🖼️",
)
registry.register(
    name="browser_vision",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_vision"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_vision",
        args,
        fallback=lambda: browser_vision(question=args.get("question", ""), annotate=args.get("annotate", False), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_vision_requirements,
    emoji="👁️",
)
registry.register(
    name="browser_console",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_console"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_console",
        args,
        fallback=lambda: browser_console(clear=args.get("clear", False), expression=args.get("expression"), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_requirements,
    emoji="🖥️",
)
