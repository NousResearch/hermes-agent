"""``browser.*`` config knobs and cloud browser provider resolution (explicit ``browser.cloud_provider``,
auto-detect, per-profile cache).

Every config read is profile-scoped: under a routed profile (multiplexed gateway, HERMES_HOME override)
the cached slots are bypassed or keyed by ``hermes_home_key()``, so one profile's engine / headed /
private-URL policy never reaches another.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from agent.browser_provider import BrowserProvider
from hermes_constants import get_hermes_home_override, hermes_home_key
from tools.tool_backend_helpers import normalize_browser_cloud_provider
from utils import is_truthy_value

logger = logging.getLogger(__name__)

DEFAULT_COMMAND_TIMEOUT = 30
_VALID_ENGINES = ("auto", "lightpanda")

# Per-profile memo for config-derived values: {name: {home_key: value}}. ``reset_caches`` clears it.
_memo_slots: dict[str, dict[str, Any]] = {}
_memo_lock = threading.Lock()

# Cloud provider cache keyed by (home_key, registry generation): a plugin (re)registering a provider
# bumps the generation and the next lookup resolves again.
_cached_providers: dict[tuple[str, Any], Optional[BrowserProvider]] = {}
_provider_lock = threading.RLock()


def browser_section() -> dict:
    """The active profile's ``browser:`` section as every runtime reader sees it: the user file, the managed
    overlay winning, ``${VAR}`` expanded, no defaults (``load_user_config_effective``); ``{}`` when absent."""
    from hermes_cli.config_effective import load_user_config_effective
    section = load_user_config_effective().get("browser")
    return section if isinstance(section, dict) else {}


def browser_cfg(key: str, default, parse: Callable[[Any], Any] = lambda v: v):
    """``parse(browser.<key>)`` from :func:`browser_section`; ``default`` when absent or unparseable."""
    try:
        section = browser_section()
        if key in section:
            return parse(section[key])
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Could not read browser.%s: %s", key, exc)
    return default


def _memo(name: str, compute: Callable[[], Any]):
    """Process cache, one slot per profile home; a routed profile always recomputes."""
    if get_hermes_home_override() is not None:
        return compute()
    home = hermes_home_key()
    with _memo_lock:
        slot = _memo_slots.setdefault(name, {})
        if home in slot:
            return slot[home]
    value = compute()
    with _memo_lock:
        _memo_slots.setdefault(name, {})[home] = value
    return value


def reset_caches(home_key: Optional[str] = None) -> None:
    """Drop cached config reads (all homes, or only ``home_key``'s) and the provider cache."""
    with _memo_lock:
        if home_key is None:
            _memo_slots.clear()
        else:
            for slot in _memo_slots.values():
                slot.pop(home_key, None)
    with _provider_lock:
        for key in [k for k in _cached_providers if home_key in (None, k[0])]:
            _cached_providers.pop(key, None)


def _truthy(value) -> bool:
    return is_truthy_value(value, default=False)


def command_timeout() -> int:
    """``browser.command_timeout`` (floored at 5 s): the floor of the Chromium launch timeout."""
    return _memo("command_timeout", lambda: browser_cfg(
        "command_timeout", DEFAULT_COMMAND_TIMEOUT,
        lambda v: DEFAULT_COMMAND_TIMEOUT if v is None else max(int(v), 5)))


def inactivity_timeout(default: int) -> int:
    """``browser.inactivity_timeout`` (30 s floor: no instant reaping)."""
    return browser_cfg("inactivity_timeout", default, lambda v: default if v is None else max(int(v), 30))


def chromium_args() -> list[str]:
    """``browser.chromium_args``: extra flags appended to every packaged/shared Chromium launch."""
    def parse(value) -> list[str]:
        if isinstance(value, str):
            value = [part for part in value.replace("\n", ",").split(",")]
        return [str(v).strip() for v in (value or []) if str(v).strip()]
    return browser_cfg("chromium_args", [], parse)


def executable_path() -> str:
    """``browser.executable_path`` ("" = the packaged Chromium)."""
    return browser_cfg("executable_path", "", lambda v: str(v or "").strip())


def user_data_dir() -> str:
    """``browser.user_data_dir``: a persistent profile every session shares ("" = fresh per session).
    ``~`` expands; a relative path is anchored at this profile's HERMES_HOME."""
    import os
    raw = browser_cfg("user_data_dir", "", lambda v: str(v or "").strip())
    if not raw:
        return ""
    raw = os.path.expanduser(raw)
    if os.path.isabs(raw):
        return raw
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / raw)


def get_browser_engine() -> str:
    """``auto`` (packaged Chromium) or ``lightpanda``; ``chrome`` is accepted as ``auto``."""
    def compute() -> str:
        engine = browser_cfg("engine", "auto", lambda v: str(v).strip().lower() if v and str(v).strip() else "auto")
        if engine == "chrome":
            return "auto"
        if engine not in _VALID_ENGINES:
            logger.warning("Unknown browser engine %r (valid: %s), using 'auto'", engine, ", ".join(_VALID_ENGINES))
            return "auto"
        return engine
    return _memo("engine", compute)


def is_headed_mode() -> bool:
    """``browser.headed``: launch a visible window (Linux: only when a display exists)."""
    return _memo("headed", lambda: browser_cfg("headed", False, _truthy))


def auto_local_for_private_urls() -> bool:
    """``browser.auto_local_for_private_urls`` (default True): route private/LAN URLs to a local sidecar
    browser even with a cloud provider configured."""
    return _memo("auto_local", lambda: browser_cfg("auto_local_for_private_urls", True, bool))


def use_real_profile() -> bool:
    """Real-profile consent. Read on EVERY call: it is a consent switch (flipping it off must not need a
    restart) and each multiplexed profile decides for itself."""
    return browser_cfg("use_real_profile", False, bool)


def allow_private_urls() -> bool:
    """Whether the browser may navigate to private/internal addresses (default False: SSRF guard on)."""
    return _memo("allow_private_urls", lambda: browser_cfg("allow_private_urls", False, _truthy))


# ── Cloud provider ───────────────────────────────────────────────────────────────────────────────────

def _registry_generation(scope: str):
    try:
        from agent.browser_registry import registry_generation
    except ImportError:  # isolated compat stubs ship only get_provider
        return (0, 0)
    return registry_generation(scope=scope)


def _ensure_browser_plugins_loaded() -> None:
    """Idempotent plugin discovery (standalone scripts/tests may never import ``model_tools``)."""
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered
        _ensure_plugins_discovered()
    except Exception as exc:  # health: allow BLE001 -- plugin discovery failure must never break browsing
        logger.debug("Browser plugin discovery failed (non-fatal): %s", exc)


def _instantiate_explicit_provider(provider_key: str) -> Optional[BrowserProvider]:
    """The provider named by ``browser.cloud_provider``. An unregistered name raises ``ValueError`` (never a
    silent reroute to auto-detect); any other instantiation error yields None so the next call retries."""
    from agent.browser_registry import get_provider
    try:
        _ensure_browser_plugins_loaded()
        resolved = get_provider(provider_key)
    except Exception:  # health: allow BLE001 -- a broken provider plugin must not take the browser down
        logger.warning("Failed to instantiate cloud_provider %r; will retry on next call", provider_key, exc_info=True)
        return None
    if resolved is None:
        from tools.tool_backend_helpers import selection_error
        raise ValueError(selection_error(
            "browser", f"'{provider_key}'",
            "no registered browser plugin has that name (install the corresponding plugin or fix the config key spelling)",
        ))
    return resolved


def _autodetect_provider() -> Optional[BrowserProvider]:
    """Browser Use, then Browserbase, from credentials; never raises. Third-party plugins are only reachable
    through an explicit ``browser.cloud_provider``."""
    # Late import: a core tool must not execute bundled plugin code before a session is resolved.
    from plugins.browser.browser_use.provider import BrowserUseBrowserProvider
    from plugins.browser.browserbase.provider import BrowserbaseBrowserProvider
    try:
        for cls in (BrowserUseBrowserProvider, BrowserbaseBrowserProvider):
            provider = cls()
            if provider.is_available():
                return provider
    except Exception:  # health: allow BLE001 -- auto-detect must never poison the cache
        logger.debug("Cloud provider auto-detect failed", exc_info=True)
    return None


def _resolve_provider_uncached() -> tuple[Optional[BrowserProvider], bool]:
    """``(provider, definitive)``. Definitive answers (explicit ``local``/``camofox``, a resolved provider) are
    cached; a transient None (unreadable config, missing credentials) is not, so it self-heals."""
    provider_key = None
    try:
        section = browser_section()
        if "cloud_provider" in section:
            provider_key = normalize_browser_cloud_provider(section.get("cloud_provider"))
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Could not read cloud_provider from config: %s", exc)
    if provider_key in ("local", "camofox"):
        return None, True  # camofox is a plugin with its own tool surface, not a CDP provider
    if provider_key == "nous":
        provider_key = "browser-use"  # the managed Nous subscription is served by the Browser Use provider
    if provider_key:
        resolved = _instantiate_explicit_provider(provider_key)
        return resolved, resolved is not None
    if provider_key is not None:  # an empty selection was written: no auto-detect
        return None, False
    resolved = _autodetect_provider()
    return resolved, resolved is not None


def get_cloud_provider() -> Optional[BrowserProvider]:
    """The cloud provider for the active profile, or None for local browsing."""
    scope = hermes_home_key()
    with _provider_lock:
        while True:
            before = _registry_generation(scope)
            if (scope, before) in _cached_providers:
                return _cached_providers[(scope, before)]
            resolved, definitive = _resolve_provider_uncached()
            if _registry_generation(scope) != before:  # a reload landed mid-resolution: resolve again
                continue
            if definitive:
                for stale in [k for k in _cached_providers if k[0] == scope]:
                    _cached_providers.pop(stale, None)
                _cached_providers[(scope, before)] = resolved
            return resolved


def is_local_backend() -> bool:
    """True when the browser runs locally AND the terminal is local too.

    SSRF protection matters only when the browser reaches networks the terminal cannot (cloud backends,
    containerized terminals). A CDP override is never trusted as local (that Chrome may live off-host).
    """
    from tools.browser_session_cdp import get_cdp_override_raw
    if get_cdp_override_raw() or get_cloud_provider() is not None:
        return False
    # Scope-aware: under multiplexing the routed profile's terminal backend lives in the per-turn scope (#68559).
    from tools.terminal_scope import terminal_env
    return terminal_env("TERMINAL_ENV", "local").strip().lower() in ("local", "")


def browser_backend_name(local_engine: Optional[str] = None) -> str:
    """Shared-metrics label for the active profile's backend in session-creation precedence
    (CDP override > cloud provider > local engine). ``local_engine``: the call ran on a LOCAL browser (the
    hybrid ``::local`` sidecar, a cloud failure's local fallback, the real profile) of that engine (``local``
    or ``lightpanda``, as its session was created), so neither a configured provider nor ``browser.engine``
    names it. Config reads only, no network I/O."""
    from tools.browser_session_cdp import get_cdp_override_raw
    if get_cdp_override_raw():
        return "cdp"
    if local_engine:
        return local_engine
    provider = get_cloud_provider()
    if provider is not None:
        return str(getattr(provider, "name", "") or "other")
    return "lightpanda" if get_browser_engine() == "lightpanda" else "local"
