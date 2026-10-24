"""``browser_exec``: drive a real browser through the Browser Use CLI engine (browser-harness,
https://browser-use.com), Hermes' only core browser tool.

The harness speaks CDP to whatever ``tools.browser_session`` resolved for the call: an operator
``BU_CDP_*`` endpoint, a ``/browser connect`` override, a cloud provider's session, or a local browser
(real profile, Lightpanda, or the packaged Chromium started by ``tools.browser_launcher``).
"""

import contextlib
import contextvars
import functools
import importlib
import importlib.util
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from hermes_constants import get_hermes_home, hermes_home_key
from tools.browser_harness_dirs import daemon_env, daemon_identity, namespace_harness_dirs
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_BROWSER_USE_PROVIDER = "browser-use"
# The one credential the harness itself reads (BU_AUTOSPAWN of a Browser Use cloud browser).
_HARNESS_PASSTHROUGH_KEYS: tuple[str, ...] = ("BROWSER_USE_API_KEY",)

# Cloud daemon names become the BU_NAME env var
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Set on the env dict by the CDP resolvers when the resolved browser is EXCLUSIVE to this named session
# (per-name provider / named BU cloud / Lightpanda). Popped before the subprocess launches — never exported.
_PRIVATE_BROWSER_SENTINEL = "_HERMES_BU_PRIVATE_BROWSER"
# Internal route provenance: this exec resolved to a browser on the Bot Desktop display and must use
# the same human-control lease fence as the built-in browser tools. Popped before launching the CLI.
_BOT_DESKTOP_BROWSER_SENTINEL = "_HERMES_BU_BOT_DESKTOP_BROWSER"
# The note when the consented real profile could not load (the route fell back). Popped before launch.
_REAL_PROFILE_FALLBACK_SENTINEL = "_HERMES_BU_REAL_PROFILE_FALLBACK"
# local=True whose real profile could not load: run on the local sidecar browser. Popped before launch.
_LOCAL_FALLBACK_SENTINEL = "_HERMES_BU_LOCAL_FALLBACK"

# Prepended to the model's code for named sessions on SHARED browsers (a /browser connect CDP override): the
# harness daemon attaches to the first existing page at startup, so two fresh named daemons can land on the
# SAME tab. Steering each onto a tab it created prevents clobbering. Runs once per daemon (marker keyed by
# BU_NAME + daemon pid).
_OWN_TAB_PREAMBLE = """\
# hermes: pin this named session to its own tab (once per daemon process)
def _hermes_ensure_own_tab():
    import os as _os, tempfile as _tf
    _name = _os.environ.get("BU_NAME", "default")
    try:
        # Key the marker by the daemon's pid so a daemon restart (which
        # re-attaches to the first shared page) re-pins automatically,
        # while agent-driven tab switches mid-session are left alone.
        from browser_harness import _ipc as _bipc
        _dpid = _bipc.pid_path(_name).read_text().strip() or "0"
    except Exception:
        _dpid = "0"
    _uid = _os.getuid() if hasattr(_os, "getuid") else 0
    _marker = _os.path.join(
        _tf.gettempdir(), "hermes-bu-owntab-%s-%s-%s" % (_uid, _name, _dpid)
    )
    if _os.path.exists(_marker):
        return
    try:
        # Force a fresh target: new_tab() would REUSE a blank current tab,
        # which is exactly the tab a sibling daemon may also hold.
        _tid = cdp("Target.createTarget", url="about:blank").get("targetId")
        if _tid:
            switch_tab(_tid)
    except Exception:
        pass  # best-effort: worst case is pre-fix behavior
    try:
        open(_marker, "w").close()
    except OSError:
        pass
_hermes_ensure_own_tab()
del _hermes_ensure_own_tab
"""

_DEFAULT_TIMEOUT_S = 300
_MIN_TIMEOUT_S = 5
_MAX_TIMEOUT_S = 1800
_STDERR_CAP_CHARS = 4000

_TASK_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")  # filesystem-safe task ids
# Screenshot paths printed by capture_screenshot(): POSIX or Windows drive-letter absolute.
_IMAGE_PATH_RE = re.compile(r"((?:[A-Za-z]:[\\/]|/)[^\s\"']+?\.(?:png|jpe?g|webp))", re.IGNORECASE)
# http(s) URL literals in exec code, checked against the browser URL policy
_URL_RE = re.compile(r"https?://[^\s'\"\\)]+", re.IGNORECASE)
_FHS_BIN_DIRS = ("/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/sbin", "/usr/local/bin", "/usr/sbin",
                 "/usr/bin", "/sbin", "/bin")


def _quiet(fn: Callable[[], Any], default: Any, log_prefix: str = "") -> Any:
    """``fn()``, or ``default`` on any exception (debug-logged when ``log_prefix`` is set)."""
    try:
        return fn()
    except Exception as e:
        if log_prefix:
            logger.debug("%s: %s", log_prefix, e)
    return default


def _lazy_call(module: str, name: str, default: Any, log_prefix: str) -> Any:
    """Call ``module.name()`` resolved at call time (tests stub the module / patch the
    attribute); on any failure log ``log_prefix`` and return ``default``."""
    return _quiet(lambda: getattr(importlib.import_module(module), name)(), default, log_prefix)


def _real_profile_consented() -> bool:
    return _lazy_call("tools.browser_session_cloud", "use_real_profile", False, "real-profile consent lookup failed")


def _set_cdp_env(env: dict, cdp: str) -> None:
    """Export a CDP endpoint under the BU_CDP_* contract (http(s) → URL, else WS)."""
    env["BU_CDP_URL" if cdp.startswith(("http://", "https://")) else "BU_CDP_WS"] = cdp


def _has_cdp_env(env: dict) -> bool:
    return bool(env.get("BU_CDP_WS") or env.get("BU_CDP_URL"))


def _export_session_cdp(env: dict, get_session_info: Callable[[str], Any], cache_key: str,
                        fail_msg: Callable[[Exception], str], no_cdp_msg: str) -> Optional[str]:
    """Export the CDP endpoint from ``get_session_info(cache_key)``; error string on failure / no CDP."""
    try:
        cdp = str((get_session_info(cache_key) or {}).get("cdp_url") or "")
    except Exception as e:
        return fail_msg(e)
    if not cdp:
        return no_cdp_msg
    _set_cdp_env(env, cdp)
    return None


class _CodeUrlVerdict(NamedTuple):
    """What one pass over a code blob's URL literals decided (see ``_code_url_verdict``)."""

    blocked: Optional[str]
    local_sidecar: bool


def _code_url_verdict(code: str) -> _CodeUrlVerdict:
    """One pass over the code's URL literals: the blocking error, if any, and whether the call must run on
    the hybrid LOCAL sidecar. Each URL's routing predicate is evaluated ONCE and feeds both answers: it
    resolves DNS unmemoised, so a rebinding host evaluated twice could relax the private-address floor and
    still route to the cloud provider."""
    from tools.browser_url_policy import evaluate_url_safety, hybrid_routes_locally
    local_sidecar = False
    for url in _URL_RE.findall(code or ""):
        routes_local = hybrid_routes_locally(url)
        local_sidecar = local_sidecar or routes_local
        err = evaluate_url_safety(url, auto_local=routes_local)
        if err:
            return _CodeUrlVerdict(err.get("error", "Blocked: unsafe URL"), local_sidecar)
    return _CodeUrlVerdict(None, local_sidecar)


def _harness_credentials() -> dict:
    """The harness's own credential from the active profile's secret scope, never ``os.environ``: under
    multiplex that holds the LAUNCH profile's key, and a served profile's browser runs on its own (or none)."""
    from agent.secret_scope import current_secret_scope, get_secret, serves_routed_profile
    routed = serves_routed_profile()
    scope = (current_secret_scope() or {}) if routed else None
    values = {key: (scope.get(key) if routed else get_secret(key)) for key in _HARNESS_PASSTHROUGH_KEYS}
    return {key: value for key, value in values.items() if value is not None}


def _base_subprocess_env() -> dict:
    from tools.browser_session import build_browser_env
    env = build_browser_env()
    env.update(_harness_credentials())
    # The harness runs on Hermes's own interpreter, but a bundled Desktop install boots that
    # interpreter with its site dir on PYTHONPATH (no venv to activate), and the harness's daemon
    # re-runs sys.executable. Point PYTHONPATH at the harness's site dir, replacing whatever the
    # agent process inherited, so both the CLI and its daemon import the same packages.
    env.pop("PYTHONHOME", None)
    site_dir = _harness_site_dir()
    if site_dir:
        env["PYTHONPATH"] = site_dir
    else:
        env.pop("PYTHONPATH", None)
    env["PATH"] = _floor_subprocess_path(env.get("PATH", ""))
    env.setdefault("ANONYMIZED_TELEMETRY", "false")
    namespace_harness_dirs(env, _served_profile_tag())  # a served profile's own harness runtime/tmp dirs
    return env


def _floor_subprocess_path(path: str) -> str:
    """Keep system commands reachable from profile workers with a minimal PATH.

    Browser subprocesses may launch shell helpers; version-manager-only paths
    omit /usr/bin. Reuse the shared browser PATH floor on POSIX.
    """
    if os.name == "nt":
        return path
    parts = [p for p in (path or "").split(os.pathsep) if p]
    return os.pathsep.join(parts + [d for d in _FHS_BIN_DIRS if d not in set(parts) and os.path.isdir(d)])


def _read_browser_cfg() -> dict:
    """Return the ``browser:`` config section, or {} on any failure."""
    try:
        from tools.browser_session_cloud import browser_section
        return browser_section()
    except Exception as e:
        logger.debug("Could not read browser config section: %s", e)
        return {}


def _use_gateway(browser_cfg: dict) -> bool:
    """True when the browser section selects the Nous Tool Gateway — by the current ``hermes tools``
    picker row (``cloud_provider: nous``) or the pre-picker ``use_gateway: true`` flag. Reading only
    the legacy flag missed every picker-configured gateway, and the direct-API branch it fell into
    holds no credentials in managed mode (#108310)."""
    if is_truthy_value(browser_cfg.get("use_gateway"), default=False):
        return True
    try:
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
    except Exception:  # pragma: no cover — helper ships with the package
        return False
    return str(browser_cfg.get("cloud_provider") or "").strip().lower() == NOUS_MANAGED_PROVIDER


def is_legacy_browser_use_cloud_config(browser_cfg: dict) -> bool:
    """True for pre-CLI direct-API Browser Use cloud configs: no cloud_provider (or ``browser-use``), not the
    Nous gateway, and BROWSER_USE_API_KEY set; the CLI then reaches Browser Use cloud natively."""
    if not isinstance(browser_cfg, dict):
        return False
    provider = str(browser_cfg.get("cloud_provider") or "").strip().lower()
    if provider not in {"browser-use", ""} or _use_gateway(browser_cfg):
        return False
    return _browser_use_api_key_set()


def _browser_use_api_key_set() -> bool:
    """BROWSER_USE_API_KEY for the served profile: the only credential Hermes autospawns BU cloud with.
    Profile-scoped so a multiplexed secondary never inherits the default's cloud mode."""
    from agent.secret_scope import get_secret
    return bool(get_secret("BROWSER_USE_API_KEY", ""))


def _harness_site_dir() -> Optional[str]:
    """The site dir Hermes's interpreter imports ``browser_harness`` from, or None."""
    spec = importlib.util.find_spec("browser_harness")
    if spec is None or not spec.origin:
        return None
    return str(Path(spec.origin).resolve().parent.parent)


def _find_cli() -> Optional[list[str]]:
    """The Browser Use CLI's engine (browser-harness) is a core dependency of Hermes's own venv,
    so every install, the Desktop bundle included, runs it on the current interpreter."""
    if _harness_site_dir() is None:
        return None
    return [sys.executable, "-m", "browser_harness.run"]


def browser_exec_available() -> bool:
    """check_fn: browser-harness is importable (a core dependency). Never launches or opens sockets."""
    return _find_cli() is not None


def _workspace_dir(task_id: Optional[str]) -> Optional[str]:
    """Stable per-task scratch dir that persists across browser_exec calls"""
    if os.environ.get("BH_AGENT_WORKSPACE"):
        return os.environ["BH_AGENT_WORKSPACE"]
    try:
        safe = _TASK_ID_SAFE_RE.sub("_", str(task_id or "default"))[:80] or "default"
        path = Path(get_hermes_home()) / "cache" / "browser-use" / "workspace" / safe
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    except Exception as e:
        logger.debug("browser_exec workspace unavailable: %s", e)
        return None


def _find_screenshot(stdout: str, since: float) -> Optional[str]:
    """Last screenshot path printed during this exec that exists and was written after
    the exec started, or None."""
    for path in reversed(_IMAGE_PATH_RE.findall(stdout or "")):
        try:
            if os.path.isfile(path) and os.path.getmtime(path) >= since - 1:
                return path
        except OSError:
            continue
    return None


def _native_screenshot_result(result: dict[str, Any], path: str) -> Optional[dict[str, Any]]:
    """Build a multimodal tool result attaching path for vision models"""
    try:
        from tools.vision_tools import (_EMBED_MAX_DIMENSION,
                                        _resize_image_for_vision, _should_use_native_vision_fast_path)
        from tools.vision_tools_history_budget import resolve_embed_target_bytes
        if not _should_use_native_vision_fast_path():
            return None
        # History-reuse cap: this data URL bakes into the tool result and is re-sent every later turn —
        # same policy as the vision_analyze native embed.
        data_url = _resize_image_for_vision(Path(path), mime_type="image/png",
                                            max_base64_bytes=resolve_embed_target_bytes(),
                                            max_dimension=_EMBED_MAX_DIMENSION, force_jpeg=True)
        text = json.dumps(result, ensure_ascii=False)
        attached = text + "\n\nThe screenshot from this call is attached — inspect it with your native vision."
        return {"_multimodal": True, "text_summary": text, "meta": {"screenshot_path": path, "native_vision": True},
                "content": [{"type": "text", "text": attached}, {"type": "image_url", "image_url": {"url": data_url}}]}
    except Exception as e:
        logger.debug("Native screenshot attach failed (falling back to text): %s", e)
        return None


def _served_profile_tag() -> str:
    """``""`` outside a served-profile scope (every legacy key stays byte-identical); under a
    multiplexed turn, the routed profile's home key — one profile's browser must never be handed
    to another that happens to use the same session name or task id (#110032)."""
    from hermes_constants import get_hermes_home_override, hermes_home_key
    return "" if get_hermes_home_override() is None else hermes_home_key()


def _backend_cache_key(task_id: Optional[str], session_name: str = "", sidecar: bool = False) -> str:
    """Session-cache key for a backend browser: named sessions get their own; served profiles get their own.
    ``sidecar`` appends the hybrid ``::local`` suffix, which makes ``_get_session_info`` build local Chromium
    even when a cloud provider is configured."""
    key = f"bu-named-{session_name}" if session_name else (task_id or "browser-exec-default")
    tag = _served_profile_tag()
    key = f"{key}@{tag}" if tag else key
    return f"{key}::local" if sidecar else key


def _sidecar_daemon_name(name: str) -> str:
    """BU_NAME of the harness daemon serving the hybrid local sidecar. The daemon reads BU_CDP_* once at
    spawn and outlives the call, so sharing the cloud route's daemon would relay a private URL to the
    cloud browser it is already attached to. User session names cannot start with ``_``."""
    return f"_local_{name or 'default'}"[:64]


def _export_real_profile_note(env: dict, key: str) -> None:
    """The session layer decides real profile vs fresh browser once per key; carry its note (if it fell
    back) so the result reports the browser this call actually drives."""
    from tools.browser_session import real_profile_note
    note = real_profile_note(key)
    if note:
        env.setdefault(_REAL_PROFILE_FALLBACK_SENTINEL, note)


def _mark_local_browser(env: dict, session: dict, key: str) -> None:
    """Flags for a LOCAL session browser: it runs on the Bot Desktop display (lease fence), and unless a
    persistent profile / the real-profile browser is shared by every key, it is exclusive to this key (no
    own-tab preamble: nothing to share a tab with)."""
    features = session.get("features") or {}
    env[_BOT_DESKTOP_BROWSER_SENTINEL] = "lightpanda" if features.get("lightpanda") else "local"
    if not session.get("shared_profile") and not features.get("real_profile"):
        env[_PRIVATE_BROWSER_SENTINEL] = "1"
    _export_real_profile_note(env, key)


def _resolve_local_cdp(env: dict, task_id: Optional[str], session_name: str = "",
                       sidecar: bool = False) -> Optional[str]:
    """Local browser for this cache key through the session layer: the consented real profile, Lightpanda
    (``browser.engine: lightpanda``), or the packaged Chromium. Never the harness's own discovery of the
    user's installed Chrome (chrome://inspect toggle + Allow popup per run, blocked on Chrome >=136)."""
    from tools.browser_lightpanda import using_lightpanda_engine
    from tools.browser_session import get_session_info
    key = _backend_cache_key(task_id, session_name, sidecar)
    session: dict = {}

    def _session_info(cache_key: str) -> dict:
        session.update(get_session_info(cache_key) or {})
        return session

    if _quiet(using_lightpanda_engine, False):
        fail = lambda e: (f"Lightpanda could not be started: {e} Set browser.engine to auto "
                          "to use local Chrome, or switch backends via `hermes tools` → Browser Automation.")
    else:
        fail = lambda e: (f"The local browser could not be started: {e} "
                          "Run `hermes tools` → Browser Automation to (re)install Chromium, or switch backends.")
    err = _export_session_cdp(env, _session_info, key, fail, "The local browser session returned no CDP endpoint.")
    if err is None:
        _mark_local_browser(env, session, key)
    return err


def _resolve_backend_cdp(env: dict, task_id: Optional[str], session_name: str = "",
                         force_local_sidecar: bool = False) -> Optional[str]:
    """Point the harness at the configured backend's CDP endpoint; error string on failure.

    Precedence: (1) ``BU_CDP_WS``/``BU_CDP_URL`` already in env (operator override); (2) ``BROWSER_CDP_URL``
    env / ``browser.cdp_url`` (``/browser connect``); (3) a cloud provider via ``browser_session.get_session_info``
    (per-key cache, expiry, reaper, atexit; a provider failure falls back to a local browser);
    (4) the local engine — ``browser.engine: lightpanda`` or Hermes' packaged Chromium via the launcher
    (never the harness's own discovery of the user's installed Chrome); (5) BU direct-API configs WITH
    ``BROWSER_USE_API_KEY`` → None (checked before (3)): the CLI reaches BU cloud natively (BU_AUTOSPAWN); a
    keyless ``browser-use`` provider (managed-gateway autodetect, or a selection missing its key) takes (3).
    ``session_name`` (BU_NAME) keys the session cache so each name gets its OWN browser — what makes named
    sessions concurrent-safe.

    ``force_local_sidecar`` skips step (3): the call runs on the local engine under the ``::local`` cache
    key (hybrid routing; see ``browser_url_policy.hybrid_routes_locally``). It sits BELOW the two operator overrides, and the
    sidecar verdict never fires while a CDP override is set."""
    if _has_cdp_env(env):
        return None
    from tools.browser_session import get_session_info
    from tools.browser_session_cdp import get_cdp_override
    from tools.browser_session_cloud import get_cloud_provider
    override = _quiet(get_cdp_override, "")
    if override:
        _set_cdp_env(env, override)
        return None
    if force_local_sidecar:
        return _resolve_local_cdp(env, task_id, session_name, sidecar=True)
    provider = _quiet(get_cloud_provider, None, "Cloud provider lookup failed")
    if provider is None:
        return _resolve_local_cdp(env, task_id, session_name)

    # Browser Use direct-API configs: the CLI talks to BU cloud natively (BU_AUTOSPAWN) — the
    # legacy provider would create a second, redundant session. Nous-gateway configs (cloud_provider: nous
    # from the picker, or the pre-picker use_gateway: true) DO resolve through the provider: the gateway
    # provisions the browser server-side and returns its CDP URL. Without BROWSER_USE_API_KEY the CLI cannot
    # autospawn, and the provider is either the managed gateway (autodetect on a never-configured install) or
    # a selection missing its key; both go through the session path (gateway CDP, or the session layer's
    # local-Chromium fallback) instead of leaving the harness to hunt for an installed Chrome.
    provider_key = str(getattr(provider, "name", "") or "").strip().lower()
    if provider_key == _BROWSER_USE_PROVIDER and not _use_gateway(_read_browser_cfg()) and _browser_use_api_key_set():
        env[_PRIVATE_BROWSER_SENTINEL] = "1"  # named BU cloud browsers are exclusive to their daemon
        return None

    provider_name = type(provider).__name__
    session: dict = {}

    def _session_info(key: str) -> dict:
        session.update(get_session_info(key) or {})
        return session

    key = _backend_cache_key(task_id, session_name)
    err = _export_session_cdp(
        env, _session_info, key,
        lambda e: (f"Cloud browser provider {provider_name} failed to provide a session: {e}. "
                   "Fix the provider configuration or switch backends via `hermes tools` → Browser Automation."),
        f"Cloud browser provider {provider_name} returned no CDP endpoint, so browser_exec "
        "cannot drive it. Pick a CDP-capable provider via `hermes tools` → Browser Automation.",
    )
    if session.get("fallback_from_cloud"):
        # The provider failed (rate limit, quota, outage) and the session layer fell back to a LOCAL
        # browser for this key; it already carries its CDP URL.
        logger.warning("browser_exec: %s failed (%s); using local Chromium for this session",
                       provider_name, session.get("fallback_reason"))
        if err is None:
            _mark_local_browser(env, session, key)
        return err
    # A provider browser keyed bu-named-<name> is exclusive to this session — the
    # own-tab preamble would just leak a blank tab into it.
    if err is None and session_name:
        env[_PRIVATE_BROWSER_SENTINEL] = "1"
    return err


def _resolve_real_profile_cdp(env: dict, force_local: bool) -> Optional[str]:
    """Point the harness at the user's real-profile copy-browser (a SNAPSHOT of their default Chromium
    profile, hermes_cli.browser_connect) when consented. Two ways in: the effective backend is already local
    (no provider, CDP override, or legacy BU cloud config) → silent upgrade; or ``force_local`` (consent-gated
    ``local`` arg) → the user's browser even under a cloud backend. Operator overrides (BU_CDP_* env,
    /browser connect, ``browser.cdp_url``) own the session either way. A launch error is returned;
    ``_route_backend`` then falls back to the normal backend and reports it, never silently."""
    if not _real_profile_consented() or _has_cdp_env(env):
        return None
    from tools.browser_session_cdp import get_cdp_override_raw
    from tools.browser_session_cloud import get_cloud_provider
    from tools.browser_session_real_profile import real_profile_cdp
    if _quiet(get_cdp_override_raw, ""):
        return None
    # Only auto-upgrade genuinely-local attaches; any cloud path (provider, provider lookup failure, or
    # legacy BU cloud config) stays on its backend unless the model passes local=true.
    if not force_local and (_quiet(get_cloud_provider, object()) is not None
                            or is_legacy_browser_use_cloud_config(_read_browser_cfg())):
        return None
    cdp, err = real_profile_cdp()
    if cdp and not err:
        _set_cdp_env(env, cdp)
        env[_BOT_DESKTOP_BROWSER_SENTINEL] = "local"  # the user's real Chromium, whatever browser.engine says
    return err or None


def _attach_vault_supervisor(env: dict, task_id: Optional[str]) -> None:
    """Attach the per-task CDP supervisor to the browser this exec drives so ``browser_vault`` fill has
    a secret-capable WebSocket (never argv) into the SAME browser. Only CDP-routed backends expose an
    endpoint; BU direct-cloud (BU_AUTOSPAWN) does not, and the vault tools report ``supervisor_required``."""
    cdp = env.get("BU_CDP_WS") or env.get("BU_CDP_URL")
    if not cdp:
        return
    try:
        from tools.browser_session_cdp import get_dialog_policy_config, resolve_cdp_override
        from tools.browser_supervisor import SUPERVISOR_REGISTRY
        policy, timeout_s = get_dialog_policy_config()
        SUPERVISOR_REGISTRY.get_or_start(task_id=task_id or "default", cdp_url=resolve_cdp_override(cdp),
                                         dialog_policy=policy, dialog_timeout_s=timeout_s)
    except Exception as exc:
        logger.debug("browser_exec: CDP supervisor attach failed (non-fatal): %s", exc)


def _route_backend(env: dict, session: str, task_id: Optional[str], local: bool,
                   force_local_sidecar: bool = False, real_profile: bool = True) -> Optional[str]:
    """Resolve where the harness connects; returns an error string or None. Real-profile consent runs
    BEFORE provider resolution so a hit short-circuits the cloud path via the BU_CDP_* env contract. Named
    sessions compose with the backend: BU_NAME namespaces the harness daemon (IPC socket, log, pid) and on
    provider backends additionally keys its own cloud browser.

    ``force_local_sidecar`` (a call carrying a private URL) declines the cloud provider for the hybrid local
    sidecar; unlike ``local`` (the consent-gated real-profile browser) it never touches the user's profile."""
    # ``real_profile=False``: this route already fell back, so it stays on its fresh browser instead
    # of swapping browsers mid-task once the profile frees up.
    rp_err = _resolve_real_profile_cdp(env, force_local=local) if real_profile else None
    # local=True is only served by the real-profile route; consent off must not pretend.
    if local and not _has_cdp_env(env) and not _real_profile_consented():
        return ("local=true was requested but browser.use_real_profile is off. Enable it in config.yaml "
                "(browser.use_real_profile: true) or the desktop Settings → Browser section, then retry.")
    # local=True promised the user's own machine: a profile that can't load runs on a fresh LOCAL browser
    # (the sidecar, which never loads the profile), never the cloud one. Without ``local`` the session
    # layer makes the one real-profile-or-fresh decision and reports it (_resolve_local_cdp).
    if local and not _has_cdp_env(env) and (rp_err or not real_profile):
        if rp_err:
            from tools.browser_session_real_profile import real_profile_fallback_note
            env[_REAL_PROFILE_FALLBACK_SENTINEL] = real_profile_fallback_note(rp_err)
        env[_LOCAL_FALLBACK_SENTINEL] = "1"
        force_local_sidecar = True
    return _resolve_backend_cdp(env, task_id, session_name=session,
                                force_local_sidecar=force_local_sidecar)


def _group_popen_kwargs() -> dict:
    """Popen kwargs starting the CLI in its own process group (a new session on POSIX) so a
    timeout can take down every process that inherited the capture pipes, not just the CLI
    child. Windows also hides the console window."""
    def _flags() -> dict:
        from hermes_cli._subprocess_compat import windows_hide_flags
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"creationflags": windows_hide_flags() | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                "startupinfo": si}
    return _quiet(_flags, {}, "Windows hide-flags unavailable") if os.name == "nt" else {"start_new_session": True}


def _clamp_timeout(timeout_s: Any) -> int:
    try:
        return max(_MIN_TIMEOUT_S, min(int(timeout_s), _MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


# After a whole-group SIGKILL, every pipe holder is dead, so the drain below is normally
# instant; the deadline only guards against a process outside the group still holding a pipe.
_POST_KILL_DRAIN_S = 10.0


def _kill_cli_process_group(proc) -> None:
    """SIGKILL the CLI's whole process group (POSIX; ``start_new_session`` made pgid == pid) or,
    on Windows, its process tree via ``taskkill /T /F`` — the only group-wide kill it offers."""
    if os.name == "nt":
        from hermes_cli._subprocess_compat import windows_hide_flags
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
                           check=False, creationflags=windows_hide_flags())
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX only, the nt branch returned above


def _run_cli_killing_process_group(cmd, code, env, timeout):
    """Run the CLI in its own process group and kill the whole group on timeout.

    ``subprocess.run`` only kills the direct child on ``TimeoutExpired``; a grandchild that
    inherited the stdout/stderr pipes (browser_harness daemon / Chrome helper) is orphaned
    still holding them, and on Windows ``run()``'s unbounded post-kill ``communicate()`` then
    blocks on pipe EOF forever — so the tool call, plus its activity heartbeat, wedges (#106244).
    """
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", env=env, **_group_popen_kwargs(),
    )
    try:
        stdout, stderr = proc.communicate(input=code, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_cli_process_group(proc)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.communicate(timeout=_POST_KILL_DRAIN_S)
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


# Harness daemons this process drove: each latches BU_CDP_* at start and outlives every call, so a backend
# swap (/browser connect|disconnect) leaves later calls in the old browser until these are stopped.
_driven_daemons: dict[tuple[str, ...], str] = {}  # daemon identity -> home
_driven_daemons_lock = threading.Lock()
_live_generation = functools.partial(_lazy_call, "tools.browser_session_cdp", "live_cdp_generation", 0,
                                     "live CDP generation lookup failed")  # this profile's publish count
# Route cache key -> whether its last browser_exec ran on the hybrid local sidecar.
_exec_route_is_sidecar: dict[str, bool] = {}
# Route cache key -> the note from its real-profile fallback; that route stays on the fallback browser.
_real_profile_fallback: dict[str, str] = {}
_real_profile_note_delivered: set = set()


def stop_harness_daemons(home_key: Optional[str] = None) -> None:
    """Stop every harness daemon this process drove (only ``home_key``'s when given), through the
    harness's own identity-checked ``--reload``; the next browser_exec respawns one on the endpoint
    it resolves then."""
    with _driven_daemons_lock:
        daemons = sorted(key for key, home in _driven_daemons.items() if home_key in (None, home))
        for key in daemons:
            _driven_daemons.pop(key, None)
    _reload_daemons(daemons)


def _reload_daemons(daemons) -> None:  # each ``daemon_identity`` daemon, with its own dirs
    cmd = _find_cli() if daemons else None
    if not cmd:
        return
    env = _base_subprocess_env()
    for daemon in daemons:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            _run_cli_killing_process_group([*cmd, "--reload"], "", daemon_env(env, daemon), 15)


def _route_exec(env: dict, session: str, task_id: Optional[str], local: bool, local_sidecar: bool,
                route_key: str) -> tuple[Optional[str], bool]:
    """``_route_backend`` for one browser_exec call; returns ``(error, runs on the local sidecar)``.
    A real-profile fallback is remembered per route: the route keeps its fresh browser and the note
    waits until a result carries it (``_take_real_profile_note``)."""
    route_err = _route_backend(env, session, task_id, local, force_local_sidecar=local_sidecar,
                               real_profile=route_key not in _real_profile_fallback)
    note = env.pop(_REAL_PROFILE_FALLBACK_SENTINEL, None)
    if note:
        _real_profile_fallback.setdefault(route_key, note)
    return route_err, local_sidecar or bool(env.pop(_LOCAL_FALLBACK_SENTINEL, None))


def _route_exec_current(env: dict, session: str, task_id: Optional[str], local: bool,
                        local_sidecar: bool, route_key: str) -> tuple[dict, Optional[str], bool, int]:
    """``_route_exec`` on a copy of ``env`` -> ``(routed env, error, sidecar, generation)``, re-routed (at
    most 3 tries) when /browser connect published meanwhile; ``_reload_if_republished`` covers dispatch."""
    for _attempt in range(3):
        routed = dict(env)
        generation = _live_generation()
        route_err, sidecar = _route_exec(routed, session, task_id, local, local_sidecar, route_key)
        if route_err or _live_generation() == generation:
            break
    return routed, route_err, sidecar, generation


def _reload_if_republished(env: dict, generation: int) -> None:
    """Stop this call's daemon if the profile published a new endpoint while it spawned on the old one."""
    if _live_generation() == generation:
        return
    daemon = daemon_identity(env)
    with _driven_daemons_lock:
        _driven_daemons.pop(daemon, None)
    _reload_daemons([daemon])


def _take_real_profile_note(route_key: str) -> dict:
    """The fallback note for the first result this route returns after falling back, then nothing."""
    if route_key not in _real_profile_fallback or route_key in _real_profile_note_delivered:
        return {}
    _real_profile_note_delivered.add(route_key)
    return {"real_profile_unavailable": _real_profile_fallback[route_key]}


def browser_exec(code: str, session: str = "", timeout_s: int = _DEFAULT_TIMEOUT_S,
                 task_id: Optional[str] = None, local: bool = False):
    """Run Python code through the browser-use CLI, and return its output"""
    from agent.redact import redact_sensitive_text
    from tools.registry import tool_error, tool_result
    if not code or not code.strip():
        return tool_error("No code provided. Pass Python that uses the pre-imported helpers, e.g. new_tab(\"https://example.com\") then print(page_info()).")

    # The verdict is carried into the route, never re-derived: the predicate resolves DNS, so a second
    # evaluation could relax the floor and still pick the cloud for a rebinding host.
    verdict = _code_url_verdict(code)
    if verdict.blocked:
        return tool_error(verdict.blocked)
    # One endpoint per call: any private URL sends the whole call to the local sidecar. A call with no URL
    # literal stays on the route the previous call took.
    route_key = _backend_cache_key(task_id, session)
    local_sidecar = verdict.local_sidecar if _URL_RE.search(code) else _exec_route_is_sidecar.get(route_key, False)

    cmd = _find_cli()
    if not cmd:
        return tool_error("browser-harness is missing from Hermes's Python environment. "
                          "Run `hermes update` to re-sync it.")

    env = _base_subprocess_env()
    if session:
        if not _SESSION_RE.match(session):
            return tool_error(f"Invalid session name {session!r}: use 1-64 letters, digits, "
                              "dashes, or underscores (e.g. 'r7k2').")
        env["BU_NAME"] = session
    env, route_err, local_sidecar, generation = _route_exec_current(
        env, session, task_id, bool(local), local_sidecar, route_key)
    if route_err:
        return tool_error(route_err)
    if local_sidecar:
        env["BU_NAME"] = _sidecar_daemon_name(session)
    _exec_route_is_sidecar[route_key] = local_sidecar
    local_engine = env.pop(_BOT_DESKTOP_BROWSER_SENTINEL, None)
    bot_desktop_browser = bool(local_engine)
    _exec_local_engine.set(local_engine)  # only local browsers carry the sentinel

    # SHARED browser (/browser connect CDP override): pin each named session to its own tab (see
    # _OWN_TAB_PREAMBLE). Private per-name browsers skip this — nothing to collide with.
    private_browser = env.pop(_PRIVATE_BROWSER_SENTINEL, None)  # always pop: never exported to the CLI
    if session and not private_browser:
        code = _OWN_TAB_PREAMBLE + code

    workspace = _workspace_dir(task_id)
    if workspace:
        env["BH_AGENT_WORKSPACE"] = workspace

    # BU_AUTOSPAWN makes the CLI start a Browser Use cloud browser when no local
    # Chrome/CDP endpoint is reachable (their API key authenticates it)
    if "BU_AUTOSPAWN" not in env and is_legacy_browser_use_cloud_config(_read_browser_cfg()):
        env["BU_AUTOSPAWN"] = "1"

    timeout = _clamp_timeout(timeout_s)
    started = time.time()

    def dispatch() -> dict[str, Any]:
        _attach_vault_supervisor(env, task_id)
        with _driven_daemons_lock:
            daemon = daemon_identity(env)
            _driven_daemons[daemon] = hermes_home_key()
        try:
            return {"proc": _run_cli_killing_process_group(cmd, code, env, timeout)}
        except subprocess.TimeoutExpired:
            return {"error_result": tool_error(
                f"browser-use exec timed out after {timeout}s. The daemon may still be working; retry "
                f"with a larger timeout_s (max {_MAX_TIMEOUT_S}), or split the work into several calls that "
                "append to workspace files — anything already written to the workspace is preserved."
            )}
        except OSError as e:
            return {"error_result": tool_error(f"Failed to launch browser-use CLI: {e}")}

    if bot_desktop_browser:
        from tools.browser_session import run_fenced
        dispatched = run_fenced({"features": {"local": True}}, dispatch)
    else:
        dispatched = dispatch()
    _reload_if_republished(env, generation)
    if "proc" not in dispatched:
        if "error_result" in dispatched:
            return dispatched["error_result"]
        return tool_result(dispatched)
    proc = dispatched["proc"]

    # browser_vault fill registers injected values with this forced model-egress
    # boundary. Preserve raw stdout only for screenshot-path detection below.
    result = {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output": redact_sensitive_text(proc.stdout, force=True),
    }
    if workspace:
        result["workspace"] = workspace
    if session:
        result["session"] = session
    result.update(_take_real_profile_note(route_key))
    stderr = redact_sensitive_text((proc.stderr or "").strip(), force=True)
    if len(stderr) > _STDERR_CAP_CHARS:
        stderr = stderr[:_STDERR_CAP_CHARS] + "\n… (stderr truncated)"
    if stderr:
        result["stderr"] = stderr
    screenshot = _find_screenshot(proc.stdout, started)
    if screenshot:
        result["screenshot_path"] = screenshot
        native = _native_screenshot_result(result, screenshot)
        if native is not None:
            return native
    return tool_result(result)


_HEADER_BASE = (
    "Drive a real browser for JS pages, logins, forms and clicks (Browser Use CLI). `code` runs as full "
    "Python (stdlib available) with pre-imported browser helpers; stdout comes back in the result. Start `code` with a one-line "
    "comment describing the step for the user in plain language, max 60 chars "
    "(e.g. `# Searching Amazon for paper towels`) — the UI shows it as the step label.\n\n"
    "STATE: the browser session and workspace persist across calls; Python variables do NOT (fresh "
    "interpreter each call). The workspace dir is $BH_AGENT_WORKSPACE (also `workspace` in every result); "
    "functions defined in agent_helpers.py there are auto-imported into every call. For multi-item tasks "
    "('all N products / every entry'), append each batch to a JSON/CSV file in the workspace, then read it "
    "back and aggregate in code — dedupe/count/sort with Python, not in your head — and verify the "
    "collected count against what was asked before answering.\n\n"
    "Batch each sub-procedure (navigate, wait, extract, act) into one call — do not spend a call per "
    "action — but for long extractions prefer several medium calls that append to workspace files over "
    "one giant call, so progress survives timeouts."
)

_HEADER_VISION = (
    " Screenshots are attached to your context automatically: when the exec output contains a "
    "capture_screenshot() path, the image arrives with this tool's result and you inspect it directly "
    "with your own vision — never send browser screenshots to a separate vision tool."
)

_HEADER_TEXT_ONLY = (
    " Your model cannot view images, so work text-first: page_info() for state, js() for "
    "reading/extracting DOM text, fill_input(selector, text) for inputs, and "
    "js(\"document.querySelector('…').click()\") for clicks — skip the screenshot-driven workflow described below."
)

# Appended when the local engine is Lightpanda: no graphical renderer, and one CDP
# connection holds one page — a second Target.createTarget fails with
# TargetAlreadyLoaded (drop the new_tab() sentence once lightpanda-io/browser#1962 lands).
_HEADER_LIGHTPANDA = (
    " The local engine is Lightpanda (no graphical renderer, one page per session): capture_screenshot() "
    "is unavailable, so work text-first; navigate with new_tab(url) exactly once, then goto_url(url) for "
    "every later navigation — a second new_tab() fails with TargetAlreadyLoaded."
)

# Pinned quick-reference for the CLI's pre-imported helpers, replacing the live
# ``browser-use skill`` fetch (uncontrolled third-party text in every schema: version
# drift, supply-chain exposure, byte-unstable prompt). A/B benchmarked ~equal.
_HELPERS_DIGEST = (
    "\n\nHELPERS (pre-imported): new_tab(url) opens/navigates (use for the FIRST navigation), goto_url(url) "
    "navigates the current tab, wait_for_load() after navigation, page_info() summarizes the current page "
    "state, js(expr) evaluates a JS expression and returns its value (js('document.title'); wrap function "
    "bodies as js('(() => {...})()') — a bare '() => {...}' returns the function itself, uncalled), "
    "fill_input(selector, text) types into inputs, click_at_xy(x, y) clicks viewport coordinates, "
    "capture_screenshot() saves and prints a screenshot path, cdp('Domain.method', **kwargs) is raw CDP — "
    "cdp('Accessibility.getFullAXTree')['nodes'] lists every element's role/name/backendDOMNodeId (filter "
    "in Python before printing; it is thousands of nodes), then cdp('DOM.getBoxModel', backendNodeId=n) "
    "gives click coordinates. ensure_real_tab() recovers from a stale/internal tab. Login walls: never guess "
    "credentials; see the vault note below if present, otherwise stop and ask the user."
)


def _description_header() -> str:
    """Header tailored to whether the active model can see images natively"""
    if _lazy_call("tools.browser_lightpanda", "lightpanda_engine_status", (False, ""),
                  "lightpanda engine status unavailable")[0]:  # no screenshots, whatever the model sees
        return _HEADER_BASE + _HEADER_TEXT_ONLY + _HEADER_LIGHTPANDA
    vision = _lazy_call("tools.vision_tools", "_should_use_native_vision_fast_path", False, "")
    return _HEADER_BASE + (_HEADER_VISION if vision else _HEADER_TEXT_ONLY)


def _dynamic_schema_overrides() -> dict:
    overrides: dict = {"description": _description_header() + _HELPERS_DIGEST}
    # ``local`` exists ONLY when the user consented to real-profile browsing — everyone
    # else's schema carries zero extra surface. The caller memoizes on config.yaml mtime,
    # so toggling consent applies next session, not mid-chat.
    if _real_profile_consented():
        props = dict(BROWSER_EXEC_SCHEMA["parameters"]["properties"])
        props["local"] = {
            "type": "boolean", "default": False,
            "description": ("Drive the user's own local browser (a Hermes-managed copy of their real "
                            "default-Chromium profile, logins/cookies included) instead of the configured "
                            "cloud browser backend. Use when the user asks to act as themselves — their "
                            "accounts, their sessions. No-op when the backend is already local. Default false."),
        }
        overrides["parameters"] = {**BROWSER_EXEC_SCHEMA["parameters"], "properties": props}
    return overrides


BROWSER_EXEC_SCHEMA = {
    "name": "browser_exec",
    # Static fallback description, used only when the managed CLI is unavailable
    "description": (_HEADER_BASE + _HELPERS_DIGEST
                    + "\n\n(The browser-use CLI is not installed yet. Install it with `hermes tools` (Browser Automation → Browser Use).)"),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute using the pre-imported browser helpers. Use print(...) for any data you need back."},
            "session": {"type": "string", "description": "Named isolated browser session — its own daemon and (on cloud backends) own browser, so concurrent tasks don't share tabs. Reuse the same name on every related call; omit for the shared default session."},
            "timeout_s": {"type": "integer", "default": _DEFAULT_TIMEOUT_S,
                          "description": f"Max seconds to wait for the code to finish (default {_DEFAULT_TIMEOUT_S}, max {_MAX_TIMEOUT_S})."},
        },
        "required": ["code"],
    },
}


# browser_exec is additionally gated at tool-definition time — sessions whose toolsets
# lack ``terminal`` never see it (model_tools._compute_tool_definitions). check_fn only
# answers "is browser-harness importable"; surface policy lives with the session.
from tools.registry import registry


# The LOCAL browser this context's browser_exec call ran on ("local" Chromium or "lightpanda", from its session),
# else None: a configured cloud provider did not serve a hybrid ::local sidecar, cloud-fallback or real-profile
# call, and browser.engine read afterwards may name another engine than the session's, so neither labels it.
_exec_local_engine: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("browser_exec_local_engine",
                                                                                   default=None)


def _handle_browser_exec(args: dict, **kw):
    """Registry handler; each call counts against the backend that served it (shared metrics, opt-in)."""
    from hermes_cli.observability.shared_metrics_loop import record_browser_call
    from tools.browser_session_cloud import browser_backend_name
    token = _exec_local_engine.set(None)
    try:
        return record_browser_call(lambda: browser_exec(
            code=args.get("code", ""), session=args.get("session", "") or "",
            timeout_s=args.get("timeout_s", _DEFAULT_TIMEOUT_S), task_id=kw.get("task_id"),
            local=bool(args.get("local", False)),
        ), lambda: browser_backend_name(local_engine=_exec_local_engine.get()))
    finally:
        _exec_local_engine.reset(token)


registry.register(
    name="browser_exec",
    toolset="browser-use",
    schema=BROWSER_EXEC_SCHEMA,
    handler=_handle_browser_exec,
    check_fn=browser_exec_available,
    dynamic_schema_overrides=_dynamic_schema_overrides,
    emoji="🌐",
)
