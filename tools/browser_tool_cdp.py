"""User-supplied CDP endpoint resolution (browser.cdp_url / real-profile), dialog-policy config and the per-task CDP supervisor lifecycle.

Split out of ``tools/browser_tool.py``. Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle."""

import contextlib
import os
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from agent.proxy_bypass import loopback_request_kwargs
from tools.browser_tool_origin import origin_module as _origin
from utils import is_truthy_value

# Same shape as browser_exec ``session=`` / BU_NAME: 1-64 letters, digits, underscore, hyphen.
_ENDPOINT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# Last-segment auto-bind must not treat a UUID-ish Hermes session id as a map key.
_UUIDISH_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)


def _resolve_cdp_override(cdp_url: str) -> str:
    """Normalize a user-supplied CDP endpoint into a concrete websocket URL.

    Full ``ws://.../devtools/browser/...`` endpoints pass through; HTTP discovery roots and bare ``ws://host:port``
    resolve via ``/json/version`` → ``webSocketDebuggerUrl`` (falls back to the raw value with a warning).
    """
    _bt = _origin()
    raw = (cdp_url or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if "/devtools/browser/" in lowered:
        return raw

    discovery_url = raw
    if lowered.startswith(("ws://", "wss://")):
        if not (raw.count(":") == 2 and raw.rstrip("/").rsplit(":", 1)[-1].isdigit() and "/" not in raw.split(":", 2)[-1]):
            return raw
        discovery_url = ("http://" if lowered.startswith("ws://") else "https://") + raw.split("://", 1)[1]
    version_url = discovery_url if discovery_url.lower().endswith("/json/version") else discovery_url.rstrip("/") + "/json/version"

    san = _bt._sanitize_url_for_logs
    try:
        import requests  # lazy — shared module object, test patches still apply
        response = requests.get(version_url, timeout=10, **loopback_request_kwargs(version_url))
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        _bt.logger.warning("Failed to resolve CDP endpoint %s via %s: %s", san(raw), san(version_url), san(exc))
        return raw
    ws_url = str(payload.get("webSocketDebuggerUrl") or "").strip()
    if ws_url:
        _bt.logger.info("Resolved CDP endpoint %s -> %s", san(raw), san(ws_url))
        return ws_url
    _bt.logger.warning("CDP discovery at %s did not return webSocketDebuggerUrl; using raw endpoint", san(version_url))
    return raw


def _parse_cdp_endpoint_records(value) -> Dict[str, Dict[str, Any]]:
    """Normalize ``browser.cdp_endpoints`` to ``{name: {url, stay_put}}``.

    String form (``lab2: http://127.0.0.1:9223``) stays allowed and is *not* stay-put.
    Object form (``primary: {url: ..., stay_put: true}``) opts that Chrome into the Bot
    Screen lease fence. Invalid names / missing URLs are dropped.
    """
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw in value.items():
        name = str(raw_name or "").strip()
        if not name or not _ENDPOINT_NAME_RE.match(name):
            continue
        if isinstance(raw, dict):
            url = str(raw.get("url") or "").strip()
            stay_put = is_truthy_value(raw.get("stay_put"), default=False)
        else:
            url = str(raw or "").strip()
            stay_put = False
        if not url:
            continue
        out[name] = {"url": url, "stay_put": stay_put}
    return out


def _parse_cdp_endpoints(value) -> Dict[str, str]:
    """Normalize ``browser.cdp_endpoints`` to ``{name: url}``. Invalid names/URLs are dropped."""
    return {name: rec["url"] for name, rec in _parse_cdp_endpoint_records(value).items()}


def _cdp_endpoint_records() -> Dict[str, Dict[str, Any]]:
    """``browser.cdp_endpoints`` records from config, or ``{}``. No network I/O."""
    return _origin()._browser_cfg(
        "cdp_endpoints", {}, _parse_cdp_endpoint_records, "browser.cdp_endpoints from config"
    )


def _cdp_endpoints_map() -> Dict[str, str]:
    """``browser.cdp_endpoints`` from config, or ``{}``. No network I/O."""
    return {name: rec["url"] for name, rec in _cdp_endpoint_records().items()}


def _unnamed_cdp_url() -> str:
    return _origin()._browser_cfg("cdp_url", "", lambda v: str(v or "").strip(), "browser.cdp_url from config")


def _unnamed_cdp_is_stay_put() -> bool:
    """``browser.cdp_stay_put`` — opt-in fence for the unnamed ``cdp_url`` default."""
    return _origin()._browser_cfg(
        "cdp_stay_put", False, lambda v: is_truthy_value(v, default=False), "browser.cdp_stay_put from config"
    )


def _cdp_urls_equal(left: str, right: str) -> bool:
    return (left or "").rstrip("/") == (right or "").rstrip("/")


def _url_is_stay_put(url: str) -> bool:
    """True when ``url`` is a stay-put CDP (named ``stay_put: true`` or unnamed ``cdp_stay_put``)."""
    url = (url or "").strip()
    if not url:
        return False
    for rec in _cdp_endpoint_records().values():
        if rec["stay_put"] and _cdp_urls_equal(rec["url"], url):
            return True
    unnamed = _unnamed_cdp_url()
    return bool(unnamed and _cdp_urls_equal(unnamed, url) and _unnamed_cdp_is_stay_put())


def _cdp_override_is_stay_put(endpoint: Optional[str] = None) -> bool:
    """True when the CDP that ``_get_cdp_override_raw`` would select is marked stay-put.

    Unmarked user/cloud CDP stays unfenced (#108914). Opt-in only — a random Browserbase
    URL is never fenced just because some other endpoint is stay-put.
    """
    try:
        raw = _get_cdp_override_raw(endpoint=endpoint)
    except TypeError:
        raw = _get_cdp_override_raw()
    return _url_is_stay_put(raw)


def _cdp_url_for_endpoint_name(name: str) -> str:
    """Exact map lookup for ``name`` (``session=`` / ``/browser connect <name>``). ``""`` on miss."""
    name = (name or "").strip()
    if not name:
        return ""
    return _cdp_endpoints_map().get(name, "")


def _hermes_session_identity_names() -> list:
    """Candidate endpoint names from the live Hermes conversation identity (no network I/O)."""
    try:
        from gateway.session_context import get_session_env
        getter = get_session_env
    except Exception:
        getter = lambda n, d="": os.environ.get(n, d)  # noqa: E731 — env fallback when gateway is absent
    names: list = []
    for var in ("HERMES_SESSION_ID", "HERMES_SESSION_KEY"):
        val = str(getter(var, "") or "").strip()
        if not val:
            continue
        names.append(val)
        for sep in (":", "/", "."):
            if sep in val:
                tail = val.rsplit(sep, 1)[-1].strip()
                if tail and tail != val and not _UUIDISH_RE.match(tail):
                    names.append(tail)
    return names


def _named_cdp_candidates(endpoint: Optional[str] = None) -> Iterable[str]:
    """Ordered names to look up in ``browser.cdp_endpoints`` (first hit wins)."""
    seen: set = set()
    ordered: list = []

    def _add(raw: str) -> None:
        name = (raw or "").strip()
        if not name or name in seen:
            return
        seen.add(name)
        ordered.append(name)

    _add(endpoint or "")
    _add(os.environ.get("BROWSER_CDP_ENDPOINT", ""))
    for name in _hermes_session_identity_names():
        _add(name)
    return ordered


def _lookup_cdp_endpoint(endpoint: Optional[str] = None) -> str:
    """First ``cdp_endpoints`` hit for ``endpoint`` / ``BROWSER_CDP_ENDPOINT`` / Hermes session identity."""
    mapping = _cdp_endpoints_map()
    if not mapping:
        return ""
    for name in _named_cdp_candidates(endpoint):
        url = mapping.get(name, "")
        if url:
            return url
    return ""


def expand_cdp_connect_target(raw: str) -> str:
    """Expand a ``cdp_endpoints`` alias used as ``/browser connect <name>``; pass URLs through."""
    raw = (raw or "").strip()
    if not raw or "://" in raw:
        return raw
    # host:port is a URL-shaped target, not a map key.
    if ":" in raw and "/" not in raw:
        host, _, port = raw.partition(":")
        if host and port.isdigit():
            return raw
    return _cdp_url_for_endpoint_name(raw) or raw


def _get_cdp_override_raw(endpoint: Optional[str] = None) -> str:
    """Return the *configured* CDP override without any network I/O.

    Precedence:
      1. ``BROWSER_CDP_URL`` env (live ``/browser connect`` URL — process-global)
      2. ``browser.cdp_endpoints`` hit for ``endpoint`` / ``BROWSER_CDP_ENDPOINT`` / Hermes session identity
      3. ``browser.cdp_url`` (unnamed default)

    Is-it-configured gates (check_fns, ``_is_local_mode`` / ``_is_local_backend``, ``hermes doctor``)
    MUST use this, not :func:`_get_cdp_override`: its 10s HTTP discovery against a stale ``cdp_url``
    would stall every startup's schema build with no error.
    """
    env_override = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_override:
        return env_override
    mapped = _lookup_cdp_endpoint(endpoint)
    if mapped:
        return mapped
    return _unnamed_cdp_url()


def _get_cdp_override(endpoint: Optional[str] = None) -> str:
    """Resolved CDP URL override, or "" (skips cloud AND local launch).

    May perform HTTP ``/json/version`` discovery — only call on paths about to *connect*; pure gates must use
    :func:`_get_cdp_override_raw`. ``endpoint`` is a ``cdp_endpoints`` / ``session=`` name; omitted uses the
    unnamed default (``BROWSER_CDP_URL`` → session identity → ``cdp_url``).
    """
    return _resolve_cdp_override(raw) if (raw := _get_cdp_override_raw(endpoint=endpoint)) else ""


def _get_dialog_policy_config() -> Tuple[str, float]:
    """Read ``browser.dialog_policy`` + ``browser.dialog_timeout_s``; supervisor defaults when absent/invalid."""
    _bt = _origin()
    # Deferred so browser_tool imports in minimal environments.
    from tools.browser_supervisor_dialogs import DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S, _VALID_POLICIES
    policy, timeout_s = DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if not isinstance(browser_cfg, dict):
            return policy, timeout_s
        candidate = str(browser_cfg.get("dialog_policy") or DEFAULT_DIALOG_POLICY)
        if candidate in _VALID_POLICIES:
            policy = candidate
        else:
            _bt.logger.debug("Invalid browser.dialog_policy=%r; using default", candidate)
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

    Idempotent (``get_or_start`` skips an existing ``(task_id, cdp_url)`` and restarts on URL change), so safe on
    every navigate / ``/browser connect``. URL precedence: the CDP override, then the session's own ``cdp_url``
    (cloud providers, e.g. Browserbase). Swallows all errors — a failed attach must not break the session;
    snapshots just lack ``pending_dialogs`` / ``frame_tree``.
    """
    _bt = _origin()
    cdp_url = _get_cdp_override()
    if not cdp_url:
        with _bt._cleanup_lock:
            session_info = _bt._active_sessions.get(task_id, {})
        maybe = str(session_info.get("cdp_url") or "")
        if maybe:
            cdp_url = _resolve_cdp_override(maybe)
    if not cdp_url:
        return
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        policy, timeout_s = _get_dialog_policy_config()
        SUPERVISOR_REGISTRY.get_or_start(task_id=task_id, cdp_url=cdp_url, dialog_policy=policy, dialog_timeout_s=timeout_s)
    except Exception as exc:
        _bt.logger.debug("CDP supervisor attach for task=%s failed (non-fatal): %s", task_id, exc)


def _stop_cdp_supervisor(task_id: str) -> None:
    """Stop the CDP supervisor for ``task_id`` if one exists. No-op otherwise."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        SUPERVISOR_REGISTRY.stop(task_id)
    except Exception as exc:
        _origin().logger.debug("CDP supervisor stop for task=%s failed (non-fatal): %s", task_id, exc)
