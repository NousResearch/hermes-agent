"""Camofox browser backend — local anti-detection browser via REST API.

Camofox-browser (https://github.com/jo-inc/camofox-browser) is a self-hosted Node.js
server wrapping Camoufox (Firefox fork with C++ fingerprint spoofing); its REST API maps
1:1 to our browser tool interface (accessibility snapshots with element refs, click/type/
scroll by ref, screenshots). Setup: ``npm start`` in a checkout or ``docker run -p 9377:9377
-e CAMOFOX_PORT=9377 jo-inc/camofox-browser``, then ``CAMOFOX_URL=http://localhost:9377`` in
``~/.hermes/.env`` (Docker: see ``CAMOFOX_REWRITE_LOOPBACK_URLS`` below).
"""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import threading
import uuid
from typing import Any, Callable, Dict, Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from hermes_cli.config import cfg_get, load_config, read_raw_config
from tools.browser_camofox_state import get_camofox_identity
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---- Configuration ----

_DEFAULT_TIMEOUT = 30  # fallback when config is unreadable
_NO_SESSION_ERROR = "No browser session. Call browser_navigate first."
_vnc_url: Optional[str] = None  # cached from /health response
_vnc_url_checked = False  # only probe once per process
_cached_cmd_timeout: Optional[int] = None  # browser.command_timeout, resolved lazily like browser_tool
_cmd_timeout_resolved = False


def _get_command_timeout() -> int:
    """``browser.command_timeout`` (floor 5s, default 30s), cached after first read."""
    global _cached_cmd_timeout, _cmd_timeout_resolved
    if _cmd_timeout_resolved:
        return _cached_cmd_timeout  # type: ignore[return-value]
    _cmd_timeout_resolved = True
    _cached_cmd_timeout = _DEFAULT_TIMEOUT
    try:
        val = cfg_get(read_raw_config(), "browser", "command_timeout")
        if val is not None:
            _cached_cmd_timeout = max(int(val), 5)
    except Exception as exc:
        logger.debug("Could not read browser.command_timeout: %s", exc)
    return _cached_cmd_timeout


def _auth_headers() -> Dict[str, str]:
    """Return Authorization header when CAMOFOX_API_KEY is set."""
    key = (get_secret("CAMOFOX_API_KEY", "") or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def get_camofox_url() -> str:
    """Return the configured Camofox server URL, or empty string."""
    return (get_secret("CAMOFOX_URL", "") or "").rstrip("/")


def _config_cdp_url() -> str:
    """Persistent ``browser.cdp_url`` from config.yaml, or "" (read here, not via
    ``browser_tool_cdp._get_cdp_override`` — circular import)."""
    try:
        from hermes_cli.config import read_raw_config  # late-bound: tests patch the source module
        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception:
        pass
    return ""


def is_camofox_mode() -> bool:
    """True when the Camofox backend is selected and no CDP override is active.

    Selection is ``browser.cloud_provider: camofox``; ``CAMOFOX_URL`` is only the address
    and never overrides a different stored selection (legacy: with no selection ever
    written, a set ``CAMOFOX_URL`` still activates Camofox). A CDP override (``BROWSER_CDP_URL``
    env or ``browser.cdp_url``, same precedence as ``browser_tool_cdp._get_cdp_override()``) wins.
    """
    if os.getenv("BROWSER_CDP_URL", "").strip() or _config_cdp_url():
        return False
    try:
        from tools.tool_backend_helpers import read_selection
        selected = read_selection("browser")
    except Exception:  # pragma: no cover — helpers are in-repo
        selected = None
    if selected is not None:
        return selected == "camofox"
    return bool(get_camofox_url())


def check_camofox_available() -> bool:
    """Verify the Camofox server is reachable (and cache its VNC URL once)."""
    global _vnc_url, _vnc_url_checked
    url = get_camofox_url()
    if not url:
        return False
    try:
        resp = requests.get(f"{url}/health", timeout=5)
    except Exception:
        return False
    if resp.status_code == 200 and not _vnc_url_checked:
        try:
            vnc_port = resp.json().get("vncPort")
            if isinstance(vnc_port, int) and 1 <= vnc_port <= 65535:
                _vnc_url = f"http://{urlsplit(url).hostname or 'localhost'}:{vnc_port}"
        except (ValueError, KeyError):
            pass
        _vnc_url_checked = True
    return resp.status_code == 200


def get_vnc_url() -> Optional[str]:
    """Return the VNC URL if the Camofox server exposes one, or None."""
    if not _vnc_url_checked:
        check_camofox_available()
    return _vnc_url


def _get_camofox_config() -> Dict[str, Any]:
    """Return the ``browser.camofox`` config block, or an empty dict."""
    try:
        camofox_cfg = load_config().get("browser", {}).get("camofox", {})
    except Exception as exc:
        logger.warning("camofox config check failed, defaulting to disabled: %s", exc)
        return {}
    return camofox_cfg if isinstance(camofox_cfg, dict) else {}


def _managed_persistence_enabled(camofox_cfg: Optional[Dict[str, Any]] = None) -> bool:
    """``browser.camofox.managed_persistence``: stable profile-scoped userId vs random per session."""
    return bool((_get_camofox_config() if camofox_cfg is None else camofox_cfg).get("managed_persistence"))


def _env_or_cfg(env_name: str, camofox_cfg: Dict[str, Any], cfg_key: str, *, secret: bool = False) -> str:
    """Env/secret-scope value first, then the ``browser.camofox`` config key, else ""."""
    raw = get_secret(env_name, "") if secret else os.getenv(env_name, "")
    return (raw or "").strip() or str(camofox_cfg.get(cfg_key) or "").strip()


def _camofox_identity_override(task_id: Optional[str], camofox_cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Externally configured identity (integrations owning the visible Camofox browser
    share a user ID so Hermes uses the same profile), or None."""
    user_id = _env_or_cfg("CAMOFOX_USER_ID", camofox_cfg, "user_id", secret=True)
    if not user_id:
        return None
    session_key = _env_or_cfg("CAMOFOX_SESSION_KEY", camofox_cfg, "session_key", secret=True)
    return {"user_id": user_id, "session_key": session_key or f"task_{(task_id or 'default')[:16]}"}


def _flag(env_name: str, camofox_cfg: Dict[str, Any], cfg_key: str) -> bool:
    """Boolean toggle: env var wins when set to a valid value, else config key."""
    raw = os.getenv(env_name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw:
        logger.debug("Ignoring invalid boolean env %s=%r", env_name, raw)
    return bool(camofox_cfg.get(cfg_key))


def _is_loopback_hostname(hostname: Optional[str]) -> bool:
    if not hostname:
        return False
    host = hostname.strip().strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _rewrite_loopback_url_for_camofox(url: str) -> tuple[str, Optional[Dict[str, str]]]:
    """Rewrite loopback page URLs for Docker-hosted Camofox, if configured.

    ``CAMOFOX_URL`` may point at a host-published Docker port, but page URLs are opened by
    the browser *inside* the container, where loopback is the container, not the host.
    Opt-in (``CAMOFOX_REWRITE_LOOPBACK_URLS`` / config) because non-Docker installs run the
    browser on the host. Returns ``(rewritten_url, metadata)``; ``metadata`` is present only
    when a rewrite happened so the tool result can disclose the change to the model.
    """
    camofox_cfg = _get_camofox_config()
    if not _flag("CAMOFOX_REWRITE_LOOPBACK_URLS", camofox_cfg, "rewrite_loopback_urls"):
        return url, None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return url, None
    alias = _env_or_cfg("CAMOFOX_LOOPBACK_HOST_ALIAS", camofox_cfg, "loopback_host_alias") or "host.docker.internal"
    if parsed.scheme not in {"http", "https"} or not _is_loopback_hostname(parsed.hostname) or not alias:
        return url, None
    userinfo = (parsed.username + (f":{parsed.password}" if parsed.password else "") + "@") if parsed.username else ""
    host_part = f"[{alias}]" if ":" in alias and not alias.startswith("[") else alias
    port_part = f":{parsed.port}" if parsed.port else ""
    rewritten = urlunsplit(
        SplitResult(parsed.scheme, f"{userinfo}{host_part}{port_part}", parsed.path, parsed.query, parsed.fragment))
    return rewritten, {"from": parsed.hostname or "", "to": alias, "original_url": url, "rewritten_url": rewritten}


# ---- Session management ----
_sessions: Dict[str, Dict[str, Any]] = {}  # task_id -> {"user_id": str, "tab_id": str|None, ...}
_sessions_lock = threading.Lock()


def _adopt_existing_tab(session: Dict[str, Any]) -> Dict[str, Any]:
    """Rehydrate tab_id from an already-open managed tab: gateway restarts empty the
    in-memory cache while Camofox still holds the integration-owned tab."""
    if session.get("tab_id") or not session.get("adopt_existing_tab") or not get_camofox_url():
        return session
    try:
        tabs = _get("/tabs", params=_user_params(session), timeout=5).get("tabs", [])
    except Exception as exc:
        logger.debug("Camofox tab adoption failed for %s: %s", session.get("user_id"), exc)
        return session
    dict_tabs = [tab for tab in tabs if isinstance(tab, dict)] if isinstance(tabs, list) else []
    candidates = [tab for tab in dict_tabs if tab.get("listItemId") == session.get("session_key")] or dict_tabs
    tab_id = candidates[-1].get("tabId") if candidates else None
    if isinstance(tab_id, str) and tab_id:
        session["tab_id"] = tab_id
        logger.debug("Adopted existing Camofox tab %s for %s", tab_id, session.get("user_id"))
    return session


def _get_session(task_id: Optional[str]) -> Dict[str, Any]:
    """Get or create the task's session. Identity precedence: external override
    (CAMOFOX_USER_ID / config) → profile-scoped identity when managed persistence
    is on → random ephemeral userId."""
    task_id = task_id or "default"
    with _sessions_lock:
        if task_id in _sessions:
            return _adopt_existing_tab(_sessions[task_id])
        camofox_cfg = _get_camofox_config()
        identity = _camofox_identity_override(task_id, camofox_cfg)
        if identity is None and _managed_persistence_enabled(camofox_cfg):
            identity = get_camofox_identity(task_id)
        if identity is None:
            identity = {"user_id": f"hermes_{uuid.uuid4().hex[:10]}", "session_key": f"task_{task_id[:16]}"}
            managed, adopt = False, False
        else:
            managed, adopt = True, _flag("CAMOFOX_ADOPT_EXISTING_TAB", camofox_cfg, "adopt_existing_tab")
        session = {"user_id": identity["user_id"], "tab_id": None, "session_key": identity["session_key"],
                   "managed": managed, "adopt_existing_tab": adopt}
        _sessions[task_id] = session
        return _adopt_existing_tab(session)


def _ensure_tab(task_id: Optional[str], url: str = "about:blank") -> Dict[str, Any]:
    """Ensure a tab exists for the session, creating one if needed."""
    session = _get_session(task_id)
    if not session["tab_id"]:
        data = _post("/tabs", {"userId": session["user_id"], "listItemId": session["session_key"], "url": url})
        session["tab_id"] = data.get("tabId")
    return session


def _drop_session(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Remove and return session info."""
    with _sessions_lock:
        return _sessions.pop(task_id or "default", None)


def camofox_soft_cleanup(task_id: Optional[str] = None) -> bool:
    """Drop only the local tracking entry (``True``) for managed profiles, which must
    survive across agent tasks; ``False`` for ephemeral sessions so the caller falls back
    to :func:`camofox_close`."""
    camofox_cfg = _get_camofox_config()
    if _managed_persistence_enabled(camofox_cfg) or _camofox_identity_override(task_id, camofox_cfg):
        _drop_session(task_id)
        logger.debug("Camofox soft cleanup for task %s (managed persistence)", task_id)
        return True
    return False


# ---- HTTP helpers ----
def _request(method: str, path: str, timeout: Optional[int] = None, **kwargs: Any) -> requests.Response:
    """Issue an authenticated request to camofox and return the raised-for-status response."""
    resp = getattr(requests, method)(f"{get_camofox_url()}{path}", headers=_auth_headers(),
                                     timeout=_get_command_timeout() if timeout is None else timeout, **kwargs)
    resp.raise_for_status()
    return resp


def _post(path: str, body: dict, timeout: Optional[int] = None) -> dict:
    return _request("post", path, timeout, json=body).json()


def _get(path: str, params: dict = None, timeout: Optional[int] = None) -> dict:
    return _request("get", path, timeout, params=params).json()


def _get_raw(path: str, params: dict = None, timeout: Optional[int] = None) -> requests.Response:
    """GET and return the raw response (for binary data)."""
    return _request("get", path, timeout, params=params)


def _delete(path: str, body: dict = None, timeout: Optional[int] = None) -> dict:
    return _request("delete", path, timeout, json=body).json()


# ---- Tool implementations ----
def _tab_path(session: Dict[str, Any], suffix: str) -> str:
    return f"/tabs/{session['tab_id']}/{suffix}"


def _user_params(session: Dict[str, Any]) -> Dict[str, str]:
    return {"userId": session["user_id"]}


def _snapshot_data(session: Dict[str, Any]) -> dict:
    return _get(_tab_path(session, "snapshot"), params=_user_params(session))


def _parse_snapshot_images(snapshot: str) -> list[Dict[str, str]]:
    """Images from an accessibility snapshot: ``img "alt" [eN]`` entries with the URL on
    the following ``/url:`` line (Camofox has no /images endpoint)."""
    images = []
    lines = snapshot.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(("- img ", "img ")):
            continue
        alt_match = re.search(r'img\s+"([^"]*)"', stripped)
        url_match = re.search(r'/url:\s*(\S+)', lines[i + 1].strip()) if i + 1 < len(lines) else None
        alt, src = (alt_match.group(1) if alt_match else ""), (url_match.group(1) if url_match else "")
        if alt or src:
            images.append({"src": src, "alt": alt})
    return images


def _fetch_snapshot(session: Dict[str, Any]) -> tuple[str, int]:
    """``(snapshot_text, refs_count)`` truncated like the main browser tool (line boundaries,
    full tree stored to cache/web, read_file pointer appended). Lazy import: ``browser_tool``
    imports this module."""
    from tools.browser_tool_snapshot import _truncate_snapshot
    from tools.browser_tool import get_browser_snapshot_threshold
    data = _snapshot_data(session)
    snapshot, threshold = data.get("snapshot", ""), get_browser_snapshot_threshold()
    if len(snapshot) > threshold:
        snapshot = _truncate_snapshot(snapshot, max_chars=threshold)
    return snapshot, data.get("refsCount", 0)


def _navigate_tab(task_id: Optional[str], browser_url: str) -> tuple[Dict[str, Any], dict]:
    """Open ``browser_url`` in the task's tab (creating it if missing) and return
    ``(session, navigate_response)``. A 404 on the existing tab means the server
    garbage-collected it — recreate instead of failing."""
    session = _get_session(task_id)
    if session["tab_id"]:
        try:
            data = _post(_tab_path(session, "navigate"), {"userId": session["user_id"], "url": browser_url}, timeout=60)
            return session, data
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise
            logger.warning("Camofox tab %s returned 404 — tab was garbage collected. Creating a fresh tab.",
                           session["tab_id"])
            session["tab_id"] = None
    return _ensure_tab(task_id, browser_url), {"ok": True, "url": browser_url}


def camofox_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to a URL via Camofox."""
    try:
        browser_url, rewrite_info = _rewrite_loopback_url_for_camofox(url)
        session, data = _navigate_tab(task_id, browser_url)
        result = {"success": True, "url": data.get("url", browser_url), "title": data.get("title", "")}
        if rewrite_info:
            result["requested_url"], result["url_rewrite"] = url, rewrite_info
            result["warning"] = ("Rewrote loopback URL for Docker-hosted Camofox: "
                                 f"{rewrite_info['from']} -> {rewrite_info['to']}")
        vnc = get_vnc_url()
        if vnc:
            result["vnc_url"] = vnc
            result["vnc_hint"] = ("Browser is visible via VNC. "
                                  "Share this link with the user so they can watch the browser live.")
        try:  # Auto-take a compact snapshot so the model can act immediately.
            result["snapshot"], result["element_count"] = _fetch_snapshot(session)
        except Exception:
            pass  # Navigation succeeded; snapshot is a bonus
        return json.dumps(result)
    except requests.HTTPError as e:
        return tool_error(f"Navigation failed: {e}", success=False)
    except requests.ConnectionError:
        return json.dumps({"success": False, "error": (
            f"Cannot connect to Camofox at {get_camofox_url()}. "
            "Is the server running? Start with: npm start (in camofox-browser dir) "
            "or: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")})
    except Exception as e:
        return tool_error(str(e), success=False)


def _camofox_private_page_block(session: Dict[str, Any], task_id: Optional[str], action: str) -> Optional[str]:
    """Blocked payload when the current page is private/internal, else None.

    Mirrors the ``_camofox_eval`` guard in browser_tool.py: page-state reads on a non-local
    backend can leak an intranet/metadata page the terminal can't reach. Only active when
    the SSRF guard applies (non-local backend, not a local sidecar, ``allow_private_urls``
    unset); fail-open on probe failure like sibling guards. Lazy import (cycle).
    """
    from tools.browser_tool_eval_policy import _camofox_current_page_private_url, _eval_ssrf_guard_active
    if not _eval_ssrf_guard_active(task_id or "default"):
        return None
    blocked_url = _camofox_current_page_private_url(session["tab_id"], session["user_id"])
    if not blocked_url:
        return None
    return json.dumps({"success": False, "error": (
        "Blocked: page URL targets a private or internal address "
        f"({blocked_url}). Refusing to {action} on this page in this browser mode.")}, ensure_ascii=False)


def _require_tab(task_id: Optional[str], action: Optional[str] = None) -> tuple[Dict[str, Any], Optional[str]]:
    """Return ``(session, error_payload)``: error when no tab exists or, if ``action`` given, the page is private."""
    session = _get_session(task_id)
    if not session["tab_id"]:
        return session, tool_error(_NO_SESSION_ERROR, success=False)
    return session, (_camofox_private_page_block(session, task_id, action) if action is not None else None)


def _with_tab(task_id: Optional[str], guard_action: Optional[str], body: Callable[[Dict[str, Any]], str]) -> str:
    """Require a tab (+ private-page guard when ``guard_action`` is set), then run ``body(session)``;
    any exception becomes a ``tool_error``."""
    try:
        session, blocked = _require_tab(task_id, guard_action)
        if blocked:
            return blocked
        return body(session)
    except Exception as e:
        return tool_error(str(e), success=False)


def _tab_action(task_id: Optional[str], guard_action: Optional[str], suffix: str,
                body: Dict[str, Any], result: Callable[[dict], dict]) -> str:
    """Simple tab action: POST ``body`` to ``/tabs/<id>/<suffix>``, build the result."""
    return _with_tab(task_id, guard_action, lambda session: json.dumps(
        result(_post(_tab_path(session, suffix), {"userId": session["user_id"], **body}))))


def camofox_snapshot(full: bool = False, task_id: Optional[str] = None, user_task: Optional[str] = None) -> str:
    """Accessibility tree snapshot. ``user_task`` is deprecated and ignored —
    oversized snapshots always truncate-and-store (no LLM summarization)."""
    def body(session):
        snapshot, refs_count = _fetch_snapshot(session)
        return json.dumps({"success": True, "snapshot": snapshot, "element_count": refs_count})
    return _with_tab(task_id, "read a page snapshot", body)


def camofox_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by ref via Camofox."""
    clean_ref = ref.lstrip("@")  # our tool convention prefixes refs with @
    return _tab_action(task_id, "click", "click", {"ref": clean_ref},
                       lambda data: {"success": True, "clicked": clean_ref, "url": data.get("url", "")})


def camofox_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an element by ref via Camofox."""
    try:
        session, blocked = _require_tab(task_id, "type")
        if blocked:
            return blocked
        clean_ref = ref.lstrip("@")
        _post(_tab_path(session, "type"), {"userId": session["user_id"], "ref": clean_ref, "text": text})
        from agent.display import redact_browser_typed_text_for_display, redact_tool_args_for_display
        # Match browser_tool.browser_type: the raw text is typed into the page, but the
        # returned display value is run through the secret-pattern redactor so API keys /
        # tokens don't leak into tool progress or chat history.
        display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]
        response = {"success": True, "typed": display_text, "element": clean_ref}
        return json.dumps(redact_browser_typed_text_for_display(response, text))
    except Exception as e:
        from agent.display import redact_browser_typed_text_for_display
        return tool_error(redact_browser_typed_text_for_display(str(e), text), success=False)


def camofox_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page via Camofox."""
    return _tab_action(task_id, None, "scroll", {"direction": direction},
                       lambda data: {"success": True, "scrolled": direction})


def camofox_back(task_id: Optional[str] = None) -> str:
    """Navigate back via Camofox."""
    return _tab_action(task_id, None, "back", {}, lambda data: {"success": True, "url": data.get("url", "")})


def camofox_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key via Camofox."""
    return _tab_action(task_id, "press", "press", {"key": key}, lambda data: {"success": True, "pressed": key})


def camofox_close(task_id: Optional[str] = None) -> str:
    """Close the browser session via Camofox."""
    try:
        session = _drop_session(task_id)
        if session:
            _delete(f"/sessions/{session['user_id']}")
        return json.dumps({"success": True, "closed": True})
    except Exception as e:
        return json.dumps({"success": True, "closed": True, "warning": str(e)})


def camofox_get_images(task_id: Optional[str] = None) -> str:
    """Get images on the current page via Camofox (parsed from the snapshot)."""
    def body(session):
        images = _parse_snapshot_images(_snapshot_data(session).get("snapshot", ""))
        return json.dumps({"success": True, "images": images, "count": len(images)})
    return _with_tab(task_id, "extract page images", body)


def _vision_llm_settings() -> tuple[float, float]:
    """``auxiliary.vision`` ``(timeout, temperature)``; defaults 120s / 0.1 on any config error."""
    try:
        cfg = cfg_get(load_config(), "auxiliary", "vision", default={})
        return float(cfg.get("timeout", 120)), float(cfg.get("temperature", 0.1))
    except Exception:
        return 120.0, 0.1


def _save_screenshot(content: bytes) -> str:
    """Write PNG bytes under ``$HERMES_HOME/browser_screenshots`` and return the path."""
    from hermes_constants import get_hermes_home
    screenshots_dir = get_hermes_home() / "browser_screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = str(screenshots_dir / f"browser_screenshot_{uuid.uuid4().hex[:8]}.png")
    with open(screenshot_path, "wb") as f:
        f.write(content)
    return screenshot_path


def camofox_vision(question: str, annotate: bool = False, task_id: Optional[str] = None) -> str:
    """Take a screenshot and analyze it with vision AI via Camofox."""
    def body(session):
        resp = _get_raw(_tab_path(session, "screenshot"), params=_user_params(session))
        screenshot_path = _save_screenshot(resp.content)
        img_b64 = base64.b64encode(resp.content).decode("utf-8")
        annotation_context = ""
        if annotate:
            try:
                snapshot = _snapshot_data(session).get("snapshot", "")
                annotation_context = f"\n\nAccessibility tree (element refs for interaction):\n{snapshot[:3000]}"
            except Exception:
                pass
        # The screenshot itself cannot be redacted, but the text-based accessibility snippet
        # sent alongside it must not leak secret values.
        from agent.redact import redact_sensitive_text
        from agent.auxiliary_client import call_llm
        vision_prompt = f"Analyze this browser screenshot and answer: {question}{redact_sensitive_text(annotation_context)}"
        timeout, temperature = _vision_llm_settings()
        response = call_llm(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}],
            task="vision", temperature=temperature, timeout=timeout)
        analysis = (response.choices[0].message.content or "").strip() if response.choices else ""
        # Redact secrets the vision LLM may have read from the screenshot.
        return json.dumps({"success": True, "analysis": redact_sensitive_text(analysis), "screenshot_path": screenshot_path})
    return _with_tab(task_id, "capture a screenshot", body)


def camofox_console(clear: bool = False, task_id: Optional[str] = None) -> str:
    """Console output is not exposed by the Camofox REST API; return an empty result with a note."""
    return json.dumps({
        "success": True, "console_messages": [], "js_errors": [], "total_messages": 0, "total_errors": 0,
        "note": "Console log capture is not available with the Camofox backend. "
                "Use browser_snapshot or browser_vision to inspect page state."})
