"""Camofox browser backend — local anti-detection browser via REST API.

Camofox-browser is a self-hosted Node.js server wrapping Camoufox (Firefox
fork with C++ fingerprint spoofing).  It exposes a REST API that maps 1:1
to our browser tool interface: accessibility snapshots with element refs,
click/type/scroll by ref, screenshots, etc.

When ``CAMOFOX_URL`` is set (e.g. ``http://localhost:9377``), the browser
tools route through this module instead of the ``agent-browser`` CLI.

Setup::

    # Option 1: npm
    git clone https://github.com/jo-inc/camofox-browser && cd camofox-browser
    npm install && npm start   # downloads Camoufox (~300MB) on first run

    # Option 2: Docker
    docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser

Then set ``CAMOFOX_URL=http://localhost:9377`` in ``~/.hermes/.env``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, Optional

import requests

from hermes_cli.config import cfg_get, load_config
from tools.browser_camofox_state import get_camofox_identity
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # seconds per HTTP request
_SNAPSHOT_MAX_CHARS = 80_000  # camofox paginates at this limit
_vnc_url: Optional[str] = None  # cached from /health response
_vnc_url_checked = False  # only probe once per process


def get_camofox_url() -> str:
    """Return the configured Camofox server URL, or empty string."""
    return os.getenv("CAMOFOX_URL", "").rstrip("/")


def is_camofox_mode() -> bool:
    """True when Camofox backend is configured and no CDP override is active.

    When the user has explicitly connected to a live Chrome instance via
    ``/browser connect`` (which sets ``BROWSER_CDP_URL``), the CDP connection
    takes priority over Camofox so the browser tools operate on the real
    browser instead of being silently routed to the Camofox backend.
    """
    if os.getenv("BROWSER_CDP_URL", "").strip():
        return False
    return bool(get_camofox_url())


def check_camofox_available() -> bool:
    """Verify the Camofox server is reachable."""
    global _vnc_url, _vnc_url_checked
    url = get_camofox_url()
    if not url:
        return False
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200 and not _vnc_url_checked:
            try:
                data = resp.json()
                vnc_port = data.get("vncPort")
                if isinstance(vnc_port, int) and 1 <= vnc_port <= 65535:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    host = parsed.hostname or "localhost"
                    _vnc_url = f"http://{host}:{vnc_port}"
            except (ValueError, KeyError):
                pass
            _vnc_url_checked = True
        return resp.status_code == 200
    except Exception:
        return False


def get_vnc_url() -> Optional[str]:
    """Return the VNC URL if the Camofox server exposes one, or None."""
    if not _vnc_url_checked:
        check_camofox_available()
    return _vnc_url


def _managed_persistence_enabled() -> bool:
    """Return whether Hermes-managed persistence is enabled for Camofox.

    When enabled, sessions use a stable profile-scoped userId so the
    Camofox server can map it to a persistent browser profile directory.
    When disabled (default), each session gets a random userId (ephemeral).

    Controlled by ``browser.camofox.managed_persistence`` in config.yaml.
    """
    try:
        camofox_cfg = load_config().get("browser", {}).get("camofox", {})
    except Exception as exc:
        logger.warning("managed_persistence check failed, defaulting to disabled: %s", exc)
        return False
    return bool(camofox_cfg.get("managed_persistence"))


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
# Maps task_id -> {"user_id": str, "tab_id": str|None}
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _get_session(task_id: Optional[str]) -> Dict[str, Any]:
    """Get or create a camofox session for the given task.

    When managed persistence is enabled, uses a deterministic userId
    derived from the Hermes profile so the Camofox server can map it
    to the same persistent browser profile across restarts.
    """
    task_id = task_id or "default"
    with _sessions_lock:
        if task_id in _sessions:
            return _sessions[task_id]
        if _managed_persistence_enabled():
            identity = get_camofox_identity(task_id)
            session = {
                "user_id": identity["user_id"],
                "tab_id": None,
                "session_key": identity["session_key"],
                "managed": True,
            }
        else:
            session = {
                "user_id": f"hermes_{uuid.uuid4().hex[:10]}",
                "tab_id": None,
                "session_key": f"task_{task_id[:16]}",
                "managed": False,
            }
        _sessions[task_id] = session
        return session


def _ensure_tab(task_id: Optional[str], url: str = "about:blank") -> Dict[str, Any]:
    """Ensure a tab exists for the session, creating one if needed."""
    session = _get_session(task_id)
    if session["tab_id"]:
        return session
    base = get_camofox_url()
    resp = requests.post(
        f"{base}/tabs",
        json={
            "userId": session["user_id"],
            "sessionKey": session["session_key"],
            "url": url,
        },
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    session["tab_id"] = data.get("tabId")
    return session


def _drop_session(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Remove and return session info."""
    task_id = task_id or "default"
    with _sessions_lock:
        return _sessions.pop(task_id, None)


def camofox_soft_cleanup(task_id: Optional[str] = None) -> bool:
    """Release the in-memory session without destroying the server-side context.

    When managed persistence is enabled the browser profile (and its cookies)
    must survive across agent tasks.  This helper drops only the local tracking
    entry and returns ``True``.  When managed persistence is *not* enabled it
    does nothing and returns ``False`` so the caller can fall back to
    :func:`camofox_close`.
    """
    if _managed_persistence_enabled():
        _drop_session(task_id)
        logger.debug("Camofox soft cleanup for task %s (managed persistence)", task_id)
        return True
    return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """POST JSON to camofox and return parsed response."""
    url = f"{get_camofox_url()}{path}"
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get(path: str, params: dict = None, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """GET from camofox and return parsed response."""
    url = f"{get_camofox_url()}{path}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get_raw(path: str, params: dict = None, timeout: int = _DEFAULT_TIMEOUT) -> requests.Response:
    """GET from camofox and return raw response (for binary data)."""
    url = f"{get_camofox_url()}{path}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def _delete(path: str, body: dict = None, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """DELETE to camofox and return parsed response."""
    url = f"{get_camofox_url()}{path}"
    resp = requests.delete(url, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def camofox_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to a URL via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            # Create tab with the target URL directly
            session = _ensure_tab(task_id, url)
            data = {"ok": True, "url": url}
        else:
            # Navigate existing tab
            data = _post(
                f"/tabs/{session['tab_id']}/navigate",
                {"userId": session["user_id"], "url": url},
                timeout=60,
            )
        result = {
            "success": True,
            "url": data.get("url", url),
            "title": data.get("title", ""),
        }
        vnc = get_vnc_url()
        if vnc:
            result["vnc_url"] = vnc
            result["vnc_hint"] = (
                "Browser is visible via VNC. "
                "Share this link with the user so they can watch the browser live."
            )

        # Auto-take a compact snapshot so the model can act immediately
        try:
            snap_data = _get(
                f"/tabs/{session['tab_id']}/snapshot",
                params={"userId": session["user_id"]},
            )
            snapshot_text = snap_data.get("snapshot", "")
            from tools.browser_tool import (
                SNAPSHOT_SUMMARIZE_THRESHOLD,
                _truncate_snapshot,
            )
            if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
                snapshot_text = _truncate_snapshot(snapshot_text)
            result["snapshot"] = snapshot_text
            result["element_count"] = snap_data.get("refsCount", 0)
        except Exception:
            pass  # Navigation succeeded; snapshot is a bonus

        return json.dumps(result)
    except requests.HTTPError as e:
        return tool_error(f"Navigation failed: {e}", success=False)
    except requests.ConnectionError:
        return json.dumps({
            "success": False,
            "error": f"Cannot connect to Camofox at {get_camofox_url()}. "
                     "Is the server running? Start with: npm start (in camofox-browser dir) "
                     "or: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser",
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_snapshot(full: bool = False, task_id: Optional[str] = None,
                     user_task: Optional[str] = None) -> str:
    """Get accessibility tree snapshot from Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        data = _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )

        snapshot = data.get("snapshot", "")
        refs_count = data.get("refsCount", 0)

        # Apply same summarization logic as the main browser tool
        from tools.browser_tool import (
            SNAPSHOT_SUMMARIZE_THRESHOLD,
            _extract_relevant_content,
            _truncate_snapshot,
        )

        if len(snapshot) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            if user_task:
                snapshot = _extract_relevant_content(snapshot, user_task)
            else:
                snapshot = _truncate_snapshot(snapshot)

        return json.dumps({
            "success": True,
            "snapshot": snapshot,
            "element_count": refs_count,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by ref via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        # Strip @ prefix if present (our tool convention)
        clean_ref = ref.lstrip("@")

        data = _post(
            f"/tabs/{session['tab_id']}/click",
            {"userId": session["user_id"], "ref": clean_ref},
        )
        return json.dumps({
            "success": True,
            "clicked": clean_ref,
            "url": data.get("url", ""),
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an element by ref via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        clean_ref = ref.lstrip("@")

        _post(
            f"/tabs/{session['tab_id']}/type",
            {"userId": session["user_id"], "ref": clean_ref, "text": text},
        )
        return json.dumps({
            "success": True,
            "typed": text,
            "element": clean_ref,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        _post(
            f"/tabs/{session['tab_id']}/scroll",
            {"userId": session["user_id"], "direction": direction},
        )
        return json.dumps({"success": True, "scrolled": direction})
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_back(task_id: Optional[str] = None) -> str:
    """Navigate back via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        data = _post(
            f"/tabs/{session['tab_id']}/back",
            {"userId": session["user_id"]},
        )
        return json.dumps({"success": True, "url": data.get("url", "")})
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        _post(
            f"/tabs/{session['tab_id']}/press",
            {"userId": session["user_id"], "key": key},
        )
        return json.dumps({"success": True, "pressed": key})
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_close(task_id: Optional[str] = None) -> str:
    """Close the browser session via Camofox."""
    try:
        session = _drop_session(task_id)
        if not session:
            return json.dumps({"success": True, "closed": True})

        _delete(
            f"/sessions/{session['user_id']}",
        )
        return json.dumps({"success": True, "closed": True})
    except Exception as e:
        return json.dumps({"success": True, "closed": True, "warning": str(e)})


def camofox_get_images(task_id: Optional[str] = None) -> str:
    """Get images on the current page via Camofox.

    Extracts image information from the accessibility tree snapshot,
    since Camofox does not expose a dedicated /images endpoint.
    """
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        import re

        data = _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )
        snapshot = data.get("snapshot", "")

        # Parse img elements from the accessibility tree.
        # Format: img "alt text" or img "alt text" [eN]
        # URLs appear on /url: lines following img entries
        images = []
        lines = snapshot.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("- img ", "img ")):
                alt_match = re.search(r'img\s+"([^"]*)"', stripped)
                alt = alt_match.group(1) if alt_match else ""
                # Look for URL on the next line
                src = ""
                if i + 1 < len(lines):
                    url_match = re.search(r'/url:\s*(\S+)', lines[i + 1].strip())
                    if url_match:
                        src = url_match.group(1)
                if alt or src:
                    images.append({"src": src, "alt": alt})

        return json.dumps({
            "success": True,
            "images": images,
            "count": len(images),
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_vision(question: str, annotate: bool = False,
                   task_id: Optional[str] = None) -> str:
    """Take a screenshot and analyze it with vision AI via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        # Get screenshot as binary PNG
        resp = _get_raw(
            f"/tabs/{session['tab_id']}/screenshot",
            params={"userId": session["user_id"]},
        )

        # Save screenshot to cache
        from hermes_constants import get_hermes_home
        screenshots_dir = get_hermes_home() / "browser_screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = str(screenshots_dir / f"browser_screenshot_{uuid.uuid4().hex[:8]}.png")

        with open(screenshot_path, "wb") as f:
            f.write(resp.content)

        # Encode for vision LLM
        img_b64 = base64.b64encode(resp.content).decode("utf-8")

        # Also get annotated snapshot if requested
        annotation_context = ""
        if annotate:
            try:
                snap_data = _get(
                    f"/tabs/{session['tab_id']}/snapshot",
                    params={"userId": session["user_id"]},
                )
                annotation_context = f"\n\nAccessibility tree (element refs for interaction):\n{snap_data.get('snapshot', '')[:3000]}"
            except Exception:
                pass

        # Redact secrets from annotation context before sending to vision LLM.
        # The screenshot image itself cannot be redacted, but at least the
        # text-based accessibility tree snippet won't leak secret values.
        from agent.redact import redact_sensitive_text
        annotation_context = redact_sensitive_text(annotation_context)

        # Send to vision LLM
        from agent.auxiliary_client import call_llm

        vision_prompt = (
            f"Analyze this browser screenshot and answer: {question}"
            f"{annotation_context}"
        )

        try:
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vision_timeout = float(_vision_cfg.get("timeout", 120))
            _vision_temperature = float(_vision_cfg.get("temperature", 0.1))
        except Exception:
            _vision_timeout = 120.0
            _vision_temperature = 0.1

        response = call_llm(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }],
            task="vision",
            temperature=_vision_temperature,
            timeout=_vision_timeout,
        )
        analysis = (response.choices[0].message.content or "").strip() if response.choices else ""

        # Redact secrets the vision LLM may have read from the screenshot.
        from agent.redact import redact_sensitive_text
        analysis = redact_sensitive_text(analysis)

        return json.dumps({
            "success": True,
            "analysis": analysis,
            "screenshot_path": screenshot_path,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_console(clear: bool = False, task_id: Optional[str] = None) -> str:
    """Get console output — limited support in Camofox.

    Camofox does not expose browser console logs via its REST API.
    Returns an empty result with a note.
    """
    return json.dumps({
        "success": True,
        "console_messages": [],
        "js_errors": [],
        "total_messages": 0,
        "total_errors": 0,
        "note": "Console log capture is not available with the Camofox backend. "
                "Use browser_snapshot or browser_vision to inspect page state.",
    })



