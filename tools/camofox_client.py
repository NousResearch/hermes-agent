"""Neutral Camofox REST client primitives.

This module owns the low-level Camofox HTTP/tab API shared by browser tools and
web-provider integrations. Higher-level adapters should keep session semantics
and tool response shaping in their own modules, and call these primitives for
actual Camofox server I/O.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_vnc_url: Optional[str] = None
_vnc_url_checked = False


def get_camofox_url() -> str:
    """Return the configured Camofox server URL, or empty string."""
    return os.getenv("CAMOFOX_URL", "").rstrip("/")


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


def post_json(path: str, body: dict, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """POST JSON to Camofox and return the parsed response."""
    url = f"{get_camofox_url()}{path}"
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_json(path: str, params: Optional[dict] = None, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """GET from Camofox and return the parsed response."""
    url = f"{get_camofox_url()}{path}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_raw(path: str, params: Optional[dict] = None, timeout: int = _DEFAULT_TIMEOUT) -> requests.Response:
    """GET from Camofox and return the raw response (for binary data)."""
    url = f"{get_camofox_url()}{path}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def delete_json(
    path: str,
    body: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """DELETE to Camofox and return the parsed response."""
    url = f"{get_camofox_url()}{path}"
    resp = requests.delete(url, json=body, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def camofox_connection_help() -> str:
    """Return the canonical Camofox connection troubleshooting message."""
    return (
        f"Cannot connect to Camofox at {get_camofox_url()}. "
        "Is the server running? Start with: npm start (in camofox-browser dir) "
        "or: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser"
    )


def camofox_create_tab(
    user_id: str,
    session_key: str,
    url: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Create a Camofox tab and return the server payload."""
    if not get_camofox_url():
        raise RuntimeError("CAMOFOX_URL is not set")
    body = {"userId": user_id, "sessionKey": session_key}
    if url:
        body["url"] = url
    data = post_json("/tabs", body, timeout=timeout)
    tab_id = data.get("tabId")
    if not isinstance(tab_id, str) or not tab_id:
        raise RuntimeError("Camofox did not return a tabId")
    return data


def camofox_tab_navigate(
    user_id: str,
    tab_id: str,
    url: str,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Navigate a specific Camofox tab without using browser tool state."""
    return post_json(
        f"/tabs/{tab_id}/navigate",
        {"userId": user_id, "url": url},
        timeout=timeout,
    )


def camofox_tab_snapshot(
    user_id: str,
    tab_id: str,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Return an accessibility snapshot for a specific Camofox tab."""
    return get_json(
        f"/tabs/{tab_id}/snapshot",
        params={"userId": user_id},
        timeout=timeout,
    )


def camofox_tab_evaluate(
    user_id: str,
    tab_id: str,
    expression: str,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Evaluate JavaScript in a specific Camofox tab."""
    return post_json(
        f"/tabs/{tab_id}/evaluate",
        {"userId": user_id, "expression": expression},
        timeout=timeout,
    )


def camofox_close_tab(user_id: str, tab_id: str, timeout: int = 10) -> None:
    """Best-effort close for an isolated Camofox tab."""
    try:
        delete_json(f"/tabs/{tab_id}", params={"userId": user_id}, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — cleanup must not mask caller errors
        logger.debug("Camofox tab cleanup failed for %s: %s", tab_id, exc)


@contextmanager
def camofox_temporary_tab(
    purpose: str,
    url: Optional[str] = None,
    user_prefix: str = "hermes_web",
):
    """Create an isolated temporary Camofox tab and always close it."""
    user_id = f"{user_prefix}_{uuid.uuid4().hex[:10]}"
    tab_id = str(camofox_create_tab(user_id, purpose, url=url)["tabId"])
    try:
        yield user_id, tab_id
    finally:
        camofox_close_tab(user_id, tab_id)
