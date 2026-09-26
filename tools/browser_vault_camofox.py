"""Private Camofox transport for vault operations; never a model-facing JS tool."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import ipaddress
import json
from typing import Any
from urllib.parse import quote, urlsplit

import requests

from hermes_constants import hermes_home_key
from tools import browser_camofox as camofox

# Private compatibility contract with the external handoff plugin.
CAMOFOX_VAULT_PROTOCOL = 1


@dataclass(frozen=True, repr=False)
class _Target:
    task_id: str
    home: str
    url: str
    user_id: str
    headers: dict[str, str]
    timeout: int


_target: ContextVar[_Target | None] = ContextVar("vault_camofox_target", default=None)


def selected() -> bool:
    return _target.get() is not None or camofox.is_camofox_mode()


def _secure_url(url: str) -> bool:
    """Plain HTTP is only safe on a numeric loopback (e.g. an SSH tunnel).

    Do not resolve DNS: localhost-like names and rebinding are not loopback proof.
    """
    try:
        parsed = urlsplit(url)
        if (not parsed.hostname or parsed.username is not None or parsed.password is not None
                or "?" in url or "#" in url or any(c.isspace() for c in url)):
            return False
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            return False
        if parsed.scheme == "https":
            return True
        address = ipaddress.ip_address(parsed.hostname)
        return parsed.scheme == "http" and address.is_loopback and (
            address.version == 4 or str(address) == "::1"
        )
    except ValueError:
        return False


def operation(func):
    """Pin one existing tab/endpoint/auth across inspection, prompts and the fill.

    Nested save-login → fill calls reuse the target. No tab creation/adoption or
    fallback is permitted inside a vault operation.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not selected():
            return func(*args, **kwargs)
        task_id = (kwargs.get("task_id") if "task_id" in kwargs else args[1] if len(args) > 1 else None) or "default"
        existing = _target.get()
        if existing is not None:
            if existing.task_id != task_id or existing.home != hermes_home_key():
                return json.dumps({"success": False, "error": "Camofox vault operation scope changed."})
            return func(*args, **kwargs)
        session_key = camofox._session_key(task_id)
        home, url, _ = session_key
        if not _secure_url(url):
            return json.dumps({
                "success": False, "error_type": "camofox_transport_required",
                "error": "Camofox vault requires HTTPS or HTTP on a numeric loopback address (for example an SSH tunnel).",
            })
        headers = camofox._auth_headers()
        if not headers.get("Authorization"):
            return json.dumps({"success": False, "error_type": "camofox_auth_required",
                               "error": "Camofox vault requires CAMOFOX_API_KEY authentication."})
        with camofox._sessions_lock:
            session = dict(camofox._sessions.get(session_key, {}))
        if not session.get("tab_id"):
            return json.dumps({"success": False, "error": "No Camofox tab. Call browser_navigate first."})
        target = _Target(
            task_id, home,
            f"{url}/tabs/{quote(session['tab_id'], safe='')}/evaluate",
            session["user_id"], headers, camofox._get_command_timeout(),
        )
        token = _target.set(target)
        try:
            return func(*args, **kwargs)
        finally:
            _target.reset(token)
    return wrapped


def evaluate(task_id: str, expression: str, *, max_filled: int | None = None) -> dict[str, Any]:
    """Use only the operation's pinned Camofox target, never a local browser."""
    try:
        target = _target.get()
        if target is None or target.task_id != task_id or target.home != hermes_home_key():
            return {"success": False, "error": "No matching Camofox vault operation."}
        if max_filled is not None:
            # Camofox logs uncaught evaluation errors. Page-defined exceptions
            # must never reach that logger; the catch uses a literal, not a
            # potentially page-overridden JSON.stringify. Await catches rejections.
            failure = json.dumps(json.dumps({"refused": "evaluation_failed"}))
            expression = f"(async () => {{ try {{ return await ({expression}); }} catch {{ return {failure}; }} }})()"
        with requests.Session() as http:
            # A loopback tunnel must not silently become a plaintext proxy hop;
            # .netrc must not override the pinned bearer credential either.
            http.trust_env = False
            response = http.post(
                target.url, headers=target.headers, timeout=target.timeout,
                allow_redirects=False, verify=True,
                json={"userId": target.user_id, "expression": expression},
            )
        if response.status_code != 200:
            return {"success": False, "error": "Camofox vault evaluation failed (redirects are refused)."}
        result = response.json().get("result")
        if max_filled is not None:
            # Only the protocol's bounded count/refusal may cross this boundary.
            # Even `found` can contain an endpoint's echo of the credential.
            if isinstance(result, str):
                result = json.loads(result)
            if not isinstance(result, dict):
                raise ValueError
            if result.get("refused") == "origin_changed":
                result = {"refused": "origin_changed"}
            elif (set(result) == {"filled"} and type(result["filled"]) is int
                  and 0 <= result["filled"] <= max_filled):
                result = {"filled": result["filled"]}
            else:
                raise ValueError
        return {"success": True, "result": result}
    except Exception:
        return {"success": False, "error": "Camofox vault evaluation failed."}
