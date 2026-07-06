"""Backend-owned KarinAI app integration tools.

These tools do not hold third-party credentials. They call the KarinAI backend
with a run-scoped bearer token that the backend injects into /v1/runs only when
the conversation has active, enabled app connections.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from gateway.session_context import get_session_env
from tools.registry import registry

logger = logging.getLogger(__name__)


LIST_SCHEMA = {
    "name": "karinai_app_tools_list",
    "description": (
        "List connected KarinAI app tools available in this chat session, "
        "including Google Gmail, Drive, and Calendar tools when connected."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


EXECUTE_SCHEMA = {
    "name": "karinai_app_tool",
    "description": (
        "Execute one connected KarinAI app tool. Call karinai_app_tools_list first "
        "to get allowed tool_slug values and whether a connection_id is needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tool_slug": {
                "type": "string",
                "description": "Exact provider tool slug returned by karinai_app_tools_list.",
            },
            "arguments": {
                "type": "object",
                "description": "JSON arguments for the provider tool.",
                "additionalProperties": True,
            },
            "connection_id": {
                "type": "string",
                "description": "Optional connection id returned by karinai_app_tools_list.",
            },
        },
        "required": ["tool_slug"],
        "additionalProperties": False,
    },
}


def _gateway_config() -> Optional[tuple[str, str]]:
    url = get_session_env("KARINAI_APP_TOOL_GATEWAY_URL", "").strip().rstrip("/")
    token = get_session_env("KARINAI_APP_TOOL_GATEWAY_TOKEN", "").strip()
    if not url or not token:
        return None
    return url, token


def _parse_expiry(raw: str) -> Optional[datetime]:
    """Parse the backend-delivered ISO-8601 gateway expiry into aware UTC.

    Returns ``None`` when the value is missing or unparseable so callers can
    fall back to the prior behaviour (still call the gateway; the backend
    enforces expiry server-side). A trailing ``Z`` designator is normalised for
    interpreters whose ``fromisoformat`` predates 3.11, and a naive timestamp is
    treated as UTC because the backend emits UTC.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _gateway_expired() -> bool:
    """Whether the bound gateway token's expiry is present AND already elapsed.

    Uses a timezone-aware UTC comparison. Missing/unparseable expiry => ``False``
    (do not block; the backend still enforces expiry).
    """
    expires_at = _parse_expiry(get_session_env("KARINAI_APP_TOOL_GATEWAY_EXPIRES_AT", ""))
    if expires_at is None:
        return False
    return datetime.now(timezone.utc) >= expires_at


def _app_tools_available() -> bool:
    """Advertise the KarinAI app tools only when a run-scoped gateway token is bound.

    The backend injects the gateway URL+token into ``/v1/runs`` only when the
    conversation has active, enabled app connections. On every other path (CLI,
    messaging platforms, ``/v1/responses``, ``/v1/chat/completions``) no token is
    bound, so without this gate the tools would appear and always report
    ``available: false``. Availability tracks token presence only — expiry is
    enforced at call time in ``_post_gateway`` so the model still gets the clean
    "please reconnect" message instead of the tool silently vanishing mid-run.
    """
    return _gateway_config() is not None


# The gateway token is bound per-request via a contextvar (gateway.session_context),
# so this check's result varies request-to-request inside a single long-lived
# gateway process. Opt out of the registry's 30 s check_fn TTL cache (which keys
# purely on function identity and assumes process-stable availability) so a
# tokenless request can never mask the token from a later /v1/runs request.
_app_tools_available._hermes_no_ttl_cache = True  # type: ignore[attr-defined]


def _gateway_endpoint(base_url: str, path: str) -> str:
    if base_url.endswith("/internal/app-tools"):
        return f"{base_url}{path}"
    return f"{base_url}/internal/app-tools{path}"


def _json_response(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _read_json_response(response: Any) -> Dict[str, Any]:
    raw = response.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {"data": parsed}


def _post_gateway(path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = _gateway_config()
    if config is None:
        return {"available": False, "error": "No KarinAI app connectors are enabled for this run."}
    if _gateway_expired():
        # Do NOT make the HTTP call with an expired token — return a clean,
        # non-leaking message the model can relay to the user.
        return {
            "available": False,
            "error": "The app connection session has expired. Please reconnect.",
        }
    base_url, token = config
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _gateway_endpoint(base_url, path),
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            parsed = _read_json_response(response)
            parsed.setdefault("available", True)
            return parsed
    except urllib.error.HTTPError as exc:
        try:
            detail = _read_json_response(exc)
        except Exception:
            detail = {}
        # Only surface a backend-provided structured message. Never echo
        # exc.reason / exc.url / the raw body verbatim — for some failures those
        # can carry the internal gateway host. The status code is safe.
        message = detail.get("detail") or detail.get("error")
        if not isinstance(message, str) or not message.strip():
            message = "The KarinAI app tool request was rejected."
        return {"available": False, "status": exc.code, "error": message}
    except Exception:
        # URL/DNS/socket errors put the internal gateway host in str(exc); log it
        # server-side and hand the model a static, non-leaking message only.
        logger.exception("KarinAI app tool gateway request failed")
        return {"available": False, "error": "The KarinAI app tool gateway is unavailable."}


def _handle_list(args: Dict[str, Any], **kwargs: Any) -> str:
    _ = (args, kwargs)
    return _json_response(_post_gateway("/list"))


def _handle_execute(args: Dict[str, Any], **kwargs: Any) -> str:
    _ = kwargs
    tool_slug = str(args.get("tool_slug") or "").strip()
    if not tool_slug:
        return _json_response({"available": False, "error": "tool_slug is required"})
    raw_arguments = args.get("arguments")
    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
    payload: Dict[str, Any] = {"tool_slug": tool_slug, "arguments": arguments}
    connection_id = str(args.get("connection_id") or "").strip()
    if connection_id:
        payload["connection_id"] = connection_id
    return _json_response(_post_gateway("/execute", payload))


registry.register(
    name="karinai_app_tools_list",
    toolset="karinai_app_integrations",
    schema=LIST_SCHEMA,
    handler=_handle_list,
    check_fn=_app_tools_available,
    max_result_size_chars=100_000,
)

registry.register(
    name="karinai_app_tool",
    toolset="karinai_app_integrations",
    schema=EXECUTE_SCHEMA,
    handler=_handle_execute,
    check_fn=_app_tools_available,
    max_result_size_chars=200_000,
)
