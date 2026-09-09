"""Typed errors for the connector tool gateway, plus THE envelope parser.

House idiom: one ``RuntimeError`` base with a small set of subclasses, one
per condition a caller actually branches on (``microsoft_graph_auth.py`` /
``image_source.py`` precedent). HTTP-level failures use the gateway's nested
error envelope ``{"error": {"code", "message", ...}, "requestId"}`` and are
parsed in exactly one place: :func:`parse_gateway_error`.

Per-tool errors inside a 200 execute envelope are NOT exceptions — they are
result entries, rendered by ``merge.py`` (CONNECTION_REQUIRED payloads via
:func:`render_connection_required`, the single producer of the connect-link
dict shown to the model; the link is deliberately not redacted).

stdlib-only: this module must not import pydantic or any sibling module.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = [
    "GatewayAuthError",
    "GatewayUnavailable",
    "IdempotencyConflict",
    "ToolGatewayError",
    "parse_gateway_error",
    "render_connection_required",
]


class ToolGatewayError(RuntimeError):
    """A connector gateway request failed at the HTTP level.

    ``retryable`` encodes the retry policy decision (at most one retry, same
    idempotency key, transport failures and 5xx only) so the client never
    re-derives it from the status code.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "GATEWAY_ERROR",
        status: Optional[int] = None,
        request_id: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_id = request_id
        self.retryable = retryable


class GatewayAuthError(ToolGatewayError):
    """401/403 — the portal token is missing, expired, or not entitled."""


class GatewayUnavailable(ToolGatewayError):
    """404 from any connector route — connectors are dark for this principal.

    This is the silent-degradation signal: callers fall back to local-only
    behavior and the model never sees a connector error.
    """


class IdempotencyConflict(ToolGatewayError):
    """409 — the idempotency key was reused with a different body.

    Always a client bug; never retried.
    """


def parse_gateway_error(status: int, body: Any) -> ToolGatewayError:
    """Parse an HTTP-level gateway failure into the right exception.

    The one place that understands the nested error envelope. Total: any
    body shape (dict, text, ``None``) produces a usable exception rather
    than raising.
    """
    code = f"HTTP_{status}"
    message = ""
    request_id = None
    if isinstance(body, Mapping):
        envelope = body.get("error")
        if isinstance(envelope, Mapping):
            code = str(envelope.get("code") or code)
            message = str(envelope.get("message") or "")
        raw_request_id = body.get("requestId")
        if raw_request_id is not None:
            request_id = str(raw_request_id)
    elif body:
        message = str(body)[:500]
    if not message:
        message = f"tool gateway request failed with status {status}"

    kwargs = {
        "code": code,
        "status": status,
        "request_id": request_id,
    }
    if status in (401, 403):
        return GatewayAuthError(message, **kwargs)
    if status == 404:
        return GatewayUnavailable(message, **kwargs)
    if status == 409:
        return IdempotencyConflict(message, **kwargs)
    return ToolGatewayError(message, retryable=status >= 500, **kwargs)


def render_connection_required(
    *,
    connector: Optional[str] = None,
    message: Optional[str] = None,
    connect_url: Optional[str] = None,
    hint: Optional[str] = None,
) -> dict[str, Any]:
    """Render the model-facing CONNECTION_REQUIRED payload.

    The single producer of this dict, shared by the execute merge and the
    connections tool so the model always sees one shape. The connect link is
    passed through un-redacted — the model is allowed to show it to the user.
    Wording beyond a fallback message is the gateway's job; this function
    does not invent instructions.
    """
    payload: dict[str, Any] = {
        "code": "CONNECTION_REQUIRED",
        "message": message
        or (
            f"The {connector} connector is not connected for this account."
            if connector
            else "This connector is not connected for this account."
        ),
    }
    if connector:
        payload["connector"] = connector
    if connect_url:
        payload["connect_url"] = connect_url
    if hint:
        payload["hint"] = hint
    return payload
