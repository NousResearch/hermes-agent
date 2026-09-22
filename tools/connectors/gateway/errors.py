"""Gateway HTTP errors and envelope parsing.

Per-tool execute failures are result entries, not exceptions; connection links are intentionally unredacted for the model to relay.
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
    """HTTP-level gateway failure; ``retryable`` is decided by the envelope parser."""

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
    """Authentication or entitlement failure."""


class GatewayUnavailable(ToolGatewayError):
    """404 signals callers to degrade silently to local-only behavior."""


class IdempotencyConflict(ToolGatewayError):
    """Never retry a reused idempotency key with a different body."""


class RateLimited(ToolGatewayError):
    """429 with the gateway's retry hint in seconds; the caller decides whether to wait."""

    def __init__(self, message: str, *, retry_after: float, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


def parse_gateway_error(status: int, body: Any, headers: Optional[Mapping[str, Any]] = None) -> ToolGatewayError:
    """Parse every gateway error envelope without raising on a malformed body.

    Three envelope shapes exist: ``{error: {code, message}}`` on most routes, ``{error: "<code>"}``
    on the account routes, and a flat ``{code, message, requestId, retryAfterMs}`` on a 429."""
    code = f"HTTP_{status}"
    message = ""
    request_id = None
    retry_after_ms: Optional[float] = None
    if isinstance(body, Mapping):
        envelope = body.get("error")
        if isinstance(envelope, Mapping):
            code = str(envelope.get("code") or code)
            message = str(envelope.get("message") or "")
        elif isinstance(envelope, str) and envelope:
            code = envelope
        elif body.get("code"):
            code = str(body["code"])
            message = str(body.get("message") or "")
        raw_request_id = body.get("requestId")
        if raw_request_id is not None:
            request_id = str(raw_request_id)
        if isinstance(body.get("retryAfterMs"), (int, float)):
            retry_after_ms = float(body["retryAfterMs"])
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
    if status == 429:
        header = (headers or {}).get("Retry-After") if headers else None
        retry_after = float(header) if header not in (None, "") else (retry_after_ms or 1000.0) / 1000.0
        return RateLimited(message, retry_after=retry_after, **kwargs)
    return ToolGatewayError(message, retryable=status >= 500, **kwargs)


def render_connection_required(
    *,
    connector: Optional[str] = None,
    message: Optional[str] = None,
    connect_url: Optional[str] = None,
    hint: Optional[str] = None,
    card: bool = False,
) -> dict[str, Any]:
    """Single shared CONNECTION_REQUIRED shape. With a card the link stays on the panel and the
    model is told a connect card is available; without one the model relays the link."""
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
    if card:
        payload["connect_card_available"] = True
    elif connect_url:
        payload["connect_url"] = connect_url
    if hint:
        payload["hint"] = hint
    return payload
