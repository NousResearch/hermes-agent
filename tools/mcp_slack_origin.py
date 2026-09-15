"""Trusted gateway Slack-origin context and opaque per-call MCP signing."""

from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping, Optional

_SECRET_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class SlackOriginSigningError(ValueError):
    """The protected MCP call lacks a valid trusted Slack origin or signing key."""


@dataclass(frozen=True)
class SlackOrigin:
    platform: str
    chat_id: str
    thread_id: Optional[str]


_slack_origin: contextvars.ContextVar[Optional[SlackOrigin]] = contextvars.ContextVar(
    "mcp_slack_origin", default=None
)
_slack_origin_header: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mcp_slack_origin_header", default=None
)


def get_slack_origin() -> Optional[SlackOrigin]:
    """Return trusted gateway origin context, never data supplied through tool arguments."""
    return _slack_origin.get()


@contextmanager
def slack_origin_context(*, platform: str, chat_id: str, thread_id: Optional[str]) -> Iterator[None]:
    """Scope trusted gateway source metadata to one agent run."""
    token = _slack_origin.set(SlackOrigin(platform=str(platform or ""), chat_id=str(chat_id or ""),
                                           thread_id=str(thread_id) if thread_id else None))
    try:
        yield
    finally:
        _slack_origin.reset(token)


@contextmanager
def slack_origin_header_context(header: str) -> Iterator[None]:
    """Scope an already-validated protected-server header to one MCP RPC."""
    token = _slack_origin_header.set(header)
    try:
        yield
    finally:
        _slack_origin_header.reset(token)


def get_slack_origin_header() -> Optional[str]:
    return _slack_origin_header.get()


def propagate_slack_origin_context(coro):
    """Re-establish caller context in Hermes' dedicated MCP event-loop task."""
    origin, header = get_slack_origin(), get_slack_origin_header()
    if origin is None and header is None:
        return coro

    async def _scoped():
        origin_token = _slack_origin.set(origin)
        header_token = _slack_origin_header.set(header)
        try:
            return await coro
        finally:
            _slack_origin_header.reset(header_token)
            _slack_origin.reset(origin_token)

    return _scoped()


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def build_slack_origin_header(policy: Mapping[str, object], *, issued_at: int) -> str:
    """Build a compact authenticated header for an explicitly protected server, or fail closed."""
    secret_env = policy.get("secret_env") if isinstance(policy, Mapping) else None
    if not isinstance(secret_env, str) or not _SECRET_ENV_RE.fullmatch(secret_env):
        raise SlackOriginSigningError("invalid Slack-origin signing policy")
    origin = get_slack_origin()
    if origin is None or origin.platform.lower() != "slack" or not origin.chat_id:
        raise SlackOriginSigningError("trusted Slack origin context is required")
    secret = os.environ.get(secret_env)
    if not secret:
        raise SlackOriginSigningError("Slack-origin signing secret is unavailable")
    payload: dict[str, object] = {"chat_id": origin.chat_id}
    if origin.thread_id:
        payload["thread_id"] = origin.thread_id
    payload["issued_at"] = int(issued_at)
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded_payload = _b64url(payload_bytes)
    signature = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    return f"{encoded_payload}.{_b64url(signature)}"


def slack_origin_request_hook(policy: Mapping[str, object], request) -> None:
    """Attach a prevalidated protected-server header at HTTP request dispatch only."""
    header = get_slack_origin_header()
    if header:
        request.headers["x-nexus-slack-origin"] = header
