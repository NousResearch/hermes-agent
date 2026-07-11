"""Versioned authorization contract for native/mobile gateway clients.

This module is deliberately transport-only: it defines what a scoped client
may ask the existing TUI JSON-RPC gateway to do. It does not introduce a new
runtime or duplicate any Hermes-owned session state.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from hermes_cli import __release_date__, __version__

MOBILE_AUDIENCE = "hermes.mobile"
LEGACY_AUDIENCE = "dashboard"
PROTOCOL_NAME = "hermes.tui.jsonrpc"
PROTOCOL_MAJOR = 1
MOBILE_CONTRACT_NAME = "hermes.mobile"
MOBILE_CONTRACT_MAJOR = 1

SERVER_VERSION = __version__
SERVER_RELEASE_DATE = __release_date__
# A new server process is a new event-stream authority.  This identity is
# intentionally stable across connections but changes after process restart;
# durable deployment identity remains outside this first contract slice.
SERVER_INSTANCE_ID = str(uuid.uuid4())

# Advertise only schemas this slice actually defines.  Conversation snapshots,
# replay, mutation idempotency, and recoverable approvals land in later slices
# and must not be inferred from Mobile Client Contract v1 alone.
CLIENT_SCHEMA_MAJORS = {
    "gateway.ready": 1,
    "authorization.grant": 1,
    "authorization.error": 1,
}

CONVERSATION_READ_SCOPE = "conversation.read"
CONVERSATION_WRITE_SCOPE = "conversation.write"
# This scope is an authorization boundary only.  It does not claim durable
# mutation idempotency; clients must wait for that separately advertised
# capability before treating retries as safe.
CONVERSATION_CONTROL_SCOPE = "conversation.control"

SUPPORTED_MOBILE_SCOPES = frozenset(
    {
        CONVERSATION_READ_SCOPE,
        CONVERSATION_WRITE_SCOPE,
        CONVERSATION_CONTROL_SCOPE,
    }
)

MOBILE_METHOD_SCOPES = {
    "session.list": CONVERSATION_READ_SCOPE,
    "prompt.submit": CONVERSATION_WRITE_SCOPE,
    "session.active_list": CONVERSATION_READ_SCOPE,
    "session.activate": CONVERSATION_READ_SCOPE,
    "session.history": CONVERSATION_READ_SCOPE,
    "session.status": CONVERSATION_READ_SCOPE,
    "session.usage": CONVERSATION_READ_SCOPE,
    "session.create": CONVERSATION_WRITE_SCOPE,
    "session.resume": CONVERSATION_WRITE_SCOPE,
    "session.interrupt": CONVERSATION_CONTROL_SCOPE,
    "session.steer": CONVERSATION_CONTROL_SCOPE,
}


def normalize_mobile_scopes(scopes: Iterable[object]) -> tuple[str, ...]:
    """Validate and de-duplicate a requested mobile authorization grant."""
    granted: list[str] = []
    for raw in scopes:
        scope = str(raw or "").strip()
        if not scope or scope not in SUPPORTED_MOBILE_SCOPES:
            raise ValueError(f"unsupported mobile scope: {scope or '<empty>'}")
        if scope not in granted:
            granted.append(scope)
    if not granted:
        raise ValueError("at least one mobile scope is required")
    return tuple(granted)


def effective_authorization(raw: dict | None = None) -> dict:
    """Return the stable, JSON-safe authorization grant exposed to a client."""
    raw = raw or {}
    scopes = raw.get("scopes", ("*",))
    if not isinstance(scopes, (list, tuple)):
        scopes = (str(scopes),)
    return {
        "subject": str(raw.get("subject") or raw.get("user_id") or "legacy"),
        "provider": str(raw.get("provider") or "legacy"),
        "audience": str(raw.get("audience") or LEGACY_AUDIENCE),
        "scopes": [str(scope) for scope in scopes],
    }


def gateway_ready_payload(*, skin: str, authorization: dict | None = None) -> dict:
    """Build the additive first-frame contract for every WebSocket client."""
    return {
        "skin": skin,
        "server": {
            "version": SERVER_VERSION,
            "release_date": SERVER_RELEASE_DATE,
            "instance_id": SERVER_INSTANCE_ID,
        },
        "protocol": {"name": PROTOCOL_NAME, "major": PROTOCOL_MAJOR},
        "contract": {
            "name": MOBILE_CONTRACT_NAME,
            "major": MOBILE_CONTRACT_MAJOR,
        },
        "schemas": dict(CLIENT_SCHEMA_MAJORS),
        "capabilities": {"auth.ws_scopes": {"version": 1}},
        "authorization": effective_authorization(authorization),
    }


def mobile_method_denial(method: str, authorization: dict | None) -> dict | None:
    """Describe why a mobile grant cannot call *method*, or return ``None``."""
    grant = effective_authorization(authorization)
    if grant["audience"] != MOBILE_AUDIENCE:
        return None

    required_scope = MOBILE_METHOD_SCOPES.get(method)
    if required_scope is None:
        return {
            "reason": "method_not_available_to_mobile",
            "method": method,
            "required_scope": None,
            "granted_scopes": grant["scopes"],
        }
    if required_scope not in grant["scopes"]:
        return {
            "reason": "missing_scope",
            "method": method,
            "required_scope": required_scope,
            "granted_scopes": grant["scopes"],
        }
    return None
