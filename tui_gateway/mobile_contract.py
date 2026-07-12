"""Versioned authorization contract for native/mobile gateway clients.

This module is deliberately transport-only: it defines what a scoped client
may ask the existing TUI JSON-RPC gateway to do. It does not introduce a new
runtime or duplicate any Hermes-owned session state.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

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

# Advertise only schemas implemented at this stacked branch. Conversation
# snapshots/replay and recoverable approvals remain separate capabilities and
# must not be inferred from Mobile Client Contract v1 alone.
CLIENT_SCHEMA_MAJORS = {
    "gateway.ready": 1,
    "authorization.grant": 1,
    "authorization.error": 1,
    "mutation.receipt": 1,
}

CONVERSATION_READ_SCOPE = "conversation.read"
CONVERSATION_WRITE_SCOPE = "conversation.write"
# This scope is an authorization boundary only. Clients must still intersect it
# with the separately advertised mutation capability before treating retries as
# safe.
CONVERSATION_CONTROL_SCOPE = "conversation.control"
CONVERSATION_DELETE_SCOPE = "conversation.delete"
# Error-only marker for a method or parameter that has no grantable mobile
# scope in this contract version.  It is deliberately absent from
# SUPPORTED_MOBILE_SCOPES; clients must also inspect ``grantable``.
MOBILE_UNAVAILABLE_SCOPE = "mobile.unavailable"

SUPPORTED_MOBILE_SCOPES = frozenset(
    {
        CONVERSATION_READ_SCOPE,
        CONVERSATION_WRITE_SCOPE,
        CONVERSATION_CONTROL_SCOPE,
        CONVERSATION_DELETE_SCOPE,
    }
)

@dataclass(frozen=True)
class MobileMethodPolicy:
    """Scopes and parameters that make one existing gateway method mobile-safe."""

    required_scopes: tuple[str, ...]
    allowed_parameters: frozenset[str]


MOBILE_METHOD_POLICIES = {
    "session.list": MobileMethodPolicy(
        (CONVERSATION_READ_SCOPE,),
        frozenset(),
    ),
    "session.active_list": MobileMethodPolicy(
        (CONVERSATION_READ_SCOPE,),
        frozenset({"current_session_id"}),
    ),
    # Activating or resuming rebinds the live transport, so read access alone
    # is intentionally insufficient even though both responses contain history.
    "session.activate": MobileMethodPolicy(
        (CONVERSATION_CONTROL_SCOPE,),
        frozenset({"session_id"}),
    ),
    "session.history": MobileMethodPolicy(
        (CONVERSATION_READ_SCOPE,),
        frozenset({"session_id"}),
    ),
    # Creating a live session starts Hermes-owned workers and hooks.  It is both
    # a conversation write and a runtime-control action, with server-derived
    # profile/source/model/cwd rather than client-selected privileged knobs.
    "session.create": MobileMethodPolicy(
        (CONVERSATION_WRITE_SCOPE, CONVERSATION_CONTROL_SCOPE),
        frozenset({"cols", "title"}),
    ),
    "session.resume": MobileMethodPolicy(
        (CONVERSATION_CONTROL_SCOPE,),
        frozenset({"cols", "session_id"}),
    ),
    "prompt.submit": MobileMethodPolicy(
        (CONVERSATION_WRITE_SCOPE,),
        frozenset(
            {
                "client_request_id",
                "expected_stored_session_id",
                "session_id",
                "text",
            }
        ),
    ),
    "session.interrupt": MobileMethodPolicy(
        (CONVERSATION_CONTROL_SCOPE,),
        frozenset(
            {
                "client_request_id",
                "expected_stored_session_id",
                "session_id",
            }
        ),
    ),
    "session.delete": MobileMethodPolicy(
        (CONVERSATION_DELETE_SCOPE,),
        frozenset({"client_request_id", "session_id"}),
    ),
    "mutation.status": MobileMethodPolicy(
        (CONVERSATION_READ_SCOPE,),
        frozenset({"client_request_id"}),
    ),
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
    if CONVERSATION_READ_SCOPE not in granted:
        raise ValueError(
            "conversation.read is required for every mobile WebSocket grant"
        )
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
        "capabilities": {
            "auth.ws_scopes": {"version": 1},
            "mutation.idempotency": {
                "version": 1,
                "methods": [
                    "prompt.submit",
                    "session.interrupt",
                    "session.delete",
                ],
                "status_method": "mutation.status",
            },
        },
        "authorization": effective_authorization(authorization),
    }


def authorization_allows_scope(authorization: dict | None, scope: str) -> bool:
    """Return whether a connection may perform a scope-gated side effect."""
    grant = effective_authorization(authorization)
    return grant["audience"] != MOBILE_AUDIENCE or scope in grant["scopes"]


def _denial(
    *,
    reason: str,
    method: str,
    grant: dict,
    required_scopes: tuple[str, ...],
    grantable: bool,
    parameter: str | None = None,
) -> dict:
    missing = (
        list(required_scopes)
        if not grantable
        else [scope for scope in required_scopes if scope not in grant["scopes"]]
    )
    data = {
        "reason": reason,
        "method": method,
        "required_scope": missing[0] if missing else required_scopes[0],
        "required_scopes": list(required_scopes),
        "missing_scopes": missing,
        "granted_scopes": grant["scopes"],
        "grantable": grantable,
    }
    if parameter is not None:
        data["parameter"] = parameter
    return data


def mobile_method_denial(
    method: str,
    authorization: dict | None,
    params: dict | None = None,
) -> dict | None:
    """Describe why a mobile grant cannot call *method*, or return ``None``."""
    grant = effective_authorization(authorization)
    if grant["audience"] != MOBILE_AUDIENCE:
        return None

    policy = MOBILE_METHOD_POLICIES.get(method)
    if policy is None:
        return _denial(
            reason="method_not_available_to_mobile",
            method=method,
            grant=grant,
            required_scopes=(MOBILE_UNAVAILABLE_SCOPE,),
            grantable=False,
        )

    params = params or {}
    unsupported_parameters = sorted(set(params) - policy.allowed_parameters)
    if unsupported_parameters:
        parameter = unsupported_parameters[0]
        required_scope = (
            CONVERSATION_DELETE_SCOPE
            if method == "prompt.submit"
            and parameter == "truncate_before_user_ordinal"
            else MOBILE_UNAVAILABLE_SCOPE
        )
        return _denial(
            reason="parameter_not_available_to_mobile",
            method=method,
            parameter=parameter,
            grant=grant,
            required_scopes=(required_scope,),
            grantable=False,
        )

    if any(scope not in grant["scopes"] for scope in policy.required_scopes):
        return _denial(
            reason="missing_scope",
            method=method,
            grant=grant,
            required_scopes=policy.required_scopes,
            grantable=True,
        )
    return None
