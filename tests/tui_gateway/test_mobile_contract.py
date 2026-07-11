"""Public JSON-RPC conformance tests for the mobile authorization contract."""

import pytest

from tui_gateway import server


class RecordingTransport:
    def __init__(self, authorization=None):
        self.authorization = authorization
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        pass


def test_mobile_dispatch_rejects_a_mapped_method_without_its_required_scope():
    transport = RecordingTransport(
        {
            "subject": "mobile-user",
            "provider": "stub",
            "audience": "hermes.mobile",
            "scopes": ("conversation.read",),
        }
    )

    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "prompt.submit",
            "params": {},
        },
        transport,
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": "request-1",
        "error": {
            "code": 4030,
            "message": "insufficient authorization scope",
            "data": {
                "reason": "missing_scope",
                "method": "prompt.submit",
                "required_scope": "conversation.write",
                "granted_scopes": ["conversation.read"],
            },
        },
    }


def test_mobile_dispatch_rejects_an_unmapped_method_fail_closed():
    transport = RecordingTransport(
        {
            "subject": "mobile-user",
            "provider": "stub",
            "audience": "hermes.mobile",
            "scopes": ("conversation.read", "conversation.write"),
        }
    )

    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "request-2",
            "method": "shell.exec",
            "params": {"command": "true"},
        },
        transport,
    )

    assert response["error"] == {
        "code": 4030,
        "message": "insufficient authorization scope",
        "data": {
            "reason": "method_not_available_to_mobile",
            "method": "shell.exec",
            "required_scope": None,
            "granted_scopes": ["conversation.read", "conversation.write"],
        },
    }


def test_mobile_dispatch_does_not_expose_legacy_fifo_approval_resolution():
    transport = RecordingTransport(
        {
            "subject": "mobile-user",
            "provider": "stub",
            "audience": "hermes.mobile",
            "scopes": ("conversation.read", "conversation.write"),
        }
    )

    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "approval-request",
            "method": "approval.respond",
            "params": {"choice": "once"},
        },
        transport,
    )

    assert response["error"]["data"] == {
        "reason": "method_not_available_to_mobile",
        "method": "approval.respond",
        "required_scope": None,
        "granted_scopes": ["conversation.read", "conversation.write"],
    }


def test_mobile_dispatch_allows_a_mapped_method_with_its_scope():
    transport = RecordingTransport(
        {
            "subject": "mobile-user",
            "provider": "stub",
            "audience": "hermes.mobile",
            "scopes": ("conversation.read",),
        }
    )
    server._sessions.clear()

    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "request-3",
            "method": "session.active_list",
            "params": {},
        },
        transport,
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": "request-3",
        "result": {"sessions": []},
    }


@pytest.mark.parametrize(
    ("method", "required_scope"),
    [
        ("session.list", "conversation.read"),
        ("session.active_list", "conversation.read"),
        ("session.activate", "conversation.read"),
        ("session.history", "conversation.read"),
        ("session.status", "conversation.read"),
        ("session.usage", "conversation.read"),
        ("session.create", "conversation.write"),
        ("session.resume", "conversation.write"),
        ("prompt.submit", "conversation.write"),
        ("session.interrupt", "conversation.control"),
        ("session.steer", "conversation.control"),
    ],
)
def test_mobile_dispatch_declares_the_minimum_conversation_scope_map(
    method,
    required_scope,
):
    transport = RecordingTransport(
        {
            "subject": "mobile-user",
            "provider": "stub",
            "audience": "hermes.mobile",
            "scopes": (),
        }
    )

    response = server.dispatch(
        {"jsonrpc": "2.0", "id": "scope-check", "method": method, "params": {}},
        transport,
    )

    assert response["error"]["data"] == {
        "reason": "missing_scope",
        "method": method,
        "required_scope": required_scope,
        "granted_scopes": [],
    }


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        {
            "subject": "server-internal",
            "provider": "server-internal",
            "audience": "dashboard",
            "scopes": ("*",),
        },
    ],
)
def test_legacy_and_internal_dispatch_keep_the_existing_behavior(authorization):
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "legacy-request",
            "method": "not.a.real.method",
            "params": {},
        },
        RecordingTransport(authorization),
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": "legacy-request",
        "error": {
            "code": -32601,
            "message": "unknown method: not.a.real.method",
        },
    }
