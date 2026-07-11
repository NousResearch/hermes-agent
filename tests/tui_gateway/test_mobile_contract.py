"""Public JSON-RPC conformance tests for the mobile authorization contract."""

import threading

import pytest

from tui_gateway import mobile_contract
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


def _mobile_authorization(*scopes):
    return {
        "subject": "mobile-user",
        "provider": "stub",
        "audience": "hermes.mobile",
        "scopes": scopes,
    }


def test_mobile_dispatch_rejects_a_mapped_method_without_its_required_scope():
    transport = RecordingTransport(_mobile_authorization("conversation.read"))

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
                "required_scopes": ["conversation.write"],
                "missing_scopes": ["conversation.write"],
                "granted_scopes": ["conversation.read"],
                "grantable": True,
            },
        },
    }


def test_mobile_dispatch_rejects_an_unmapped_method_fail_closed():
    transport = RecordingTransport(
        _mobile_authorization("conversation.read", "conversation.write")
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
            "required_scope": "mobile.unavailable",
            "required_scopes": ["mobile.unavailable"],
            "missing_scopes": ["mobile.unavailable"],
            "granted_scopes": ["conversation.read", "conversation.write"],
            "grantable": False,
        },
    }


def test_mobile_dispatch_does_not_expose_legacy_fifo_approval_resolution():
    transport = RecordingTransport(
        _mobile_authorization("conversation.read", "conversation.write")
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
        "required_scope": "mobile.unavailable",
        "required_scopes": ["mobile.unavailable"],
        "missing_scopes": ["mobile.unavailable"],
        "granted_scopes": ["conversation.read", "conversation.write"],
        "grantable": False,
    }


def test_mobile_dispatch_allows_a_mapped_method_with_its_scope():
    transport = RecordingTransport(_mobile_authorization("conversation.read"))
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
        ("session.activate", "conversation.control"),
        ("session.history", "conversation.read"),
        ("session.create", "conversation.write"),
        ("session.resume", "conversation.control"),
        ("prompt.submit", "conversation.write"),
        ("session.interrupt", "conversation.control"),
    ],
)
def test_mobile_dispatch_declares_the_minimum_conversation_scope_map(
    method,
    required_scope,
):
    transport = RecordingTransport(_mobile_authorization())

    response = server.dispatch(
        {"jsonrpc": "2.0", "id": "scope-check", "method": method, "params": {}},
        transport,
    )

    assert response["error"]["data"] == {
        "reason": "missing_scope",
        "method": method,
        "required_scope": required_scope,
        "required_scopes": (
            ["conversation.write", "conversation.control"]
            if method == "session.create"
            else [required_scope]
        ),
        "missing_scopes": (
            ["conversation.write", "conversation.control"]
            if method == "session.create"
            else [required_scope]
        ),
        "granted_scopes": [],
        "grantable": True,
    }


def test_mobile_create_requires_write_and_control():
    transport = RecordingTransport(
        _mobile_authorization("conversation.read", "conversation.write")
    )

    response = server.dispatch(
        {"jsonrpc": "2.0", "id": "create", "method": "session.create", "params": {}},
        transport,
    )

    assert response["error"]["data"] == {
        "reason": "missing_scope",
        "method": "session.create",
        "required_scope": "conversation.control",
        "required_scopes": ["conversation.write", "conversation.control"],
        "missing_scopes": ["conversation.control"],
        "granted_scopes": ["conversation.read", "conversation.write"],
        "grantable": True,
    }


@pytest.mark.parametrize("method", ["session.status", "session.usage", "session.steer"])
def test_mobile_contract_keeps_nonessential_or_account_methods_unavailable(method):
    response = server.dispatch(
        {"jsonrpc": "2.0", "id": "unavailable", "method": method, "params": {}},
        RecordingTransport(
            _mobile_authorization(
                "conversation.read",
                "conversation.write",
                "conversation.control",
            )
        ),
    )

    assert response["error"]["data"]["required_scope"] == "mobile.unavailable"
    assert response["error"]["data"]["grantable"] is False


def test_mobile_dispatch_rejects_hidden_history_deletion():
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "hidden-delete",
            "method": "prompt.submit",
            "params": {
                "session_id": "known",
                "text": "replace",
                "truncate_before_user_ordinal": 1,
            },
        },
        RecordingTransport(
            _mobile_authorization(
                "conversation.read",
                "conversation.write",
                "conversation.control",
            )
        ),
    )

    assert response["error"]["data"] == {
        "reason": "parameter_not_available_to_mobile",
        "method": "prompt.submit",
        "parameter": "truncate_before_user_ordinal",
        "required_scope": "conversation.delete",
        "required_scopes": ["conversation.delete"],
        "missing_scopes": ["conversation.delete"],
        "granted_scopes": [
            "conversation.read",
            "conversation.write",
            "conversation.control",
        ],
        "grantable": False,
    }


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("close_on_disconnect", True),
        ("cwd", "/tmp/outside"),
        ("fast", True),
        ("messages", [{"role": "system", "content": "override"}]),
        ("model", "provider/model"),
        ("parent_session_id", "parent"),
        ("profile", "../../outside"),
        ("provider", "custom"),
        ("reasoning_effort", "high"),
        ("source", "spoofed-platform"),
    ],
)
def test_mobile_dispatch_rejects_each_privileged_create_parameter(
    monkeypatch,
    parameter,
    value,
):
    def fail_if_handler_runs(_profile):
        raise AssertionError("session.create handler ran before mobile authorization")

    monkeypatch.setattr(server, "_profile_home", fail_if_handler_runs)
    server._sessions.clear()
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "privileged-create",
            "method": "session.create",
            "params": {parameter: value},
        },
        RecordingTransport(
            _mobile_authorization(
                "conversation.read",
                "conversation.write",
                "conversation.control",
            )
        ),
    )

    denial = response["error"]["data"]
    assert denial["reason"] == "parameter_not_available_to_mobile"
    assert denial["parameter"] == parameter
    assert denial["required_scope"] == "mobile.unavailable"
    assert denial["grantable"] is False
    assert server._sessions == {}


def test_mobile_write_without_control_queues_a_busy_prompt_without_interrupting(
    monkeypatch,
):
    class Agent:
        interrupted = False

        def interrupt(self):
            self.interrupted = True

    agent = Agent()
    original_transport = RecordingTransport()
    session = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "queued_prompt": None,
        "running": True,
        "transport": original_transport,
    }
    server._sessions.clear()
    server._sessions["busy"] = session
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    transport = RecordingTransport(
        _mobile_authorization("conversation.read", "conversation.write")
    )

    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "busy-write",
            "method": "prompt.submit",
            "params": {"session_id": "busy", "text": "next"},
        },
        transport,
    )

    assert response["result"] == {"status": "queued"}
    assert agent.interrupted is False
    assert session["transport"] is original_transport
    assert session["queued_prompt"]["text"] == "next"
    assert session["queued_prompt"]["transport"] is transport


def test_mobile_scope_grants_require_read_access():
    with pytest.raises(
        ValueError,
        match="conversation.read is required for every mobile WebSocket grant",
    ):
        mobile_contract.normalize_mobile_scopes(["conversation.write"])


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
