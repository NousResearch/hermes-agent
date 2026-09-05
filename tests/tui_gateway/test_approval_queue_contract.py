"""Zero-model proof of the public gateway approval queue contract."""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.tui_gateway.approval_queue_contract import (
    ApprovalProtocolViolation,
    EXPECTED_CHOICES,
    EXPECTED_EXECUTIONS,
    GatewayApprovalTrace,
    render_normalized_trace,
)


class _CaptureTransport:
    def __init__(self) -> None:
        self.frames: queue.Queue[dict[str, Any]] = queue.Queue()

    def write(self, frame: dict[str, Any]) -> bool:
        self.frames.put(frame)
        return True

    def next_event(self, event_type: str) -> dict[str, Any]:
        frame = self.frames.get(timeout=5)
        assert frame["method"] == "event"
        assert frame["params"]["type"] == event_type
        return frame


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    for sid in list(mod._sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")
    mod._pending.clear()
    mod._answers.clear()
    mod._live_transports.clear()


def _event(event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "type": event_type,
        "session_id": "synthetic-ui-session",
    }
    if payload is not None:
        params["payload"] = payload
    return {"jsonrpc": "2.0", "method": "event", "params": params}


def _resolved(_request: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"result": {"resolved": 1}}


def test_once_deny_once_uses_public_gateway_protocol_without_provider_calls(
    server, monkeypatch
):
    from tools import approval, approval_context
    from tools.approval_gateway_wait import _await_gateway_decision

    sid = "synthetic-ui-session"
    session_key = "synthetic-runtime-session"
    transport = _CaptureTransport()
    server._sessions[sid] = {
        "history": [],
        "session_key": session_key,
        "tool_progress_mode": "all",
        "transport": transport,
    }
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 5)

    trace = GatewayApprovalTrace(session_id=sid)
    execution_counts = [0, 0, 0]
    decisions: list[dict[str, Any] | None] = [None, None, None]

    try:
        for index, expected_choice in enumerate(EXPECTED_CHOICES):
            def _wait_for_decision(turn_index: int = index) -> None:
                decision = _await_gateway_decision(
                    session_key,
                    lambda data: server._emit_approval_request(sid, data),
                    {
                        "command": f"synthetic-fixture-{turn_index + 1}",
                        "description": "deterministic approval fixture",
                        "pattern_key": "deterministic-fixture",
                        "pattern_keys": ["deterministic-fixture"],
                    },
                    surface="tui_gateway_contract",
                )
                decisions[turn_index] = decision
                if decision["choice"] != "deny":
                    execution_counts[turn_index] += 1
                    tool_id = f"synthetic-tool-{turn_index + 1}"
                    server._on_tool_start(sid, tool_id, "fixture_tool", {})
                    server._on_tool_complete(
                        sid, tool_id, "fixture_tool", {}, '{"ok":true}'
                    )

            worker = threading.Thread(target=_wait_for_decision)
            worker.start()

            approval_event = transport.next_event("approval.request")
            request_id = trace.observe_event(approval_event)
            assert request_id
            assert trace.respond(request_id, server.handle_request) == expected_choice

            worker.join(timeout=5)
            assert not worker.is_alive()
            assert decisions[index] == {"resolved": True, "choice": expected_choice, "reason": None}
            if expected_choice != "deny":
                trace.observe_event(transport.next_event("tool.start"))
                trace.observe_event(transport.next_event("tool.complete"))

        server._emit("message.complete", sid, {"text": "synthetic completion"})
        trace.observe_event(transport.next_event("message.complete"))
    finally:
        approval.resolve_gateway_approval(session_key, "deny", resolve_all=True)

    normalized = trace.normalized_trace(tuple(execution_counts))
    rendered = render_normalized_trace(normalized)

    assert tuple(execution_counts) == EXPECTED_EXECUTIONS
    assert normalized["provider_calls"] == 0
    assert normalized["approval_request_before_tool_start"] is True
    assert normalized["terminal_outcomes"] == ["completed"]
    assert rendered == render_normalized_trace(normalized)
    assert sid not in rendered
    assert session_key not in rendered
    assert "synthetic-tool" not in rendered
    assert "synthetic-fixture" not in rendered


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "non-empty request_id"),
        ("duplicate", "duplicate approval request_id"),
        ("unknown", "unknown approval request_id"),
        ("out_of_order", "out-of-order approval request_id"),
        ("unresolved", "did not resolve exactly one"),
        ("duplicate_response", "duplicate approval response"),
        ("pending_terminal", "unresolved approval request"),
        ("post_terminal", "event received after turn terminal"),
    ],
)
def test_invalid_approval_sequences_fail_closed(case: str, message: str):
    trace = GatewayApprovalTrace(session_id="synthetic-ui-session")

    with pytest.raises(ApprovalProtocolViolation, match=message):
        if case == "missing":
            trace.observe_event(_event("approval.request", {"request_id": ""}))
        elif case == "duplicate":
            trace.observe_event(_event("approval.request", {"request_id": "a"}))
            trace.observe_event(_event("approval.request", {"request_id": "a"}))
        elif case == "unknown":
            trace.respond("unknown", _resolved)
        elif case == "out_of_order":
            trace.observe_event(_event("approval.request", {"request_id": "a"}))
            trace.observe_event(_event("approval.request", {"request_id": "b"}))
            trace.respond("b", _resolved)
        elif case == "unresolved":
            trace.observe_event(_event("approval.request", {"request_id": "a"}))
            trace.respond("a", lambda _request: {"result": {"resolved": 0}})
        elif case == "duplicate_response":
            trace.observe_event(_event("approval.request", {"request_id": "a"}))
            trace.respond("a", _resolved)
            trace.respond("a", _resolved)
        elif case == "pending_terminal":
            trace.observe_event(_event("approval.request", {"request_id": "a"}))
            trace.observe_event(_event("message.complete"))
        else:
            for request_id in ("a", "b", "c"):
                trace.observe_event(
                    _event("approval.request", {"request_id": request_id})
                )
                trace.respond(request_id, _resolved)
            trace.observe_event(_event("message.complete"))
            trace.observe_event(_event("status.update"))
