"""Regression tests: malformed JSON-RPC `session_id` must not crash handlers.

Some TUI JSON-RPC methods directly do `_sessions.get(params["session_id"])`.
If a client sends an unhashable type (e.g. list/dict), Python raises
`TypeError` before any handler error handling — crashing the request.

This test covers a representative path (`process.list`) and a second
helper path (`_completion_cwd`).
"""

import pytest

import tui_gateway.server as server


@pytest.fixture(autouse=True)
def _disable_deferred_agent_build(monkeypatch):
    # Avoid background timers arming during these unit tests.
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)


def test_process_list_rejects_unhashable_session_id(monkeypatch):
    resp = server.handle_request(
        {
            "id": "r1",
            "method": "process.list",
            "params": {"session_id": ["not", "hashable"]},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4001


def test_completion_cwd_never_crashes_on_unhashable_session_id():
    cwd = server._completion_cwd({"session_id": {"x": 1}})
    assert isinstance(cwd, str)
    assert cwd

