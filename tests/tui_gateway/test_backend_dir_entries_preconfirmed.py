"""Backend path completion must arrive at ``terminal_tool`` pre-confirmed.

The composer's remote-backend directory listing wraps a fixed read-only script in
``sh -c``, which the dangerous-command guard flags as "shell command via -c/-lc
flag". Under smart approvals that verdict fires an auxiliary-LLM call per
completion — the main model when no auxiliary approval model is configured — and
Desktop's websocket reconnect loop re-runs completion on every reconnect, so an
idle Desktop makes model calls around the clock (#115478).
"""

import json

import pytest

from tui_gateway.methods_complete import _backend_dir_entries


def _fake_terminal_tool(seen, output="src/\nREADME.md\n", exit_code=0):
    def _capture(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return json.dumps({"output": output, "exit_code": exit_code})

    return _capture




def test_backend_dir_entries_preconfirms_internal_listing(monkeypatch):
    """The listing is Hermes-owned plumbing: it must skip the approval gate, not consult it."""
    seen = {}
    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _fake_terminal_tool(seen))
    entries = _backend_dir_entries("/workspace", session_key="sess")

    assert seen["kwargs"]["force"] is True
    assert entries == [("README.md", False), ("src", True)]


def test_backend_dir_entries_keeps_session_routing(monkeypatch):
    seen = {}
    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _fake_terminal_tool(seen))
    _backend_dir_entries("~/proj", session_key="sess-42")

    assert seen["kwargs"]["task_id"] == "sess-42"


@pytest.mark.parametrize("payload", [
    {"output": "boom", "exit_code": 1},
    {"error": "backend unreachable"},
])
def test_backend_dir_entries_swallows_failed_listings(monkeypatch, payload):
    def _raw(command, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _raw)
    assert _backend_dir_entries("/workspace", session_key="sess") == []
