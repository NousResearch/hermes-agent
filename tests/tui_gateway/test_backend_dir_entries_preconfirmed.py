"""Internal path listings bypass model approvals and keep their session identity (#115478)."""

from types import SimpleNamespace

import pytest

from gateway.session_context import get_session_env
from tui_gateway.methods_complete import _backend_dir_entries


def _live_environment(monkeypatch, seen, payload=None):
    def execute(command, **kwargs):
        seen.update(command=command, kwargs=kwargs, session=get_session_env("HERMES_SESSION_KEY"))
        return payload if payload is not None else {"output": "src/\nREADME.md\n", "returncode": 0}

    monkeypatch.setattr("tools.terminal_tool_lifecycle.get_active_env",
                        lambda task_id: SimpleNamespace(execute=execute))


def test_backend_dir_entries_preconfirms_internal_listing(monkeypatch):
    seen = {}
    _live_environment(monkeypatch, seen)

    def approval(*args, **kwargs):
        raise AssertionError("internal listing entered the model approval path")

    monkeypatch.setattr("tools.terminal_tool.terminal_tool", approval)
    assert _backend_dir_entries("/workspace", session_key="sess") == [("README.md", False), ("src", True)]
    assert seen["kwargs"] == {"timeout": 3, "allow_start": False}


def test_backend_dir_entries_keeps_session_routing(monkeypatch):
    seen = {}
    _live_environment(monkeypatch, seen)
    _backend_dir_entries("~/proj", session_key="sess-42")
    assert seen["session"] == "sess-42"


@pytest.mark.parametrize("payload", [{"output": "boom", "returncode": 1}, {"error": "backend unreachable"}])
def test_backend_dir_entries_swallows_failed_listings(monkeypatch, payload):
    _live_environment(monkeypatch, {}, payload)
    assert _backend_dir_entries("/workspace", session_key="sess") == []
