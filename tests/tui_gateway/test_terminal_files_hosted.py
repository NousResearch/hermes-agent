"""Attachment lookup follows hosted sandbox identity, not local session cache keys."""
from types import SimpleNamespace

import pytest

from agent.runtime_scope import bind_state
from hermes_scope import Principal, Scope
from tools.environments.sandbox_scope import (
    CrossAgentHandleRefused, ScopedEnvironment, sandbox_identity,
)
from tui_gateway.terminal_files import scoped_session_fs


def test_hosted_reader_uses_existing_own_sandbox_and_rejects_foreign_handle(monkeypatch):
    from tools import terminal_tool

    owner = Scope("acme", "agent-a", Principal("u1"), owner=Principal("u1"), can_administer=True)
    other = Scope("acme", "agent-b", Principal("u2"), owner=Principal("u2"), can_administer=True)
    calls = []

    def read(command, *, scope, **kwargs):
        calls.append(scope)
        return {"output": "remote bytes", "returncode": 0}

    inner = SimpleNamespace(cwd="/workspace", timeout=10, _remote_home="/workspace",
                            execute=read)
    own_env = ScopedEnvironment(inner, sandbox_identity(owner), owner=owner)
    foreign_env = ScopedEnvironment(inner, sandbox_identity(other), owner=other)
    key = sandbox_identity(owner).key
    monkeypatch.setattr(terminal_tool, "_active_environments", {key: own_env})
    monkeypatch.setattr(terminal_tool, "ensure_task_env", lambda *a, **k: pytest.fail("created environment"))
    with bind_state(owner):
        fs = scoped_session_fs({"id": "durable"}, "test_remote")
        assert fs._execute("read protocol")["output"] == "remote bytes"
        assert calls == [owner]
        terminal_tool._active_environments[key] = foreign_env
        with pytest.raises(CrossAgentHandleRefused):
            scoped_session_fs({"id": "durable"}, "test_remote")
        assert calls == [owner]
