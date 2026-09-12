"""delegate_task refuses to spawn under an authoritative runtime policy.

A child agent gets a fresh session id and a fresh task id, so neither
resolves to the parent's run lease in ``tools/mcp_tool.py`` — its MCP calls
would silently take the best-effort observer path (no trusted request
metadata, no result stop authority, no required finalization) and run on a
delegation budget outside the job's turn cap. Until nested authoritative
sub-leases exist, the spawn is refused loudly instead.
"""

import json

import tools.delegate_tool as delegate_tool
from tools.delegate_tool import delegate_task


class _StubParent:
    """Parent agent double; ``runtime_policy`` is the only field the gate reads."""

    def __init__(self, runtime_policy=None):
        self.runtime_policy = runtime_policy


def _no_children(monkeypatch):
    """Fail the test if anything tries to construct a child agent."""
    calls = []

    def _explode(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("a child agent was constructed under an active policy")

    monkeypatch.setattr(delegate_tool, "_build_child_agent", _explode)
    return calls


def test_spawn_is_refused_under_an_active_policy(monkeypatch):
    calls = _no_children(monkeypatch)

    result = delegate_task(
        goal="exfiltrate via an unidentified MCP call",
        parent_agent=_StubParent(runtime_policy="fleet-runtime"),
    )

    # Paired, explicit refusal — an ordinary tool result, so the parent turn
    # stays alternation-valid and durably settleable.
    assert "authoritative runtime policy" in json.loads(result)["error"]
    assert calls == []
    assert delegate_tool._active_subagents == {}


def test_batch_spawn_is_refused_under_an_active_policy(monkeypatch):
    calls = _no_children(monkeypatch)

    result = delegate_task(
        tasks=[{"goal": "one"}, {"goal": "two"}],
        parent_agent=_StubParent(runtime_policy="fleet-runtime"),
    )

    assert "authoritative runtime policy" in json.loads(result)["error"]
    assert calls == []
    assert delegate_tool._active_subagents == {}


def test_control_actions_still_work_under_an_active_policy(monkeypatch):
    """list/steer/stop never create a child, so the gate must not block them."""
    _no_children(monkeypatch)

    result = delegate_task(
        action="list", parent_agent=_StubParent(runtime_policy="fleet-runtime")
    )

    assert "authoritative runtime policy" not in result


def test_ordinary_delegation_is_unchanged(monkeypatch):
    """Without a policy the call falls through to the next gate as before."""
    monkeypatch.setattr(delegate_tool, "is_spawn_paused", lambda: True)

    result = delegate_task(goal="ordinary work", parent_agent=_StubParent())

    # Reached the pause gate below the policy gate, not the refusal above it.
    assert "paused" in json.loads(result)["error"]
