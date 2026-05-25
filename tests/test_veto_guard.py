from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass

import pytest

from agent import veto_guard
from tools.registry import registry


@dataclass
class _FakeGuardResult:
    decision: str
    reason: str | None = None
    rule_id: str | None = None


class _FakeVetoOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeVetoClient:
    guard_calls = 0

    def __init__(self, decision: str = "allow"):
        self.decision = decision

    async def guard(self, tool_name, args, **kwargs):
        type(self).guard_calls += 1
        return _FakeGuardResult(
            decision=self.decision,
            reason="test policy",
            rule_id="test-rule",
        )


class _FakeVeto:
    decision = "allow"
    init_calls = 0
    from_rules_calls = 0

    @classmethod
    async def init(cls, options=None):
        cls.init_calls += 1
        return _FakeVetoClient(cls.decision)

    @classmethod
    def from_rules(cls, **kwargs):
        cls.from_rules_calls += 1
        return _FakeVetoClient(cls.decision)


@pytest.fixture(autouse=True)
def _reset_veto_guard(monkeypatch, tmp_path):
    veto_guard.reset_for_tests()
    _FakeVeto.decision = "allow"
    _FakeVeto.init_calls = 0
    _FakeVeto.from_rules_calls = 0
    _FakeVetoClient.guard_calls = 0

    module = types.ModuleType("veto")
    module.Veto = _FakeVeto
    module.VetoOptions = _FakeVetoOptions
    monkeypatch.setitem(sys.modules, "veto", module)
    monkeypatch.setattr(veto_guard, "_default_config_dir", lambda: tmp_path / "veto")
    monkeypatch.setattr(
        veto_guard,
        "_load_hermes_config",
        lambda: {"veto": {"enabled": True, "validation_mode": "local"}},
    )
    yield
    registry.deregister("veto_test_tool")
    veto_guard.reset_for_tests()


def _register_test_tool(calls: list[dict]):
    registry.register(
        name="veto_test_tool",
        toolset="test",
        schema={
            "name": "veto_test_tool",
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kw: calls.append(args) or json.dumps({"ok": True}),
    )


def test_registry_dispatch_blocks_denied_tool_call():
    _FakeVeto.decision = "deny"
    calls: list[dict] = []
    _register_test_tool(calls)

    result = json.loads(registry.dispatch("veto_test_tool", {"danger": True}))

    assert calls == []
    assert "Veto blocked veto_test_tool" in result["error"]
    assert "test-rule" in result["error"]
    assert _FakeVetoClient.guard_calls == 1


def test_plugin_precheck_marks_allowed_call_so_registry_does_not_double_check():
    calls: list[dict] = []
    _register_test_tool(calls)
    veto_guard.enable_for_process()

    precheck = veto_guard.precheck_tool_call("veto_test_tool", {"safe": True})
    result = json.loads(registry.dispatch("veto_test_tool", {"safe": True}))

    assert precheck.allowed is True
    assert result == {"ok": True}
    assert calls == [{"safe": True}]
    assert _FakeVetoClient.guard_calls == 1


def test_veto_guard_fail_open_allows_when_sdk_check_fails(monkeypatch):
    calls: list[dict] = []
    _register_test_tool(calls)
    monkeypatch.setattr(veto_guard, "_get_veto_client", lambda settings: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        veto_guard,
        "_load_hermes_config",
        lambda: {
            "veto": {
                "enabled": True,
                "validation_mode": "local",
                "fail_open": True,
            }
        },
    )

    result = json.loads(registry.dispatch("veto_test_tool", {"safe": True}))

    assert result == {"ok": True}
    assert calls == [{"safe": True}]
