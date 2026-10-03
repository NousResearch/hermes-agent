"""Selected approval transport must serve protected instruction-file writes
and plugin-escalated tool calls (#120859).

A write to AGENTS.md/CLAUDE.md/SOUL.md/.cursorrules entered the human gate
with ``transport=False`` hard-coded (and the standalone protected-write guard
never consulted the transport at all), so an operator-selected transport was
silently bypassed and the request stalled on a built-in surface nobody watches.
"""

import json

import pytest

import tools.approval as approval
import tools.approval_context as approval_context
import tools.approval_prompt as approval_prompt
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _manager_with_transport(seen):
    manager = PluginManager()
    manifest = PluginManifest(
        name="fixture-approval", version="1.0.0", description="fixture",
        source="user", key="fixture-approval",
    )

    def present(request):
        seen.append(request)
        return request.respond("once")

    PluginContext(manifest, manager).register_approval_transport("phone", present)
    return manager


@pytest.fixture
def _transport_selected(monkeypatch):
    seen = []
    manager = _manager_with_transport(seen)
    monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(
        approval_context, "_get_approval_transport_config", lambda: ("phone", None))
    return seen


def _builtin_must_not_materialize(*args, **kwargs):
    raise AssertionError("builtin prompt must not materialize when a transport is selected")


class TestProtectedWriteTransport:
    @pytest.fixture(autouse=True)
    def _gate_on(self, monkeypatch):
        import tools.file_tools_write_guards as ft
        monkeypatch.setattr(ft, "_protected_instruction_config", lambda: (True, []))
        yield

    def test_protected_write_routes_through_selected_transport(
        self, tmp_path, monkeypatch, _transport_selected,
    ):
        """write_file_tool(AGENTS.md) is answered by the selected transport."""
        from tools.file_tools import write_file_tool
        from tools.terminal_tool import set_approval_callback

        seen = _transport_selected
        set_approval_callback(_builtin_must_not_materialize)
        try:
            target = tmp_path / "AGENTS.md"
            res = json.loads(write_file_tool(str(target), "approved content"))
        finally:
            set_approval_callback(None)
        assert not res.get("error"), res
        assert target.read_text(encoding="utf-8") == "approved content"
        assert len(seen) == 1
        assert seen[0].pattern_key == "protected_instruction_file"

    def test_protected_write_transport_deny_blocks(
        self, tmp_path, monkeypatch, _transport_selected,
    ):
        """A transport 'deny' still fails the protected write closed."""
        import tools.file_tools_write_guards as ft
        from tools.terminal_tool import set_approval_callback

        manager = PluginManager()
        manifest = PluginManifest(
            name="fixture-deny", version="1.0.0", description="fixture",
            source="user", key="fixture-deny",
        )
        PluginContext(manifest, manager).register_approval_transport(
            "phone", lambda request: request.respond("deny"))
        monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
        set_approval_callback(_builtin_must_not_materialize)
        try:
            target = tmp_path / "AGENTS.md"
            res = ft._check_protected_instruction_write([str(target)])
        finally:
            set_approval_callback(None)
        assert res is not None and "BLOCKED" in res
        assert not target.exists()

    def test_protected_write_transport_consult_failure_blocks(
        self, tmp_path, monkeypatch, _transport_selected,
    ):
        """A transport that cannot be consulted fails the write CLOSED.

        Falling back to the built-in prompt on a transport error would
        reintroduce #120859: the operator selected a surface and is not watching
        the built-in one, so silence there is not consent.
        """
        import tools.file_tools_write_guards as ft
        from tools.terminal_tool import set_approval_callback

        def boom(*args, **kwargs):
            raise RuntimeError("transport exploded")

        monkeypatch.setattr(approval_prompt, "_present_with_selected_transport", boom)
        set_approval_callback(_builtin_must_not_materialize)
        try:
            target = tmp_path / "AGENTS.md"
            res = ft._check_protected_instruction_write([str(target)])
        finally:
            set_approval_callback(None)
        assert res is not None and "BLOCKED" in res, res
        assert not target.exists()


class TestActionGateTransport:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        monkeypatch.setattr(
            approval, "get_current_session_key", lambda default="default": "test-session")
        monkeypatch.setattr(
            approval_context, "get_current_session_key",
            lambda default="default": "test-session")
        monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
        monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval_context, "_is_gateway_approval_context", lambda: False)
        yield

    def test_plugin_escalated_call_routes_through_selected_transport(
        self, monkeypatch, _transport_selected,
    ):
        """request_tool_approval honors the operator-selected transport."""
        seen = _transport_selected
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", _builtin_must_not_materialize)
        monkeypatch.setattr(
            approval_prompt, "prompt_dangerous_approval", _builtin_must_not_materialize)
        res = approval.request_tool_approval("write_file", "sensitive path", rule_key="ssh")
        assert res["approved"] is True
        assert len(seen) == 1
