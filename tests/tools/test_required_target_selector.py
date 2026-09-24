"""Provider execution requires an explicit read-only selection contract."""
import json

import pytest


@pytest.mark.parametrize("kind", ["cua", "terminal"])
@pytest.mark.parametrize("direct", [False, True])
def test_missing_selector_refuses_without_resolver_or_compute(monkeypatch, kind, direct):
    from hermes_cli import session_execution as execution
    from tools.computer_use import targets, tool
    from tools import terminal_targets, terminal_tool
    from tools.registry import registry
    import tools.computer_use_tool

    execution.register_session_execution_context("selector-required", execution.SessionExecutionContext())
    lease = execution.resolve_session_execution_context(session_id="selector-required")
    calls = []
    def allocating(**kw):
        calls.append(kw)
        return lease
    monkeypatch.setattr(tool, "_new_backend", lambda *a, **kw: tool._NoopBackend())
    monkeypatch.setattr(tool, "_request_approval", lambda *_: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **kw: {"approved": True})
    register = targets.register_target_resolver if kind == "cua" else terminal_targets.register_terminal_target_resolver
    dispose = register("selector-required", allocating)
    try:
        if direct:
            with pytest.raises(execution.SessionExecutionError, match="selector"):
                if kind == "cua":
                    targets.resolve_target_context("caller")
                else:
                    terminal_targets.resolve_terminal_target("selector-required", command="true", session_id="caller")
        else:
            args = {"action": "click", "coordinate": [1, 2], "capture_after": False} if kind == "cua" else {"command": "true", "target": "selector-required"}
            result = registry.dispatch("computer_use" if kind == "cua" else "terminal", args, session_id="caller")
            result = json.loads(result) if isinstance(result, str) else result
            assert "selector" in str(result).lower(), result
        assert calls == []
    finally:
        dispose()
        execution.remove_session_execution_context("selector-required")
