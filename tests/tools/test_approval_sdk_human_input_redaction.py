"""SDK-surface redaction extends to the human-input observers around the gateway approval wait.

``_await_gateway_decision`` wraps every wait in ``human_input_request`` (``on_human_input_request`` /
``on_human_input_resolved``). On ``surface="claude_sdk"`` the approval payload is hostile SDK data, so a
failing observer logs only the fixed SDK line, never its exception text, while the hook pair still fires;
every other surface keeps the normal failure report.
"""

import logging

import pytest

from hermes_cli import plugins
from tools import approval as mod
from tools import approval_context as ctx
from tools import approval_gateway_wait as wait_mod

MARKER = "SDK_HUMAN_INPUT_OBSERVER_SECRET_65982"
APPROVAL = {"command": "Bash(command=true)", "description": "d", "pattern_key": "claude_sdk_tool",
            "pattern_keys": ["claude_sdk_tool"], "no_coalesce": True}


@pytest.fixture
def broken_observers(monkeypatch):
    """Both human-input hooks registered on a fresh manager; each records its call, then raises."""
    manager = plugins.PluginManager()
    manager._discovered = True
    seen = []

    def observer(hook_name):
        def callback(**kwargs):
            seen.append((hook_name, kwargs.get("kind"), kwargs.get("outcome")))
            raise RuntimeError(f"{MARKER} prompt={kwargs.get('prompt')}")
        return callback

    for hook_name in ("on_human_input_request", "on_human_input_resolved"):
        manager._hooks.setdefault(hook_name, []).append(observer(hook_name))
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(ctx, "_fire_approval_hook", lambda name, **kw: None)
    return seen


def _decide(session_key, surface):
    """A prompt the operator answers at once: the notifier resolves it before the wait polls."""
    def notify(_data):
        mod.resolve_gateway_approval(session_key, "once")

    try:
        return wait_mod._await_gateway_decision(session_key, notify, dict(APPROVAL), surface=surface)
    finally:
        with mod._lock:
            mod._gateway_queues.pop(session_key, None)


def test_sdk_human_input_observer_failure_logs_fixed_line(broken_observers, caplog):
    with caplog.at_level(logging.DEBUG):
        decision = _decide("sdk-human-input-redaction", "claude_sdk")

    assert decision["resolved"] is True and decision["choice"] == "once"
    assert broken_observers == [("on_human_input_request", "approval", None),
                                ("on_human_input_resolved", "approval", "once")]
    assert MARKER not in caplog.text
    assert not any(record.exc_info for record in caplog.records)
    fixed = [record for record in caplog.records if record.getMessage() == ctx.SDK_HUMAN_INPUT_OBSERVER_FAILURE_LOG]
    assert len(fixed) == 2


def test_gateway_surface_keeps_the_normal_failure_report(broken_observers, caplog):
    with caplog.at_level(logging.DEBUG):
        decision = _decide("gateway-human-input-report", "gateway")

    assert decision["choice"] == "once"
    assert len(broken_observers) == 2
    assert MARKER in caplog.text
    assert ctx.SDK_HUMAN_INPUT_OBSERVER_FAILURE_LOG not in caplog.text
