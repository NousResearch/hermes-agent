"""The shared action gate presents through the selected approval transport.

``security.approval.transport`` replaces every built-in prompt surface: without the explicit
``transport_fallback: builtin`` opt-in, Hermes never materializes the prompt on another surface
(website/docs/user-guide/features/plugins.md, "Approval transports"). The terminal and execute_code
gates honour that, but ``_run_approval_gate`` — plugin ``pre_tool_call`` ``approve`` escalations,
computer_use, SSH-config writes and ``check_dangerous_command`` — still painted the built-in
CLI/gateway prompt, on the surface the operator routed approvals away from. Regression for #133946.

Real discovery, config and dispatch against a temp HERMES_HOME (AGENTS.md: E2E with real imports).
"""

import json
import os

import pytest

import hermes_yaml as yaml

_PLUGIN = '''\
import json
import os
from pathlib import Path


def _home():
    return Path(os.environ["HERMES_HOME"])


def pre_tool_call(tool_name, **kwargs):
    if tool_name == "fixture_tool":
        return {"action": "approve", "message": "fixture_tool needs a human", "rule_key": "fixture-rule"}
    return None


def present(request):
    with (_home() / "transport-requests.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pattern_key": request.pattern_key, "surface": request.surface}) + "\\n")
    return request.respond((_home() / "transport-choice").read_text(encoding="utf-8").strip())


def register(ctx):
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_approval_transport("fixture", present)
'''


@pytest.fixture
def transport_home(tmp_path, monkeypatch):
    """A HERMES_HOME whose only enabled plugin escalates ``fixture_tool`` and is the selected transport.

    Yields ``(home, builtin)``: ``builtin`` records every built-in CLI prompt or gateway notification.
    """
    import hermes_cli.plugins as plugins_module
    from tools import approval, approval_context
    from tools.terminal_tool import set_approval_callback

    home = tmp_path / "hermes-home"
    plugin_dir = home / "plugins" / "fixture-escalation"
    bundled = tmp_path / "empty-bundled"
    plugin_dir.mkdir(parents=True)
    bundled.mkdir()
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(
        {"name": "fixture-escalation", "version": "1.0.0", "description": "escalation + transport fixture"}),
        encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["fixture-escalation"]},
        "approvals": {"mode": "manual", "timeout": 2},
        "security": {"approval": {"transport": "fixture"}},
    }), encoding="utf-8")
    (home / "transport-choice").write_text("once", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(plugins_module, "_plugin_manager", plugins_module.PluginManager())

    builtin = []
    session_key = approval_context.get_current_session_key()
    set_approval_callback(lambda *args, **kwargs: builtin.append("cli") or "once")
    approval.register_gateway_notify(session_key, lambda data: builtin.append("gateway"))
    token = approval_context.set_hermes_interactive_context(True)
    try:
        yield home, builtin
    finally:
        approval_context.reset_hermes_interactive_context(token)
        approval.unregister_gateway_notify(session_key)
        set_approval_callback(None)


def _transport_requests(home):
    log = home / "transport-requests.jsonl"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


def test_plugin_escalation_asks_through_the_selected_transport(transport_home, monkeypatch):
    """A plugin ``approve`` escalation reaches the selected transport on the CLI and gateway surfaces, never a
    built-in prompt, and a transport deny comes back to the model as the user's refusal."""
    from hermes_cli.plugins import resolve_pre_tool_block
    from tools import approval_context

    home, builtin = transport_home
    approved_on_cli = resolve_pre_tool_block("fixture_tool", {})
    (home / "transport-choice").write_text("deny", encoding="utf-8")
    denied_on_cli = resolve_pre_tool_block("fixture_tool", {})

    (home / "transport-choice").write_text("once", encoding="utf-8")
    token = approval_context.set_hermes_interactive_context(False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    try:
        approved_on_gateway = resolve_pre_tool_block("fixture_tool", {})
    finally:
        approval_context.reset_hermes_interactive_context(token)

    assert builtin == []
    assert [(r["pattern_key"], r["surface"]) for r in _transport_requests(home)] == [
        ("plugin_rule:fixture-rule", "cli"), ("plugin_rule:fixture-rule", "cli"),
        ("plugin_rule:fixture-rule", "gateway"),
    ]
    assert approved_on_cli is None
    assert approved_on_gateway is None
    assert denied_on_cli and "denied" in denied_on_cli.lower()


def _computer_use_click():
    from tools.computer_use.tool import _request_approval
    return _request_approval("click", {}) is None


def _ssh_config_write():
    from tools.file_tools_write_guards import _check_approval_required_write
    return _check_approval_required_write([os.path.expanduser("~/.ssh/config")]) is None


def _dangerous_command():
    from tools.approval import check_dangerous_command
    return check_dangerous_command("rm -rf /tmp/hermes-transport-fixture", "local")["approved"]


@pytest.mark.parametrize("approve", [_computer_use_click, _ssh_config_write, _dangerous_command],
                         ids=["computer_use", "ssh_config_write", "check_dangerous_command"])
def test_every_action_gate_caller_asks_through_the_selected_transport(transport_home, approve):
    """The other ``_run_approval_gate`` callers share the plugin escalation's gate, so they share its surface."""
    home, builtin = transport_home

    assert approve() is True
    assert builtin == []
    assert [r["surface"] for r in _transport_requests(home)] == ["cli"]
