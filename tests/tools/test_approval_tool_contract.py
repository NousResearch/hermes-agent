"""Approval-tool contract: a tool that declares ``preview`` + ``apply`` at registration.

A fixture plugin registers its tools through the published ``ctx.register_tool``; the calls go
through the REAL dispatcher (``model_tools.handle_function_call``) against a temp HERMES_HOME
whose config keeps the default ``approvals.scope: dangerous``. Only the human is a stand-in.
"""
import json

import pytest

import hermes_cli.config as hc
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from model_tools import handle_function_call
from tools import approval, approval_decisions
from tools.approval_context import (
    reset_current_session_key, reset_hermes_interactive_context, set_current_session_key,
    set_hermes_interactive_context,
)
from tools.registry import registry
from tools.terminal_tool import set_approval_callback

_SCHEMA = {"name": "", "description": "Send a message.", "parameters": {
    "type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}},
    "required": ["to", "text"]}}


class _Human:
    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.cards: list[str] = []

    def __call__(self, command, description, **_kwargs):
        self.cards.append(command)
        return self.answers.pop(0) if self.answers else "deny"


@pytest.fixture
def plugin(tmp_path, monkeypatch, request):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("approvals:\n  mode: manual\n", encoding="utf-8")  # scope left at its default
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    session = f"contract-{request.node.name}"
    session_token = set_current_session_key(session)
    interactive_token = set_hermes_interactive_context(True)
    scope = registry.current_scope_key()
    manager = PluginManager(scope_key=scope)
    ctx = PluginContext(PluginManifest(name="contract_fixture", key="contract_fixture"), manager)
    sent: list[tuple[dict, str]] = []
    plain: list[dict] = []

    def apply(args, decision_id):
        sent.append((dict(args), decision_id))
        return json.dumps({"sent": True, "to": args["to"]})

    ctx.register_tool(name="fixture_send", toolset="contract_fixture", schema={**_SCHEMA, "name": "fixture_send"},
                      preview=lambda args: f"Send to {args['to']}:\n{args['text']}", apply=apply)
    ctx.register_tool(name="fixture_plain", toolset="contract_fixture", schema={**_SCHEMA, "name": "fixture_plain"},
                      handler=lambda args, **_: plain.append(dict(args)) or json.dumps({"sent": True}))
    yield {"sent": sent, "plain": plain, "home": home}
    manager.unload("contract_fixture")
    set_approval_callback(None)
    approval.clear_session(session)
    reset_hermes_interactive_context(interactive_token)
    reset_current_session_key(session_token)
    hc._LOAD_CONFIG_CACHE.clear()


def _call(name: str, args: dict, tool_call_id: str) -> dict:
    return json.loads(handle_function_call(name, args, task_id="contract-task", tool_call_id=tool_call_id))


def test_declared_preview_is_the_card_and_apply_runs_once_per_decision(plugin):
    human = _Human("once")
    set_approval_callback(human)
    args = {"to": "#ops", "text": "deploy finished"}

    first = _call("fixture_send", args, "call-1")
    replay = _call("fixture_send", args, "call-1")  # same tool call replayed: same decision

    assert human.cards == ["Send to #ops:\ndeploy finished"]  # asked under the default scope, once
    assert first == replay == {"sent": True, "to": "#ops"}
    assert len(plugin["sent"]) == 1 and plugin["sent"][0][0] == args
    decision_id = plugin["sent"][0][1]
    assert approval_decisions.get(decision_id)["status"] == "applied"

    denied = _Human("deny")
    set_approval_callback(denied)
    refused = _call("fixture_send", {"to": "#ops", "text": "other"}, "call-2")
    assert "error" in refused and len(plugin["sent"]) == 1


def test_apply_interrupted_mid_flight_is_unconfirmed_not_rerun(plugin):
    args = {"to": "#ops", "text": "pay invoice 7"}
    digest = approval_decisions.args_digest(args)
    decision_id = approval_decisions.decision_id_for(
        f"contract-{test_apply_interrupted_mid_flight_is_unconfirmed_not_rerun.__name__}", "call-9",
        "fixture_send", digest)
    approval_decisions.mint(decision_id, "fixture_send", digest, "user")
    with approval_decisions._connect() as con:  # a process that claimed it and died (no such pid)
        con.execute("UPDATE decisions SET status='applying', owner_pid=?, owner_start=NULL WHERE id=?",
                    (2**22 + 12345, decision_id))
    set_approval_callback(_Human())

    result = _call("fixture_send", args, "call-9")

    assert "UNCONFIRMED" in json.dumps(result) and result.get("decision_status") == "unconfirmed"
    assert plugin["sent"] == []


def test_tool_without_a_declaration_is_unchanged(plugin):
    set_approval_callback(lambda *a, **k: pytest.fail("a non-declaring tool must not prompt"))
    assert _call("fixture_plain", {"to": "#ops", "text": "hi"}, "call-3") == {"sent": True}
    assert plugin["plain"] == [{"to": "#ops", "text": "hi"}]
    assert not (plugin["home"] / "approvals").exists()
