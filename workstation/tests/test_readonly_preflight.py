"""Incident regression: twelve card descriptions, no live provider or browser."""
import json
import sqlite3
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.effects import ToolEffect, capability_key, observe_capability, operation_capability, tool_effect
from tools.registry import registry
from workstation.artifacts import ArtifactStore
from workstation.batch_detection import detects_fan_out, mutation_summary, record_mutation
from workstation.durable_tasks import DurableTaskStore
from workstation.task_compiler import TaskCompiler, execution_context, execute_compiled_work, requires_compilation


def call(name, args, ident="call"):
    return SimpleNamespace(id=ident, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


@pytest.fixture
def incident(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(registry, "_tools", dict(registry._tools))
    for name, effect, method, channel in (("incident_inspect", ToolEffect.DISCOVERY, "GET", "api"),
                                         ("incident_get", ToolEffect.PURE_READ, "GET", "api"),
                                         ("incident_put", ToolEffect.MUTATION, "PUT", "api"),
                                         ("incident_ui", ToolEffect.MUTATION, "UPDATE", "ui")):
        registry.register(name, "incident", {"name": name, "parameters": {"type": "object"},
            "capability": {"provider": "fake_cards", "channel": channel, "method": method},
            "mutation_target": {"provider": "fake_cards", "scope": "external", "kind": "card",
                                "identifier_field": "id", "field": "description", "operation": "description.update",
                                "identity_fields": ["id", "description"]}},
            lambda **kw: "{}", effect=effect, routes=["fake_cards." + channel])
    database = tmp_path / "kanban.db"
    compiler = TaskCompiler(DurableTaskStore(conn=sqlite3.connect(database)), ArtifactStore())
    state, writes, sequence = {}, Counter(), []
    environment = {"selector": None}
    def dispatch(name, args, *_):
        sequence.append(name)
        if name == "incident_inspect":
            environment["selector"] = "textarea.description"
            return {"selector": "textarea.description", "status_code": 200}
        if name == "incident_ui":
            assert environment["selector"] is not None and args["selector"] == environment["selector"]
            state[args["id"]] = args["description"]
            writes[args["id"]] += 1
            return {"ok": True, "status_code": 200}
        if name == "incident_get":
            return {"description": state.get(args["id"]), "status_code": 200}
        return {"error": "CSRF", "status_code": 403}
    req = {"operation_key": "twelve-descriptions", "items": [{"id": str(i), "description": f"Description {i}"} for i in range(12)],
           "mutation_target": {"scope": "external", "provider": "fake_cards", "kind": "card", "field": "description"},
           "constraints": {"mutation_forbidden_routes": ["fake_cards.api"]},
           "setup_steps": [{"id": "inspect", "tool": "incident_inspect", "args": {}}],
           "steps": [{"id": "write", "tool": "incident_ui", "args": {"id": "$item.id", "description": "$item.description", "selector": "$setup.inspect.selector"},
                      "expect": {"ok": True}},
                     {"id": "readback", "tool": "incident_get", "args": {"id": "$item.id"},
                      "verifies": ["write"], "expect": {"description": "$item.description"}}]}
    yield compiler, dispatch, req, state, writes, sequence, database
    compiler.store.close()


def test_twelve_cards_discover_canary_readback_checkpoint_restart(incident):
    compiler, dispatch, req, state, writes, sequence, database = incident
    preflight = compiler.discover({"preflight": [{"tool": "incident_inspect", "args": {}}]},
                                 task_id="task", session_id="owner", dispatch=dispatch)
    assert preflight["code"] == "PREFLIGHT_COMPLETE" and not preflight["plan_frozen"]
    assert compiler.store.get_connection().execute("SELECT count(*) FROM work_plans").fetchone()[0] == 0
    def interrupt():
        if len(writes) == 4:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        compiler.execute(req, task_id="task", session_id="owner", dispatch=dispatch, progress=interrupt)
    compiler.store.close()
    restored = TaskCompiler(DurableTaskStore(conn=sqlite3.connect(database)), compiler.artifacts)
    try:
        result = restored.execute(req, task_id="task", session_id="owner", dispatch=dispatch)
        assert result["completed"] == 12 and result["ledger"]["canary"]["verified"]
        assert writes == Counter({str(i): 1 for i in range(12)})
        assert [name for name in sequence if name != "incident_inspect"][:4] == ["incident_ui", "incident_get", "incident_ui", "incident_get"]
        for item in restored.store.get_work_items(result["plan_id"]):
            if item.input_payload.get("_work_phase") != "fan_out":
                continue
            identity = item.checkpoints["step_0_meta"]["mutation_identity"]
            assert identity["persisted"] and identity["verifier_status"] == "verified"
            assert identity["target_identifier"] == item.input_payload["id"]
    finally:
        restored.store.close()


def test_missing_verifier_rejected_before_canary(incident):
    compiler, _, req, *_ = incident
    req["steps"] = req["steps"][:1]
    with pytest.raises(ValueError, match="read verifier"):
        compiler.execute(req, task_id="task", session_id="owner", dispatch=lambda *a: pytest.fail("dispatched"))


@pytest.mark.parametrize("expected", [{"ok": True}, {"result.exit_code": 0}])
def test_preparation_success_cannot_admit_persistence_verifier(incident, expected):
    compiler, _, req, *_ = incident
    req["steps"][1]["expect"] = expected
    with pytest.raises(ValueError, match="persisted resource"):
        compiler.execute(req, task_id="task", session_id="owner", dispatch=lambda *a: pytest.fail("dispatched"))


def test_readback_can_map_different_serialized_field(incident):
    compiler, dispatch, req, *_ = incident
    req["steps"][1].update(readback={"field": "description", "path": "data.text"}, expect={"data.text": "$item.description"})
    def mapped(name, args, *rest):
        result = dispatch(name, args, *rest)
        return {"data": {"text": result["description"]}} if name == "incident_get" else result
    result = compiler.execute(req, task_id="task", session_id="owner", dispatch=mapped)
    assert result["completed"] == 12


def test_external_transaction_cannot_be_replaced_by_script_preparation(incident):
    compiler, _, req, *_ = incident
    req["steps"] = [{"id": "script", "tool": "terminal", "args": {"command": "generate scripts"}, "expect": {"exit_code": 0}},
                    {"id": "verify", "tool": "read_file", "args": {"path": "scratch.js"}, "verifies": ["script"], "expect": {"content": "OK"}}]
    with pytest.raises(ValueError, match="real mutation"):
        compiler.execute(req, task_id="task", session_id="owner", dispatch=lambda *a: pytest.fail("preparation canary"))


def test_opaque_preparation_cannot_omit_intended_target(incident):
    compiler, _, req, *_ = incident
    req.pop("mutation_target")
    req.pop("setup_steps")
    req["steps"] = [{"id": "script", "tool": "terminal", "args": {"command": "generate scripts"}, "expect": {"exit_code": 0}},
                    {"id": "verify", "tool": "read_file", "args": {"path": "scratch.js"}, "verifies": ["script"], "expect": {"content": "OK"}}]
    with pytest.raises(ValueError, match="preflight_required"):
        compiler.execute(req, task_id="task", session_id="owner", dispatch=lambda *a: pytest.fail("ambiguous canary"))


def test_recipe_keeps_intended_external_target(incident):
    compiler, dispatch, req, *_ = incident
    req["recipe_key"] = "incident.descriptions.v1"
    compiler.execute(req, task_id="task", session_id="owner", dispatch=dispatch)
    assert compiler.recipes.get(req["recipe_key"])["mutation_target"] == req["mutation_target"]
    with pytest.raises(ValueError, match="mutation target differs"):
        compiler.execute({"recipe_key": req["recipe_key"], "items": [{"id": "other"}],
                          "mutation_target": {"scope": "local"}}, task_id="task", session_id="owner",
                         dispatch=lambda *a: pytest.fail("changed target"))


def test_semantic_replay_identity_ignores_incidental_selector(incident):
    from workstation.batch_detection import call_key
    args = {"id": "card", "description": "expected"}
    assert call_key("incident_ui", {**args, "selector": "old"}) == call_key("incident_ui", {**args, "selector": "new"})
    assert call_key("incident_ui", args) != call_key("incident_ui", {**args, "description": "new update"})


def test_unpersisted_canary_blocks_eleven(incident):
    compiler, dispatch, req, state, writes, *_ = incident
    def lost_write(name, args, *rest):
        result = dispatch(name, args, *rest)
        if name == "incident_ui":
            state.clear()
        return result
    result = compiler.execute(req, task_id="task", session_id="owner", dispatch=lost_write)
    assert sum(writes.values()) == 1 and result["code"] == "canary_failed"
    assert not result["ledger"]["canary"]["verified"]


def test_repeated_read_discovery_not_counted_and_loose_writes_blocked(incident):
    agent = SimpleNamespace(valid_tool_names={"work_execute", "incident_inspect", "incident_get", "incident_ui"}, _work_batch_candidate=True)
    for name in ("incident_inspect", "incident_get"):
        calls = [call(name, {"id": str(i)}) for i in range(12)]
        assert not detects_fan_out(agent, calls) and not requires_compilation(agent, calls)
    writes = [call("incident_ui", {"id": str(i)}) for i in range(12)]
    assert detects_fan_out(agent, writes) and requires_compilation(agent, writes)


def test_uncertain_dispatched_effects_still_count_toward_fanout(incident):
    agent = SimpleNamespace(valid_tool_names={"work_execute"}, session_id="owner", _conversation_root_id=lambda: "owner")
    for i in range(2):
        record_mutation(agent, "incident_ui", {"id": str(i)}, '{"error":"connection lost"}')
    assert detects_fan_out(agent, [call("incident_ui", {"id": "2"})])
    assert len(agent._work_mutation_evidence) == 2


@pytest.mark.parametrize("name", ["browser_console", "terminal", "browser_exec"])
def test_arbitrary_execution_cannot_claim_discovery(incident, name):
    compiler, *_ = incident
    result = compiler.discover({"preflight": [{"tool": name, "effect": "PURE_READ",
                             "args": {"code": "fetch('/cards', {method:'PUT'})"}}]},
        task_id="task", session_id="owner", dispatch=lambda *a: pytest.fail("escape hatch"))
    assert result["code"] == "GUARD_BOOTSTRAP_BLOCKED" and tool_effect(name) == ToolEffect.MUTATION


def test_uncertain_mutation_never_retried(incident):
    compiler, dispatch, req, _, writes, *_ = incident
    def crash(name, args, *rest):
        result = dispatch(name, args, *rest)
        if name == "incident_ui":
            raise KeyboardInterrupt()
        return result
    with pytest.raises(KeyboardInterrupt):
        compiler.execute(req, task_id="task", session_id="owner", dispatch=crash)
    result = compiler.execute(req, task_id="task", session_id="owner", dispatch=dispatch)
    assert sum(writes.values()) == 1
    assert result["ledger"]["canary"]["status"] == "uncertain"


def test_get_200_never_authorizes_put_and_put_403_never_invalidates_get(incident):
    capabilities = {}
    observe_capability(capabilities, "incident_get", {}, {"status_code": 200})
    assert operation_capability(capabilities, "incident_get", {}) == "VERIFIED"
    assert operation_capability(capabilities, "incident_put", {}) == "UNKNOWN"
    observe_capability(capabilities, "incident_put", {}, {"status_code": 403, "error": "CSRF"})
    observe_capability(capabilities, "incident_get", {}, {"status_code": 200})
    assert operation_capability(capabilities, "incident_put", {}) == "REJECTED"
    assert operation_capability(capabilities, "incident_get", {}) == "VERIFIED"
    assert capabilities[capability_key("fake_cards", "api", "PUT")]["http_status"] == 403


def test_mutation_channel_constraint_does_not_restrict_api_readback(incident):
    from agent.turn_constraints import user_constraints
    from workstation.task_compiler import merge_constraints
    policy = user_constraints('mutation_forbidden_routes=["fake_cards.api"]')
    assert merge_constraints(policy, {})["mutation_forbidden_routes"] == ["fake_cards.api"]
    compiler, dispatch, req, *_ = incident
    result = compiler.discover({"preflight": [{"tool": "incident_get", "args": {"id": "0"}}], "constraints": policy},
        task_id="task", session_id="owner", dispatch=dispatch)
    assert result["code"] == "PREFLIGHT_COMPLETE"
    req["steps"][0]["tool"] = "incident_put"
    from workstation.routing import ConstraintViolation
    with pytest.raises(ConstraintViolation):
        compiler.execute(req, task_id="task", session_id="owner", dispatch=lambda *a: pytest.fail("forbidden API write"))


def test_historical_seven_effects_do_not_claim_seven_persisted_cards(incident):
    agent = SimpleNamespace(valid_tool_names={"work_execute"}, session_id="owner", _conversation_root_id=lambda: "owner")
    for i in range(6):
        record_mutation(agent, "write_file", {"path": f"scratch/{i}.json"}, '{"ok":true}')
    record_mutation(agent, "incident_ui", {"id": "historical-card"}, '{"ok":true}')
    result = mutation_summary(agent)
    assert result["completed_mutation_count"] == 7
    assert result["effect_summary"] == {"local_mutation": 6, "external_mutation": 1}
    external = [r for r in result["mutation_identities"] if r["external"]]
    assert len(external) == 1 and external[0]["target_identifier"] == "historical-card"
    assert external[0]["persisted"] is None


def test_repeated_compile_refusal_keeps_agent_discovery_available(incident):
    from workstation.tests.test_durable_agent_integration import make_agent
    agent = make_agent()
    agent.valid_tool_names.update({"incident_inspect", "incident_ui"})
    from workstation.batch_detection import structural_signature
    agent._work_mutation_shapes = {structural_signature('incident_ui', {'id': '0'}): 2}
    messages = []
    def handler(name, args, task, **kw):
        return execute_compiled_work(args, task_id=task) if name == "work_execute" else '{"selector":"textarea.description"}'
    with patch("run_agent.handle_function_call", side_effect=handler):
        for i in range(3):
            agent._execute_tool_calls(SimpleNamespace(tool_calls=[call("incident_ui", {"id": str(i)})]), messages, "task")
        assert agent._tool_guardrail_halt_decision is None
        agent._execute_tool_calls(SimpleNamespace(tool_calls=[call("incident_inspect", {})]), messages, "task")
    assert json.loads(messages[-1]["content"])["selector"] == "textarea.description"
    assert json.loads(messages[0]["content"])["preflight_status"] == "PREFLIGHT_REQUIRED"


def test_discovery_action_uses_real_scoped_dispatch_without_freezing(incident):
    compiler, dispatch, *_ = incident
    with execution_context(dispatch, "owner"):
        from workstation.task_compiler import _execute_compiled_work, _dispatch_context
        result = json.loads(_execute_compiled_work(compiler, _dispatch_context.get(),
            {"action": "discover", "preflight": [{"tool": "incident_inspect", "args": {}}]}, {"task_id": "task"}))
    assert result["code"] == "PREFLIGHT_COMPLETE" and not result["plan_frozen"]
