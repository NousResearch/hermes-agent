import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.registry import registry
from tools.effects import ToolEffect, tool_effect
from workstation.artifacts import ArtifactStore
from workstation.durable_tasks import DurableTaskStore
from workstation.task_compiler import TaskCompiler, batch_intent, requires_compilation
from agent.turn_constraints import TurnConstraintContext, guard_provider_call
from workstation.routing import ConstraintViolation
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(registry, "_tools", dict(registry._tools))
    for name, effect in (("trello_search_board", ToolEffect.DISCOVERY),
                         ("trello_list_columns", ToolEffect.PURE_READ),
                         ("trello_search_cards", ToolEffect.DISCOVERY),
                         ("trello_verify_card", ToolEffect.PURE_READ),
                         ("trello_create_card", ToolEffect.MUTATION)):
        registry.register(name, "test", {"name": name}, lambda **kw: "{}", effect=effect)
    return TaskCompiler(DurableTaskStore(conn=sqlite3.connect(tmp_path / "kanban.db")), ArtifactStore())


def call(name, args, call_id="test"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def graph_request(n=100):
    return {"operation_key": "graph", "items": [{"id": i} for i in range(n)],
            "setup_steps": [{"id": "resolve_board", "tool": "trello_search_board", "args": {"name": "ACIRV"}}],
            "steps": [{"id": "create", "tool": "trello_create_card", "args": {"board": "$setup.resolve_board.board_id", "id": "$item.id"}, "expect": {"ok": True}},
                      {"id": "verify", "tool": "trello_verify_card", "verifies": ["create"], "args": {"id": "$item.id"}, "expect": {"id": "$item.id"}}],
            "finalize_steps": [{"id": "summary", "depends_on": ["create"], "tool": "trello_search_cards", "args": {"results": "$items_ref"}}]}


def execute(compiler, request, dispatch):
    return compiler.execute(request, task_id="task", session_id="session", dispatch=dispatch)


def test_discovery_before_compilation(compiler):
    agent = SimpleNamespace(_work_batch_candidate=batch_intent("Crie 12 cards no board X."), valid_tool_names={"work_execute", "tool_call"})
    reads = [call(name, {}) for name in ("trello_search_board", "trello_list_columns", "trello_search_cards")]
    assert not requires_compilation(agent, reads)
    assert not requires_compilation(agent, [call("tool_call", {"name": "trello_list_columns", "arguments": {}})])
    assert requires_compilation(agent, [call("trello_create_card", {"id": i}) for i in range(12)])
    assert not requires_compilation(agent, [call("clarify", {})])
    # An unrelated cognitive operation cannot inherit another candidate's gate.
    assert not requires_compilation(agent, [call("delegate_task", {})])


@pytest.mark.parametrize("effect", list(ToolEffect))
def test_taxonomy_metadata(compiler, effect):
    registry.register("future_connector", "test", {}, lambda: "{}", effect=effect, idempotency_key="operation_id")
    assert tool_effect("future_connector") == effect
    assert tool_effect("unknown_get_data") == ToolEffect.MUTATION
    assert tool_effect("mcp_read", schema={"annotations": {"readOnlyHint": True}}) == ToolEffect.PURE_READ
    registry.register("unsafe_idempotent", "test", {}, lambda: "{}", effect=ToolEffect.IDEMPOTENT_WRITE)
    assert tool_effect("unsafe_idempotent") == ToolEffect.MUTATION


def test_graph_setup_once_finalize_after_items_reference_first(compiler):
    calls = []
    def dispatch(name, args, task, cid):
        calls.append(name)
        if name == "trello_search_board":
            return {"board_id": "B", "large": "z" * 50000}
        if name == "trello_create_card":
            assert args["board"] == "B"
            return {"ok": True, "large": "x" * 10000}
        if name == "trello_verify_card":
            return {"id": args["id"]}
        assert calls.count("trello_create_card") == 100
        assert args["results"].startswith("artifact://")
        return {"ok": True}
    result = execute(compiler, graph_request(), dispatch)
    assert calls.count("trello_search_board") == 1
    assert calls.count("trello_create_card") == 100
    assert calls.count("trello_search_cards") == 1
    assert result["completed"] == 100
    assert result["ledger"]["setup"]["completed"] == 1
    assert result["ledger"]["finalize"]["completed"] == 1
    assert result["ledger"]["items"]["completed"] == 100
    assert "x" * 100 not in json.dumps(result)
    assert len(json.dumps(result)) < 6000
    execute(compiler, graph_request(), lambda *a: pytest.fail("replay"))


@pytest.mark.parametrize("crash_at", ["setup", 37])
def test_restart_shared_and_fan_out(compiler, crash_at):
    request = graph_request()
    if crash_at == "setup":
        request["setup_steps"].append({"id": "second", "depends_on": ["resolve_board"], "tool": "trello_list_columns", "args": {}})
    writes, reads = [], []
    def crash(name, args, task, cid):
        if name == "trello_search_board":
            reads.append(name)
            return {"board_id": "B"}
        if crash_at == "setup" and name == "trello_list_columns":
            raise KeyboardInterrupt()
        if name == "trello_create_card":
            writes.append(args["id"])
        if name == "trello_verify_card":
            return {"id": args["id"]}
        return {"ok": True}
    with pytest.raises(KeyboardInterrupt):
        if crash_at == 37:
            def progress():
                if len(writes) == 37:
                    raise KeyboardInterrupt()
            compiler.execute(request, task_id="task", session_id="session", dispatch=crash, progress=progress)
        else:
            execute(compiler, request, crash)
    def resume(name, args, *rest):
        if name == "trello_create_card":
            writes.append(args["id"])
        if name == "trello_verify_card":
            return {"id": args["id"]}
        return {"ok": True, "board_id": "B"}
    result = execute(compiler, request, resume)
    assert len(reads) == 1
    if crash_at == "setup":
        assert writes == list(range(100))
    else:
        assert writes == list(range(100))
        assert result["ledger"]["uncertain"] == 0
        assert result["ledger"]["finalize"]["completed"] == 1


def test_uncertain_shared_mutation_never_replayed(compiler):
    req = graph_request(2)
    req["setup_steps"] = [{"id": "resolve_board", "tool": "trello_create_card", "args": {}, "expect": {"ok": True}}]
    calls = []
    def crash(*args):
        calls.append(args)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        execute(compiler, req, crash)
    result = execute(compiler, req, lambda *a: pytest.fail("uncertain replay"))
    assert len(calls) == 1
    assert result["anomalies"][0]["reason"] == "uncertain_mutation_requires_review"
    assert result["ledger"]["setup"]["uncertain"] == 1
    assert result["ledger"]["items"]["completed"] == 0


@pytest.mark.parametrize("deps", [(["B"], ["A"]), (["missing"], []), (["summary"], [])])
def test_dag_rejects_cycle_missing_impossible_before_dispatch(compiler, deps):
    req = graph_request(2)
    req["setup_steps"] = [{"id": name, "tool": "trello_search_board", "args": {}, "depends_on": dep} for name, dep in zip(("A", "B"), deps)]
    with pytest.raises(ValueError):
        execute(compiler, req, lambda *a: pytest.fail("invalid DAG dispatched"))
    assert compiler.store.get_connection().execute("SELECT COUNT(*) FROM work_plans").fetchone()[0] == 0


@pytest.mark.parametrize("prompt", ["crie 12 cartões", "publique 12 posts", "envie 30 mensagens", "adicione 40 participantes", "cadastre 100 leads", "responda essas 20 pessoas", "processe todos esses arquivos", "aplique isso em cada registro", "faça a mesma alteração nesses usuários"])
def test_varied_batch_language(prompt):
    assert batch_intent(prompt)


def test_dynamic_fan_out_blocks_third_and_adopts_completed(compiler):
    from workstation.tests.test_durable_agent_integration import make_agent
    from workstation.task_compiler import execute_compiled_work
    agent = make_agent()
    agent.valid_tool_names.add("trello_create_card")
    agent.valid_tool_names.add("trello_verify_card")
    writes, messages = [], []
    def handler(name, args, task, **kw):
        if name == "work_execute":
            return execute_compiled_work(args, task_id=task)
        if name == "trello_create_card":
            writes.append(args["id"])
        if name == "trello_verify_card":
            return json.dumps({"id": args["id"]})
        return json.dumps({"ok": True})
    with patch("run_agent.handle_function_call", side_effect=handler):
        for i in range(3):
            agent._execute_tool_calls(SimpleNamespace(tool_calls=[call("trello_create_card", {"id": i}, str(i))]), messages, "task")
        assert writes == [0, 1]
        assert json.loads(messages[-1]["content"])["code"] == "durable_compile_required"
        # Deliberately include all items: adopted verified results prevent replay.
        req = {"operation_key": "remaining", "items": [{"id": i} for i in range(5)], "steps": [{"id": "create", "tool": "trello_create_card", "args": {"id": "$item.id"}, "expect": {"ok": True}},
               {"id": "verify", "tool": "trello_verify_card", "verifies": ["create"], "args": {"id": "$item.id"}, "expect": {"id": "$item.id"}}]}
        agent._execute_tool_calls(SimpleNamespace(tool_calls=[call("work_execute", req)]), messages, "task")
    assert writes == list(range(5))
    assert json.loads(messages[-1]["content"])["completed"] == 5


def test_mixed_response_discovery_is_not_blocked_with_mutations(compiler):
    from workstation.tests.test_durable_agent_integration import make_agent
    agent = make_agent()
    agent.valid_tool_names.update({"trello_search_board", "trello_create_card"})
    from workstation.batch_detection import structural_signature
    agent._work_mutation_shapes = {structural_signature('trello_create_card', {}): 2}
    messages, calls = [], []
    with patch("run_agent.handle_function_call", side_effect=lambda name, *args, **kw: calls.append(name) or '{"board_id":"B"}'):
        agent._execute_tool_calls(SimpleNamespace(tool_calls=[call("trello_search_board", {}), call("trello_create_card", {})]), messages, "task")
    assert calls == ["trello_search_board"]
    assert json.loads(messages[-1]["content"])["code"] == "durable_compile_required"


def test_provider_constraint_prunes_primary_and_fallback_before_call():
    context = TurnConstraintContext.from_user('forbidden_routes=["openai_api"]')
    agent = SimpleNamespace(provider="openai", base_url="https://api.openai.com/v1")
    def fallback():
        agent.provider, agent.base_url = "anthropic", "https://api.anthropic.com"
        return True
    agent._try_activate_fallback = fallback
    context.select_before_call(agent)
    assert agent.provider == "anthropic"
    agent._turn_constraints = context
    guard_provider_call(agent)
    agent.provider, agent.base_url = "openai", "https://api.openai.com/v1"
    with pytest.raises(ConstraintViolation):
        guard_provider_call(agent)
    agent._try_activate_fallback = lambda: False
    with pytest.raises(ConstraintViolation, match="No provider"):
        context.select_before_call(agent)


def test_no_provider_allowed_full_turn_no_network(compiler):
    from workstation.tests.test_durable_agent_integration import make_agent
    agent = make_agent()
    agent.provider = "openai"
    agent.client = MagicMock()
    agent._fallback_chain = []
    result = agent.run_conversation("Não use OpenAI API.")
    assert result["error"] == "constraint_violation"
    assert not agent.client.chat.completions.create.called


def test_guardrails_verified_edit_starts_new_experiment():
    guard = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=3))
    for _ in range(2):
        guard.after_call("terminal", {"command": "pytest"}, '{"exit_code":1}', failed=True)
    guard.after_call("patch", {}, '{"success":true}')
    assert guard.before_call("terminal", {"command": "pytest"}).action == "allow"
    for _ in range(3):
        guard.after_call("terminal", {"command": "pytest"}, '{"exit_code":1}', failed=True)
    assert guard.before_call("terminal", {"command": "pytest"}).action == "block"
    # A textual declaration does not reset evidence.
    guard.after_call("terminal", {"command": "echo"}, '{"actual_delta":true,"exit_code":0}')
    assert guard.before_call("terminal", {"command": "pytest"}).action == "block"


def test_unattended_guardrail_configuration():
    assert ToolCallGuardrailConfig.from_mapping({}, platform="telegram").hard_stop_enabled
    assert not ToolCallGuardrailConfig.from_mapping({"non_interactive_hard_stop_enabled": False}, platform="cron").hard_stop_enabled
    assert not ToolCallGuardrailConfig.from_mapping({}, platform="desktop").hard_stop_enabled


def test_finalize_checkpoint_and_failure_barrier(compiler):
    req = graph_request(3)
    req["finalize_steps"] = [
        {"id": "update_parent", "depends_on": ["create"], "tool": "trello_create_card", "args": {"parent": True}, "expect": {"ok": True}},
        {"id": "summary", "depends_on": ["update_parent"], "tool": "trello_search_cards", "args": {}}]
    parent_writes = []
    def dispatch(name, args, *rest):
        if "parent" in args:
            parent_writes.append(args)
        if name == "trello_search_cards":
            raise KeyboardInterrupt()
        if name == "trello_verify_card":
            return {"id": args["id"]}
        return {"ok": True, "board_id": "B"}
    with pytest.raises(KeyboardInterrupt):
        execute(compiler, req, dispatch)
    result = execute(compiler, req, lambda *a: {"ok": True, "board_id": "B"})
    assert len(parent_writes) == 1
    assert result["ledger"]["finalize"]["completed"] == 2
    assert result["ledger"]["phase"] == "completed"


def test_fallback_pruning_happens_before_client_resolution(compiler):
    from workstation.tests.test_durable_agent_integration import make_agent
    agent = make_agent()
    agent._turn_constraints = TurnConstraintContext({"forbidden_routes": ["openai_api", "anthropic"]})
    agent._fallback_chain = [{"provider": "openai", "model": "gpt"}, {"provider": "anthropic", "model": "claude"}]
    agent._fallback_index = 0
    with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
        assert agent._try_activate_fallback() is False
    assert not resolve.called


def test_auxiliary_constraints_cover_cached_and_new_clients():
    from agent.turn_constraints import scoped_turn_constraints, publish_turn_constraints
    from agent.auxiliary_client import resolve_provider_client, _get_cached_client
    with scoped_turn_constraints():
        publish_turn_constraints(TurnConstraintContext({"forbidden_routes": ["openai_api"]}))
        for resolve in (resolve_provider_client, _get_cached_client):
            with pytest.raises(ConstraintViolation):
                resolve("openai", model="gpt")
    from agent.turn_constraints import guard_auxiliary_route
    guard_auxiliary_route("openai")


def test_full_turn_primary_is_pruned_before_first_provider_call(compiler):
    from workstation.tests.test_durable_agent_integration import make_agent
    agent = make_agent()
    agent.provider = "openai"
    agent.client = MagicMock()
    permitted_client = MagicMock()
    permitted_client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop")], model="test/model", usage=None)
    def activate():
        agent.provider, agent.base_url, agent.client = "anthropic", "https://api.anthropic.com", permitted_client
        return True
    original_client = agent.client
    agent._try_activate_fallback = activate
    agent._cached_system_prompt = "helpful"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    result = agent.run_conversation('forbidden_routes=["openai_api"]')
    assert result["final_response"] == "done"
    assert not original_client.chat.completions.create.called
    assert permitted_client.chat.completions.create.call_count == 1


def test_acirv_trello_replay_constant_provider_and_setup(compiler, tmp_path):
    from workstation.benchmarks.trello_regression import run
    result = run(tmp_path / "trello-replay")
    metrics = result["durable_measured"]
    assert metrics["provider_calls"] == 2
    assert metrics["LLM_interventions"] < result["items"]
    assert metrics["setup_calls"] == 3
    assert metrics["mutations"] == metrics["completed_items"] == 12
    assert metrics["replayed_mutations"] == 0
    assert metrics["cache_hits"] >= 11
    assert metrics["inline_context_bytes"] < 8000
    assert metrics["bytes_avoided_by_refs"] > 1000000
    assert metrics["artifact_bytes"] > metrics["inline_context_bytes"]
    assert metrics["token_count"] is None
