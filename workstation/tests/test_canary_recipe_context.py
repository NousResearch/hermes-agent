"""Behavior contracts for canary admission, recipes and durable wire context."""
import copy
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.registry import registry
from tools.effects import ToolEffect, tool_effect
from workstation.artifacts import ArtifactStore
from workstation.durable_tasks import DurableTaskStore
from workstation.task_compiler import TaskCompiler, execution_context, execute_compiled_work, requires_compilation
from workstation.recipes import RecipeStore
from workstation.continuation import build_durable_handoff, planner_projection, project_for_provider, durable_compaction
from agent.tool_guardrails import ExperimentKey, ToolCallGuardrailController, ToolCallGuardrailConfig


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(registry, "_tools", dict(registry._tools))
    for name, effect in (("fake_setup", ToolEffect.DISCOVERY), ("fake_create", ToolEffect.MUTATION),
                        ("fake_verify", ToolEffect.PURE_READ), ("fake_preflight", ToolEffect.DISCOVERY)):
        registry.register(name, "canary_test", {"name": name, "parameters": {"type": "object"}}, lambda **kw: "{}", effect=effect)
    conn = sqlite3.connect(tmp_path / "test-kanban.db")
    value = TaskCompiler(DurableTaskStore(conn=conn), ArtifactStore())
    yield value
    conn.close()


def request(n=12, key="new"):
    return {"operation_key": key, "recipe_key": "fake.cards.v1", "items": [{"id": i} for i in range(n)],
            "recipe_scope": {"route": "tool.fake_create", "host": "example.test", "path_family": "/cards/:card"},
            "preflight": [{"tool": "fake_preflight", "args": {}, "expect": {"capability": "v1"}}],
            "setup_steps": [{"id": "setup", "tool": "fake_setup", "args": {}}],
            "steps": [{"id": "create", "tool": "fake_create", "args": {"id": "$item.id"}, "expect": {"ok": True}},
                      {"id": "verify", "tool": "fake_verify", "verifies": ["create"], "args": {"id": "$steps.create.id"}, "expect": {"persisted": True}}],
            "finalize_steps": [{"id": "finish", "tool": "fake_setup", "args": {"results": "$items_ref"}}]}


def adapter(calls, *, persisted=True, capability="v1"):
    def dispatch(name, args, *rest):
        calls.append((name, args))
        if name == "fake_preflight":
            return {"capability": capability}
        if name == "fake_create":
            return {"ok": True, "id": args["id"], "large": "Z" * 10000}
        if name == "fake_verify":
            return {"persisted": persisted, "id": args["id"]}
        return {"ok": True}
    return dispatch


def execute(compiler, req, dispatch, **kwargs):
    return compiler.execute(req, task_id="task", session_id="owner", dispatch=dispatch, **kwargs)


def test_canary_success_order_and_no_llm(compiler):
    calls = []
    result = execute(compiler, request(), adapter(calls))
    names = [n for n, _ in calls]
    assert names[:4] == ["fake_setup", "fake_preflight", "fake_create", "fake_verify"]
    assert names.count("fake_create") == result["completed"] == 12
    assert names.count("fake_setup") == 2  # shared setup + finalize
    assert result["ledger"]["canary"]["verified"]
    assert result["metrics"]["canary_attempts"] == result["metrics"]["canary_successes"] == 1
    assert result["metrics"]["executor_llm_calls"] == 0


def test_invalid_canary_does_not_multiply_failure(compiler):
    calls = []
    result = execute(compiler, request(), adapter(calls, persisted=False))
    assert [n for n, _ in calls].count("fake_create") == 1
    assert result["ledger"]["items"]["pending"] == 11
    assert result["metrics"]["failed_fanout_items"] == 0
    assert result["ledger"]["next_action"] == "review_exceptions"
    assert result["ledger"]["canary"]["status"] == "failed"
    assert result["code"] == "canary_failed"


@pytest.mark.parametrize("verified", [False, True])
def test_canary_restart_never_replays_confirmed_or_uncertain(compiler, verified):
    calls = []
    regular = adapter(calls)
    def crash(name, args, *rest):
        if not verified and name == "fake_create":
            calls.append((name, args))
            raise KeyboardInterrupt()
        if verified and name == "fake_create" and args["id"] == 1:
            raise KeyboardInterrupt()
        return regular(name, args, *rest)
    with pytest.raises(KeyboardInterrupt):
        execute(compiler, request(), crash)
    if verified:
        # Crash before the next mutation dispatch still records intent; use a
        # progress crash after canary's verified read instead for safe resume.
        assert build_durable_handoff(compiler.store, "work_" + __import__("hashlib").sha256(b"owner:task:new").hexdigest())["canary"]["verified"]
    result = execute(compiler, request(), regular)
    assert len([a for n, a in calls if n == "fake_create" and a["id"] == 0]) == 1
    if not verified:
        assert result["anomalies"][0]["reason"] == "uncertain_mutation_requires_review"
        assert result["ledger"]["canary"]["status"] == "uncertain"


def test_restart_after_canary_checkpoint_continues_remaining(compiler):
    calls = []
    def progress():
        if calls and calls[-1][0] == "fake_verify" and calls[-1][1]["id"] == 0:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        execute(compiler, request(), adapter(calls), progress=progress)
    result = execute(compiler, request(), adapter(calls))
    assert result["completed"] == 12
    assert [a["id"] for n, a in calls if n == "fake_create"] == list(range(12))


def test_recipe_restart_hit_preflight_and_graph_not_resent(compiler):
    execute(compiler, request(), adapter([]))
    restored = TaskCompiler(compiler.store, ArtifactStore(compiler.artifacts.root), RecipeStore(ArtifactStore(compiler.artifacts.root)))
    assert restored.recipes.get("fake.cards.v1")["status"] == "VERIFIED"
    calls = []
    result = execute(restored, {"recipe_key": "fake.cards.v1", "items": [{"id": i} for i in range(12,24)]}, adapter(calls))
    assert result["completed"] == 12
    assert result["metrics"]["recipe_cache_hits"] == 1
    assert result["metrics"]["canary_attempts"] == 0
    assert [n for n, _ in calls].count("fake_preflight") == 1
    assert [n for n, _ in calls].count("fake_create") == 12


def test_stale_preflight_blocks_all_mutations(compiler):
    execute(compiler, request(), adapter([]))
    calls = []
    result = execute(compiler, {"recipe_key": "fake.cards.v1", "items": [{"id": 20}, {"id": 21}]}, adapter(calls, capability="v2"))
    assert not any(n == "fake_create" for n, _ in calls)
    assert result["metrics"]["recipe_invalidations"] == 1
    assert compiler.recipes.get("fake.cards.v1")["status"] == "STALE"


def test_changed_schema_invalidates_before_dispatch(compiler):
    execute(compiler, request(), adapter([]))
    entry = registry.get_entry("fake_verify")
    entry.schema = {**entry.schema, "version": "v2"}
    with pytest.raises(ValueError, match="recipe_fingerprint_mismatch"):
        execute(compiler, {"recipe_key": "fake.cards.v1", "items": [{"id": 20}, {"id": 21}]}, lambda *a: pytest.fail("stale dispatch"))
    assert compiler.recipes.get("fake.cards.v1")["status"] == "STALE"


def test_secret_stripping_in_all_persisted_recipe_files(compiler):
    secrets = {key: "SECRET_" + key for key in ("api_key", "token", "Authorization", "cookie", "password", "session_cookie", "credentials")}
    body = {"status": "VERIFIED", "scope": secrets, "graph": {"args": {"nested": [secrets], "encoded": json.dumps(secrets)}}}
    compiler.recipes.put("sanitized", body)
    persisted = "".join(p.read_text(errors="replace") for base in (compiler.recipes.root, compiler.artifacts.root / "recipes") for p in base.rglob("*.json"))
    assert "SECRET_" not in persisted


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_handoff_bound_independent_of_transcript(compiler, n):
    plan = compiler.store.create_plan("many", "many", [{"id": i} for i in range(n)], session_id="owner")
    first = build_durable_handoff(compiler.store, plan.id)
    restored = DurableTaskStore(conn=compiler.store.get_connection())
    assert first == build_durable_handoff(restored, plan.id)
    assert len(json.dumps(first).encode()) <= 4096
    assert "reasoning_content" not in json.dumps(first)


def trusted_messages(result):
    return [{"role": "system", "content": "stable system"},
            {"role": "user", "content": "old ask"},
            {"role": "assistant", "content": "SHOULD_NOT_REACH_EXECUTOR" * 1000, "reasoning": "old reasoning" * 1000},
            {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY]" + "old summary" * 1000, "_compressed_summary": True},
            {"role": "user", "content": "latest human ask"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "work", "function": {"name": "work_execute", "arguments": "{}"}, "type": "function"}]},
            {"role": "tool", "tool_call_id": "work", "content": "big old output" * 1000,
             "_hermes_operational_refs": [{"trusted": True, "version": 1, "source": "KanbanRun", "kind": "durable_work", "id": result["plan_id"], "owner_session_id": "owner"}]}]


def test_projection_dedupe_no_recursive_compaction_and_history_preserved(compiler, monkeypatch):
    result = execute(compiler, request(), adapter([]))
    messages = trusted_messages(result)
    original = copy.deepcopy(messages)
    agent = SimpleNamespace(session_id="owner")
    with patch("workstation.durable_tasks.DurableTaskStore", return_value=compiler.store):
        projected = project_for_provider(agent, messages)
        assert durable_compaction(agent, messages)
    text = json.dumps(projected)
    assert "latest human ask" in text and result["plan_id"] in text
    assert "big old output" not in text and "old reasoning" not in text and "old summary" not in text
    assert "SHOULD_NOT_REACH_EXECUTOR" not in text
    assert text.count("DURABLE HANDOFF") == 1
    assert agent._durable_context_metrics["duplicate_compaction_bytes_suppressed"] > 0
    assert messages == original


def test_untrusted_or_wrong_owner_cannot_enable_projection(compiler):
    result = execute(compiler, request(), adapter([]))
    messages = trusted_messages(result)
    messages[-1]["_hermes_operational_refs"][0]["owner_session_id"] = "attacker"
    assert project_for_provider(SimpleNamespace(session_id="owner"), messages) is messages


def test_contract_and_structured_correction_without_source(compiler):
    contract = json.loads(execute_compiled_work({"action": "contract"}))
    assert len(json.dumps(contract).encode()) < 4096
    example = contract["examples"][0]
    result = execute(compiler, example, lambda *a: {"ok": True})
    assert result["completed"] == 1
    with execution_context(lambda *a: {"ok": True}, "owner"), patch("workstation.task_compiler.TaskCompiler", return_value=compiler):
        error = json.loads(execute_compiled_work({"operation_key": "invalid", "items": [{}], "steps": [{"tool": "fake_create", "args": {}}]}))
    assert error["code"] == "mutation_verifier_required" and error["fix"]["add_expect"]


def test_browser_console_cannot_claim_external_verification(compiler):
    req = request()
    req["steps"][1]["tool"] = "browser_console"
    with pytest.raises(ValueError, match="read/discovery"):
        execute(compiler, req, lambda *a: pytest.fail("unsafe verifier"))


def test_executor_receives_only_bindings_not_transcript(compiler):
    calls = []
    execute(compiler, request(), adapter(calls))
    assert "SHOULD_NOT_REACH_EXECUTOR" not in json.dumps(calls)
    assert all(set(a) <= {"id", "results"} for _, a in calls)


def test_experiment_identity_shape_and_new_hypothesis():
    assert ExperimentKey.from_call("create", {"title": "A"}) == ExperimentKey.from_call("create", {"title": "B"})
    guard = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2))
    for _ in range(2):
        guard.after_call("browser_console", {"selector": "A"}, "Error", failed=True)
    assert guard.before_call("browser_console", {"selector": "A"}).action == "block"
    assert guard.before_call("browser_console", {"selector": "B"}).action == "allow"
    guard.after_call("browser_console", {"selector": "B"}, '{"actual_delta":true}', failed=False)
    assert guard.before_call("browser_console", {"selector": "A"}).action == "block"
    guard.mark_verified_progress()
    assert guard.before_call("browser_console", {"selector": "A"}).action == "allow"


def test_skill_discovery_and_unknown_effects():
    from agent.tool_result_classification import tool_may_have_side_effect
    agent = SimpleNamespace(_work_batch_candidate=True, valid_tool_names={"work_execute"})
    for name in ("skill_view", "skills_list", "read_file", "tool_describe", "tool_search"):
        call = SimpleNamespace(function=SimpleNamespace(name=name, arguments="{}"))
        assert not requires_compilation(agent, [call])
        assert not tool_may_have_side_effect(name)
    assert tool_effect("unknown_external_read") == ToolEffect.MUTATION
    assert tool_may_have_side_effect("browser_console")


def test_registered_builtin_effect_coverage(tmp_path):
    # Earlier registry-isolation fixtures may import then restore modules.
    # Audit real import-time registrations in a clean process, not that cache.
    import os
    import subprocess
    import sys
    source = "from tools.registry import registry,discover_builtin_tools; discover_builtin_tools(); entries=[e for e in registry.get_all_entries() if e.handler.__module__.startswith('tools.') and e.handler.__module__ != 'tools.mcp_tool']; assert entries; assert all(e.effect is not None for e in entries), [(e.name,e.effect) for e in entries if e.effect is None]; print(len(entries))"
    completed = subprocess.run([sys.executable, "-c", source], env={**os.environ, "HERMES_HOME": str(tmp_path)},
                               capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    assert int(completed.stdout.strip().splitlines()[-1]) > 0


def test_turn_constraint_reset_through_real_conversation(compiler):
    from workstation.tests.test_durable_agent_integration import make_agent
    from agent.turn_constraints import guard_auxiliary_route
    agent = make_agent()
    agent.provider = "openai"
    agent._fallback_chain = []
    agent.client = MagicMock()
    assert agent.run_conversation("Não use OpenAI API.")["error"] == "constraint_violation"
    guard_auxiliary_route("openai")  # ambient route restored after rejection
    agent._cached_system_prompt = "stable"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop")], model="fake", usage=None)
    assert agent.run_conversation("Olá.")["final_response"] == "done"
    assert agent._turn_constraints.routes == {}


def test_telemetry_cached_and_uncached_distinguished(compiler):
    result = execute(compiler, request(), adapter([]), provider_usage={"input_tokens": 1000, "cache_read_tokens": 900, "output_tokens": 20, "total_tokens": 1020})
    metrics = result["metrics"]
    assert metrics["uncached_input_tokens"] == 100
    assert metrics["cached_input_tokens"] == 900
    assert metrics["uncached_input_tokens_per_verified_transition"] == 100/12
    assert metrics["estimated_cost"] is None and metrics["cost_status"] == "unknown"


@pytest.mark.parametrize("n", [100, 1000])
def test_large_fanout_no_executor_model_calls_or_confirmed_replay(compiler, n):
    calls = []
    req = request(n)
    def dispatch(name, args, *rest):
        calls.append((name, args))
        if name == "fake_preflight":
            return {"capability": "v1"}
        if name == "fake_create":
            return {"ok": True, "id": args["id"]}
        return {"ok": True, "persisted": True}
    result = execute(compiler, req, dispatch)
    assert result["completed"] == n and result["metrics"]["executor_llm_calls"] == 0
    assert [a["id"] for name, a in calls if name == "fake_create"] == list(range(n))
    restored = TaskCompiler(compiler.store, compiler.artifacts)
    resumed = restored.resume(result["plan_id"], session_id="owner", dispatch=lambda *a: pytest.fail("confirmed replay"))
    assert resumed["completed"] == n


def test_real_provider_projection_preserves_sessiondb_history(compiler):
    from workstation.tests.test_durable_agent_integration import make_agent
    from hermes_state import SessionDB
    agent = make_agent()
    result = compiler.execute(request(), task_id="task", session_id=agent.session_id, dispatch=adapter([], persisted=False))
    agent._session_db = SessionDB(compiler.artifacts.root.parent / "state.db")
    agent._ensure_db_session()
    history = trusted_messages(result)
    history[-1]["_hermes_operational_refs"][0]["owner_session_id"] = agent.session_id
    agent._flush_messages_to_session_db(history)
    persisted_before = agent._session_db.get_messages(agent.session_id)
    agent._cached_system_prompt = "stable system"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop")], model="fake", usage=None)
    with patch("workstation.durable_tasks.DurableTaskStore", return_value=compiler.store):
        agent.run_conversation("continue latest ask", conversation_history=history[1:])
    wire = json.dumps(agent.client.chat.completions.create.call_args.kwargs["messages"])
    assert "continue latest ask" in wire and "DURABLE HANDOFF" in wire
    assert "SHOULD_NOT_REACH_EXECUTOR" not in wire and "big old output" not in wire
    rows = agent._session_db.get_messages(agent.session_id)
    assert len(rows) >= len(persisted_before)
    assert any("SHOULD_NOT_REACH_EXECUTOR" in str(row.get("content", "")) for row in rows)
    agent._session_db.close()


def test_three_controlled_trello_scenarios(tmp_path):
    from workstation.benchmarks.trello_regression import run_scenarios
    scenarios = run_scenarios(tmp_path / "three-replays")
    assert scenarios["broken"]["durable_measured"]["mutations"] == 1
    assert scenarios["corrected"]["durable_measured"]["canary_successes"] == 1
    assert scenarios["known_after_restart"]["durable_measured"]["recipe_cache_hits"] == 1


def test_browser_scope_and_stable_entity_fingerprint(compiler):
    from workstation.recipes import require_browser_scope, recipe_fingerprint
    scope = {"route": "native_browser", "host": "trello.com", "path_family": "/c/:card"}
    require_browser_scope({"url": "https://trello.com/c/abc"}, scope)
    for url in ("https://evil.test/c/abc", "https://trello.com/settings", "file:///c/abc"):
        with pytest.raises(ValueError, match="recipe_scope"):
            require_browser_scope({"url": url}, scope)
    one = {"setup": [], "fan_out": [{"id": "read", "tool": "read_file", "args": {"card_id": "A"}}], "finalize": []}
    two = copy.deepcopy(one)
    two["fan_out"][0]["args"]["card_id"] = "B"
    assert recipe_fingerprint(one, scope) == recipe_fingerprint(two, scope)


def test_confirmed_preflight_cannot_resume_globally_stale_recipe(compiler):
    execute(compiler, request(), adapter([]))
    known = {"operation_key": "hit", "recipe_key": "fake.cards.v1", "items": [{"id": 50}, {"id": 51}]}
    def interrupted(name, args, *rest):
        if name == "fake_create" and args["id"] == 51:
            raise KeyboardInterrupt()
        return adapter([])(name, args, *rest)
    with pytest.raises(KeyboardInterrupt):
        execute(compiler, known, interrupted)
    compiler.recipes.invalidate("fake.cards.v1")
    plan_id = compiler.store.get_connection().execute("SELECT id FROM work_plans WHERE status != 'completed'").fetchone()[0]
    with pytest.raises(ValueError, match="recipe_stale"):
        compiler.resume(plan_id, session_id="owner", dispatch=lambda *a: pytest.fail("stale resume"))


def test_arbitrary_successful_read_does_not_reset_progress(compiler):
    progress = MagicMock()
    req = {"operation_key": "read-only", "items": [{"path": "a"}, {"path": "b"}], "steps": [{"tool": "read_file", "args": {"path": "$item.path"}}]}
    result = execute(compiler, req, lambda *a: {"same": True}, progress=progress)
    assert result["completed"] == 2 and result["metrics"]["verified_state_transitions"] == 0
    progress.assert_not_called()


def test_fake_planner_uses_only_published_schema_and_contract(compiler):
    import importlib
    import tools.workstation_work as work_tool
    importlib.reload(work_tool)
    from workstation.tests.test_durable_agent_integration import make_agent
    agent = make_agent()
    agent.tools = [{"type": "function", "function": registry.get_schema("work_execute")}]
    agent._cached_system_prompt = "Use the published work contract."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    guide = json.loads(execute_compiled_work({"action": "contract"}))
    reads = []
    def response(text, tools, finish):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=tools), finish_reason=finish)], model="fake", usage=None)
    def model(**kwargs):
        if agent.client.chat.completions.create.call_count == 1:
            description = kwargs["tools"][0]["function"]["description"]
            assert "expect" in description and "recipe_key" in description
            decided = guide["examples"][0]
            call = SimpleNamespace(id="planned", type="function", function=SimpleNamespace(name="work_execute", arguments=json.dumps(decided)))
            return response("", [call], "tool_calls")
        return response("verified", None, "stop")
    def handler(name, args, task, **kw):
        if name == "work_execute":
            return execute_compiled_work(args, task_id=task)
        reads.append(args["path"])
        return '{"ok":true}'
    agent.client.chat.completions.create.side_effect = model
    with patch("run_agent.handle_function_call", side_effect=handler):
        result = agent.run_conversation("Read a.txt using the published contract.")
    assert result["final_response"] == "verified" and reads == ["a.txt"]
    assert agent.client.chat.completions.create.call_count == 2


def test_graph_changes_are_new_experiments_without_entity_value_identity():
    first = {"steps": [{"tool": "fake_create", "args": {"title": "$item.title"}, "expect": {"ok": True}}]}
    second = copy.deepcopy(first)
    second["steps"][0]["tool"] = "different_create"
    repeated = {"steps": [*first["steps"], *first["steps"]]}
    changed_expect = copy.deepcopy(first)
    changed_expect["steps"][0]["expect"] = {"persisted": True}
    key = ExperimentKey.from_call("work_execute", first)
    assert all(ExperimentKey.from_call("work_execute", args) != key for args in (second, repeated, changed_expect))
