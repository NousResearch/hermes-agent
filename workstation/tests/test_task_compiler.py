import json
import sqlite3

import pytest

from workstation.artifacts import ArtifactStore
from workstation.durable_tasks import DurableTaskStore
from workstation.task_compiler import TaskCompiler, WorkClass, classify
from workstation.routing import ConstraintViolation
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = DurableTaskStore(conn=sqlite3.connect(tmp_path / "kanban.db"))
    return TaskCompiler(store, ArtifactStore(tmp_path / "artifacts"))


def request(n=100):
    return {"operation_key": "records-v1", "items": [{"id": i} for i in range(n)],
            "steps": [{"tool": "read_file", "args": {"path": "$item.id"}}]}


def run(compiler, req, dispatch):
    return compiler.execute(req, task_id="browser-task", session_id="session", dispatch=dispatch)


def test_100_items_no_llm_and_reference_boundary(compiler):
    calls = []
    def dispatch(tool, args, task, call):
        calls.append((tool, args, task))
        return {"status": "ok", "text": "z" * 20000}
    result = run(compiler, request(), dispatch)
    assert len(calls) == 100
    assert result["completed"] == 100
    assert result["metrics"]["LLM_interventions"] == 0
    assert len(json.dumps(result)) < 4000
    assert "z" * 100 not in json.dumps(result)
    assert compiler.artifacts.resolve_ref(result["results_ref"])
    assert len(compiler.store.get_work_items(result["plan_id"])) == 100
    assert all(i.evidence_refs for i in compiler.store.get_work_items(result["plan_id"]))


def test_restart_at_37_and_ledger_independent_of_transcript(compiler, tmp_path):
    calls = []
    def crash(tool, args, task, call):
        if len(calls) == 37:
            raise KeyboardInterrupt("restart")
        calls.append(args["path"])
        return {"ok": True}
    with pytest.raises(KeyboardInterrupt):
        run(compiler, request(), crash)
    restored = TaskCompiler(DurableTaskStore(conn=sqlite3.connect(tmp_path / "kanban.db")), compiler.artifacts)
    def resume(tool, args, task, call):
        calls.append(args["path"])
        return {"ok": True}
    plan_id = restored.store.get_connection().execute("SELECT id FROM work_plans").fetchone()[0]
    result = restored.resume(plan_id, session_id="session", dispatch=resume)
    assert calls == list(range(100))
    assert result["ledger"]["completed"] == 100
    assert result["ledger"]["pending"] == 0
    assert restored.store.operational_ledger(result["plan_id"]) == result["ledger"]
    with pytest.raises(ValueError, match="owning conversation"):
        restored.resume(plan_id, session_id="another-session", dispatch=resume)


def test_step_checkpoint_restart_does_not_repeat_mutation(compiler):
    req = request(1)
    req["steps"] = [{"tool": "browser_type", "args": {}, "expect": {"ok": True}},
                    {"tool": "browser_snapshot", "args": {}}]
    calls = []
    def crash(tool, args, task, call):
        calls.append(tool)
        if tool == "browser_snapshot":
            raise KeyboardInterrupt()
        return {"ok": True}
    with pytest.raises(KeyboardInterrupt):
        run(compiler, req, crash)
    result = run(compiler, req, lambda tool, *a: calls.append(tool) or {"ok": True})
    assert calls.count("browser_type") == 1
    assert result["completed"] == 1


def test_browser_batch_one_task_and_prompt_queue(compiler):
    req = request(12)
    req["kind"] = "prompt_queue"
    req["steps"] = [{"tool": "browser_type", "args": {"text": "$item.id"}, "expect": {"ok": True}},
                    {"tool": "browser_snapshot", "verifies": ["fan_out_0"], "args": {}, "expect": {"done": True},
                     "wait": {"interval_seconds": 0, "max_polls": 2}}]
    owners = []
    result = run(compiler, req, lambda tool, args, task, call: owners.append(task) or {"ok": True, "done": True})
    assert set(owners) == {"browser-task"}
    assert len(owners) == 24
    assert result["completed"] == 12
    assert classify(req) == WorkClass.PROMPT_QUEUE


def test_constraint_pruning_no_probe(compiler):
    req = request()
    req["constraints"] = {"forbidden_routes": ["tool.read_file", "openai_api"]}
    calls = []
    with pytest.raises(ConstraintViolation):
        run(compiler, req, lambda *a: calls.append(a))
    assert calls == []


def test_failed_verifier_escalates_and_restart_does_not_replay(compiler):
    req = request(1)
    req["steps"][0]["expect"] = {"ok": True}
    calls = []
    def failed(*a):
        calls.append(a)
        return {"ok": False}
    result = run(compiler, req, failed)
    assert result["completed"] == 0
    assert result["needs_reasoning"] == 1
    result = run(compiler, req, failed)
    assert len(calls) == 1
    assert result["ledger"]["next_action"] == "review_exceptions"


def test_operation_identity_conflict_and_mutation_requires_verifier(compiler):
    run(compiler, request(1), lambda *a: {"ok": True})
    with pytest.raises(ValueError, match="conflicts"):
        run(compiler, request(2), lambda *a: {"ok": True})
    req = request(1)
    req["steps"][0]["tool"] = "browser_click"
    with pytest.raises(ValueError, match="verifier"):
        run(compiler, req, lambda *a: {"ok": True})


@pytest.mark.parametrize("period", [2, 3, 4])
def test_no_progress_cycles_halt(period):
    guard = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    for _ in range(5):
        for i in range(period):
            guard.observe_call("terminal", {"command": str(i)}, "same output")
    assert guard.halt_decision.code == "identical_cycle_halt"


def test_progress_resets_failed_retry():
    guard = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2))
    for _ in range(2):
        guard.after_call("read_file", {"path": "a"}, "Error", failed=True)
    assert guard.before_call("read_file", {"path": "a"}).action == "block"
    guard.reset_for_turn()
    for _ in range(2):
        guard.after_call("read_file", {"path": "a"}, "Error", failed=True)
    guard.after_call("patch", {}, '{}', failed=False, actual_delta=True)
    assert guard.before_call("read_file", {"path": "a"}).action == "allow"


def test_durable_regression_benchmark_property(tmp_path):
    from workstation.benchmarks.durable_execution import run
    measured = run(tmp_path)
    assert measured["durable"]["completed"] == 99
    assert measured["durable"]["exceptions"] == 1
    assert measured["durable"]["planner_interventions"] < measured["items"]
    assert measured["durable"]["runtime_llm_calls"] == 0
    assert measured["durable"]["restart_completed_items_replayed"] == 0
    assert measured["durable"]["cache_hits"] > 0
    assert measured["durable"]["inline_bytes"] < measured["baseline"]["inline_bytes"]


def test_prompt_queue_bounded_completion_wait(compiler):
    req = request(3)
    req["kind"] = "prompt_queue"
    req["steps"] = [{"tool": "browser_type", "args": {"text": "$item.id"}, "expect": {"ok": True}},
                    {"tool": "browser_snapshot", "verifies": ["fan_out_0"], "args": {}, "expect": {"done": True},
                     "wait": {"interval_seconds": 0, "max_polls": 3}}]
    polls = []
    def dispatch(tool, args, task, call):
        if tool == "browser_type":
            return {"ok": True}
        polls.append(call)
        return {"done": "poll_" in call, "capture": "evidence"}
    result = run(compiler, req, dispatch)
    assert result["completed"] == 3
    assert len(polls) == 6


def test_dataset_ref_compiles_without_inline_model_payload(compiler):
    records = compiler.artifacts.store("input", "records.json", [{"id": i} for i in range(50)])
    req = request(1)
    del req["items"]
    req["items_ref"] = records.ref
    result = run(compiler, req, lambda *a: {"ok": True})
    assert result["total"] == result["completed"] == 50


def test_runtime_usage_only_reported_scope(compiler):
    known = compiler.execute(request(10), task_id="t", session_id="s", dispatch=lambda *a: {"ok": True},
        provider_usage={"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 50,
                        "reasoning_tokens": 5, "total_tokens": 120})
    assert known["metrics"]["tokens_per_successful_state_transition"] == 12
    assert known["metrics"]["token_usage_scope"] == "compile_request"
    unknown = compiler.execute(request(10), task_id="t", session_id="s", dispatch=lambda *a: {"ok": True})
    assert unknown["metrics"]["usage_status"] == "unknown"
    assert "tokens_per_successful_state_transition" not in unknown["metrics"]


def test_captured_output_persistence_retry_preserves_type_without_reexecution(compiler, monkeypatch):
    from workstation.batch_runner import DurableBatchRunner
    calls = []
    persist_failures = []
    original = compiler.artifacts.store
    def store(*a, **kw):
        if kw.get("schema") == "normalized_batch_output" and not persist_failures:
            persist_failures.append(True)
            raise OSError("transient disk failure")
        return original(*a, **kw)
    monkeypatch.setattr(compiler.artifacts, "store", store)
    runner = DurableBatchRunner("capture", task_store=compiler.store, artifact_store=compiler.artifacts,
                                max_retries=1, backoff_seconds=0)
    summary = runner.execute_batch("capture", [{}],
        worker_fn=lambda *a: calls.append(True) or '{"ok":true}',
        validator_fn=lambda raw, _: {"valid": isinstance(raw, str)})
    assert len(calls) == 1
    assert summary.retry_success_count == 1
