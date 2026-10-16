"""Completed child output must be redacted before delegation consumers (#137594)."""

import json
import queue
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import async_delegation as ad
from tools import delegate_tool as delegate
from tools import delegate_tool_results as results
from tools.process_registry import process_registry
from tests.tools.test_delegate_output_schema import _StubChild, _StubParent

SECRET = "sk-ant-" + "syntheticdelegationcredential" * 3
PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    + "c3ludGhldGlj" * 100
    + "\n-----END PRIVATE KEY-----"
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


class _Parent(_StubParent):
    session_id = "parent-session"

    def __init__(self):
        self._memory_manager = SimpleNamespace(on_delegation=Mock())


@pytest.mark.parametrize("text", [SECRET, PEM])
def test_child_result_events_memory_hooks_and_spill_use_redacted_output(
    text, tmp_path, monkeypatch
):
    import hermes_cli.plugins
    from agent.redact import redact_for_egress

    hooks = Mock()
    monkeypatch.setattr(hermes_cli.plugins, "invoke_hook", hooks)
    monkeypatch.setattr(delegate, "_load_config", lambda: {"max_summary_chars": 2000})
    events = []
    summary = text + "\n" + "ordinary explanation\n" * 180
    child = _StubChild([summary])
    child.tool_progress_callback = lambda event, **kw: events.append((event, kw))
    parent = _Parent()

    entry = results._run_child_lifecycle(0, "summarize the task", child, parent)

    assert entry["status"] == "completed"
    assert text not in entry["summary"]
    completion = next(
        payload for name, payload in events if name == "subagent.complete"
    )
    assert completion["preview"] == redact_for_egress(summary)[:160]
    assert text not in parent._memory_manager.on_delegation.call_args.kwargs["result"]
    stop = next(
        call.kwargs for call in hooks.call_args_list if call.args[0] == "subagent_stop"
    )
    assert text not in stop["child_summary"]
    spills = list((tmp_path / "cache" / "delegation").glob("subagent-summary-*.txt"))
    assert spills
    assert text not in spills[0].read_text(encoding="utf-8-sig")


def _record():
    record = {
        "delegation_id": "synthetic-delegation",
        "dispatched_at": time.time(),
        "session_key": "agent:example:dm:route",
        "origin_session_id": "origin-session",
        "parent_session_id": "parent-session",
        "goal": "retain the parent goal",
        "context": "retain the parent context",
        "model": "test-model",
    }
    ad._persist_dispatch(record)
    return record


def test_async_queue_and_durable_result_redact_child_output_without_changing_routing(
    monkeypatch,
):
    record = _record()
    target = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", target)
    payload = {
        "status": "completed",
        "summary": SECRET,
        "error": SECRET,
        "future_child_field": SECRET,
    }

    ad._push_completion_event(record, payload, "completed")

    event = target.get_nowait()
    assert SECRET not in json.dumps(event)
    assert all(
        event[key] == record[key]
        for key in ("delegation_id", "session_key", "goal", "context")
    )
    stored = ad.get_durable_delegation(record["delegation_id"])
    assert stored is not None
    assert SECRET not in json.dumps(stored["result"])
    assert payload["summary"] == SECRET  # the producer's own value is not mutated
    with ad._transaction() as conn:
        row = conn.execute(
            "SELECT event_json,result_json FROM async_delegations"
        ).fetchone()
    assert SECRET not in "".join(row)


def test_old_durable_results_are_redacted_on_read_replay_and_failure_roster():
    record = _record()
    event = {
        **record,
        "type": "async_delegation",
        "status": "failed",
        "summary": SECRET,
        "error": SECRET,
    }
    result = {"status": "failed", "summary": SECRET, "error": SECRET}
    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET state='failed',completed_at=?,event_json=?,result_json=?",
            (time.time(), json.dumps(event), json.dumps(result)),
        )

    stored = ad.get_durable_delegation(record["delegation_id"])
    assert stored is not None
    assert SECRET not in json.dumps(stored["result"])
    target = queue.Queue()
    assert ad.restore_undelivered_completions(target) == 1
    replay = target.get_nowait()
    assert replay["restored"] is True
    assert replay["session_key"] == record["session_key"]
    assert SECRET not in json.dumps(replay)
    failures = ad.failed_delegations_for_session(parent_session_id="parent-session")
    assert len(failures) == 1 and failures[0]["goal"] == record["goal"]
    assert SECRET not in json.dumps(failures)


def test_partial_child_receipt_is_sanitized_before_recovery_storage():
    record = _record()
    entry = {
        "task_index": 0,
        "status": "completed",
        "summary": SECRET,
        "schema_valid": True,
    }

    ad.record_unit_child(record["delegation_id"], entry)

    saved = ad.get_durable_delegation(record["delegation_id"])
    assert saved is not None
    stored = saved["result"]
    assert SECRET not in json.dumps(stored)
    assert stored["results"][0]["schema_valid"] is False


def test_redacted_schema_verdict_describes_delivered_summary_without_rerunning_child():
    child = _StubChild([json.dumps({"value": SECRET})])
    child._delegate_output_schema = {
        "type": "object",
        "properties": {"value": {"const": SECRET}},
        "required": ["value"],
    }

    entry = delegate._run_single_child(0, "produce structured output", child, _Parent())

    assert entry["status"] == "completed"
    assert entry["schema_valid"] is False
    assert SECRET not in json.dumps(entry)
    assert len(child.calls) == 1


def test_redacted_json_can_still_satisfy_its_schema():
    child = _StubChild([json.dumps({"value": SECRET})])
    child._delegate_output_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    entry = delegate._run_single_child(0, "produce structured output", child, _Parent())

    assert entry["schema_valid"] is True
    assert SECRET not in entry["summary"]
    assert "schema_note" not in entry and len(child.calls) == 1


def test_schema_revalidation_fault_does_not_leak_the_exception_or_raw_summary(
    monkeypatch,
):
    from tools.delegation_redaction import redact_child_result
    import tools.delegation_output_schema as schema

    def fail(*args):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(schema, "validate_output", fail)
    entry = redact_child_result({"summary": SECRET, "schema_valid": True}, {})

    assert entry["schema_valid"] is False
    assert SECRET not in json.dumps(entry)
    assert "unavailable" in entry["schema_errors"][0]


def test_redactor_failure_never_falls_back_to_raw_output(monkeypatch):
    import agent.redact

    def fail(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(agent.redact, "redact_sensitive_text", fail)
    child = _StubChild([SECRET])
    events = []
    child.tool_progress_callback = lambda name, **kw: events.append((name, kw))
    entry = delegate._run_single_child(0, "finish safely", child, _Parent())
    assert SECRET not in json.dumps(entry)
    assert agent.redact.REDACTION_UNAVAILABLE in entry["summary"]
    assert entry["status"] == "completed"
    completion = next(kw for name, kw in events if name == "subagent.complete")
    assert completion["status"] == "completed"
    assert agent.redact.REDACTION_UNAVAILABLE in completion["preview"]
    assert SECRET not in json.dumps(completion)


def test_ordinary_text_and_accounting_survive_redaction():
    child = _StubChild(["Completed the requested analysis."])
    entry = delegate._run_single_child(0, "ordinary task", child, _Parent())
    assert entry["summary"] == "Completed the requested analysis."
    assert entry["status"] == "completed"
    assert entry["api_calls"] == 1
    assert "schema_valid" not in entry


def test_raised_child_error_and_completion_event_are_scrubbed():
    child = _StubChild([])
    child.run_conversation = Mock(side_effect=RuntimeError(SECRET))
    events = []
    child.tool_progress_callback = lambda name, **kw: events.append((name, kw))

    entry = delegate._run_single_child(0, "finish the task", child, _Parent())

    assert entry["status"] == "error"
    assert entry["error"] and SECRET not in json.dumps(entry)
    complete = next(kw for name, kw in events if name == "subagent.complete")
    assert complete["status"] == "error"
    assert complete["preview"] and SECRET not in json.dumps(complete)


def test_tool_preview_and_hook_targets_are_redacted_before_shortening():
    from agent.redact import redact_for_egress

    tail = results._extract_output_tail(
        {"messages": [{"role": "tool", "content": PEM}]}, max_chars=160
    )
    assert tail[0]["preview"] == redact_for_egress(PEM)[:160]
    arguments = json.dumps({"path": PEM, PEM: "ordinary payload"})
    summary = results._summarize_tool_arguments(arguments)
    assert summary["targets"]["path"] == redact_for_egress(PEM)[:1024]
    assert summary["argument_keys"] == [redact_for_egress(PEM)[:128], "path"]
    history = results._subagent_stop_tool_call_history(
        [{"tool": "read_file", "input_summary": summary}]
    )
    assert "c3ludGhldGlj" not in json.dumps(history)


def test_process_receipts_redact_complete_buffers_before_taking_tails(monkeypatch):
    from agent.redact import redact_for_egress
    from tools.delegate_tool_child_run import _ChildRun

    child = _StubChild([])
    run = _ChildRun(child, _Parent(), 0, "task", None, None)
    process = SimpleNamespace(
        id="process-id",
        command=PEM,
        started_at=time.time(),
        exit_code=0,
        output_buffer=PEM,
    )
    monkeypatch.setattr(process_registry, "running_owned_by", lambda _: [process])
    monkeypatch.setattr(
        process_registry, "unread_completions_owned_by", lambda _: [process]
    )
    entry = {}

    run.account_background_processes(entry)

    assert entry["orphaned_processes"][0]["command"] == redact_for_egress(PEM)[:200]
    receipt = entry["unread_completions"][0]
    assert receipt["output_tail"] == redact_for_egress(PEM)[-600:]
    assert receipt["session_id"] == "process-id" and receipt["exit_code"] == 0


def test_parent_instructions_and_delivery_identity_are_not_redacted():
    from tools.delegation_redaction import redact_completion_event

    parent_fields = {
        key: SECRET
        for key in (
            "delegation_id",
            "session_key",
            "parent_session_id",
            "origin_session_id",
            "origin_ui_session_id",
            "goal",
            "context",
            "model",
            "scope_id",
            "user_id",
            "user_name",
            "group",
        )
    }
    event = {**parent_fields, "goals": [SECRET], "results": [{"summary": SECRET}]}

    cleaned = redact_completion_event(event)

    assert all(cleaned[key] == value for key, value in parent_fields.items())
    assert cleaned["goals"] == [SECRET]
    assert SECRET not in json.dumps(cleaned["results"])
    assert event["results"][0]["summary"] == SECRET


def test_worker_ipc_receives_only_redacted_results_and_events(monkeypatch):
    remote = Mock()
    monkeypatch.setattr(ad, "worker_ledger", lambda: remote)

    ad._persist_completion({"goal": SECRET, "summary": SECRET}, {"summary": SECRET})
    ad.record_unit_child("delegation-id", {"task_index": 2, "summary": SECRET})

    event, result = remote.complete.call_args.args
    assert event["goal"] == SECRET
    assert SECRET not in event["summary"] and SECRET not in result["summary"]
    assert SECRET not in json.dumps(remote.child.call_args.args[1])
    remote.read.return_value = {"result": {"summary": SECRET}, "origin_session": SECRET}
    saved = ad.get_durable_delegation("delegation-id")
    assert saved is not None
    assert saved["origin_session"] == SECRET
    assert SECRET not in saved["result"]["summary"]


def test_owner_recovery_sanitizes_old_partial_results_and_diagnostics(monkeypatch):
    import tools.async_delegation_recovery_hints as hints

    record = _record()
    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET task_json=?,result_json=?",
            (
                json.dumps({"is_batch": True, "goals": [record["goal"]]}),
                json.dumps(
                    {
                        "partial": True,
                        "results": [
                            {"task_index": 0, "summary": SECRET, "status": "completed"}
                        ],
                    }
                ),
            ),
        )
    monkeypatch.setattr(ad, "_owner_liveness", lambda: lambda *_: False)
    monkeypatch.setattr(hints, "transcript_tails", lambda _: {"0": SECRET})
    monkeypatch.setattr(hints, "git_state_hint", lambda _: {"status": SECRET})

    assert ad.recover_abandoned_delegations() == 1

    with ad._transaction() as conn:
        event_json, result_json = conn.execute(
            "SELECT event_json,result_json FROM async_delegations"
        ).fetchone()
    assert SECRET not in event_json + result_json
    assert json.loads(result_json)["results"][0]["status"] == "completed"


def test_batch_failure_notice_scrubs_entry_without_finishing_batch(monkeypatch):
    record = _record()
    record["status"] = "running"
    with ad._records_lock:
        ad._records[record["delegation_id"]] = record
    target = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", target)

    ad.push_task_failure_notice(
        record["delegation_id"],
        {"task_index": 1, "status": "failed", "error": SECRET},
        n_tasks=2,
    )

    event = target.get_nowait()
    assert event["status"] == "running" and event["task_failure_notice"] is True
    assert event["results"][0]["status"] == "failed"
    assert SECRET not in json.dumps(event)
    assert ad._records[record["delegation_id"]]["status"] == "running"


def test_structured_summary_is_redacted_per_value_and_stays_valid_json():
    """A text pass over JSON lets ``KEY=***`` swallow the closing quote and comma."""
    from tools.delegation_redaction import redact_child_result

    answer = {"env": "OPENAI_API_KEY=sk-proj-" + "a" * 40, "n": 2}
    entry = redact_child_result(
        {"summary": json.dumps(answer), "schema_valid": True}, {"type": "object"}
    )

    parsed = json.loads(entry["summary"])
    assert parsed["n"] == 2 and "sk-proj-" + "a" * 40 not in parsed["env"]
    assert entry["schema_valid"] is True


def test_parent_copy_honours_opt_out_while_persisted_copies_stay_forced(
    tmp_path, monkeypatch
):
    """security.redact_secrets=false: the parent model reads the child's text as-is;
    spill file, memory provider, lifecycle hook and the durable row are still masked."""
    import hermes_cli.plugins

    monkeypatch.setattr("agent.redact._redact_enabled", lambda: False)
    hooks = Mock()
    monkeypatch.setattr(hermes_cli.plugins, "invoke_hook", hooks)
    monkeypatch.setattr(delegate, "_load_config", lambda: {"max_summary_chars": 2000})
    summary = SECRET + "\n" + "ordinary explanation\n" * 180
    parent = _Parent()

    entry = results._run_child_lifecycle(0, "summarize", _StubChild([summary]), parent)

    assert SECRET in entry["summary"]
    assert SECRET not in parent._memory_manager.on_delegation.call_args.kwargs["result"]
    stop = next(c.kwargs for c in hooks.call_args_list if c.args[0] == "subagent_stop")
    assert SECRET not in stop["child_summary"]
    spill = next((tmp_path / "cache" / "delegation").glob("subagent-summary-*.txt"))
    assert SECRET not in spill.read_text(encoding="utf-8-sig")

    record, target = _record(), queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", target)
    ad._push_completion_event(record, {"status": "completed", "summary": SECRET}, "completed")
    assert SECRET in target.get_nowait()["summary"]
    with ad._transaction() as conn:
        row = conn.execute("SELECT event_json,result_json FROM async_delegations").fetchone()
    assert SECRET not in "".join(row)
