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


@pytest.mark.parametrize(
    "wrap",
    [lambda s: s, lambda s: "```json\n" + s + "\n```", lambda s: "Here is the answer:\n" + s + "\nDone."],
    ids=["bare", "fenced", "prose"],
)
def test_structured_summary_is_redacted_per_value_and_stays_valid_json(wrap):
    """A text pass over JSON lets ``KEY=***`` swallow the closing quote and comma; every
    wrapper the schema validator accepts must survive redaction the same way."""
    from tools.delegation_output_schema import extract_json_candidate
    from tools.delegation_redaction import redact_child_result

    answer = {"env": "OPENAI_API_KEY=sk-proj-" + "a" * 40, "n": 2}
    entry = redact_child_result(
        {"summary": wrap(json.dumps(answer)), "schema_valid": True}, {"type": "object"}
    )

    parsed = json.loads(extract_json_candidate(entry["summary"]))
    assert parsed["n"] == 2 and "sk-proj-" + "a" * 40 not in parsed["env"]
    assert entry["schema_valid"] is True


@pytest.mark.parametrize(
    "answer,schema",
    [
        ("OPENAI_API_KEY=opaqueSyntheticValueXYZ1234", {"type": "string"}),
        ("ordinary scalar answer", {"type": "string"}),
        (None, {"type": "null"}),
        (True, {"type": "boolean"}),
        (17, {"type": "integer"}),
        (1.5, {"type": "number"}),
    ],
)
def test_structured_scalar_summary_stays_valid_json(answer, schema):
    summary = json.dumps(answer)
    child = _StubChild([summary])
    child._delegate_output_schema = schema

    entry = delegate._run_single_child(0, "produce structured output", child, _Parent())

    delivered = json.loads(entry["summary"])
    assert entry["schema_valid"] is True
    assert len(child.calls) == 1
    if isinstance(answer, str) and answer.startswith("OPENAI_API_KEY="):
        assert "opaqueSyntheticValueXYZ1234" not in delivered
        assert delivered.startswith("OPENAI_API_KEY=")
    else:
        assert entry["summary"] == summary


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


def test_forced_copies_are_projected_from_complete_output_before_the_parent_budget(
    tmp_path, monkeypatch
):
    """Opt-out + budget: the trimmed parent tail can start after a PEM BEGIN marker; the
    memory/durable copies must still hide the key body, and the spill must stay valid JSON."""
    monkeypatch.setattr("agent.redact._redact_enabled", lambda: False)
    monkeypatch.setattr(delegate, "_load_config", lambda: {"max_summary_chars": 2000})
    body = "\n".join("MC4CAQAwBQYDK2VwBCIEI" + chr(65 + i) * 43 for i in range(4))
    pem = "-----BEGIN PRIVATE KEY-----\n" + body + "\n-----END PRIVATE KEY-----\n"
    tail = pem[pem.index("KEY-----\n") :] + "ordinary ending\n" * 10
    tail += "x" * (500 - len(tail))  # the parent's 500-char tail starts inside the BEGIN line
    summaries = ["ordinary explanation\n" * 200 + pem[: pem.index("KEY-----\n")] + tail,
                 json.dumps({"env": "OPENAI_API_KEY=" + SECRET, "pad": "z" * 3000, "n": 2})]
    parent = _Parent()
    entries = [{"task_index": i, "status": "completed", "summary": s} for i, s in enumerate(summaries)]
    entries[1]["schema_valid"] = True
    schema_child = SimpleNamespace(_delegate_output_schema={"type": "object"})
    results._finalize_child_results(
        entries, [{"goal": "a"}, {"goal": "b"}], [(0, {}, None), (1, {}, schema_child)], parent
    )

    assert all(e["summary_truncated"] for e in entries) and body in entries[0]["summary"]
    memory = parent._memory_manager.on_delegation.call_args_list[0].kwargs["result"]
    assert body[-43:] not in memory
    spilled = json.loads(Path(entries[1]["summary_full_path"]).read_text(encoding="utf-8-sig"))
    assert spilled["n"] == 2 and SECRET not in spilled["env"]

    record, target = {**_record(), "is_batch": True}, queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", target)
    ad._push_completion_event(record, {"results": entries}, "completed")
    assert body in target.get_nowait()["results"][0]["summary"]
    with ad._transaction() as conn:
        row = conn.execute("SELECT event_json,result_json FROM async_delegations").fetchone()
    assert body[-43:] not in "".join(row) and SECRET not in "".join(row)
