"""A background child's spend and its classified stop reason must survive into the completion text.

A `delegate_task` fan-out that finishes as one unit is the parent's only view of children it never
watches: the tool result carries `cost_usd`/`tokens`/`failure_reason`, but the batch block rendered
for the parent dropped all three, so an expensive branch and a provider quota wall both read the
same as a cheap, ordinary failure.
"""

import queue
import time
from pathlib import Path

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry
from tools.process_registry_notifications import _format_async_delegation


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _evt(results):
    now = time.time()
    return {"type": "async_delegation", "delegation_id": "deleg-triage", "session_key": "",
            "goal": "review the diff", "goals": ["a", "b"], "is_batch": True,
            "results": results, "model": "test-model", "dispatched_at": now - 4, "completed_at": now}


def test_batch_block_names_cost_and_token_usage_per_child():
    text = _format_async_delegation(_evt([
        {"task_index": 0, "status": "completed", "summary": '{"verdict": "ok"}',
         "tokens": {"input": 12400, "output": 800}, "cost_usd": 0.00228, "cost_status": "estimated"},
    ]))

    assert "Usage: 12,400 in / 800 out · $0.00228 (estimated)" in text


def test_unpriced_route_says_unknown_instead_of_a_fake_zero():
    text = _format_async_delegation(_evt([
        {"task_index": 0, "status": "completed", "summary": "done",
         "tokens": {"input": 900, "output": 40}, "cost_usd": 0.0, "cost_status": "unknown"},
    ]))

    assert "cost unknown (route has no pricing entry)" in text
    assert "$0.0000" not in text


def test_failed_child_reports_its_classified_reason():
    text = _format_async_delegation(_evt([
        {"task_index": 0, "status": "failed", "error": "429 Too Many Requests",
         "failure_reason": "rate_limit", "tokens": {"input": 10, "output": 2},
         "cost_usd": 0.0, "cost_status": "unknown"},
    ]))

    assert "Reason: rate_limit" in text


def test_legacy_result_without_triage_fields_renders_no_extra_lines():
    text = _format_async_delegation(_evt([
        {"task_index": 0, "status": "completed", "summary": "done"},
    ]))

    assert "Usage:" not in text and "Reason:" not in text


def test_single_task_completion_payload_carries_triage_fields(monkeypatch):
    """The single-task payload whitelists its fields; a dropped key never reaches the formatter."""
    record = {"delegation_id": "deleg-single", "dispatched_at": time.time(), "session_key": "",
              "goal": "run the audit", "model": "test-model"}
    ad._persist_dispatch(record)
    target = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", target)
    ad._push_completion_event(record, {
        "status": "failed", "summary": "partial", "error": "429 Too Many Requests",
        "failure_reason": "rate_limit", "api_calls": 3, "duration_seconds": 2.5,
        "tokens": {"input": 5000, "output": 120}, "cost_usd": 0.0155, "cost_status": "estimated",
    }, "failed")

    text = _format_async_delegation(target.get_nowait())

    assert "Reason: rate_limit" in text
    assert "Usage: 5,000 in / 120 out · $0.0155 (estimated)" in text
    assert text.index("--- RESULT ---") < text.index("Usage:")
