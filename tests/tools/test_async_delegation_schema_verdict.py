"""A background delegate_task whose answer broke its output_schema must not read as a clean result."""

import queue
import time
from pathlib import Path

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry
from tools.process_registry_notifications import _format_async_delegation

ERROR = "'verdict' is a required property"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _complete(monkeypatch, record_extra, result, *, replay=False):
    """Format the live queued event, or (``replay=True``) the copy a restarted process restores from state.db."""
    record = {"delegation_id": "deleg-schema", "dispatched_at": time.time(), "session_key": "",
              "goal": "review the diff", "model": "test-model", **record_extra}
    ad._persist_dispatch(record)
    target = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", target)
    ad._push_completion_event(record, result, "completed")
    if replay:
        ad._reset_for_tests()
        target = queue.Queue()
        assert ad.restore_undelivered_completions(target) == 1
    return _format_async_delegation(target.get_nowait())


@pytest.mark.parametrize("replay", [False, True], ids=["live", "restart_replay"])
def test_single_async_completion_names_the_failed_schema_verdict(monkeypatch, replay):
    text = _complete(monkeypatch, {}, {
        "status": "completed", "summary": "looks fine to me", "schema_valid": False,
        "schema_errors": [ERROR], "schema_retries": 1}, replay=replay)

    assert "SCHEMA: invalid" in text and ERROR in text and "(after 1 retry)" in text
    assert text.index("SCHEMA: invalid") < text.index("looks fine to me")


@pytest.mark.parametrize("replay", [False, True], ids=["live", "restart_replay"])
def test_batch_async_completion_flags_only_the_task_that_failed_its_schema(monkeypatch, replay):
    text = _complete(monkeypatch, {"is_batch": True, "goals": ["a", "b"]}, replay=replay, result={"results": [
        {"task_index": 0, "status": "completed", "summary": "{\"verdict\": \"ok\"}", "schema_valid": True},
        {"task_index": 1, "status": "completed", "summary": "prose", "schema_valid": False,
         "schema_errors": [ERROR]},
    ]})

    assert "--- ✓ TASK 1/2: a" in text
    assert "--- ⚠ TASK 2/2: b" in text and "SCHEMA INVALID" in text and ERROR in text
    assert text.count("SCHEMA: invalid") == 1
