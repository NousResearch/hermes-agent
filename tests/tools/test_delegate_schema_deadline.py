"""Exercise schema correction through the real child worker and deadline."""

import threading
import uuid
from types import SimpleNamespace

import pytest

from tools import delegate_tool as dt
from tools import delegate_tool_registry as registry
from tools.delegate_tool_child_run import _ChildRun
from tools.delegation_output_schema import validate_output

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


class Child:
    def __init__(self, responses, *, retry_delay=0.0, initial_delay=0.0):
        self._subagent_id = "schema-deadline-" + uuid.uuid4().hex
        self.session_id = self._subagent_id
        self._delegate_output_schema = SCHEMA
        self.responses = responses
        self.retry_delay = retry_delay
        self.initial_delay = initial_delay
        self.calls = []
        self.stop = threading.Event()
        self.closed = threading.Event()
        self.close_count = 0
        self.active = False
        self.events = []
        self.tool_progress_callback = lambda name, **kw: self.events.append((name, kw))

    def run_conversation(self, user_message, task_id, **kwargs):
        lease = getattr(self, "_delegate_reviewed_deadline", None)
        self.calls.append({"thread": threading.current_thread(), "task_id": task_id,
                           "lease": lease, "snapshot": lease.snapshot() if lease else None,
                           "registered": self._subagent_id in registry._active_subagents,
                           "message": user_message})
        self.active = True
        try:
            if len(self.calls) == 1:
                self.stop.wait(self.initial_delay)
            else:
                self.stop.wait(self.retry_delay)
            response = self.responses[len(self.calls) - 1]
            if isinstance(response, Exception):
                raise response
            return {"final_response": response, "completed": True, "api_calls": len(self.calls),
                    "messages": [], "interrupted": self.stop.is_set()}
        finally:
            self.active = False

    def get_activity_summary(self):
        return {"api_call_count": len(self.calls)}

    def interrupt(self):
        self.stop.set()

    def close(self):
        assert not self.active
        self.close_count += 1
        self.closed.set()


@pytest.fixture
def run_child(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Keep real await/schema/registry/cleanup; only avoid workspace side effects.
    monkeypatch.setattr(_ChildRun, "seed_workspace", lambda self: setattr(self, "child_task_id", self.subagent_id))
    # Load jsonschema outside the deliberately short worker deadline.
    validate_output('{"ok":true}', SCHEMA)

    def run(child, *, reviewed, timeout=0.15):
        monkeypatch.setattr(dt, "_load_config", lambda: {"reviewed_timeout": reviewed})
        monkeypatch.setattr(dt, "_get_child_timeout", lambda: timeout)
        parent = SimpleNamespace(session_id="schema-test-parent", _active_children=[child])
        entry = dt._run_single_child(0, "return JSON", child, parent)
        assert child.closed.wait(2), "worker never settled/closed"
        worker = child.calls[0]["thread"]
        worker.join(2)
        assert not worker.is_alive(), "timed-out worker still running"
        assert child.close_count == 1
        assert child not in parent._active_children
        assert child._subagent_id not in registry._active_subagents
        assert not child.active
        return entry

    return run


@pytest.mark.parametrize("reviewed", [True, False])
def test_schema_retry_obeys_original_deadline(run_child, reviewed):
    child = Child(["invalid", '{"ok":true}'], retry_delay=0.45)
    entry = run_child(child, reviewed=reviewed)
    assert entry["status"] == "timeout"
    assert entry["exit_reason"] == "timeout"
    assert entry["summary"] is None
    assert entry["api_calls"] == 2
    assert entry["timeout_seconds"] == 0.15
    assert child.stop.is_set()
    assert len(child.calls) == 2
    assert all(call["registered"] for call in child.calls)
    assert child.calls[0]["thread"] is child.calls[1]["thread"]
    assert child.calls[1]["task_id"] == child.calls[0]["task_id"]
    assert [event[1]["status"] for event in child.events if event[0] == "subagent.complete"] == ["timeout"]
    if reviewed:
        assert "Parent-reviewed deadline expired" in entry["error"]
        assert not child.calls[1]["snapshot"]["closed"]
        assert child.calls[0]["lease"] is child.calls[1]["lease"]
        assert child.calls[0]["lease"].snapshot()["closed"]
        assert child.calls[0]["lease"].snapshot()["renewals"] == 0
    else:
        assert not hasattr(child, "_delegate_reviewed_deadline")
        assert "Subagent timed out" in entry["error"]


@pytest.mark.parametrize("reviewed", [True, False])
def test_expiry_during_validation_does_not_start_correction(run_child, monkeypatch, reviewed):
    from tools import delegation_output_schema

    entered = threading.Event()
    release = threading.Event()
    original = delegation_output_schema.validate_output

    def slow_validation(text, schema):
        if text == "invalid":
            entered.set()
            release.wait(1)
        return original(text, schema)

    child = Child(["invalid", '{"ok":true}'])
    # Release validation only AFTER timeout has sent the original interrupt.
    child.interrupt = release.set
    monkeypatch.setattr(delegation_output_schema, "validate_output", slow_validation)
    try:
        entry = run_child(child, reviewed=reviewed)
    finally:
        release.set()
    assert entered.is_set()
    assert entry["status"] == "timeout"
    assert len(child.calls) == 1, "expired worker started a new correction turn"


@pytest.mark.parametrize("reviewed", [True, False])
def test_initial_turn_and_retry_share_budget(run_child, reviewed):
    child = Child(["invalid", '{"ok":true}'], initial_delay=0.18, retry_delay=0.22)
    entry = run_child(child, reviewed=reviewed, timeout=0.3)
    # Reviewed mode owns both turns; ordinary mode retains upstream inactivity renewal.
    assert entry["status"] == ("timeout" if reviewed else "completed")
    assert child.stop.is_set() is reviewed
    assert len(child.calls) == 2


@pytest.mark.parametrize("reviewed", [True, False])
@pytest.mark.parametrize("responses,valid,retries,status,api_calls", [
    (['{"ok":true}'], True, 0, "completed", 1),
    (["invalid", '{"ok":true}'], True, 1, "completed", 3),
    (["invalid", "{}"], False, 1, "completed", 3),
    (["invalid", RuntimeError("retry failed")], False, 1, "completed", 1),
    (["invalid", TimeoutError("provider timeout")], False, 1, "completed", 1),
    ([""], False, 0, "failed", 1),
])
def test_schema_result_contract(run_child, monkeypatch, reviewed, responses, valid, retries, status, api_calls):
    from tools import delegation_output_schema

    validations = []
    original = delegation_output_schema.validate_output

    def record_validation(text, schema):
        validations.append(text)
        return original(text, schema)

    monkeypatch.setattr(delegation_output_schema, "validate_output", record_validation)
    child = Child(responses)
    entry = run_child(child, reviewed=reviewed, timeout=1.0)
    assert entry["status"] == status
    assert entry["schema_valid"] is valid
    assert entry.get("schema_retries", 0) == retries
    assert bool(entry.get("schema_errors")) is not valid
    assert entry["api_calls"] == api_calls
    expected_texts = [response for response in responses if isinstance(response, str)]
    assert validations == expected_texts  # No second validation/correction outside the worker.
    assert entry["summary"] == expected_texts[-1]
    assert len(child.calls) == 1 + retries
    assert all(call["registered"] for call in child.calls)
    assert all(call["thread"] is child.calls[0]["thread"] for call in child.calls)
    assert all(call["thread"] is not threading.current_thread() for call in child.calls)
    if reviewed:
        assert all(not call["snapshot"]["closed"] for call in child.calls)
        assert all(call["lease"] is child.calls[0]["lease"] for call in child.calls)
        assert child.calls[0]["lease"].snapshot()["closed"]


@pytest.mark.parametrize("reviewed,timeout", [(True, 1.0), (False, 1.0), (False, None)])
def test_schema_less_entry_keeps_original_shape(run_child, reviewed, timeout):
    child = Child(["plain text"])
    del child._delegate_output_schema
    entry = run_child(child, reviewed=reviewed, timeout=timeout)
    assert entry["status"] == "completed"
    assert entry["summary"] == "plain text"
    assert entry["api_calls"] == 1
    assert len(child.calls) == 1
    assert not {"schema_valid", "schema_errors", "schema_retries"}.intersection(entry)


def test_schema_retry_keeps_unlimited_nonreviewed_config(run_child):
    child = Child(["invalid", '{"ok":true}'])
    entry = run_child(child, reviewed=False, timeout=None)
    assert entry["status"] == "completed"
    assert entry["schema_retries"] == 1
    assert not hasattr(child, "_delegate_reviewed_deadline")


def test_parent_can_renew_original_lease_during_schema_correction(run_child):
    child = Child(["invalid", '{"ok":true}'], initial_delay=0.15, retry_delay=0.45)
    original = child.run_conversation
    timers = []
    reviews = []

    def conversation(user_message, task_id, **kwargs):
        if child.calls:
            lease = child.calls[0]["lease"]
            lease.report(1)

            def parent_review():
                try:
                    reviews.append(lease.review(1, approve=True))
                except Exception as exc:
                    reviews.append(exc)

            timer = threading.Timer(0.2, parent_review)
            timers.append(timer)
            timer.start()
        return original(user_message, task_id, **kwargs)

    child.run_conversation = conversation
    try:
        entry = run_child(child, reviewed=True, timeout=0.5)
    finally:
        for timer in timers:
            timer.join(2)
    assert entry["status"] == "completed"
    assert entry["schema_valid"] is True
    assert entry["schema_retries"] == 1
    assert entry["duration_seconds"] >= 0.5
    assert len(reviews) == 1 and reviews[0]["approved"] is True
    lease = child.calls[0]["lease"]
    assert lease is child.calls[1]["lease"]
    assert lease.snapshot()["renewals"] == 1
    assert lease.snapshot()["closed"]
    assert not child.stop.is_set()


def test_report_during_correction_is_preserved_but_does_not_renew(run_child):
    child = Child(["invalid", '{"ok":true}'], retry_delay=0.45)
    original = child.run_conversation

    def conversation(user_message, task_id, **kwargs):
        if child.calls:
            child.calls[0]["lease"].report(1)
        return original(user_message, task_id, **kwargs)

    child.run_conversation = conversation
    entry = run_child(child, reviewed=True)
    assert entry["status"] == "timeout"
    lease = child.calls[0]["lease"]
    snapshot = lease.snapshot()
    assert snapshot["closed"]
    assert snapshot["pending_checkpoint"] == 1
    assert snapshot["renewals"] == 0
    assert child.stop.is_set()
