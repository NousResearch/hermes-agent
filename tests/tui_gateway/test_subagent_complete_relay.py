"""Regression: a delegated child's completion must survive the typed relay.

``_ChildRun.emit_complete`` always puts ``cost_usd`` on the event (the child's cost
counter starts at ``0.0``) and adds ``failure_reason`` on a classified failure. Both
ride the relay's typed frame, whose model sets ``extra="forbid"``: with the fields
undeclared the payload build raised, ``_safe_progress`` swallowed it at debug, and
NEITHER terminal event was published — ``subagent.complete`` never reached the
gateway, so the child's live-registry entry was never cleared and the child's next
prompt answered ``4009 subagent still running`` until the 3600 s stale window.

The pre-existing subagent tests hand-build the payload (see
``test_subagent_child_mirror.py::_relay``), which is exactly why they never saw it.
These drive the REAL producer through the REAL relay into the gateway's
``_on_tool_progress`` sink.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool_child_run import _ChildRun
from tools.delegate_tool_progress import _build_child_progress_callback


@pytest.fixture()
def server():
    # Mocks are scoped to the initial import only (see tests/tui_gateway/test_protocol.py).
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_subagent_complete")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")

    yield mod
    mod._sessions.clear()
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()
    mod._child_mirrors.clear()
    mod._active_child_runs.clear()


@pytest.fixture()
def emits(server, monkeypatch):
    """Captured ``(event, sid, payload)`` with the payload model dumped to wire keys."""
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append(
            (event, sid, None if payload is None else payload.model_dump(exclude_none=True))
        ),
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda sid: True)
    return captured


class _ChildStub:
    """Minimal child agent: ``emit_complete`` reads the session counters off it."""

    session_estimated_cost_usd = 0.0
    session_prompt_tokens = 12
    session_completion_tokens = 34
    session_reasoning_tokens = 5


class _ParentStub:
    def __init__(self, cb):
        self.tool_progress_callback = cb
        self._delegate_spinner = None
        self.session_id = "parent-session"


def _child_run(server, *, child=None):
    """A real ``_ChildRun`` wired to the real relay; the relay's parent callback is the gateway sink."""
    parent = _ParentStub(
        lambda event_type, name=None, preview=None, args=None, **kw: server._on_tool_progress(
            "parent-sid", event_type, name, preview, args, **kw
        )
    )
    relay = _build_child_progress_callback(
        0, "research X", parent, 1, subagent_id="sa-1", session_ref={"session_id": "child-1"}
    )
    assert relay is not None  # the parent callback above is what builds it
    return (
        _ChildRun(
            child=child or _ChildStub(),
            parent_agent=parent,
            task_index=0,
            goal="research X",
            subagent_id="sa-1",
            child_progress_cb=relay,
        ),
        relay,
    )


def _entry(**extra):
    entry = {"status": "completed", "summary": "done deal", "api_calls": 2, "error": ""}
    entry.update(extra)
    return entry


def test_emit_complete_publishes_both_terminal_events_and_clears_the_live_entry(server, emits):
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}
    run, relay = _child_run(server)

    # The child started: the relay stamps its liveness entry.
    relay("subagent.start", preview="research X")
    assert server._child_run_active("child-1") is True

    run.emit_complete({"result": "ok"}, _entry(), 1.5)

    assert [e for e, s, _ in emits if s == "parent-sid"] == ["subagent.start", "subagent.complete"], emits
    # The child's window mirrors the run and closes on the summary. Asserted as a shape (first/last
    # + the delta between) rather than an exact list so a future mirror event is not a false red.
    child_events = [e for e, s, _ in emits if s == "live-1"]
    assert child_events[0] == "message.start" and child_events[-1] == "message.complete", emits
    assert "message.delta" in child_events, emits
    # Liveness is cleared, so a resume of the child is not refused with 4009.
    assert server._child_run_active("child-1") is False
    assert "child-1" not in server._active_child_runs
    assert server._child_mirrors == {}

    payload = next(p for e, s, p in emits if e == "subagent.complete")
    assert payload["cost_usd"] == 0.0
    assert payload["status"] == "completed"
    assert payload["child_session_id"] == "child-1"


def test_emit_complete_carries_a_classified_failure_reason(server, emits):
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}
    run, relay = _child_run(server)
    relay("subagent.start", preview="research X")

    run.emit_complete({"result": ""}, _entry(status="failed", error="boom", failure_reason="timeout"), 0.7)

    payload = next(p for e, s, p in emits if e == "subagent.complete")
    assert payload["failure_reason"] == "timeout"
    assert payload["cost_usd"] == 0.0
    assert [e for e, s, _ in emits if s == "live-1"][-1] == "message.complete"
    assert server._child_run_active("child-1") is False


def test_a_frame_the_contract_refuses_is_refused_loudly(server, emits, caplog):
    """The undeclared-field case that started this: the frame is still dropped, but not silently.

    Before the fields were declared the build escaped into ``_safe_progress``, which logged at DEBUG —
    the terminal event vanished and the child stayed "running". Any future producer field now says so
    at WARNING, which is how this class gets caught instead of shipped.
    """
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}
    _run, relay = _child_run(server)
    relay("subagent.start", preview="research X")

    with caplog.at_level(logging.WARNING):
        relay("subagent.complete", preview="done", status="completed", undeclared_field=1)

    assert "subagent.complete" not in [e for e, _, _ in emits], emits
    # The warning names the offending field, so the operator does not have to reproduce it.
    assert any("undeclared_field" in message for message in caplog.messages), caplog.messages
    # A refused completion still leaves the child pinned "running" — the entry stamped on
    # `subagent.start` is reclaimed only once it ages past the stale window (see
    # test_subagent_child_mirror.py::test_stale_child_run_not_reported_active). Forwarding a
    # stripped frame instead is deliberate: the producer bug gets fixed, never papered over.
    assert server._child_run_active("child-1") is True


def test_the_sink_reports_a_frame_it_cannot_build(server, emits, caplog):
    """The gateway side of the same class: junk from a producer used to vanish at DEBUG."""
    with caplog.at_level(logging.WARNING):
        server._on_tool_progress(
            "parent-sid", "subagent.complete", None, None, None,
            goal="research X", task_count=1, task_index=0, duration_seconds="not-a-number",
        )

    assert "subagent.complete" not in [e for e, _, _ in emits], emits
    assert any("gateway sink" in message for message in caplog.messages), caplog.messages


def test_the_sink_allow_list_covers_every_declared_payload_field():
    """The drift guard: the kwargs-rebuild path may only omit the fields it builds by hand.

    This is the pair that silently diverged and shipped the dropped completion — the model and the
    allow-list have to agree, so assert the relationship rather than either side alone.
    """
    from tui_gateway.contracts.events import SubagentEventPayload
    from tui_gateway.tool_progress import _SUBAGENT_FIELDS

    handled_positionally = {"goal", "task_count", "task_index", "tool_preview"}
    assert {key for key, _, _ in _SUBAGENT_FIELDS} | handled_positionally == set(SubagentEventPayload.model_fields)
