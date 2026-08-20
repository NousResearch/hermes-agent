from dataclasses import fields
from typing import Any

from tools.delegation_status import (
    DetachedStatusEvent,
    DetachedStatusPhase,
    attach_detached_status_sink,
    bind_detached_status_owner,
    get_detached_status_owner,
)


def test_detached_status_contract_is_privacy_empty_and_turn_scoped():
    owner = object()

    assert get_detached_status_owner() is None
    with bind_detached_status_owner(owner):
        assert get_detached_status_owner() is owner
        assert tuple(field.name for field in fields(DetachedStatusEvent)) == (
            "task_index",
            "depth",
            "phase",
            "at",
        )
        assert DetachedStatusEvent(
            task_index=0,
            depth=1,
            phase=DetachedStatusPhase.WAITING,
            at=10.0,
        ).phase is DetachedStatusPhase.WAITING
    assert get_detached_status_owner() is None


def test_structural_tee_preserves_display_callback_without_copying_content():
    class Child:
        def __init__(self):
            self.tool_progress_callback: Any = None

    class Sink:
        def __init__(self):
            self.events = []

        def observe(self, event):
            self.events.append(event)

    child = Child()
    display_events = []
    child.tool_progress_callback = (
        lambda event_type, *args, **kwargs: display_events.append(
            (event_type, args, kwargs)
        )
    )
    sink = Sink()

    attach_detached_status_sink(child, sink, task_index=2, clock=lambda: 42.0)
    child.tool_progress_callback(
        "subagent.start", preview="secret goal", model="secret model"
    )
    child.tool_progress_callback("_thinking", "secret reasoning", depth=2)
    child.tool_progress_callback(
        "subagent.complete", summary="secret summary", status="failed"
    )

    assert [event[0] for event in display_events] == [
        "subagent.start",
        "_thinking",
        "subagent.complete",
    ]
    assert sink.events == [
        DetachedStatusEvent(2, 1, DetachedStatusPhase.WAITING, 42.0),
        DetachedStatusEvent(2, 2, DetachedStatusPhase.THINKING, 42.0),
        DetachedStatusEvent(2, 1, DetachedStatusPhase.FAILED, 42.0),
    ]


def test_display_callback_failure_does_not_block_structural_status():
    class Child:
        def __init__(self):
            self.tool_progress_callback: Any = None

    class Sink:
        def __init__(self):
            self.events = []

        def observe(self, event):
            self.events.append(event)

    child = Child()

    def broken_display(*args, **kwargs):
        raise RuntimeError("display broke")

    child.tool_progress_callback = broken_display
    sink = Sink()
    attach_detached_status_sink(child, sink, task_index=0, clock=lambda: 7.0)

    child.tool_progress_callback("_thinking", "private reasoning")

    assert sink.events == [
        DetachedStatusEvent(0, 1, DetachedStatusPhase.THINKING, 7.0)
    ]


def test_reasoning_tool_and_progress_events_project_as_thinking():
    class Child:
        def __init__(self):
            self.tool_progress_callback: Any = None

    class Sink:
        def __init__(self):
            self.events = []

        def observe(self, event):
            self.events.append(event)

    child = Child()
    sink = Sink()
    attach_detached_status_sink(child, sink, task_index=1, clock=lambda: 8.0)

    event_types = (
        "_thinking",
        "reasoning.available",
        "tool.started",
        "tool.completed",
        "subagent_progress",
        "delegate.task_thinking",
        "delegate.tool_started",
        "delegate.tool_completed",
        "delegate.task_progress",
    )
    for event_type in event_types:
        child.tool_progress_callback(event_type)

    assert [event.phase for event in sink.events] == [
        DetachedStatusPhase.THINKING for _ in event_types
    ]


def test_only_explicit_success_completion_projects_as_done():
    class Child:
        def __init__(self):
            self.tool_progress_callback: Any = None

    class Sink:
        def __init__(self):
            self.events = []

        def observe(self, event):
            self.events.append(event)

    child = Child()
    sink = Sink()
    attach_detached_status_sink(child, sink, task_index=0, clock=lambda: 9.0)

    statuses = ("completed", "success", "failed", "error", "timeout", "stalled", "cancelled")
    for status in statuses:
        child.tool_progress_callback("subagent.complete", status=status)

    assert [event.phase for event in sink.events] == [
        DetachedStatusPhase.DONE,
        DetachedStatusPhase.DONE,
        DetachedStatusPhase.FAILED,
        DetachedStatusPhase.FAILED,
        DetachedStatusPhase.FAILED,
        DetachedStatusPhase.FAILED,
        DetachedStatusPhase.FAILED,
    ]
