"""Channel-neutral, privacy-empty detached status contracts."""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator


class DetachedStatusPhase(str, Enum):
    WAITING = "waiting"
    THINKING = "thinking"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DetachedStatusEvent:
    task_index: int
    depth: int
    phase: DetachedStatusPhase
    at: float


_THINKING_EVENTS = frozenset(
    {
        "_thinking",
        "reasoning.available",
        "tool.started",
        "tool.completed",
        "subagent_progress",
        "delegate.task_thinking",
        "delegate.tool_started",
        "delegate.tool_completed",
        "delegate.task_progress",
    }
)


_DETACHED_STATUS_OWNER: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "hermes_detached_status_owner", default=None
)


def get_detached_status_owner() -> object | None:
    return _DETACHED_STATUS_OWNER.get()


def attach_detached_status_sink(
    child: Any,
    sink: Any,
    *,
    task_index: int,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    display_callback = getattr(child, "tool_progress_callback", None)

    def _tee(event_type: Any, *args: Any, **kwargs: Any) -> None:
        if display_callback is not None:
            try:
                display_callback(event_type, *args, **kwargs)
            except Exception:
                pass

        name = getattr(event_type, "value", event_type)
        if name == "subagent.start":
            phase = DetachedStatusPhase.WAITING
        elif name in _THINKING_EVENTS:
            phase = DetachedStatusPhase.THINKING
        elif name == "subagent.complete":
            phase = (
                DetachedStatusPhase.DONE
                if kwargs.get("status") in ("completed", "success")
                else DetachedStatusPhase.FAILED
            )
        else:
            return
        depth = kwargs.get("depth", 1)
        sink.observe(
            DetachedStatusEvent(
                task_index=task_index,
                depth=depth if isinstance(depth, int) else 1,
                phase=phase,
                at=clock(),
            )
        )

    child.tool_progress_callback = _tee


@contextmanager
def bind_detached_status_owner(owner: object | None) -> Iterator[None]:
    token = _DETACHED_STATUS_OWNER.set(owner)
    try:
        yield
    finally:
        _DETACHED_STATUS_OWNER.reset(token)
