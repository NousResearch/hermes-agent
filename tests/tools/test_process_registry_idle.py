"""``list_sessions`` reports how long a running process has been silent.

A status surface uses ``idle_seconds`` to tell live work from a quiet leftover (a preview server
nobody is hitting, a ``find`` hung on a dead network mount) that still keeps the session "running".
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _running_session(registry, started_at: float):
    from tools.process_registry import ProcessSession

    session = ProcessSession(
        id="proc_quiet",
        command="python -m http.server 8731",
        task_id="task",
        owner_task_id="task",
        session_key="sess",
        started_at=started_at,
    )
    session.process = cast(Any, SimpleNamespace(poll=lambda: None, stdout=None, stderr=None, stdin=None))
    registry._running[session.id] = session
    return session


@pytest.fixture
def clock(monkeypatch):
    import tools.process_registry as module

    fake = _Clock(1_000.0)
    monkeypatch.setattr(module.time, "time", fake)
    return fake


@pytest.mark.platforms("posix")
def test_idle_counts_from_the_last_output_not_the_spawn(clock):
    from tools.process_registry import ProcessRegistry

    registry = ProcessRegistry()
    session = _running_session(registry, started_at=clock.now)

    clock.now += 100
    session.append_output("Serving HTTP on 127.0.0.1 port 8731\n")
    clock.now += 300

    [entry] = registry.list_sessions(session_key="sess")

    assert entry["status"] == "running"
    assert entry["uptime_seconds"] == 400
    assert entry["idle_seconds"] == 300


@pytest.mark.platforms("posix")
def test_a_process_that_never_printed_is_idle_since_spawn(clock):
    from tools.process_registry import ProcessRegistry

    registry = ProcessRegistry()
    _running_session(registry, started_at=clock.now)
    clock.now += 9_900

    [entry] = registry.list_sessions(session_key="sess")

    assert entry["idle_seconds"] == entry["uptime_seconds"] == 9_900
