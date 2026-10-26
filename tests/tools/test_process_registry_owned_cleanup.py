"""Native timeout cleanup preserves earlier work and carries the captured owner scope."""
from tools.process_registry import ProcessRegistry
from tests.tools.test_process_registry import _make_session


def test_kill_started_since_preserves_preexisting_and_foreign_processes():
    registry = ProcessRegistry()
    from tools.process_registry_handoff import ownership_identity
    old = _make_session(sid="proc_old", task_id="session-a")
    finished = _make_session(
        sid="proc_finished", task_id="session-a", exited=True, exit_code=0
    )
    registry._running[old.id] = old
    registry._finished[finished.id] = finished
    baseline = registry.snapshot_running_ids("session-a")

    new = _make_session(sid="proc_new", task_id="session-a")
    foreign = _make_session(sid="proc_foreign", task_id="session-b")
    registry._running[new.id] = new
    registry._running[foreign.id] = foreign

    calls = []

    def fake_kill(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"status": "killed"}

    registry.kill_process = fake_kill

    assert baseline == frozenset({"proc_old"})
    assert registry.kill_started_since(
        "session-a", baseline, source="gateway_turn_timeout"
    ) == 1
    assert calls == [
        (
            "proc_new",
            {
                "source": "gateway_turn_timeout",
                "consume_output": True,
                "expected_ownership": ownership_identity(new),
            },
        )
    ]
