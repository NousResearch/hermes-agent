"""Display-only recovery of untyped delegation completions."""

from copy import deepcopy

import pytest

from hermes_state import SessionDB
from tools.process_registry_notifications import format_process_notification
from tui_gateway import server


@pytest.mark.parametrize("is_batch", [False, True])
def test_legacy_completion_projection_preserves_durable_context(tmp_path, is_batch):
    text = format_process_notification({
        "type": "async_delegation",
        "delegation_id": "deleg_example",
        "is_batch": is_batch,
        "summary": "A meaningful result",
        "status": "completed",
        "results": [{"status": "failed", "error": "test failure"}]
        if is_batch
        else None,
    })
    assert text is not None
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(session_id="legacy", source="cli")
        db.append_message("legacy", "user", text)
        db.append_message(
            "legacy", "assistant", "The test failed; more work is needed."
        )
        history = db.get_messages_as_conversation("legacy")
        before = deepcopy(history)
        projected = server._history_to_messages(history)
        assert projected[0].get("display_kind") == "async_delegation_complete"
        assert projected[0]["text"] == text
        assert projected[1]["text"] == before[1]["content"]
        assert history == before
        assert db.get_messages_as_conversation("legacy") == before


@pytest.mark.parametrize(
    "role,text",
    [
        ("assistant", "[ASYNC DELEGATION BATCH COMPLETE — deleg_example]\nResult"),
        ("user", "Explain [ASYNC DELEGATION BATCH COMPLETE — deleg_example]"),
        ("user", "[ASYNC DELEGATION BATCH COMPLETE — deleg_example] is a marker"),
        ("user", "[ASYNC DELEGATION BATCH COMPLETE — not-a-delegation]\nResult"),
        ("user", "[ASYNC DELEGATION BATCH COMPLETE — deleg_]\nResult"),
    ],
)
def test_legacy_projection_does_not_type_marker_mentions(role, text):
    assert server._legacy_display_kind(role, text) is None


def test_explicit_display_kind_wins_over_legacy_marker():
    projected = server._history_to_messages([
        {
            "role": "user",
            "content": "[ASYNC DELEGATION COMPLETE — deleg_example]\nResult",
            "display_kind": "internal_notification",
        }
    ])
    assert projected[0]["display_kind"] == "internal_notification"


@pytest.mark.parametrize("shutdown", [False, True])
def test_completion_start_is_typed_only_at_dispatch(monkeypatch, shutdown):
    import queue
    import threading

    from tools import async_delegation
    from tools.process_registry import process_registry

    stop = threading.Event()
    if shutdown:
        stop.set()
    session = {
        "session_key": "owner",
        "running": not shutdown,
        "history_lock": threading.Lock(),
    }
    event = {
        "type": "async_delegation",
        "session_key": "owner",
        "delegation_id": "deleg_live",
        "status": "completed",
        "summary": "Result",
    }
    pending = queue.Queue()
    pending.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", pending)
    monkeypatch.setattr(server, "_sessions", {"sid": session})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_loop_tick", lambda *a: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_heartbeat_tick", lambda *a: None)
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda *a: [])
    monkeypatch.setattr(async_delegation, "claim_event_delivery", lambda *a: "claim")
    completed = []
    monkeypatch.setattr(
        async_delegation, "complete_event_delivery", lambda *a: completed.append(a)
    )
    emitted = []
    dispatched = []
    monkeypatch.setattr(
        server, "_emit", lambda kind, sid, payload=None: emitted.append((kind, payload))
    )

    def finish_previous_turn(_seconds):
        # The completion can be announced while normal work is still running;
        # it must not reclassify that normal assistant turn.
        assert emitted and all(kind == "status.update" for kind, _ in emitted)
        session["running"] = False

    monkeypatch.setattr(server.time, "sleep", finish_previous_turn)

    def submit(rid, sid, target, text, **kwargs):
        dispatched.append((text, kwargs))
        stop.set()

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    server._notification_poller_loop(stop, "sid", session)

    assert len(dispatched) == 1
    starts = [payload for kind, payload in emitted if kind == "message.start"]
    assert starts == [
        {
            "display_kind": "async_delegation_complete",
            "display_metadata": server._async_delegation_display_metadata(event),
        }
    ]
    assert dispatched[0][1] == starts[0]
    assert dispatched[0][0] == format_process_notification(event)
    assert completed == [(event, "claim")]


@pytest.mark.parametrize(
    "event,expected",
    [
        ({"status": "failed"}, (1, 0, 1)),
        ({"status": "interrupted"}, (1, 0, 0)),
        ({"status": "completed", "truncated": True}, (1, 1, 0)),
        (
            {"is_batch": True, "status": "failed", "goals": ["a", "b"], "results": []},
            (2, 0, 2),
        ),
        (
            {
                "is_batch": True,
                "results": [{"status": "interrupted"}, {"status": "failed"}],
            },
            (2, 0, 1),
        ),
        (
            {
                "is_batch": True,
                "results": [{"status": "completed"}, {"status": "running"}],
            },
            (2, 1, 0),
        ),
    ],
)
def test_completion_metadata_does_not_invent_success(event, expected):
    metadata = server._async_delegation_display_metadata({
        "delegation_id": "deleg_truth",
        **event,
    })
    assert (
        tuple(
            metadata[key] for key in ("task_count", "completed_count", "failed_count")
        )
        == expected
    )


def test_completion_metadata_keeps_zero_duration():
    assert (
        server._async_delegation_display_metadata({"duration_seconds": 0})[
            "duration_seconds"
        ]
        == 0
    )
