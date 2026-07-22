from __future__ import annotations

import threading

from tui_gateway import server


def _context(text: str, event_id: str) -> dict:
    return {
        "surface": "desktop",
        "platform_account": "installation-1",
        "sender_id": "darwin:501",
        "chat_id": "desktop:installation-1",
        "chat_type": "private",
        "thread_id": "",
        "profile": "default",
        "app_identity": "TEAM:io.hermes.desktop@0.17.0",
        "app_instance_id": "instance-1",
        "window_id": "7",
        "gateway_session_id": "runtime-1",
        "accepted_text": text,
        "source_messages": [{"raw_event_id": event_id}],
    }


def test_prompt_hook_accepts_one_exact_plugin_verified_context(monkeypatch):
    text = "log procedure"
    context = _context(text, "event-1")

    def invoke(name, **kwargs):
        assert name == "pre_prompt_submit"
        assert kwargs["session_id"] == "runtime-1"
        assert kwargs["task_id"] == "stored-1"
        assert kwargs["user_message"] == text
        return [{"surface_context": context}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke)
    assert server._prompt_surface_context(
        {"desktop_provenance": {"signed": True}},
        sid="runtime-1",
        session={"session_key": "stored-1"},
        text=text,
    ) == context


def test_prompt_hook_fails_closed_for_mismatch_or_multiple_authorities(monkeypatch):
    text = "log procedure"
    context = _context(text, "event-1")
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [
            {"surface_context": context},
            {"surface_context": context},
        ],
    )
    assert server._prompt_surface_context(
        {"desktop_provenance": {}}, sid="r", session={"session_key": "s"}, text=text
    ) is None

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [
            {"surface_context": {**context, "accepted_text": "changed"}}
        ],
    )
    assert server._prompt_surface_context(
        {"desktop_provenance": {}}, sid="r", session={"session_key": "s"}, text=text
    ) is None


def test_queued_prompts_preserve_ordered_sources_only_for_same_principal():
    session = {"queued_prompt": None}
    first = _context("first", "event-1")
    second = _context("second", "event-2")
    server._enqueue_prompt(session, "first", object(), first)
    server._enqueue_prompt(session, "second", object(), second)
    queued = session["queued_prompt"]
    assert queued["text"] == "first\n\nsecond"
    assert queued["surface_context"]["accepted_text"] == "first\n\nsecond"
    assert [row["raw_event_id"] for row in queued["surface_context"]["source_messages"]] == [
        "event-1",
        "event-2",
    ]

    foreign = _context("third", "event-3")
    foreign["profile"] = "other"
    server._enqueue_prompt(session, "third", object(), foreign)
    assert session["queued_prompt"]["surface_context"] is None


def test_compute_host_frame_carries_only_in_memory_surface_context():
    context = _context("hello", "event-1")
    session = {
        "history": [],
        "history_version": 0,
        "attached_images": [],
        "history_lock": threading.Lock(),
        "session_key": "stored-1",
        "cols": 80,
        "cwd": "/tmp",
        "source": "desktop",
    }
    frame = server._compute_host_turn_frame("rid", "sid", session, "hello", context)
    assert frame["surface_context"] == context
    assert frame["text"] == "hello"


def test_plugin_event_emitter_resolves_only_exact_durable_session(monkeypatch):
    writes = []

    class Transport:
        def write(self, frame):
            writes.append(frame)
            return True

    server._sessions["runtime-1"] = {
        "agent": None,
        "session_key": "stored-1",
        "transport": Transport(),
    }
    try:
        assert server.emit_plugin_event_to_session_key(
            "stored-1", "plugin.medcloud.status", {"operation_id": "op-1"}
        )
        assert writes[0]["params"]["session_id"] == "runtime-1"
        assert writes[0]["params"]["type"] == "plugin.medcloud.status"
        assert not server.emit_plugin_event_to_session_key(
            "stored-1", "message.complete", {"text": "forged"}
        )
        assert not server.emit_plugin_event_to_session_key(
            "foreign", "plugin.medcloud.status", {"operation_id": "op-1"}
        )
    finally:
        server._sessions.pop("runtime-1", None)


def test_session_attach_replay_accepts_only_bounded_namespaced_events(monkeypatch):
    writes = []

    class Transport:
        def write(self, frame):
            writes.append(frame)
            return True

    class ImmediateTimer:
        daemon = False

        def __init__(self, _delay, callback):
            self.callback = callback

        def start(self):
            self.callback()

    server._sessions["runtime-1"] = {
        "agent": None,
        "session_key": "stored-1",
        "transport": Transport(),
    }
    monkeypatch.setattr(server.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: [
            {
                "events": [
                    {
                        "type": "plugin.medcloud.status",
                        "payload": {"operation_id": "op-1"},
                    },
                    {"type": "message.complete", "payload": {"text": "forged"}},
                ]
            }
        ],
    )
    try:
        server._schedule_plugin_session_events("runtime-1")
        assert [frame["params"]["type"] for frame in writes] == [
            "plugin.medcloud.status"
        ]
    finally:
        server._sessions.pop("runtime-1", None)
