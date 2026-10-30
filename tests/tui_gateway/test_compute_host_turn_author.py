"""turn_author across the compute-host boundary (author half of #122297).

With turn isolation on, relayed-bot DMs must stay attributed exactly like
inline turns: the frame carries ``turn_author``, both dispatch sites pass it,
and the child forwards it into ``_run_prompt_submit`` (which already threads
it into ``run_conversation`` via ``_invoke_agent``).
"""

from __future__ import annotations

import io
import threading
import types

from tools.bot_relay import DeliveryAuthor
from tui_gateway import server
from tui_gateway.compute_host import ComputeHost

AUTHOR = {"id": "bot:coder", "name": "coder", "is_bot": True}


def _session(**extra):
    return {
        "agent": None,
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        "transport": None,
        **extra,
    }


def test_turn_frame_carries_author_when_set():
    frame = server._compute_host_turn_frame("r", "sid", _session(), "hi", turn_author=dict(AUTHOR))

    assert frame["turn_author"] == AUTHOR


def test_turn_frame_omits_author_when_unset():
    assert "turn_author" not in server._compute_host_turn_frame("r", "sid", _session(), "hi")


def test_immediate_isolated_submit_carries_author(monkeypatch):
    """prompt.submit with a relay-stamped author dispatches an attributed frame."""

    class FakeSupervisor:
        def __init__(self):
            self.frames = []

        def submit_turn(self, frame, *, on_complete=None):
            self.frames.append(frame)
            return frame["request_id"]

    fake_supervisor = FakeSupervisor()
    server._sessions["iso-author"] = _session(agent_ready=threading.Event())
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"dashboard": {"turn_isolation": True}},
    )
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: fake_supervisor)
    try:
        resp = server._methods["prompt.submit"](
            "submit",
            {"session_id": "iso-author", "text": "hello", "_turn_author": DeliveryAuthor(AUTHOR)},
        )
        assert resp["result"] == {"status": "streaming", "turn_isolation": True}
        assert fake_supervisor.frames[0]["turn_author"] == AUTHOR
    finally:
        server._sessions.pop("iso-author", None)


def test_queued_drain_carries_author_to_compute_host(monkeypatch):
    """A drained queued prompt keeps its author on the isolated branch too."""
    captured = {}
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(
        server,
        "_submit_prompt_to_compute_host",
        lambda rid, sid, session, text, **kwargs: captured.update(
            rid=rid, sid=sid, text=text, turn_author=kwargs.get("turn_author")
        )
        or {"result": {"status": "started"}},
    )
    session = _session(
        queued_prompt={"text": "follow-up", "transport": "ws-9", "turn_author": dict(AUTHOR)}
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert captured == {"rid": "r1", "sid": "sid", "text": "follow-up", "turn_author": AUTHOR}


def test_child_forwards_frame_author_into_run_prompt_submit(monkeypatch):
    """The compute host consumes the frame's author into the far-side turn."""
    out = io.StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)
    seen = {}
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, session, text, **kwargs: seen.update(kwargs) or True,
    )
    server._sessions["s1"] = _session(
        agent=types.SimpleNamespace(session_id="s1-key"),
        session_key="s1-key",
        active_session_lease=object(),
    )
    try:
        host._run_real_turn(
            {"type": "turn.start", "sid": "s1", "request_id": "turn",
             "text": "hello", "turn_author": dict(AUTHOR)}
        )
    finally:
        server._sessions.pop("s1", None)
        host.close()

    assert seen.get("turn_author") == AUTHOR
