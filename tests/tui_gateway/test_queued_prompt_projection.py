"""A reconnecting client rebuilds a still-queued prompt from the live projection (``queued``).

The prompt exists only in the queued envelope until it drains, so the projection must carry
everything a client needs to show it: the text and the staged paths of its images.
"""

import threading
import types

import pytest

from tui_gateway import server


def _busy_session(**extra):
    return {
        "agent": types.SimpleNamespace(model="model-live"),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": {"assistant": "partial answer", "streaming": True, "user": "current prompt"},
        **extra,
    }


@pytest.mark.parametrize("text", ["what is in these", ""])
def test_activate_returns_the_images_of_a_prompt_queued_during_a_busy_turn(monkeypatch, text):
    """Images of a queued prompt reach a reconnecting client; an image-only prompt is not dropped."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": agent.model})
    session = _busy_session(attached_images=["/staged/a.png", "/staged/b.png"])
    server._sessions["sid-live"] = session
    try:
        queued = server._handle_busy_submit("submit", "sid-live", session, text, object())
        assert queued["result"]["status"] == "queued"

        activated = server.handle_request(
            {"id": "activate", "method": "session.activate", "params": {"session_id": "sid-live"}}
        )

        assert activated["result"]["queued"] == {"user": text, "images": ["/staged/a.png", "/staged/b.png"]}
    finally:
        server._sessions.pop("sid-live", None)
