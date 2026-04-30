"""Tests for tui_gateway background-review summary delivery.

When the self-improvement background review fires and saves a skill or
memory entry, it calls ``agent.background_review_callback(message)``. In
the CLI that routes through a prompt_toolkit-safe ``_cprint``; in the TUI
there is no print surface, so without a callback wired up the review
writes the change silently. ``_init_session`` attaches a callback that
emits a ``review.summary`` event which Ink renders as a persistent
transcript line.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_review_summary")
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
        mod._pending.clear()
        mod._answers.clear()
        mod._methods.clear()
        importlib.reload(mod)


def test_init_session_attaches_background_review_callback(server, monkeypatch):
    """After _init_session, agent.background_review_callback is set to a
    function that emits 'review.summary' for the session's sid."""
    # Neutralize side-effect calls inside _init_session so we're testing
    # just the callback wiring.
    monkeypatch.setattr(server, "_SlashWorker", lambda *a, **kw: object())
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": "m"})
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "all")

    captured_emits: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured_emits.append(
            (event, sid, payload)
        ),
    )

    class FakeAgent:
        model = "fake/model"
        # Presence of the attribute is all the Python side needs; the real
        # AIAgent has it defaulted to None in __init__.
        background_review_callback = None

    agent = FakeAgent()
    server._init_session("sid-abc", "session-key", agent, [], cols=80)

    cb = getattr(agent, "background_review_callback", None)
    assert callable(cb), (
        "_init_session must attach a background_review_callback to the "
        "agent so the self-improvement review is visible in the TUI."
    )

    # Clear the session.info emit captured during _init_session.
    captured_emits.clear()

    # Invoke the callback the way AIAgent._spawn_background_review would.
    cb("💾 Self-improvement review: Skill 'hermes-release' patched")

    # Exactly one review.summary event should have been emitted, bound to
    # the session id we passed in, carrying the full message text.
    matched = [e for e in captured_emits if e[0] == "review.summary"]
    assert len(matched) == 1, captured_emits
    event, sid, payload = matched[0]
    assert sid == "sid-abc"
    assert payload == {
        "text": "💾 Self-improvement review: Skill 'hermes-release' patched"
    }


def test_review_summary_callback_survives_agent_without_attribute(server, monkeypatch):
    """If the agent is a bare object that doesn't allow attribute
    assignment (e.g. some stubbed test double), _init_session must not
    raise — session startup stays robust."""
    monkeypatch.setattr(server, "_SlashWorker", lambda *a, **kw: object())
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": "m"})
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "all")
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    class LockedAgent:
        __slots__ = ("model",)

        def __init__(self):
            self.model = "fake/model"

    # LockedAgent's __slots__ blocks background_review_callback assignment.
    server._init_session("sid-x", "key-x", LockedAgent(), [], cols=80)
    # If we got here, _init_session swallowed the AttributeError gracefully.
