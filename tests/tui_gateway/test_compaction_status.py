"""Auto-compaction status re-tagging for the desktop "Summarizing…" indicator.

Auto-compaction reaches the gateway as a generic ``lifecycle`` status. The
gateway re-tags it as ``kind="compacting"`` so drivers (the desktop app) can
show an explicit summarizing indicator instead of the transcript appearing to
silently reset mid-turn.
"""

from __future__ import annotations

import importlib

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_compaction")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        yield importlib.import_module("tui_gateway.server")


def _capture(server, monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(
        server, "_emit", lambda event, sid, payload=None: events.append(payload or {})
    )
    return events


def test_compaction_lifecycle_is_retagged(server, monkeypatch):
    from agent.conversation_compression import COMPACTION_STATUS

    events = _capture(server, monkeypatch)
    server._status_update("sid", "lifecycle", COMPACTION_STATUS)

    assert events == [{"kind": "compacting", "text": COMPACTION_STATUS}]


def test_other_lifecycle_status_stays_lifecycle(server, monkeypatch):
    events = _capture(server, monkeypatch)
    server._status_update("sid", "lifecycle", "❌ Rate limited after 5 retries")

    assert events[0]["kind"] == "lifecycle"


def test_manual_compressing_kind_is_preserved(server, monkeypatch):
    events = _capture(server, monkeypatch)
    server._status_update("sid", "compressing", "⠋ compressing 40 messages…")

    assert events[0]["kind"] == "compressing"


def test_compaction_status_contains_marker():
    # Contract: the gateway matches COMPACTION_STATUS_MARKER inside the emitted
    # status text. If the message is reworded, the marker must survive.
    from agent.conversation_compression import (
        COMPACTION_STATUS,
        COMPACTION_STATUS_MARKER,
    )

    assert COMPACTION_STATUS_MARKER in COMPACTION_STATUS


def test_compaction_done_kind_is_preserved(server, monkeypatch):
    from agent.conversation_compression import COMPACTION_DONE_STATUS

    events = _capture(server, monkeypatch)
    server._status_update("sid", "compacted", COMPACTION_DONE_STATUS)

    assert events == [{"kind": "compacted", "text": COMPACTION_DONE_STATUS}]
