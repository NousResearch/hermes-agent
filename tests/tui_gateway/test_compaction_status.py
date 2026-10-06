"""Auto-compaction status re-tagging for TUI/desktop summarizing indicators.

Auto-compaction reaches the gateway as a generic ``lifecycle`` status. The
gateway re-tags in-progress lines as ``kind="compacting"`` so drivers can
show an explicit summarizing indicator instead of the turn looking hung
(#97239). The marker-only match missed idle/preflight/retry wording.
"""

from __future__ import annotations

import importlib

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
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
        mod = importlib.import_module("tui_gateway.server")
    yield mod


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


def test_idle_compaction_lifecycle_is_retagged(server, monkeypatch):
    from agent.conversation_compression import IDLE_COMPACTION_STATUS_TEMPLATE

    events = _capture(server, monkeypatch)
    text = IDLE_COMPACTION_STATUS_TEMPLATE.format(idle_seconds=747, tokens=44_579)
    server._status_update("sid", "lifecycle", text)

    assert events == [{"kind": "compacting", "text": text}]


def test_preflight_compaction_lifecycle_is_retagged(server, monkeypatch):
    from agent.conversation_compression import PREFLIGHT_COMPRESSION_STATUS_TEMPLATE

    events = _capture(server, monkeypatch)
    text = PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(
        tokens=120_000, threshold=100_000
    )
    server._status_update("sid", "lifecycle", text)

    assert events == [{"kind": "compacting", "text": text}]


def test_compaction_done_is_not_retagged_as_compacting(server, monkeypatch):
    from agent.conversation_compression import COMPACTION_DONE_STATUS

    events = _capture(server, monkeypatch)
    server._status_update("sid", "lifecycle", COMPACTION_DONE_STATUS)
    assert events[0]["kind"] == "lifecycle"

    events.clear()
    server._status_update("sid", "compacted", COMPACTION_DONE_STATUS)
    assert events[0]["kind"] == "compacted"


def test_other_lifecycle_status_stays_lifecycle(server, monkeypatch):
    events = _capture(server, monkeypatch)
    server._status_update("sid", "lifecycle", "❌ Rate limited after 5 retries")

    assert events[0]["kind"] == "lifecycle"


def test_overflow_blocked_warning_stays_lifecycle(server, monkeypatch):
    from agent.conversation_compression import CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE

    events = _capture(server, monkeypatch)
    text = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason="cooldown:30"
    )
    server._status_update("sid", "lifecycle", text)

    assert events[0]["kind"] == "lifecycle"


def test_manual_compressing_kind_is_preserved(server, monkeypatch):
    events = _capture(server, monkeypatch)
    server._status_update("sid", "compressing", "⠋ compressing 40 messages…")

    assert events[0]["kind"] == "compressing"


def test_is_compaction_progress_status_covers_routine_in_progress_lines():
    from agent.conversation_compression import (
        COMPACTION_DONE_STATUS,
        CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE,
        ROUTINE_COMPRESSION_STATUS_SAMPLES,
        is_compaction_progress_status,
    )

    for message in ROUTINE_COMPRESSION_STATUS_SAMPLES:
        if message == COMPACTION_DONE_STATUS:
            assert is_compaction_progress_status(message) is False
        else:
            assert is_compaction_progress_status(message) is True, message

    assert is_compaction_progress_status(None) is False
    assert is_compaction_progress_status("") is False
    assert is_compaction_progress_status("❌ Rate limited after 5 retries") is False
    overflow = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason="ineffective"
    )
    assert is_compaction_progress_status(overflow) is False
