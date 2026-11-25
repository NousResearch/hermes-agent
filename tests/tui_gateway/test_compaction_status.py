"""Auto-compaction status re-tagging for TUI/desktop summarizing indicators.

Auto-compaction reaches the gateway as a generic ``lifecycle`` status. The agent's
status callback re-tags the lines their producer marked as compaction progress as
``kind="compacting"`` so drivers show a summarizing indicator instead of the turn
looking hung (#97239). Warnings that only mention compression keep the kind they
were sent with: a client shown one as ``compacting`` draws a compaction that never ran.
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
    # The agent callbacks run against server's globals (tui_gateway/method_ctx.py).
    monkeypatch.setattr(server, "_agent_presentation_enabled", lambda sid, diagnostic=False: True)
    return events


class _RewordingEngine:
    def get_automatic_compaction_status_message(self, *, phase, default_message, **context):
        return f"LCM is folding older turns ({phase})"


def _progress_lines():
    from agent.context_engine import automatic_compaction_status_message
    from agent.conversation_compression import (
        COMPACTION_HEARTBEAT_STATUS,
        COMPACTION_STATUS,
        IDLE_COMPACTION_STATUS_TEMPLATE,
        PRE_API_COMPRESSION_STATUS_TEMPLATE,
        PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
    )

    defaults = {
        "pre_api": PRE_API_COMPRESSION_STATUS_TEMPLATE.format(tokens=123_456),
        "preflight": PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(tokens=120_000, threshold=100_000),
        "idle": IDLE_COMPACTION_STATUS_TEMPLATE.format(idle_seconds=747, tokens=44_579),
        "compress": COMPACTION_STATUS,
    }
    lines = [COMPACTION_STATUS, COMPACTION_HEARTBEAT_STATUS]
    for engine in (object(), _RewordingEngine()):
        lines += [automatic_compaction_status_message(engine, phase=phase, default_message=text)
                  for phase, text in defaults.items()]
    return lines


def test_every_compaction_progress_line_is_retagged(server, monkeypatch):
    events = _capture(server, monkeypatch)
    lines = _progress_lines()
    for line in lines:
        server._agent_status_update("sid", "lifecycle", line)

    assert events == [{"kind": "compacting", "text": str(line)} for line in lines]


def test_warnings_that_mention_compression_are_not_compaction(server, monkeypatch):
    from agent.conversation_compression import (
        COMPRESSION_REFUSED_WOULD_GROW_WARNING,
        COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE,
        COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE,
        CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE,
    )
    from gateway.warning_notifications import DiagnosticText

    events = _capture(server, monkeypatch)
    lines = [
        "⚠ Configured auxiliary compression provider 'nous' is unavailable, so older messages in long "
        "chats will be cut without a summary. Sign in to that provider again, or change "
        "auxiliary.compression in your config.",
        "⚠️  Session compressed 3 times — accuracy may degrade. Consider /new to start fresh.",
        COMPRESSION_REFUSED_WOULD_GROW_WARNING,
        CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(tokens=85_000, threshold=72_000, reason="cooldown:30"),
        # Retry lines reach a client only when the buffer flushes after the turn failed.
        DiagnosticText(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=30, after=12)),
        DiagnosticText(COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE.format(new_ctx=1, old_ctx=2)),
    ]
    for line in lines:
        server._agent_status_update("sid", "lifecycle", line)

    assert [e["kind"] for e in events] == ["lifecycle"] * len(lines)


def test_compaction_done_edge_keeps_its_kind(server, monkeypatch):
    from agent.conversation_compression import COMPACTION_DONE_STATUS

    events = _capture(server, monkeypatch)
    server._agent_status_update("sid", "compacted", COMPACTION_DONE_STATUS)
    assert events == [{"kind": "compacted", "text": COMPACTION_DONE_STATUS}]


def test_manual_compressing_kind_is_preserved(server, monkeypatch):
    events = _capture(server, monkeypatch)
    server._status_update("sid", "compressing", "⠋ compressing 40 messages…")

    assert events[0]["kind"] == "compressing"
