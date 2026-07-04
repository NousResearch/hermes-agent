"""Tests for the TUI gateway relaying MoA reference events to the client.

When a MoA preset is the active model, the agent's tool_progress_callback emits
``moa.reference`` (one per reference model, before the aggregator acts) and a
single ``moa.aggregating`` marker. ``_on_tool_progress`` must forward these to
the Ink/desktop client as labelled events so each reference renders like a
thinking block tagged with its source model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_moa_emit")
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


@pytest.fixture()
def emits(server, monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda sid: True)
    return captured


def test_moa_reference_relayed_with_label_and_index(server, emits):
    server._on_tool_progress(
        "sid-1",
        "moa.reference",
        "openrouter:openai/gpt-5.5",
        "Paris is the capital of France.",
        None,
        moa_index=1,
        moa_count=2,
    )

    assert len(emits) == 1
    event, sid, payload = emits[0]
    assert event == "moa.reference"
    assert sid == "sid-1"
    assert payload["label"] == "openrouter:openai/gpt-5.5"
    assert payload["text"] == "Paris is the capital of France."
    assert payload["index"] == 1
    assert payload["count"] == 2


def test_moa_aggregating_relayed(server, emits):
    server._on_tool_progress(
        "sid-1",
        "moa.aggregating",
        "openrouter:anthropic/claude-opus-4.8",
        None,
        None,
    )

    assert len(emits) == 1
    event, sid, payload = emits[0]
    assert event == "moa.aggregating"
    assert payload["aggregator"] == "openrouter:anthropic/claude-opus-4.8"


def test_moa_reference_without_index_omits_index(server, emits):
    server._on_tool_progress(
        "sid-1",
        "moa.reference",
        "openrouter:anthropic/claude-opus-4.8",
        "The capital is Paris.",
        None,
    )

    assert len(emits) == 1
    _event, _sid, payload = emits[0]
    assert "index" not in payload
    assert "count" not in payload
    assert payload["label"] == "openrouter:anthropic/claude-opus-4.8"
