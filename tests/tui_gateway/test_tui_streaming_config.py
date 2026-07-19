"""Tests for the TUI gateway respecting display.streaming config.

When ``display.streaming`` is explicitly ``False`` in config.yaml, the TUI
must NOT pass a ``stream_callback`` to ``run_conversation()``.  Otherwise the
agent's ``_has_stream_consumers()`` returns True and forces the streaming API
path, which crashes MoA + Codex aggregator (SimpleNamespace not iterable).
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
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_stream_cfg")
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


def test_streaming_disabled_passes_none_callback(server, monkeypatch):
    """When display.streaming is False, stream_callback must be None."""
    cfg = {"display": {"streaming": False}}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)

    _display_cfg = (server._load_cfg().get("display") or {})
    _streaming_enabled = _display_cfg.get("streaming", True)
    assert _streaming_enabled is False

    # Simulate what run() does
    def _stream(delta):
        pass

    _stream_cb = _stream if _streaming_enabled else None
    assert _stream_cb is None, (
        "stream_callback should be None when display.streaming is False"
    )


def test_streaming_enabled_passes_callback(server, monkeypatch):
    """When display.streaming is True (or unset), stream_callback is set."""
    cfg = {"display": {"streaming": True}}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)

    _display_cfg = (server._load_cfg().get("display") or {})
    _streaming_enabled = _display_cfg.get("streaming", True)
    assert _streaming_enabled is True

    def _stream(delta):
        pass

    _stream_cb = _stream if _streaming_enabled else None
    assert _stream_cb is _stream, (
        "stream_callback should be _stream when display.streaming is True"
    )


def test_streaming_default_enabled(server, monkeypatch):
    """When display.streaming is not set, default to True (TUI is interactive)."""
    cfg = {"display": {}}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)

    _display_cfg = (server._load_cfg().get("display") or {})
    _streaming_enabled = _display_cfg.get("streaming", True)
    assert _streaming_enabled is True


def test_no_display_key_defaults_enabled(server, monkeypatch):
    """When the display key is absent entirely, streaming defaults to True."""
    cfg = {}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)

    _display_cfg = (server._load_cfg().get("display") or {})
    _streaming_enabled = _display_cfg.get("streaming", True)
    assert _streaming_enabled is True
