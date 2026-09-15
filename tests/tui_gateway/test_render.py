"""Tests for tui_gateway.render — rendering bridge fallback behavior."""

from unittest.mock import MagicMock, patch

from tui_gateway.render import make_stream_renderer, render_diff, render_message


def _stub_rich(mock_mod):
    return patch.dict("sys.modules", {"agent.rich_output": mock_mod})


def _no_rich():
    return patch.dict("sys.modules", {"agent.rich_output": None})


# ── render_message ───────────────────────────────────────────────────


def test_render_message_none_without_module():
    with _no_rich():
        assert render_message("hello") is None


# ── render_diff / make_stream_renderer ───────────────────────────────


def test_stream_renderer_none_without_module():
    with _no_rich():
        assert make_stream_renderer() is None


