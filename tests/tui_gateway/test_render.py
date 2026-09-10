"""Tests for tui_gateway.render — rendering bridge fallback behavior."""

from types import ModuleType
from unittest.mock import patch

from tui_gateway.render import make_stream_renderer, render_diff, render_message


def _stub_rich(mock_mod: ModuleType):
    return patch.dict("sys.modules", {"agent.rich_output": mock_mod})


def _no_rich():
    return patch.dict("sys.modules", {"agent.rich_output": None})


# ── render_message ───────────────────────────────────────────────────


def test_render_message_none_without_module():
    with _no_rich():
        assert render_message("hello") is None


def test_render_message_falls_back_for_legacy_signature():
    mod = ModuleType("agent.rich_output")

    def format_response(text):
        return text.upper()

    setattr(mod, "format_response", format_response)

    with _stub_rich(mod):
        assert render_message("hello", cols=42) == "HELLO"


# ── render_diff / make_stream_renderer ───────────────────────────────


def test_render_diff_falls_back_for_legacy_signature():
    mod = ModuleType("agent.rich_output")

    def legacy_render_diff(text):
        return f"legacy:{text}"

    setattr(mod, "render_diff", legacy_render_diff)

    with _stub_rich(mod):
        assert render_diff("patch", cols=120) == "legacy:patch"


def test_stream_renderer_none_without_module():
    with _no_rich():
        assert make_stream_renderer() is None


def test_stream_renderer_falls_back_for_legacy_signature():
    mod = ModuleType("agent.rich_output")

    class StreamingRenderer:
        pass

    setattr(mod, "StreamingRenderer", StreamingRenderer)

    with _stub_rich(mod):
        assert isinstance(make_stream_renderer(cols=132), StreamingRenderer)
