"""Tests for tui_gateway.render safe fallback behavior."""

from tui_gateway.render import make_stream_renderer, render_diff, render_message


def test_render_message_uses_client_fallback():
    assert render_message("hello") is None


def test_render_diff_uses_client_fallback():
    assert render_diff("+line") is None


def test_stream_renderer_uses_client_fallback():
    assert make_stream_renderer() is None


def test_renderer_does_not_import_agent_rich_output(monkeypatch):
    imported = []
    real_import = __import__

    def tracking_import(name, *args, **kwargs):
        imported.append(name)
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if name == "agent.rich_output":
            raise AssertionError("agent.rich_output must not be imported (absolute)")
        if name == "agent" and "rich_output" in (fromlist or ()):
            raise AssertionError("agent.rich_output must not be imported (from-import)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    assert render_message("final", 77) is None
    assert render_diff("+diff", 77) is None
    assert make_stream_renderer(77) is None
    assert "agent.rich_output" not in imported
