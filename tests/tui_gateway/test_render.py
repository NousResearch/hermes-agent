"""Tests for tui_gateway.render rich-output bridge."""

from __future__ import annotations

import builtins
import sys
import types

from tui_gateway import render


def _install_rich_output(monkeypatch, **attrs):
    module = types.ModuleType("agent.rich_output")
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "agent.rich_output", module)
    return module


def test_render_message_uses_cols_when_supported(monkeypatch):
    calls = []

    def format_response(text, *, cols):
        calls.append((text, cols))
        return f"{cols}:{text}"

    _install_rich_output(monkeypatch, format_response=format_response)

    assert render.render_message("hello", cols=42) == "42:hello"
    assert calls == [("hello", 42)]


def test_render_message_falls_back_for_legacy_signature(monkeypatch):
    def format_response(text, *, cols=None):
        if cols is not None:
            raise TypeError("cols unsupported")
        return text.upper()

    _install_rich_output(monkeypatch, format_response=format_response)

    assert render.render_message("hello", cols=42) == "HELLO"


def test_render_message_returns_none_when_renderer_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "agent.rich_output":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("agent.rich_output", None)

    assert render.render_message("hello") is None


def test_render_message_returns_none_on_renderer_error(monkeypatch):
    def format_response(_text, *, cols):
        raise RuntimeError("renderer failed")

    _install_rich_output(monkeypatch, format_response=format_response)

    assert render.render_message("hello") is None


def test_render_diff_uses_cols_when_supported(monkeypatch):
    calls = []

    def render_diff(text, *, cols):
        calls.append((text, cols))
        return f"diff:{cols}:{text}"

    _install_rich_output(monkeypatch, render_diff=render_diff)

    assert render.render_diff("-old\n+new", cols=100) == "diff:100:-old\n+new"
    assert calls == [("-old\n+new", 100)]


def test_render_diff_falls_back_for_legacy_signature(monkeypatch):
    def render_diff(text, *, cols=None):
        if cols is not None:
            raise TypeError("cols unsupported")
        return f"legacy:{text}"

    _install_rich_output(monkeypatch, render_diff=render_diff)

    assert render.render_diff("patch", cols=120) == "legacy:patch"


def test_render_diff_returns_none_on_renderer_error(monkeypatch):
    def render_diff(_text, *, cols):
        raise RuntimeError("renderer failed")

    _install_rich_output(monkeypatch, render_diff=render_diff)

    assert render.render_diff("patch") is None


def test_make_stream_renderer_uses_cols_when_supported(monkeypatch):
    calls = []

    class StreamingRenderer:
        def __init__(self, *, cols):
            calls.append(cols)
            self.cols = cols

    _install_rich_output(monkeypatch, StreamingRenderer=StreamingRenderer)

    renderer = render.make_stream_renderer(cols=132)
    assert isinstance(renderer, StreamingRenderer)
    assert renderer.cols == 132
    assert calls == [132]


def test_make_stream_renderer_falls_back_for_legacy_signature(monkeypatch):
    class StreamingRenderer:
        def __init__(self, *, cols=None):
            if cols is not None:
                raise TypeError("cols unsupported")
            self.legacy = True

    _install_rich_output(monkeypatch, StreamingRenderer=StreamingRenderer)

    renderer = render.make_stream_renderer(cols=132)
    assert isinstance(renderer, StreamingRenderer)
    assert renderer.legacy is True


def test_make_stream_renderer_returns_none_on_renderer_error(monkeypatch):
    class StreamingRenderer:
        def __init__(self, *, cols):
            raise RuntimeError("renderer failed")

    _install_rich_output(monkeypatch, StreamingRenderer=StreamingRenderer)

    assert render.make_stream_renderer() is None
