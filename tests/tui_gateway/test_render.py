"""The rich-render bridge must not mistake a renderer's own TypeError for an
old signature (Regression for the render.py audit).

``_rich`` used to catch ``TypeError`` around the call to fall back to a
pre-``cols`` signature. A ``TypeError`` raised *inside* the renderer was
therefore mistaken for a signature mismatch: the renderer ran a second time and
any error from the retry leaked out of ``_rich``. The fallback is now chosen from
the actual signature, and a failed renderer is logged instead of swallowed blind.
"""
from __future__ import annotations

import sys
import types

import pytest

from tui_gateway import render


@pytest.fixture
def fake_rich(monkeypatch):
    def install(**functions):
        module = types.ModuleType("agent.rich_output")
        for name, fn in functions.items():
            setattr(module, name, fn)
        monkeypatch.setitem(sys.modules, "agent.rich_output", module)
        return module

    return install


def test_internal_typeerror_is_not_retried_or_leaked(fake_rich, caplog):
    calls = []

    def format_response(text, cols=80):
        calls.append((text, cols))
        raise TypeError("renderer bug, not a signature mismatch")

    fake_rich(format_response=format_response)
    with caplog.at_level("DEBUG", logger="tui_gateway.render"):
        assert render.render_message("hi", cols=40) is None
    assert calls == [("hi", 40)]
    assert any(record.levelname == "DEBUG" for record in caplog.records)


def test_renderer_without_cols_is_called_without_it(fake_rich):
    def format_response(text):
        return f"plain:{text}"

    fake_rich(format_response=format_response)
    assert render.render_message("hi", cols=40) == "plain:hi"


def test_renderer_with_cols_receives_it(fake_rich):
    def format_response(text, cols=80):
        return f"{text}:{cols}"

    fake_rich(format_response=format_response)
    assert render.render_message("hi", cols=40) == "hi:40"


def test_non_typeerror_inside_renderer_returns_none(fake_rich):
    def format_response(text, cols=80):
        raise RuntimeError("boom")

    fake_rich(format_response=format_response)
    assert render.render_message("hi") is None


def test_missing_module_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.rich_output", None)
    assert render.render_message("hi") is None
