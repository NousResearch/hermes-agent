"""Tests for the desktop-gated ``open_preview`` tool."""

import json

import pytest

from tools import desktop_ui, open_preview_tool as op


@pytest.fixture(autouse=True)
def _reset_emitter():
    """Each test controls the emitter; never leak one across tests."""
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)


def test_gated_on_desktop(monkeypatch):
    """Hidden unless HERMES_DESKTOP is set (mirrors read_terminal/close_terminal)."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert op.check_open_preview_requirements() is False

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    assert op.check_open_preview_requirements() is True


def test_requires_url():
    desktop_ui.set_emitter(lambda *a: None)
    assert json.loads(op.open_preview_tool("   "))["error"]


def test_desktop_only_without_emitter():
    """No emitter wired (CLI/messaging) → clear desktop-only error, no raise."""
    result = json.loads(op.open_preview_tool("https://example.com"))
    assert "desktop" in result["error"].lower()


def test_emits_preview_open(monkeypatch):
    calls = []
    desktop_ui.set_emitter(lambda sid, event, payload: calls.append((event, payload)))

    out = json.loads(op.open_preview_tool("https://example.com/app", label="Docs"))

    assert out == {"success": True, "url": "https://example.com/app", "label": "Docs"}
    assert calls == [("preview.open", {"url": "https://example.com/app", "label": "Docs"})]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("www.cnn.com", "https://www.cnn.com"),
        ("example.com/path", "https://example.com/path"),
        ("localhost:3000", "http://localhost:3000"),
        ("127.0.0.1:8080/x", "http://127.0.0.1:8080/x"),
        ("https://already.example", "https://already.example"),
        ("/abs/path/index.html", "/abs/path/index.html"),
        ("./rel/page.html", "./rel/page.html"),
        ("`https://tick.example`", "https://tick.example"),
    ],
)
def test_normalizes_bare_targets(raw, expected):
    seen = {}
    desktop_ui.set_emitter(lambda sid, event, payload: seen.update(payload))

    op.open_preview_tool(raw)

    assert seen["url"] == expected


def test_emitter_failure_is_reported():
    def _boom(*_a):
        raise RuntimeError("no window")

    desktop_ui.set_emitter(_boom)
    assert "no window" in json.loads(op.open_preview_tool("https://x.example"))["error"]
