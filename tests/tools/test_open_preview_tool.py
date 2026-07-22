"""Tests for the desktop-gated ``open_preview`` tool."""

import json

import pytest

import tools.open_preview_tool as op


@pytest.fixture(autouse=True)
def _reset_emitter():
    """Each test controls the emitter; never leak one across tests."""
    op.set_preview_emitter(None)
    yield
    op.set_preview_emitter(None)


def test_gated_on_desktop(monkeypatch):
    """Hidden unless HERMES_DESKTOP is set (mirrors read_terminal/close_terminal)."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert op.check_open_preview_requirements() is False

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    assert op.check_open_preview_requirements() is True


def test_requires_url():
    op.set_preview_emitter(lambda *a: None)
    assert json.loads(op.open_preview_tool("   "))["error"]


def test_desktop_only_without_emitter():
    """No emitter wired (CLI/messaging) → clear desktop-only error, no raise."""
    result = json.loads(op.open_preview_tool("https://example.com"))
    assert "desktop" in result["error"].lower()


def test_emits_with_ui_session_id(monkeypatch):
    """The tool routes (sid, url, label) to the wired emitter, sid from context."""
    monkeypatch.setattr(op, "get_session_env", lambda name, default="": "win-42" if name == "HERMES_UI_SESSION_ID" else default)
    calls = []
    op.set_preview_emitter(lambda sid, url, label: calls.append((sid, url, label)))

    out = json.loads(op.open_preview_tool("https://example.com/app", label="Docs"))

    assert out["success"] is True
    assert out["url"] == "https://example.com/app"
    assert calls == [("win-42", "https://example.com/app", "Docs")]


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
    op.set_preview_emitter(lambda sid, url, label: seen.update(url=url))

    op.open_preview_tool(raw)

    assert seen["url"] == expected


def test_emitter_failure_is_reported(monkeypatch):
    def _boom(*_a):
        raise RuntimeError("no window")

    op.set_preview_emitter(_boom)
    assert "no window" in json.loads(op.open_preview_tool("https://x.example"))["error"]
