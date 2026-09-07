"""Tests for the GUI-surface ``read_preview`` tool."""

import json

from tools import read_preview_tool as rp
from tools.registry import registry


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    import tools.preview_tool  # noqa: F401 — registers desktop_preview
    """Consolidated (#95681): this module's tool became an action of the
    single `desktop_preview` tool in desktop_ui; the old registration is gone and
    `preview` reaches a desktop client on ANY backend (no env gate)."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert registry.get_entry("read_preview") is None
    entry = registry.get_entry("desktop_preview")
    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


def test_requires_callback():
    """Outside the desktop GUI there is no bridge — a clear error, no crash."""
    result = json.loads(rp.read_preview_tool(callback=None))
    assert "desktop" in result["error"]


def test_empty_answer_means_nothing_open():
    result = json.loads(rp.read_preview_tool(callback=lambda **_: ""))
    assert "error" in result


def test_passes_json_through():
    payload = {"kind": "url", "url": "https://news.ycombinator.com", "title": "HN", "text": "hello"}
    result = json.loads(rp.read_preview_tool(callback=lambda **_: json.dumps(payload)))
    assert result == payload


def test_wraps_non_json_text():
    result = json.loads(rp.read_preview_tool(callback=lambda **_: "plain words"))
    assert result == {"text": "plain words"}


def test_window_forwarded_and_validated():
    seen = {}

    def cb(**kwargs):
        seen.update(kwargs)
        return json.dumps({"kind": "url"})

    rp.read_preview_tool(start=100, count=500, callback=cb)
    assert seen == {"start": 100, "count": 500}

    # Floors mirror read_terminal: start >= 0, count >= 1.
    seen.clear()
    rp.read_preview_tool(start=-5, count=0, callback=cb)
    assert seen == {"start": 0, "count": 1}

    result = json.loads(rp.read_preview_tool(start="lots", callback=cb))
    assert "integers" in result["error"]


def test_callback_failure_is_reported():
    def _boom(**_kwargs):
        raise RuntimeError("renderer went away")

    result = json.loads(rp.read_preview_tool(callback=_boom))
    assert "renderer went away" in result["error"]
