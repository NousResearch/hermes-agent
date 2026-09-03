"""Tests for the GUI-surface ``open_preview`` tool."""

import json

import pytest

from tools import desktop_ui, open_preview_tool as op
from tools.registry import registry


@pytest.fixture(autouse=True)
def _reset_emitter():
    """Each test controls the emitter; never leak one across tests."""
    desktop_ui.set_emitter(None)
    yield
    desktop_ui.set_emitter(None)


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    import tools.preview_tool  # noqa: F401 — registers desktop_preview
    """Consolidated (#95681): this module's tool became an action of the
    single `desktop_preview` tool in desktop_ui; the old registration is gone and
    `preview` reaches a desktop client on ANY backend (no env gate)."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert registry.get_entry("open_preview") is None
    entry = registry.get_entry("desktop_preview")
    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


def test_emitter_failure_is_reported():
    def _boom(*_a):
        raise RuntimeError("no window")

    desktop_ui.set_emitter(_boom)
    assert "no window" in json.loads(op.open_preview_tool("https://x.example"))["error"]


def _capture_emits():
    emitted: list = []

    def _emit(_sid, event, payload):
        emitted.append((event, payload))

    desktop_ui.set_emitter(_emit)
    return emitted


def test_existing_directory_is_an_error_not_success(tmp_path):
    """#95853: a directory must not report success while opening nothing."""
    emitted = _capture_emits()
    folder = tmp_path / "Active"
    folder.mkdir()

    result = json.loads(op.open_preview_tool(str(folder)))

    assert "error" in result
    assert "director" in result["error"].lower()
    assert result.get("success") is not True
    assert emitted == []


def test_existing_file_still_opens(tmp_path):
    emitted = _capture_emits()
    path = tmp_path / "notes.md"
    path.write_text("hi", encoding="utf-8")

    result = json.loads(op.open_preview_tool(str(path)))

    assert result["success"] is True
    assert result["url"] == str(path)
    assert emitted == [("preview.open", {"url": str(path), "label": ""})]


def test_https_url_is_not_treated_as_a_directory():
    emitted = _capture_emits()
    result = json.loads(op.open_preview_tool("https://example.com/docs"))

    assert result["success"] is True
    assert emitted[0][0] == "preview.open"


def test_file_uri_directory_is_an_error(tmp_path):
    emitted = _capture_emits()
    folder = tmp_path / "docs"
    folder.mkdir()
    uri = folder.resolve().as_uri()

    result = json.loads(op.open_preview_tool(uri))

    assert "error" in result
    assert "director" in result["error"].lower()
    assert emitted == []


def test_missing_path_still_emits(tmp_path):
    """Reject only existing directories — a missing path is the renderer's call."""
    emitted = _capture_emits()
    missing = tmp_path / "no-such-folder"

    result = json.loads(op.open_preview_tool(str(missing)))

    assert result["success"] is True
    assert emitted == [("preview.open", {"url": str(missing), "label": ""})]
