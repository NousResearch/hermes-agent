"""Tests for tools/handoff_tool.py — write-only session handoff document.

The handoff tool is deliberately write-only: it writes a document to disk so
a human can manually continue a session, and it must never gain a reset or
injection code path. These tests dispatch through the real registered tool
entry (registry.get_entry("handoff").handler) rather than calling the
private module functions directly, so they also guard the registration
wiring in toolsets.py / tools/registry.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.handoff_tool  # noqa: F401 -- import registers the tool
from tools.registry import registry


def _call_handoff(**args) -> dict:
    entry = registry.get_entry("handoff")
    assert entry is not None, "handoff tool must be registered"
    raw = entry.handler(args)
    return json.loads(raw)


class TestSchema:
    def test_action_enum_is_write_only(self):
        """Hard safety-boundary check: the schema must never advertise any
        action besides 'write' — no reset/injection action can exist."""
        entry = registry.get_entry("handoff")
        assert entry is not None
        action_schema = entry.schema["parameters"]["properties"]["action"]
        assert action_schema["enum"] == ["write"]

    def test_content_required_path_optional(self):
        entry = registry.get_entry("handoff")
        assert entry is not None
        required = entry.schema["parameters"]["required"]
        assert "content" in required
        assert "action" in required
        assert "path" not in required


class TestWriteSucceeds:
    def test_write_creates_file_with_exact_content(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        content = "# Handoff\n\nState: working on issue #114126.\n"

        result = _call_handoff(action="write", content=content, path="my-handoff.md")

        assert result["success"] is True
        target = Path(result["path"])
        assert target.is_file()
        assert target.read_text(encoding="utf-8") == content

    def test_instructions_mention_new_and_manual_continuation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = _call_handoff(action="write", content="some content", path="h.md")

        assert "/new" in result["instructions"]
        assert result["path"] in result["instructions"]


class TestContentValidation:
    def test_missing_content_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = _call_handoff(action="write")

        assert "error" in result
        assert "content" in result["error"].lower()
        # Nothing should have been written to disk.
        assert not (tmp_path / "handoffs").exists()

    def test_empty_content_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = _call_handoff(action="write", content="")

        assert "error" in result
        assert not (tmp_path / "handoffs").exists()

    def test_whitespace_only_content_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = _call_handoff(action="write", content="   \n\t  ")

        assert "error" in result
        assert not (tmp_path / "handoffs").exists()


class TestPathResolution:
    def test_bare_filename_resolves_under_handoffs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = _call_handoff(action="write", content="content", path="notes.md")

        expected = (tmp_path / "handoffs" / "notes.md").resolve()
        assert Path(result["path"]) == expected
        assert expected.is_file()

    def test_explicit_absolute_path_used_as_is(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        other_dir = tmp_path / "elsewhere"
        other_dir.mkdir()
        absolute_target = other_dir / "abs-handoff.md"

        result = _call_handoff(action="write", content="content", path=str(absolute_target))

        assert Path(result["path"]) == absolute_target.resolve()
        assert absolute_target.is_file()
        # Must NOT have been redirected under handoffs/.
        assert not (tmp_path / "handoffs" / "abs-handoff.md").exists()

    def test_no_path_defaults_to_timestamped_file_under_handoffs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = _call_handoff(action="write", content="content")

        target = Path(result["path"])
        assert target.is_file()
        assert target.parent == (tmp_path / "handoffs").resolve()
        assert target.name.endswith("-handoff.md")

    def test_directory_actually_created_on_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert not (tmp_path / "handoffs").exists()

        _call_handoff(action="write", content="content", path="a.md")

        assert (tmp_path / "handoffs").is_dir()


class TestSafetyBoundary:
    def test_unsupported_action_rejected_defensively(self, tmp_path, monkeypatch):
        """Schema enum already blocks non-'write' values, but the handler's
        own defensive check must also reject them if ever bypassed."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = _call_handoff(action="reset", content="content")

        assert "error" in result
        assert not (tmp_path / "handoffs").exists()

    def test_no_reset_or_inject_action_exists_anywhere(self):
        entry = registry.get_entry("handoff")
        assert entry is not None
        enum_values = entry.schema["parameters"]["properties"]["action"]["enum"]
        for forbidden in ("reset", "inject", "restart", "new", "continue", "resume"):
            assert forbidden not in enum_values
        assert enum_values == ["write"]
