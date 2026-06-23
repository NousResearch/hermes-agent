"""Tests for the session_summary tool — running session summary for context reduction.

The tool maintains a compact running summary of the current session in
~/.hermes/sessions/<session_id>/running_summary.md.  Three actions:
  write   — overwrite with new content
  append  — add a timestamped entry
  read    — return current summary (or empty string if none exists)
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session_dir():
    """Create a temp session directory and return its path."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def summary_path(session_dir):
    """Return the expected summary file path inside the session dir."""
    return session_dir / "running_summary.md"


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def test_write_creates_file(summary_path):
    """write action creates the summary file with the given content."""
    from tools.session_summary_tool import session_summary

    result = json.loads(session_summary(
        action="write",
        content="Key decisions: use sqlite-vec for embeddings.",
        session_id=str(summary_path.parent),
    ))
    assert result["success"] is True
    assert summary_path.exists()
    assert summary_path.read_text() == "Key decisions: use sqlite-vec for embeddings."


def test_write_overwrites_existing(summary_path):
    """write action replaces existing content entirely."""
    from tools.session_summary_tool import session_summary

    summary_path.write_text("old content")
    result = json.loads(session_summary(
        action="write",
        content="new content",
        session_id=str(summary_path.parent),
    ))
    assert result["success"] is True
    assert summary_path.read_text() == "new content"


def test_write_empty_content_clears_file(summary_path):
    """write with empty content clears the file."""
    from tools.session_summary_tool import session_summary

    summary_path.write_text("some content")
    result = json.loads(session_summary(
        action="write",
        content="",
        session_id=str(summary_path.parent),
    ))
    assert result["success"] is True
    assert summary_path.read_text() == ""


def test_write_missing_content_is_error():
    """write without content parameter returns an error."""
    from tools.session_summary_tool import session_summary

    result = json.loads(session_summary(
        action="write",
        session_id="/tmp/fake_session",
    ))
    assert result["success"] is False
    assert "content" in result["error"].lower()


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------

def test_append_creates_file_if_missing(summary_path):
    """append creates the file if it doesn't exist."""
    from tools.session_summary_tool import session_summary

    result = json.loads(session_summary(
        action="append",
        content="First entry.",
        session_id=str(summary_path.parent),
    ))
    assert result["success"] is True
    text = summary_path.read_text()
    assert "First entry." in text


def test_append_adds_timestamped_entry(summary_path):
    """append adds a timestamped entry to existing content."""
    from tools.session_summary_tool import session_summary

    summary_path.write_text("Existing summary.\n")
    result = json.loads(session_summary(
        action="append",
        content="New observation.",
        session_id=str(summary_path.parent),
    ))
    assert result["success"] is True
    text = summary_path.read_text()
    assert "Existing summary." in text
    assert "New observation." in text
    # Should have a timestamp line
    assert any(char.isdigit() for char in text.split("New observation.")[0])


def test_append_multiple_entries(summary_path):
    """multiple appends accumulate in order."""
    from tools.session_summary_tool import session_summary

    for i, entry in enumerate(["Entry A", "Entry B", "Entry C"]):
        result = json.loads(session_summary(
            action="append",
            content=entry,
            session_id=str(summary_path.parent),
        ))
        assert result["success"] is True

    text = summary_path.read_text()
    # All three entries present in order
    idx_a = text.index("Entry A")
    idx_b = text.index("Entry B")
    idx_c = text.index("Entry C")
    assert idx_a < idx_b < idx_c


def test_append_missing_content_is_error():
    """append without content returns an error."""
    from tools.session_summary_tool import session_summary

    result = json.loads(session_summary(
        action="append",
        session_id="/tmp/fake_session",
    ))
    assert result["success"] is False
    assert "content" in result["error"].lower()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def test_read_returns_content(summary_path):
    """read returns the current summary content."""
    from tools.session_summary_tool import session_summary

    summary_path.write_text("Summary content here.")
    result = json.loads(session_summary(
        action="read",
        session_id=str(summary_path.parent),
    ))
    assert result["success"] is True
    assert result["content"] == "Summary content here."


def test_read_missing_file_returns_empty(summary_path):
    """read returns empty string when no summary file exists."""
    from tools.session_summary_tool import session_summary

    result = json.loads(session_summary(
        action="read",
        session_id=str(summary_path.parent),
    ))
    assert result["success"] is True
    assert result["content"] == ""


def test_read_empty_file_returns_empty(summary_path):
    """read returns empty string for an empty file."""
    from tools.session_summary_tool import session_summary

    summary_path.write_text("")
    result = json.loads(session_summary(
        action="read",
        session_id=str(summary_path.parent),
    ))
    assert result["success"] is True
    assert result["content"] == ""


# ---------------------------------------------------------------------------
# Invalid actions
# ---------------------------------------------------------------------------

def test_invalid_action_is_error():
    """unknown action returns an error."""
    from tools.session_summary_tool import session_summary

    result = json.loads(session_summary(
        action="delete",
        session_id="/tmp/fake_session",
    ))
    assert result["success"] is False
    assert "unknown action" in result["error"].lower()


def test_missing_session_id_is_error():
    """missing session_id returns an error."""
    from tools.session_summary_tool import session_summary

    result = json.loads(session_summary(
        action="read",
    ))
    assert result["success"] is False
    assert "session_id" in result["error"].lower()


# ---------------------------------------------------------------------------
# Schema registration
# ---------------------------------------------------------------------------

def test_tool_is_registered():
    """session_summary is registered in the tool registry."""
    from tools.registry import registry
    entry = registry.get_entry("session_summary")
    assert entry is not None, "session_summary not found in registry"
    assert entry.name == "session_summary"
    assert entry.schema is not None
    assert entry.handler is not None


def test_schema_has_required_params():
    """schema includes action, content, and session_id parameters."""
    from tools.registry import registry
    entry = registry.get_entry("session_summary")
    params = entry.schema["parameters"]["properties"]
    assert "action" in params
    assert "content" in params
    assert "session_id" in params
    assert params["action"]["enum"] == ["write", "append", "read"]
