"""Regression tests for surfacing child stderr on failed stdio MCP spawns (#73299).

The Windows stdio-MCP bug reporter's central complaint is that a heavy-import
Node MCP server (Context7, Playwright) fails the handshake with a generic
``Connection closed`` while the *real* error lives in the child's stderr —
``Cannot find module 'express'``, ``SyntaxError: ``, ``EADDRINUSE``, etc. —
that Hermes silently redirects to ``~/.hermes/logs/mcp-stderr.log``.

These tests verify the contract added in this fix: ``_capture_stderr_since``
returns whatever the child wrote between the snapshot offset and now, and
``MCPServerTask._attach_child_stderr`` attaches that tail to the exception
via ``add_note`` (Python 3.11+) without mutating the exception's args.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools import mcp_tool


# ---------------------------------------------------------------------------
# _capture_stderr_since — pure function on the log file
# ---------------------------------------------------------------------------


def test_capture_returns_empty_when_offset_equals_current_size(tmp_path, monkeypatch):
    log = tmp_path / "mcp-stderr.log"
    log.write_text("===== [ts] starting MCP server 'x' =====\nold output\n", encoding="utf-8")
    monkeypatch.setattr(mcp_tool, "_resolve_log_path_for_test", lambda: log, raising=False)

    # Fall back to monkeypatching the import inside the helper. The helper
    # imports ``get_hermes_home`` from hermes_constants; we patch the symbol
    # on that module so the helper sees our tmp log file.
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    offset = log.stat().st_size
    assert mcp_tool._capture_stderr_since(offset) == ""


def test_capture_returns_tail_after_offset(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    log = tmp_path / "logs" / "mcp-stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("===== header =====\n", encoding="utf-8")
    offset = log.stat().st_size
    with open(log, "a", encoding="utf-8") as f:
        f.write("Cannot find module 'express'\n    at /srv/dist/index.js:1\n")

    tail = mcp_tool._capture_stderr_since(offset)
    assert "Cannot find module 'express'" in tail
    assert "at /srv/dist/index.js:1" in tail


def test_capture_returns_empty_when_log_missing(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    assert mcp_tool._capture_stderr_since(0) == ""


def test_capture_returns_empty_when_offset_is_negative(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    log = tmp_path / "logs" / "mcp-stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("anything", encoding="utf-8")
    assert mcp_tool._capture_stderr_since(-1) == ""


def test_capture_truncates_when_tail_exceeds_max_bytes(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    log = tmp_path / "logs" / "mcp-stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("h" * 200, encoding="utf-8")
    offset = log.stat().st_size
    log.write_text("x" * 8000, encoding="utf-8")

    tail = mcp_tool._capture_stderr_since(offset, max_bytes=4096)
    assert len(tail) <= 4096
    assert tail  # non-empty


# ---------------------------------------------------------------------------
# _attach_child_stderr — MCPServerTask method
# ---------------------------------------------------------------------------


def _make_task_stub():
    """Minimal stand-in for MCPServerTask — only the method under test is real."""

    class _Stub:
        _attach_child_stderr = mcp_tool.MCPServerTask._attach_child_stderr

    return _Stub()


def test_attach_adds_note_with_child_stderr(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    log = tmp_path / "logs" / "mcp-stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("===== header =====\n", encoding="utf-8")
    offset = log.stat().st_size
    with open(log, "a", encoding="utf-8") as f:
        f.write("Error: Cannot find module 'express'\n")

    task = _make_task_stub()
    exc = RuntimeError("Connection closed")
    task._attach_child_stderr(exc, offset)

    assert exc.args == ("Connection closed",)
    notes = getattr(exc, "__notes__", [])
    assert any("Cannot find module 'express'" in n for n in notes)


def test_attach_noop_when_log_has_no_new_bytes(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    log = tmp_path / "logs" / "mcp-stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("only-this\n", encoding="utf-8")
    offset = log.stat().st_size

    task = _make_task_stub()
    exc = RuntimeError("Connection closed")
    task._attach_child_stderr(exc, offset)

    assert exc.args == ("Connection closed",)
    assert getattr(exc, "__notes__", []) == []


def test_attach_noop_when_log_missing(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    task = _make_task_stub()
    exc = RuntimeError("Connection closed")
    task._attach_child_stderr(exc, 0)
    assert getattr(exc, "__notes__", []) == []


def test_attach_preserves_args_so_classifiers_unchanged(tmp_path, monkeypatch):
    """``str(exc)`` must remain the original to keep auth/permanence classifiers stable."""

    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    log = tmp_path / "logs" / "mcp-stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("===== header =====\n", encoding="utf-8")
    offset = log.stat().st_size
    with open(log, "a", encoding="utf-8") as f:
        f.write("child: port already in use\n")

    task = _make_task_stub()
    exc = ConnectionError("Connection closed")
    task._attach_child_stderr(exc, offset)

    assert str(exc) == "Connection closed"
    assert repr(exc).startswith("ConnectionError('Connection closed')")
    # Notes carry the context; the original ``args`` tuple is untouched.
    assert any("port already in use" in n for n in getattr(exc, "__notes__", []))
