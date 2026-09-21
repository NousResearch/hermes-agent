"""Regression: terminal_tool must fail closed on a command carrying a context
truncation marker rather than execute the surviving fragment or the literal
marker text. Restored 2026-09-21 alongside the context-compressor arg-payload
exemption (#8adf333d1b) — see agent/context_compressor.py's
_COMPRESSION_MARKER_PREFIX for the marker this guards against.
"""

import json

import tools.terminal_tool as terminal_tool


def test_rejects_command_containing_compression_marker():
    command = "echo hello ⟪HERMES-CONTEXT-COMPRESSION: 400 chars elided⟫ world"
    result = json.loads(terminal_tool.terminal_tool(command=command, force=True))
    assert result["status"] == "error"
    assert result["exit_code"] == -1
    assert "truncation marker" in result["error"]


def test_rejects_command_containing_legacy_bare_ellipsis_marker():
    command = "cat somefile...[truncated]"
    result = json.loads(terminal_tool.terminal_tool(command=command, force=True))
    assert result["status"] == "error"
    assert "truncation marker" in result["error"]


def test_ordinary_command_without_marker_is_not_rejected_by_this_guard():
    # Should not be rejected for the truncation-marker reason (may still be
    # rejected/handled downstream for unrelated reasons in this unit context).
    command = "echo hello world"
    result = json.loads(terminal_tool.terminal_tool(command=command, force=True))
    assert "truncation marker" not in (result.get("error") or "")
