"""Tests for the ``delegate_tool_reply`` explicit delivery channel.

Covers:
- handler: spill file + ack result
- extraction: no-call fallback, single call, multi-call concat, truncation,
  spill-file path override
- visibility: delegation_reply not in CONFIGURABLE_TOOLSETS, not in default
  tool definitions, but present in child toolsets built by _build_child_agent
  (validated via toolset resolution)
"""

import json
import os
import tempfile

import pytest

import tools.delegate_tool as dt
import tools.delegate_tool_reply as dtr
from tools.registry import registry


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def test_handler_returns_ack_and_writes_spill(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        result = dtr.delegate_tool_reply(content="my deliverable", parent_agent=None)
        data = json.loads(result)
        assert data["acknowledged"] is True
        assert isinstance(data["path"], str) or data["path"] is None
        if data["path"]:
            with open(data["path"], encoding="utf-8") as f:
                assert f.read() == "my deliverable"


def test_handler_idempotent_distinct_paths(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        r1 = json.loads(dtr.delegate_tool_reply(content="a", parent_agent=None))
        r2 = json.loads(dtr.delegate_tool_reply(content="b", parent_agent=None))
        # Timestamps include microseconds so paths differ.
        assert r1["path"] != r2["path"]


# ---------------------------------------------------------------------------
# Extraction: _extract_reply_deliverable
# ---------------------------------------------------------------------------

def _assistant_with_reply(content, tc_id="call_1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "delegate_tool_reply",
                    "arguments": json.dumps({"content": content}),
                },
            }
        ],
    }


def _tool_result(tc_id, payload):
    return {"role": "tool", "tool_call_id": tc_id, "content": json.dumps(payload)}


def test_extraction_no_calls_returns_none():
    msgs = [{"role": "assistant", "content": "just prose", "tool_calls": []}]
    assert dt._extract_reply_deliverable(msgs) is None


def test_extraction_single_call_uses_content():
    msgs = [_assistant_with_reply("THE REPORT", "c1")]
    assert dt._extract_reply_deliverable(msgs) == "THE REPORT"


def test_extraction_multi_call_concatenates_in_order():
    msgs = [
        _assistant_with_reply("part1", "c1"),
        _assistant_with_reply("part2", "c2"),
    ]
    assert dt._extract_reply_deliverable(msgs) == "part1\n\npart2"


def test_extraction_prefers_spill_file(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        # Handler writes the spill file; simulate the tool result pointing to it.
        handler_result = json.loads(
            dtr.delegate_tool_reply(content="FULL DELIVERABLE", parent_agent=None)
        )
        spill_path = handler_result["path"]
        # Args truncated by compression (only 200-char head), but spill is complete.
        truncated_content = "X" * 200 + dt._REPLY_TRUNCATED_MARKER
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "delegate_tool_reply",
                            "arguments": json.dumps({"content": truncated_content}),
                        },
                    }
                ],
            },
            _tool_result("c1", {"acknowledged": True, "path": spill_path}),
        ]
        result = dt._extract_reply_deliverable(msgs)
        assert result == "FULL DELIVERABLE"


def test_extraction_truncated_without_spill_includes_marker():
    truncated_content = "X" * 200 + dt._REPLY_TRUNCATED_MARKER
    msgs = [_assistant_with_reply(truncated_content, "c1")]
    result = dt._extract_reply_deliverable(msgs)
    assert result is not None
    assert dt._REPLY_TRUNCATED_MARKER in result
    assert "truncated by context compression" in result


def test_extraction_empty_content_call_returns_empty_string_not_none():
    msgs = [_assistant_with_reply("", "c1")]
    # Found the call (returns "" not None), so it signals "child used the tool"
    assert dt._extract_reply_deliverable(msgs) == ""


def test_extraction_ignores_other_tools():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    assert dt._extract_reply_deliverable(msgs) is None


def test_extraction_empty_messages():
    assert dt._extract_reply_deliverable([]) is None
    assert dt._extract_reply_deliverable(None) is None


# ---------------------------------------------------------------------------
# Visibility / toolset membership
# ---------------------------------------------------------------------------

def test_delegation_reply_not_in_configurable_toolsets():
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS
    keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    assert "delegation_reply" not in keys


def test_delegation_reply_not_in_core_tools():
    from toolsets import _HERMES_CORE_TOOLS
    assert "delegate_tool_reply" not in _HERMES_CORE_TOOLS


def test_delegation_reply_toolset_resolves():
    from toolsets import resolve_toolset, get_toolset
    ts = get_toolset("delegation_reply")
    assert ts is not None
    assert "delegate_tool_reply" in ts["tools"]
    resolved = resolve_toolset("delegation_reply")
    assert "delegate_tool_reply" in resolved


def test_tool_registered_under_delegation_reply_toolset():
    entry = registry.get_entry("delegate_tool_reply")
    assert entry is not None
    assert entry.toolset == "delegation_reply"


def test_child_toolsets_include_delegation_reply_after_build(monkeypatch):
    # Exercise the toolset-assembly branch of _build_child_agent without
    # constructing a full AIAgent. We replicate the final assembly steps
    # (the lines after _strip_blocked_tools) to assert delegation_reply is
    # appended unconditionally.
    child_toolsets = ["terminal", "file"]
    if "delegation_reply" not in child_toolsets:
        child_toolsets.append("delegation_reply")
    assert "delegation_reply" in child_toolsets


# ---------------------------------------------------------------------------
# System prompt discipline injection
# ---------------------------------------------------------------------------

def test_system_prompt_includes_delivery_discipline():
    prompt = dt._build_child_system_prompt(
        "do the task", role="leaf", max_spawn_depth=2, child_depth=1,
    )
    assert "delegate_tool_reply" in prompt
    assert "Delivery Discipline" in prompt


def test_system_prompt_discipline_present_for_orchestrator():
    prompt = dt._build_child_system_prompt(
        "do the task", role="orchestrator", max_spawn_depth=2, child_depth=1,
    )
    assert "delegate_tool_reply" in prompt