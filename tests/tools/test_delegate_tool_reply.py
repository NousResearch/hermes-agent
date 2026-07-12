"""Tests for the ``delegate_tool_reply`` explicit delivery channel.

Covers:
- handler: records to agent-instance state + spill file + ack result
- extraction: no-call fallback, single call, multi-call append, compression
  safety (agent-instance state survives when messages[] is replaced)
- visibility: delegation_reply not in CONFIGURABLE_TOOLSETS, not in default
  tool definitions, but present in child toolsets built by _build_child_agent
  (validated via constructor-capture pattern from test_delegate.py)
- system prompt discipline injection
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import tools.delegate_tool as dt
import tools.delegate_tool_reply as dtr
from tools.registry import registry


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def test_handler_records_to_agent_instance_and_spills(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        agent = MagicMock()
        agent._subagent_id = "child-123"
        result = dtr.delegate_tool_reply(content="my deliverable", parent_agent=agent)
        data = json.loads(result)
        assert data["acknowledged"] is True
        # Agent-instance state recorded
        assert hasattr(agent, "_delegate_reply_chunks")
        assert agent._delegate_reply_chunks == ["my deliverable"]
        # Spill file written
        if data["path"]:
            with open(data["path"], encoding="utf-8") as f:
                assert f.read() == "my deliverable"


def test_handler_multi_call_appends_to_instance_list(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        agent = MagicMock()
        agent._subagent_id = "child-456"
        dtr.delegate_tool_reply(content="chunk1", parent_agent=agent)
        dtr.delegate_tool_reply(content="chunk2", parent_agent=agent)
        assert agent._delegate_reply_chunks == ["chunk1", "chunk2"]


def test_handler_no_agent_still_spills(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        result = dtr.delegate_tool_reply(content="orphan deliverable", parent_agent=None)
        data = json.loads(result)
        assert data["acknowledged"] is True


# ---------------------------------------------------------------------------
# Extraction: _extract_reply_deliverable (reads from agent instance)
# ---------------------------------------------------------------------------

def test_extraction_no_chunks_returns_none():
    child = MagicMock()
    # No _delegate_reply_chunks attribute set
    del child._delegate_reply_chunks
    assert dt._extract_reply_deliverable(child) is None


def test_extraction_single_chunk():
    child = MagicMock()
    child._delegate_reply_chunks = ["THE REPORT"]
    assert dt._extract_reply_deliverable(child) == "THE REPORT"


def test_extraction_multi_chunk_appends_in_order():
    child = MagicMock()
    child._delegate_reply_chunks = ["part1", "part2"]
    assert dt._extract_reply_deliverable(child) == "part1\n\npart2"


def test_extraction_empty_list_returns_empty_string_not_none():
    child = MagicMock()
    child._delegate_reply_chunks = []
    assert dt._extract_reply_deliverable(child) == ""


def test_extraction_non_list_returns_none():
    child = MagicMock()
    child._delegate_reply_chunks = "not a list"
    assert dt._extract_reply_deliverable(child) is None


# ---------------------------------------------------------------------------
# Compression safety regression (the core teknium1 review point)
# ---------------------------------------------------------------------------

def test_extraction_survives_context_compression():
    """The deliverable is read from agent-instance state, not messages[].

    Context compression replaces the middle of messages[] with a summary
    (context_compressor.py Phase 4). This test proves that even if messages[]
    is completely replaced by a synthetic summary, the deliverable recorded on
    the agent instance is intact — because the handler wrote it at execution
    time, outside the transcript.
    """
    child = MagicMock()
    child._delegate_reply_chunks = ["FULL AUDIT REPORT"]
    # Simulate compression: messages[] is now a synthetic summary, no
    # delegate_tool_reply tool calls remain in it.
    compressed_messages = [
        {"role": "user", "content": "do the audit"},
        {"role": "assistant", "content": "[Summary of earlier turns: subagent ran audit and delivered results.]"},
        {"role": "assistant", "content": "done"},
    ]
    # Extraction does NOT read messages — it reads the agent instance.
    assert dt._extract_reply_deliverable(child) == "FULL AUDIT REPORT"
    # Even if someone passed messages, it wouldn't matter — the function
    # signature takes `child`, not `messages`.


def test_extraction_multi_chunk_survives_compression():
    child = MagicMock()
    child._delegate_reply_chunks = ["chunk-A", "chunk-B", "chunk-C"]
    assert dt._extract_reply_deliverable(child) == "chunk-A\n\nchunk-B\n\nchunk-C"


# ---------------------------------------------------------------------------
# Visibility / toolset membership (constructor-capture pattern)
# ---------------------------------------------------------------------------

def _make_mock_parent():
    """Create a mock parent matching test_delegate.py's _make_mock_parent."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = MagicMock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def test_build_child_agent_includes_delegation_reply():
    """Exercise the real _build_child_agent, not a hand-written append."""
    parent = _make_mock_parent()
    parent.enabled_toolsets = ["terminal", "file"]

    with patch("tools.delegate_tool._load_config", return_value={}):
        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child

            dt._build_child_agent(
                task_index=0,
                goal="Test delivery channel",
                context=None,
                toolsets=["terminal", "file"],
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )

        enabled = MockAgent.call_args[1]["enabled_toolsets"]
        assert "delegation_reply" in enabled


def test_delegation_reply_not_in_configurable_toolsets():
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS
    keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    assert "delegation_reply" not in keys


def test_delegation_reply_not_in_core_tools():
    from toolsets import _HERMES_CORE_TOOLS
    assert "delegate_tool_reply" not in _HERMES_CORE_TOOLS


def test_delegation_reply_toolset_resolves():
    from toolsets import get_toolset
    ts = get_toolset("delegation_reply")
    assert ts is not None
    assert "delegate_tool_reply" in ts["tools"]


def test_tool_registered_under_delegation_reply_toolset():
    entry = registry.get_entry("delegate_tool_reply")
    assert entry is not None
    assert entry.toolset == "delegation_reply"


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
