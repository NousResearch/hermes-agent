"""Unit tests for tool schema minifier, fast tool resolver, and high-SNR output reducer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tools.schema_minifier import (
    minify_and_canonicalize_tools,
    minify_and_sort_schema_node,
    minify_tool_definition,
)
from tools.tool_output_reducer import extract_high_snr_blocks, reduce_tool_output


class TestSchemaMinifierAndPrefixCache:
    def test_prune_redundant_meta_and_sort_keys(self):
        raw_schema = {
            "z_param": {"type": "string", "title": "Z Parameter", "description": "some param"},
            "a_param": {
                "type": "object",
                "$schema": "http://json-schema.org/draft-07/schema#",
                "properties": {},
                "enum": [],
                "additionalProperties": False,
            },
        }
        minified = minify_and_sort_schema_node(raw_schema)
        # Check sorted keys
        assert list(minified.keys()) == ["a_param", "z_param"]
        # Check pruned metadata
        assert "$schema" not in minified["a_param"]
        assert "enum" not in minified["a_param"]
        assert "additionalProperties" not in minified["a_param"]

    def test_minify_and_canonicalize_tools_ordering(self):
        tools = [
            {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "execute_code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "browser_navigate", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}},
        ]
        canonical = minify_and_canonicalize_tools(tools)
        names = [t["function"]["name"] for t in canonical]
        assert names == ["browser_navigate", "execute_code", "web_search"]

    def test_deterministic_serialization_hash(self):
        tool_a = {"type": "function", "function": {"parameters": {"b": 2, "a": 1}, "name": "my_tool", "description": "desc"}}
        tool_b = {"function": {"description": "desc", "name": "my_tool", "parameters": {"a": 1, "b": 2}}, "type": "function"}

        can_a = minify_tool_definition(tool_a)
        can_b = minify_tool_definition(tool_b)

        assert json.dumps(can_a, sort_keys=False) == json.dumps(can_b, sort_keys=False)


class TestHighSNROutputReducer:
    def test_output_under_limit_unchanged(self):
        short_text = "hello world error: none"
        assert reduce_tool_output(short_text, max_chars=1000) == short_text

    def test_extract_traceback_and_panics(self):
        log_text = """
Some noisy build logs line 1
Some noisy build logs line 2
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    run_app()
  File "app.py", line 10, in run_app
    raise ValueError("Invalid configuration provided")
ValueError: Invalid configuration provided
More noisy logs line 3
More noisy logs line 4
thread 'main' panicked at 'assertion failed: `(left == right)`', src/lib.rs:15:5
stack backtrace:
   0: rust_begin_unwind
   1: core::panicking::panic_fmt
Even more noisy logs
"""
        blocks = extract_high_snr_blocks(log_text)
        assert len(blocks) >= 2
        assert any("ValueError: Invalid configuration provided" in b for b in blocks)
        assert any("panicked at" in b for b in blocks)

    def test_reduce_tool_output_preserves_error_block_in_truncated_window(self):
        noise_head = "INITIAL_SETUP_START\n" + "noise line\n" * 200
        error_block = """Traceback (most recent call last):
  File "calculator.py", line 99, in compute
ZeroDivisionError: division by zero"""
        noise_tail = "\nnoise tail\n" * 200 + "\nFINAL_STATUS_FAILED"
        full_text = noise_head + "\n" + error_block + "\n" + noise_tail

        reduced = reduce_tool_output(full_text, max_chars=800)
        assert len(reduced) <= 1200
        assert "ZeroDivisionError: division by zero" in reduced
        assert "INITIAL_SETUP_START" in reduced
        assert "FINAL_STATUS_FAILED" in reduced


class TestFastToolResolver:
    def test_fast_tool_resolver_hydrates_deferred_name(self, monkeypatch):
        from agent.agent_runtime_helpers import repair_tool_call

        agent = MagicMock()
        agent.valid_tool_names = {"tool_search", "tool_describe", "tool_call", "terminal"}
        agent.enabled_toolsets = None
        agent.disabled_toolsets = None

        # Mock scoped deferrable names to include 'process_manage'
        import agent.agent_runtime_helpers as arh
        import agent.tool_executor as te

        monkeypatch.setattr(te, "_tool_search_scoped_names", lambda a: frozenset({"process_manage", "session_search", "cronjob_manage"}))

        # Direct invocation of deferred tool without search roundtrip
        assert repair_tool_call(agent, "process_manage") == "process_manage"
        assert repair_tool_call(agent, "session_search") == "session_search"
        assert repair_tool_call(agent, "SessionSearch") == "session_search"
