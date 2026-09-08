"""Unit tests for tool schema minifier and high-SNR output reducer."""

from __future__ import annotations

import json
import pytest

from tools.schema_minifier import (
    minify_and_canonicalize_tools,
    minify_and_sort_schema_node,
    minify_tool_definition,
)
from tools.tool_output_reducer import extract_high_snr_blocks, reduce_tool_output


class TestSchemaMinifierAndPrefixCache:
    def test_prune_redundant_meta_and_preserve_additional_properties(self):
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
        # Check sorted keys deterministically
        assert list(minified.keys()) == ["a_param", "z_param"]
        # Check pruned metadata
        assert "$schema" not in minified["a_param"]
        assert "enum" not in minified["a_param"]
        # Must PRESERVE additionalProperties: False for structured outputs
        assert minified["a_param"]["additionalProperties"] is False

    def test_minify_and_canonicalize_tools_preserves_tool_ordering(self):
        tools = [
            {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "execute_code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "browser_navigate", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}},
        ]
        canonical = minify_and_canonicalize_tools(tools)
        # Verify tool list order is preserved (no reordering of tool definitions)
        names = [t["function"]["name"] for t in canonical]
        assert names == ["web_search", "execute_code", "browser_navigate"]

    def test_cycle_and_depth_protection(self):
        # Recursive self-referencing dictionary
        recursive_dict = {"a": 1}
        recursive_dict["self"] = recursive_dict
        # Should not raise RecursionError and return safe bounded structure
        result = minify_and_sort_schema_node(recursive_dict)
        assert result["a"] == 1

        # Deeply nested dict (> 30 levels)
        deep = curr = {}
        for i in range(40):
            curr["nested"] = {}
            curr = curr["nested"]
        curr["leaf"] = "done"

        deep_minified = minify_and_sort_schema_node(deep)
        assert isinstance(deep_minified, dict)

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

    def test_zero_tail_chars_does_not_return_entire_output(self):
        text = "ABC" * 500  # 1500 chars
        # max_chars=251 -> available_chars = 1, head_chars=0, tail_chars=0
        reduced = reduce_tool_output(text, max_chars=251)
        assert len(reduced) < 500
        assert text not in reduced

    def test_json_payload_skips_snr_banners(self):
        json_payload = json.dumps({"status": "error", "logs": ["something error line" for _ in range(200)], "code": 500})
        reduced = reduce_tool_output(json_payload, max_chars=300)
        assert "--- [HIGH-SNR DIAGNOSTIC" not in reduced
        assert len(reduced) <= 400

    def test_redos_guard_large_text(self):
        # Huge noisy text (300k chars)
        large_text = "info: line item\n" * 20000
        # Should finish instantaneously without ReDoS
        reduced = reduce_tool_output(large_text, max_chars=1000)
        assert len(reduced) <= 1500

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
