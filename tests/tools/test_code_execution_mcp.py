#!/usr/bin/env python3
"""Tests for exposing read-only MCP tools inside execute_code (#97044).

Covers all 5 maintainer design gates:
  1. Security / read-only gate: only read-only-hinted MCP tools (and read-only utilities) are exposed.
  2. Schema budget: compact name-list in tool description, no per-tool doc bloat.
  3. Config gate: code_execution.expose_mcp_tools default false; boolean or list filter.
  4. Call budget interaction: teaching error names the blocked tool.
  5. RPC allow-list enforcement: unexposed tools called via RPC are refused.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest

from tools.code_execution_rpc import _handle_rpc_request
from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    _sandbox_failure_hint,
    _sandbox_mcp_tools,
    _sandbox_tools_for,
    build_execute_code_schema,
    generate_hermes_tools_module,
)
from tools.mcp_tool_discovery import get_read_only_mcp_tools, is_mcp_tool_read_only


class TestMcpReadOnlyDetection(unittest.TestCase):
    """Gate 1: Verify read-only detection from discovery metadata and utilities."""

    def test_read_only_hint_detection(self):
        with patch("tools.mcp_tool._mcp_tool_server_names", {
            "linear_search": "linear",
            "linear_create_issue": "linear",
            "notion_query": "notion",
            "notion_write_page": "notion",
        }), patch("tools.mcp_tool._tool_read_only_hints", {
            "linear": {"search": True, "create_issue": False},
            "notion": {"notion_query": True, "notion_write_page": False},
        }):
            self.assertTrue(is_mcp_tool_read_only("linear_search"))
            self.assertFalse(is_mcp_tool_read_only("linear_create_issue"))
            self.assertTrue(is_mcp_tool_read_only("notion_query"))
            self.assertFalse(is_mcp_tool_read_only("notion_write_page"))
            self.assertFalse(is_mcp_tool_read_only("unknown_tool"))

    def test_read_only_utilities(self):
        with patch("tools.mcp_tool._mcp_tool_server_names", {
            "linear_list_resources": "linear",
            "linear_read_resource": "linear",
        }), patch("tools.mcp_tool._tool_read_only_hints", {"linear": {}}):
            self.assertTrue(is_mcp_tool_read_only("linear_list_resources"))
            self.assertTrue(is_mcp_tool_read_only("linear_read_resource"))

    def test_get_read_only_mcp_tools_filtering(self):
        with patch("tools.mcp_tool._mcp_tool_server_names", {
            "linear_search": "linear",
            "linear_create_issue": "linear",
            "notion_query": "notion",
        }), patch("tools.mcp_tool._tool_read_only_hints", {
            "linear": {"search": True, "create_issue": False},
            "notion": {"notion_query": True},
        }):
            # All read-only
            all_ro = get_read_only_mcp_tools()
            self.assertEqual(all_ro, {"linear_search", "notion_query"})

            # Filter by server name
            linear_only = get_read_only_mcp_tools(server_or_tool_filter=["linear"])
            self.assertEqual(linear_only, {"linear_search"})

            # Filter by tool name
            notion_only = get_read_only_mcp_tools(server_or_tool_filter=["notion_query"])
            self.assertEqual(notion_only, {"notion_query"})


class TestConfigGate(unittest.TestCase):
    """Gate 3: code_execution.expose_mcp_tools default false; boolean or list filter."""

    def test_default_is_disabled(self):
        with patch("tools.code_execution_tool._load_config", return_value={}),              patch("tools.mcp_tool._mcp_tool_server_names", {"linear_search": "linear"}),              patch("tools.mcp_tool._tool_read_only_hints", {"linear": {"search": True}}):
            self.assertEqual(_sandbox_mcp_tools(), set())
            self.assertEqual(_sandbox_tools_for(None), SANDBOX_ALLOWED_TOOLS)

    def test_explicit_false_is_disabled(self):
        with patch("tools.code_execution_tool._load_config", return_value={"expose_mcp_tools": False}),              patch("tools.mcp_tool._mcp_tool_server_names", {"linear_search": "linear"}),              patch("tools.mcp_tool._tool_read_only_hints", {"linear": {"search": True}}):
            self.assertEqual(_sandbox_mcp_tools(), set())

    def test_enabled_boolean_true(self):
        with patch("tools.code_execution_tool._load_config", return_value={"expose_mcp_tools": True}),              patch("tools.mcp_tool._mcp_tool_server_names", {
                 "linear_search": "linear",
                 "linear_create": "linear",
                 "notion_query": "notion",
             }),              patch("tools.mcp_tool._tool_read_only_hints", {
                 "linear": {"search": True, "create": False},
                 "notion": {"query": True},
             }):
            # Exposes all read-only MCP tools
            self.assertEqual(_sandbox_mcp_tools(), {"linear_search", "notion_query"})

            # Session enabled_tools intersection
            self.assertEqual(
                _sandbox_mcp_tools(enabled_tools=["web_search", "linear_search"]),
                {"linear_search"},
            )

    def test_enabled_list_filter(self):
        with patch("tools.code_execution_tool._load_config", return_value={"expose_mcp_tools": ["linear"]}),              patch("tools.mcp_tool._mcp_tool_server_names", {
                 "linear_search": "linear",
                 "notion_query": "notion",
             }),              patch("tools.mcp_tool._tool_read_only_hints", {
                 "linear": {"search": True},
                 "notion": {"query": True},
             }):
            self.assertEqual(_sandbox_mcp_tools(), {"linear_search"})


class TestSchemaBudget(unittest.TestCase):
    """Gate 2: Schema diet — compact name-list only, no per-tool doc bloat."""

    def test_schema_without_mcp(self):
        schema = build_execute_code_schema(enabled_mcp_tools=set())
        desc = schema["description"]
        self.assertNotIn("Also callable via `from hermes_tools import ...`", desc)
        self.assertIn("web_search", desc)

    def test_schema_with_mcp_compact_note(self):
        schema = build_execute_code_schema(
            enabled_sandbox_tools={"read_file", "terminal"},
            enabled_mcp_tools={"linear_search", "notion_query"},
        )
        desc = schema["description"]
        self.assertIn(
            "Also callable via `from hermes_tools import ...`: linear_search(...), notion_query(...) "
            "(same arguments as the model-visible tools).",
            desc,
        )
        # Verify no multi-line docstring bloat for MCP tools
        self.assertNotIn("linear_search(query: str", desc)


class TestStubGeneration(unittest.TestCase):
    """Verify hermes_tools.py code generation with MCP stubs."""

    def test_generate_module_with_mcp_stubs(self):
        with patch("tools.code_execution_tool._load_config", return_value={"expose_mcp_tools": True}),              patch("tools.mcp_tool._mcp_tool_server_names", {"linear_search": "linear"}),              patch("tools.mcp_tool._tool_read_only_hints", {"linear": {"search": True}}):
            code = generate_hermes_tools_module(
                enabled_tools=["read_file", "linear_search"],
                transport="uds",
            )
            self.assertIn("def read_file(path: str", code)
            self.assertIn("def linear_search(*args, **kwargs):", code)
            self.assertIn("return _call('linear_search', _payload)", code)

            # Check that code compiles cleanly without syntax errors
            compiled = compile(code, "<string>", "exec")
            self.assertIsNotNone(compiled)


class TestRpcAllowListEnforcement(unittest.TestCase):
    """Verify server-side RPC allow-list enforcement and budget messaging."""

    def test_unexposed_tool_rejected(self):
        counter = [0]
        logs = []
        resp = _handle_rpc_request(
            {"tool": "linear_write_issue", "args": {"title": "x"}},
            allowed_tools=frozenset(["read_file", "linear_search"]),
            tool_call_counter=counter,
            max_tool_calls=50,
            dispatch=lambda t, a: "ok",
            tool_call_log=logs,
            call_start=0.0,
            where="test",
        )
        parsed = json.loads(resp)
        self.assertIn("Tool 'linear_write_issue' is not available in execute_code", parsed["error"])
        self.assertEqual(counter[0], 0)
        self.assertEqual(len(logs), 0)

    def test_allowed_mcp_tool_dispatched(self):
        counter = [0]
        logs = []
        mock_dispatch = MagicMock(return_value=json.dumps({"results": [1, 2, 3]}))
        resp = _handle_rpc_request(
            {"tool": "linear_search", "args": {"query": "bug"}},
            allowed_tools=frozenset(["read_file", "linear_search"]),
            tool_call_counter=counter,
            max_tool_calls=50,
            dispatch=mock_dispatch,
            tool_call_log=logs,
            call_start=0.0,
            where="test",
        )
        self.assertEqual(json.loads(resp), {"results": [1, 2, 3]})
        mock_dispatch.assert_called_once_with("linear_search", {"query": "bug"})
        self.assertEqual(counter[0], 1)
        self.assertEqual(len(logs), 1)

    def test_budget_exhaustion_names_tool(self):
        counter = [50]
        logs = []
        resp = _handle_rpc_request(
            {"tool": "linear_search", "args": {}},
            allowed_tools=frozenset(["read_file", "linear_search"]),
            tool_call_counter=counter,
            max_tool_calls=50,
            dispatch=lambda t, a: "ok",
            tool_call_log=logs,
            call_start=0.0,
            where="test",
        )
        parsed = json.loads(resp)
        self.assertIn("Tool call limit reached (50)", parsed["error"])
        self.assertIn("Call to 'linear_search' blocked", parsed["error"])


class TestFailureHintMcp(unittest.TestCase):
    """Verify actionable failure hints mention available MCP tools."""

    def test_failure_hint_lists_mcp_tools(self):
        with patch("tools.code_execution_tool._load_config", return_value={"expose_mcp_tools": True}),              patch("tools.mcp_tool._mcp_tool_server_names", {"linear_search": "linear"}),              patch("tools.mcp_tool._tool_read_only_hints", {"linear": {"search": True}}):
            hint = _sandbox_failure_hint(
                "ImportError: cannot import name 'foobar' from 'hermes_tools'",
                enabled_tools=["read_file", "linear_search"],
            )
            self.assertIn("linear_search", hint)
            self.assertIn("read_file", hint)


if __name__ == "__main__":
    unittest.main()
