#!/usr/bin/env python3
"""
Tests for the subagent delegation tool.

Uses mock AIAgent instances to test the delegation logic without
requiring API keys or real LLM calls.

Run with:  python -m pytest tests/test_delegate.py -v
   or:     python tests/test_delegate.py
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_BLOCKED_TOOLS,
    DELEGATE_TASK_SCHEMA,
    MAX_CONCURRENT_CHILDREN,
    MAX_DEPTH,
    check_delegate_requirements,
    delegate_task,
    _build_child_system_prompt,
    _strip_blocked_tools,
    _resolve_delegation_credentials,
)


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    return parent


class TestDelegateRequirements(unittest.TestCase):
    def test_always_available(self):
        self.assertTrue(check_delegate_requirements())

    def test_schema_valid(self):
        self.assertEqual(DELEGATE_TASK_SCHEMA["name"], "delegate_task")
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("goal", props)
        self.assertIn("tasks", props)
        self.assertIn("context", props)
        self.assertIn("toolsets", props)
        self.assertIn("max_iterations", props)
        self.assertEqual(props["tasks"]["maxItems"], 3)


class TestChildSystemPrompt(unittest.TestCase):
    def test_goal_only(self):
        prompt = _build_child_system_prompt("Fix the tests")
        self.assertIn("Fix the tests", prompt)
        self.assertIn("YOUR TASK", prompt)
        self.assertNotIn("CONTEXT", prompt)

    def test_goal_with_context(self):
        prompt = _build_child_system_prompt("Fix the tests", "Error: assertion failed in test_foo.py line 42")
        self.assertIn("Fix the tests", prompt)
        self.assertIn("CONTEXT", prompt)
        self.assertIn("assertion failed", prompt)

    def test_empty_context_ignored(self):
        prompt = _build_child_system_prompt("Do something", "  ")
        self.assertNotIn("CONTEXT", prompt)


class TestStripBlockedTools(unittest.TestCase):
    def test_removes_blocked_toolsets(self):
        result = _strip_blocked_tools(["terminal", "file", "delegation", "clarify", "memory", "code_execution"])
        self.assertEqual(sorted(result), ["file", "terminal"])

    def test_preserves_allowed_toolsets(self):
        result = _strip_blocked_tools(["terminal", "file", "web", "browser"])
        self.assertEqual(sorted(result), ["browser", "file", "terminal", "web"])

    def test_empty_input(self):
        result = _strip_blocked_tools([])
        self.assertEqual(result, [])


class TestDelegateTask(unittest.TestCase):
    def test_no_parent_agent(self):
        result = json.loads(delegate_task(goal="test"))
        self.assertIn("error", result)
        self.assertIn("parent agent", result["error"])

    def test_depth_limit(self):
        parent = _make_mock_parent(depth=2)
        result = json.loads(delegate_task(goal="test", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("depth limit", result["error"].lower())

    def test_no_goal_or_tasks(self):
        parent = _make_mock_parent()
        result = json.loads(delegate_task(parent_agent=parent))
        self.assertIn("error", result)

    def test_empty_goal(self):
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="  ", parent_agent=parent))
        self.assertIn("error", result)

    def test_task_missing_goal(self):
        parent = _make_mock_parent()
        result = json.loads(delegate_task(tasks=[{"context": "no goal here"}], parent_agent=parent))
        self.assertIn("error", result)

    @patch("tools.delegate_tool._run_single_child")
    def test_single_task_mode(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done!", "api_calls": 3, "duration_seconds": 5.0
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Fix tests", context="error log...", parent_agent=parent))
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["status"], "completed")
        self.assertEqual(result["results"][0]["summary"], "Done!")
        mock_run.assert_called_once()

    @patch("tools.delegate_tool._run_single_child")
    def test_batch_mode(self, mock_run):
        mock_run.side_effect = [
            {"task_index": 0, "status": "completed", "summary": "Result A", "api_calls": 2, "duration_seconds": 3.0},
            {"task_index": 1, "status": "completed", "summary": "Result B", "api_calls": 4, "duration_seconds": 6.0},
        ]
        parent = _make_mock_parent()
        tasks = [
            {"goal": "Research topic A"},
            {"goal": "Research topic B"},
        ]
        result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["summary"], "Result A")
        self.assertEqual(result["results"][1]["summary"], "Result B")
        self.assertIn("total_duration_seconds", result)

    @patch("tools.delegate_tool._run_single_child")
    def test_batch_capped_at_3(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
        }
        parent = _make_mock_parent()
        tasks = [{"goal": f"Task {i}"} for i in range(5)]
        result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))
        # Should only run 3 tasks (MAX_CONCURRENT_CHILDREN)
        self.assertEqual(mock_run.call_count, 3)

    @patch("tools.delegate_tool._run_single_child")
    def test_batch_ignores_toplevel_goal(self, mock_run):
        """When tasks array is provided, top-level goal/context/toolsets are ignored."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(
            goal="This should be ignored",
            tasks=[{"goal": "Actual task"}],
            parent_agent=parent,
        ))
        # The mock was called with the tasks array item, not the top-level goal
        call_args = mock_run.call_args
        self.assertEqual(call_args.kwargs.get("goal") or call_args[1].get("goal", call_args[0][1] if len(call_args[0]) > 1 else None), "Actual task")

    @patch("tools.delegate_tool._run_single_child")
    def test_failed_child_included_in_results(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "error",
            "summary": None, "error": "Something broke",
            "api_calls": 0, "duration_seconds": 0.5
        }
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="Break things", parent_agent=parent))
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertIn("Something broke", result["results"][0]["error"])

    def test_depth_increments(self):
        """Verify child gets parent's depth + 1."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test depth", parent_agent=parent)
            self.assertEqual(mock_child._delegate_depth, 1)

    def test_active_children_tracking(self):
        """Verify children are registered/unregistered for interrupt propagation."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test tracking", parent_agent=parent)
            self.assertEqual(len(parent._active_children), 0)

    def test_child_inherits_runtime_credentials(self):
        parent = _make_mock_parent(depth=0)
        parent.base_url = "https://chatgpt.com/backend-api/codex"
        parent.api_key = "codex-token"
        parent.provider = "openai-codex"
        parent.api_mode = "codex_responses"

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test runtime inheritance", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["base_url"], parent.base_url)
            self.assertEqual(kwargs["api_key"], parent.api_key)
            self.assertEqual(kwargs["provider"], parent.provider)
            self.assertEqual(kwargs["api_mode"], parent.api_mode)


class TestDelegateObservability(unittest.TestCase):
    """Tests for enriched metadata returned by _run_single_child."""

    def test_observability_fields_present(self):
        """Completed child should return tool_trace, tokens, model, exit_reason."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 5000
            mock_child.session_completion_tokens = 1200
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 3,
                "messages": [
                    {"role": "user", "content": "do something"},
                    {"role": "assistant", "tool_calls": [
                        {"id": "tc_1", "function": {"name": "web_search", "arguments": '{"query": "test"}'}}
                    ]},
                    {"role": "tool", "tool_call_id": "tc_1", "content": '{"results": [1,2,3]}'},
                    {"role": "assistant", "content": "done"},
                ],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test observability", parent_agent=parent))
            entry = result["results"][0]

            # Core observability fields
            self.assertEqual(entry["model"], "claude-sonnet-4-6")
            self.assertEqual(entry["exit_reason"], "completed")
            self.assertEqual(entry["tokens"]["input"], 5000)
            self.assertEqual(entry["tokens"]["output"], 1200)

            # Tool trace
            self.assertEqual(len(entry["tool_trace"]), 1)
            self.assertEqual(entry["tool_trace"][0]["tool"], "web_search")
            self.assertIn("args_bytes", entry["tool_trace"][0])
            self.assertIn("result_bytes", entry["tool_trace"][0])
            self.assertEqual(entry["tool_trace"][0]["status"], "ok")

    def test_tool_trace_detects_error(self):
        """Tool results containing 'error' should be marked as error status."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 0
            mock_child.session_completion_tokens = 0
            mock_child.run_conversation.return_value = {
                "final_response": "failed",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [
                    {"role": "assistant", "tool_calls": [
                        {"id": "tc_1", "function": {"name": "terminal", "arguments": '{"cmd": "ls"}'}}
                    ]},
                    {"role": "tool", "tool_call_id": "tc_1", "content": "Error: command not found"},
                ],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test error trace", parent_agent=parent))
            trace = result["results"][0]["tool_trace"]
            self.assertEqual(trace[0]["status"], "error")

    def test_parallel_tool_calls_paired_correctly(self):
        """Parallel tool calls should each get their own result via tool_call_id matching."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 3000
            mock_child.session_completion_tokens = 800
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [
                    {"role": "assistant", "tool_calls": [
                        {"id": "tc_a", "function": {"name": "web_search", "arguments": '{"q": "a"}'}},
                        {"id": "tc_b", "function": {"name": "web_search", "arguments": '{"q": "b"}'}},
                        {"id": "tc_c", "function": {"name": "terminal", "arguments": '{"cmd": "ls"}'}},
                    ]},
                    {"role": "tool", "tool_call_id": "tc_a", "content": '{"ok": true}'},
                    {"role": "tool", "tool_call_id": "tc_b", "content": "Error: rate limited"},
                    {"role": "tool", "tool_call_id": "tc_c", "content": "file1.txt\nfile2.txt"},
                    {"role": "assistant", "content": "done"},
                ],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test parallel", parent_agent=parent))
            trace = result["results"][0]["tool_trace"]

            # All three tool calls should have results
            self.assertEqual(len(trace), 3)

            # First: web_search → ok
            self.assertEqual(trace[0]["tool"], "web_search")
            self.assertEqual(trace[0]["status"], "ok")
            self.assertIn("result_bytes", trace[0])

            # Second: web_search → error
            self.assertEqual(trace[1]["tool"], "web_search")
            self.assertEqual(trace[1]["status"], "error")
            self.assertIn("result_bytes", trace[1])

            # Third: terminal → ok
            self.assertEqual(trace[2]["tool"], "terminal")
            self.assertEqual(trace[2]["status"], "ok")
            self.assertIn("result_bytes", trace[2])

    def test_exit_reason_interrupted(self):
        """Interrupted child should report exit_reason='interrupted'."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 0
            mock_child.session_completion_tokens = 0
            mock_child.run_conversation.return_value = {
                "final_response": "",
                "completed": False,
                "interrupted": True,
                "api_calls": 2,
                "messages": [],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test interrupt", parent_agent=parent))
            self.assertEqual(result["results"][0]["exit_reason"], "interrupted")

    def test_exit_reason_max_iterations(self):
        """Child that didn't complete and wasn't interrupted hit max_iterations."""
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 0
            mock_child.session_completion_tokens = 0
            mock_child.run_conversation.return_value = {
                "final_response": "",
                "completed": False,
                "interrupted": False,
                "api_calls": 50,
                "messages": [],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="Test max iter", parent_agent=parent))
            self.assertEqual(result["results"][0]["exit_reason"], "max_iterations")


class TestBlockedTools(unittest.TestCase):
    def test_blocked_tools_constant(self):
        for tool in ["delegate_task", "clarify", "memory", "send_message", "execute_code"]:
            self.assertIn(tool, DELEGATE_BLOCKED_TOOLS)

    def test_constants(self):
        self.assertEqual(MAX_CONCURRENT_CHILDREN, 3)
        self.assertEqual(MAX_DEPTH, 2)


class TestDelegationCredentialResolution(unittest.TestCase):
    """Tests for provider:model credential resolution in delegation config."""

    def test_no_provider_returns_none_credentials(self):
        """When delegation.provider is empty, all credentials are None (inherit parent)."""
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "", "provider": ""}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertIsNone(creds["provider"])
        self.assertIsNone(creds["base_url"])
        self.assertIsNone(creds["api_key"])
        self.assertIsNone(creds["api_mode"])
        self.assertIsNone(creds["model"])

    def test_model_only_no_provider(self):
        """When only model is set (no provider), model is returned but credentials are None."""
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "google/gemini-3-flash-preview", "provider": ""}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "google/gemini-3-flash-preview")
        self.assertIsNone(creds["provider"])
        self.assertIsNone(creds["base_url"])
        self.assertIsNone(creds["api_key"])

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_provider_resolves_full_credentials(self, mock_resolve):
        """When delegation.provider is set, full credentials are resolved."""
        mock_resolve.return_value = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-test-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "google/gemini-3-flash-preview", "provider": "openrouter"}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "google/gemini-3-flash-preview")
        self.assertEqual(creds["provider"], "openrouter")
        self.assertEqual(creds["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(creds["api_key"], "sk-or-test-key")
        self.assertEqual(creds["api_mode"], "chat_completions")
        mock_resolve.assert_called_once_with(requested="openrouter")

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_nous_provider_resolves_nous_credentials(self, mock_resolve):
        """Nous provider resolves Nous Portal base_url and api_key."""
        mock_resolve.return_value = {
            "provider": "nous",
            "base_url": "https://inference-api.nousresearch.com/v1",
            "api_key": "nous-agent-key-xyz",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "hermes-3-llama-3.1-8b", "provider": "nous"}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["provider"], "nous")
        self.assertEqual(creds["base_url"], "https://inference-api.nousresearch.com/v1")
        self.assertEqual(creds["api_key"], "nous-agent-key-xyz")
        mock_resolve.assert_called_once_with(requested="nous")

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_provider_resolution_failure_raises_valueerror(self, mock_resolve):
        """When provider resolution fails, ValueError is raised with helpful message."""
        mock_resolve.side_effect = RuntimeError("OPENROUTER_API_KEY not set")
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "some-model", "provider": "openrouter"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_delegation_credentials(cfg, parent)
        self.assertIn("openrouter", str(ctx.exception).lower())
        self.assertIn("Cannot resolve", str(ctx.exception))

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_provider_resolves_but_no_api_key_raises(self, mock_resolve):
        """When provider resolves but has no API key, ValueError is raised."""
        mock_resolve.return_value = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        cfg = {"model": "some-model", "provider": "openrouter"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_delegation_credentials(cfg, parent)
        self.assertIn("no API key", str(ctx.exception))

    def test_missing_config_keys_inherit_parent(self):
        """When config dict has no model/provider keys at all, inherits parent."""
        parent = _make_mock_parent(depth=0)
        cfg = {"max_iterations": 45}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertIsNone(creds["model"])
        self.assertIsNone(creds["provider"])


class TestDelegationProviderIntegration(unittest.TestCase):
    """Integration tests: delegation config → _run_single_child → AIAgent construction."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_config_provider_credentials_reach_child_agent(self, mock_creds, mock_cfg):
        """When delegation.provider is configured, child agent gets resolved credentials."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "google/gemini-3-flash-preview",
            "provider": "openrouter",
        }
        mock_creds.return_value = {
            "model": "google/gemini-3-flash-preview",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-delegation-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test provider routing", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], "google/gemini-3-flash-preview")
            self.assertEqual(kwargs["provider"], "openrouter")
            self.assertEqual(kwargs["base_url"], "https://openrouter.ai/api/v1")
            self.assertEqual(kwargs["api_key"], "sk-or-delegation-key")
            self.assertEqual(kwargs["api_mode"], "chat_completions")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_cross_provider_delegation(self, mock_creds, mock_cfg):
        """Parent on Nous, subagent on OpenRouter — full credential switch."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "google/gemini-3-flash-preview",
            "provider": "openrouter",
        }
        mock_creds.return_value = {
            "model": "google/gemini-3-flash-preview",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        parent.provider = "nous"
        parent.base_url = "https://inference-api.nousresearch.com/v1"
        parent.api_key = "nous-key-abc"

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Cross-provider test", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            # Child should use OpenRouter, NOT Nous
            self.assertEqual(kwargs["provider"], "openrouter")
            self.assertEqual(kwargs["base_url"], "https://openrouter.ai/api/v1")
            self.assertEqual(kwargs["api_key"], "sk-or-key")
            self.assertNotEqual(kwargs["base_url"], parent.base_url)
            self.assertNotEqual(kwargs["api_key"], parent.api_key)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_empty_config_inherits_parent(self, mock_creds, mock_cfg):
        """When delegation config is empty, child inherits parent credentials."""
        mock_cfg.return_value = {"max_iterations": 45, "model": "", "provider": ""}
        mock_creds.return_value = {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Test inherit", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], parent.model)
            self.assertEqual(kwargs["provider"], parent.provider)
            self.assertEqual(kwargs["base_url"], parent.base_url)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_credential_error_returns_json_error(self, mock_creds, mock_cfg):
        """When credential resolution fails, delegate_task returns a JSON error."""
        mock_cfg.return_value = {"model": "bad-model", "provider": "nonexistent"}
        mock_creds.side_effect = ValueError(
            "Cannot resolve delegation provider 'nonexistent': Unknown provider"
        )
        parent = _make_mock_parent(depth=0)

        result = json.loads(delegate_task(goal="Should fail", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("Cannot resolve", result["error"])
        self.assertIn("nonexistent", result["error"])

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_batch_mode_all_children_get_credentials(self, mock_creds, mock_cfg):
        """In batch mode, all children receive the resolved credentials."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "meta-llama/llama-4-scout",
            "provider": "openrouter",
        }
        mock_creds.return_value = {
            "model": "meta-llama/llama-4-scout",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-batch",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "Done", "api_calls": 1, "duration_seconds": 1.0
            }

            tasks = [{"goal": "Task A"}, {"goal": "Task B"}]
            delegate_task(tasks=tasks, parent_agent=parent)

            for call in mock_run.call_args_list:
                self.assertEqual(call.kwargs.get("model"), "meta-llama/llama-4-scout")
                self.assertEqual(call.kwargs.get("override_provider"), "openrouter")
                self.assertEqual(call.kwargs.get("override_base_url"), "https://openrouter.ai/api/v1")
                self.assertEqual(call.kwargs.get("override_api_key"), "sk-or-batch")
                self.assertEqual(call.kwargs.get("override_api_mode"), "chat_completions")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_model_only_no_provider_inherits_parent_credentials(self, mock_creds, mock_cfg):
        """Setting only model (no provider) changes model but keeps parent credentials."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "google/gemini-3-flash-preview",
            "provider": "",
        }
        mock_creds.return_value = {
            "model": "google/gemini-3-flash-preview",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1
            }
            MockAgent.return_value = mock_child

            delegate_task(goal="Model only test", parent_agent=parent)

            _, kwargs = MockAgent.call_args
            # Model should be overridden
            self.assertEqual(kwargs["model"], "google/gemini-3-flash-preview")
            # But provider/base_url/api_key should inherit from parent
            self.assertEqual(kwargs["provider"], parent.provider)
            self.assertEqual(kwargs["base_url"], parent.base_url)


if __name__ == "__main__":
    unittest.main()
