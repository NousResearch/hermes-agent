#!/usr/bin/env python3
"""
Tests for per-task model/provider override in delegate_task.

Issue: https://github.com/NousResearch/hermes-agent/issues/18591

Validates the fallback chain for per-task model routing:
  1. Per-task `model` field (highest priority)
  2. `delegation.model` from config
  3. Inherit parent agent model (current default)

And per-task `provider` credential resolution:
  - Per-task provider triggers independent credential resolution
  - Per-task provider + model combo works end-to-end
  - Invalid provider raises ValueError with helpful message

Uses mock AIAgent instances — no API keys, no network calls.

Run with:  python -m pytest tests/tools/test_delegate_task_model_override.py -v
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_agent,
    _resolve_delegation_credentials,
    delegate_task,
)


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "sk-or-test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.openrouter_min_coding_score = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.max_tokens = None
    parent.prefill_messages = None
    parent.reasoning_config = None
    parent._fallback_chain = None
    parent._client_kwargs = {}
    parent.enabled_toolsets = None
    parent.valid_tool_names = ["terminal", "read_file", "web_search"]
    return parent


def _default_creds():
    """Standard delegation credentials mock return."""
    return {
        "model": None,
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
    }


def _openrouter_creds(model="google/gemini-3-flash-preview"):
    """OpenRouter delegation credentials mock return."""
    return {
        "model": model,
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-delegation-key",
        "api_mode": "chat_completions",
    }


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------
class TestPerTaskModelSchema(unittest.TestCase):
    """Verify the per-task model/provider fields in the delegate_task schema."""

    def test_tasks_items_has_model_field(self):
        """Each task item should accept an optional 'model' string."""
        task_props = (
            DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        )
        self.assertIn("model", task_props)
        self.assertEqual(task_props["model"]["type"], "string")

    def test_tasks_items_has_provider_field(self):
        """Each task item should accept an optional 'provider' string."""
        task_props = (
            DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        )
        self.assertIn("provider", task_props)
        self.assertEqual(task_props["provider"]["type"], "string")

    def test_model_and_provider_not_required(self):
        """model and provider must be optional — backward compatible."""
        task_required = (
            DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"].get("required", [])
        )
        self.assertNotIn("model", task_required)
        self.assertNotIn("provider", task_required)

    def test_goal_still_required_in_task(self):
        """'goal' must remain the only required field in each task."""
        task_required = (
            DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["required"]
        )
        self.assertEqual(task_required, ["goal"])


# ---------------------------------------------------------------------------
# Fallback chain tests — model resolution
# ---------------------------------------------------------------------------
class TestPerTaskModelFallback(unittest.TestCase):
    """Per-task model overrides delegation.model, which overrides parent model."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_per_task_model_overrides_delegation_model(self, mock_creds, mock_cfg):
        """When a task specifies model='glm-5-turbo', that model wins over
        delegation.model='glm-5.1'."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "glm-5.1",
            "provider": "",
        }
        mock_creds.return_value = {
            "model": "glm-5.1",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(
                    tasks=[{"goal": "Search docs", "model": "glm-5-turbo"}],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            _, kwargs = MockAgent.call_args
            # Per-task model should win over delegation.model
            self.assertEqual(kwargs["model"], "glm-5-turbo")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_per_task_model_overrides_parent_model(self, mock_creds, mock_cfg):
        """When no delegation.model is set but task specifies model, task wins
        over parent model."""
        mock_cfg.return_value = {"max_iterations": 45, "model": "", "provider": ""}
        mock_creds.return_value = _default_creds()

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(
                    tasks=[{"goal": "Lightweight task", "model": "glm-5-turbo"}],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], "glm-5-turbo")
            self.assertNotEqual(kwargs["model"], parent.model)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_no_per_task_model_inherits_delegation_model(self, mock_creds, mock_cfg):
        """When task has no model field, delegation.model is used."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "glm-5",
            "provider": "",
        }
        mock_creds.return_value = {
            "model": "glm-5",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(
                    tasks=[{"goal": "Default model task"}],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], "glm-5")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_no_model_anywhere_inherits_parent(self, mock_creds, mock_cfg):
        """When no model anywhere, child inherits parent model (current behavior)."""
        mock_cfg.return_value = {"max_iterations": 45, "model": "", "provider": ""}
        mock_creds.return_value = _default_creds()

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(
                    tasks=[{"goal": "Inherit from parent"}],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], parent.model)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_empty_string_model_treated_as_not_specified(self, mock_creds, mock_cfg):
        """Empty string model should fall through to delegation.model or parent."""
        mock_cfg.return_value = {"max_iterations": 45, "model": "glm-5", "provider": ""}
        mock_creds.return_value = {
            "model": "glm-5",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(
                    tasks=[{"goal": "Empty model", "model": ""}],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            _, kwargs = MockAgent.call_args
            # Empty string should NOT override delegation.model
            self.assertEqual(kwargs["model"], "glm-5")


# ---------------------------------------------------------------------------
# Per-task provider credential resolution
# ---------------------------------------------------------------------------
class TestPerTaskProviderResolution(unittest.TestCase):
    """Per-task provider triggers independent credential resolution."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_per_task_provider_triggers_credential_resolution(
        self, mock_creds, mock_cfg
    ):
        """When a task specifies provider='openrouter', credentials are resolved
        for that provider even though delegation config has no provider."""
        mock_cfg.return_value = {"max_iterations": 45, "model": "", "provider": ""}
        # First call: delegation config (no provider)
        # Second call: per-task provider resolution
        mock_creds.side_effect = [
            _default_creds(),
            _openrouter_creds("google/gemini-3-flash-preview"),
        ]

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(
                    tasks=[{
                        "goal": "Research topic",
                        "model": "google/gemini-3-flash-preview",
                        "provider": "openrouter",
                    }],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            _, kwargs = MockAgent.call_args
            self.assertEqual(kwargs["model"], "google/gemini-3-flash-preview")
            self.assertEqual(kwargs["provider"], "openrouter")
            self.assertEqual(
                kwargs["base_url"], "https://openrouter.ai/api/v1"
            )

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_per_task_provider_same_as_delegation_no_re_resolve(
        self, mock_creds, mock_cfg
    ):
        """When per-task provider matches delegation.provider, no extra
        credential resolution needed — reuse delegation creds."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "glm-5",
            "provider": "openrouter",
        }
        mock_creds.return_value = _openrouter_creds("glm-5")

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(
                    tasks=[{
                        "goal": "Same provider",
                        "model": "glm-5-turbo",
                        "provider": "openrouter",
                    }],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            # _resolve_delegation_credentials should only be called once
            # (no re-resolution for same provider)
            self.assertLessEqual(mock_creds.call_count, 1)


# ---------------------------------------------------------------------------
# Multi-task with different models
# ---------------------------------------------------------------------------
class TestMultiTaskDifferentModels(unittest.TestCase):
    """Multiple tasks with different models should each get their own model."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_batch_with_different_models_per_task(self, mock_creds, mock_cfg):
        """3 tasks: one with glm-5.1, one with glm-5-turbo, one inheriting."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "glm-5",
            "provider": "",
        }
        mock_creds.return_value = {
            "model": "glm-5",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

        parent = _make_mock_parent()
        captured_models = []

        def make_child(*args, **kwargs):
            captured_models.append(kwargs.get("model"))
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            return mock_child

        with patch("run_agent.AIAgent", side_effect=make_child):
            result = json.loads(
                delegate_task(
                    tasks=[
                        {"goal": "Complex analysis", "model": "glm-5.1"},
                        {"goal": "Quick search", "model": "glm-5-turbo"},
                        {"goal": "Default task"},
                    ],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 3)
            self.assertEqual(captured_models[0], "glm-5.1")
            self.assertEqual(captured_models[1], "glm-5-turbo")
            self.assertEqual(captured_models[2], "glm-5")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestPerTaskModelErrors(unittest.TestCase):
    """Error cases for per-task model/provider override."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_invalid_provider_returns_error(self, mock_creds, mock_cfg):
        """Invalid provider in task should produce a clear error, not a crash."""
        mock_cfg.return_value = {"max_iterations": 45, "model": "", "provider": ""}
        mock_creds.side_effect = [
            _default_creds(),
            ValueError("Cannot resolve delegation provider 'nonexistent': ..."),
        ]

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            result = json.loads(
                delegate_task(
                    tasks=[{
                        "goal": "Bad provider",
                        "model": "test-model",
                        "provider": "nonexistent",
                    }],
                    parent_agent=parent,
                )
            )

            self.assertIn("error", result)
            self.assertIn("nonexistent", result["error"])

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_model_without_provider_inherits_delegation_provider(
        self, mock_creds, mock_cfg
    ):
        """Per-task model without provider should use delegation.provider's
        credentials with the per-task model name."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "glm-5",
            "provider": "openrouter",
        }
        mock_creds.return_value = _openrouter_creds("glm-5")

        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            result = json.loads(
                delegate_task(
                    tasks=[{
                        "goal": "Use different model same provider",
                        "model": "google/gemini-3-flash-preview",
                    }],
                    parent_agent=parent,
                )
            )

            self.assertIn("results", result)
            _, kwargs = MockAgent.call_args
            # Model should be per-task override
            self.assertEqual(kwargs["model"], "google/gemini-3-flash-preview")
            # Provider should be from delegation config
            self.assertEqual(kwargs["provider"], "openrouter")


if __name__ == "__main__":
    unittest.main()
