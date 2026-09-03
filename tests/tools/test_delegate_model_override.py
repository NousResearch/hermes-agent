#!/usr/bin/env python3
"""
Per-dispatch model/provider override for delegate_task subagents.

Issue #97653: each entry in a delegate_task `tasks` batch may carry an
optional per-task `model` and/or `provider`. The override is resolved through
the same runtime-provider path as delegation config (resolve_runtime_provider),
applies ONLY to that dispatch's child(ren), and is NEVER persisted to config.
Precedence: per-task override > delegation config > parent inheritance.

Run with: python -m pytest tests/tools/test_delegate_model_override.py -v
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _resolve_child_credential_pool,
    delegate_task,
)


def _make_mock_parent(depth=0):
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
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _fake_resolve(requested=None, target_model=None):
    """Synthetic runtime-provider bundle keyed on the requested provider name."""
    provider = requested or "openrouter"
    return {
        "provider": provider,
        "model": target_model or "resolved-model",
        "base_url": f"https://{provider}.example/v1",
        "api_key": f"{provider}-key",
        "api_mode": "chat_completions",
    }


def _completed(idx):
    return {
        "task_index": idx,
        "status": "completed",
        "summary": "ok",
        "api_calls": 1,
        "duration_seconds": 1.0,
        "_child_role": None,
    }


class TestPerDispatchOverrideSchema(unittest.TestCase):
    def test_tasks_items_accept_model_and_provider_overrides(self):
        """The tasks[] item schema advertises the optional override params."""
        task_props = (
            DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        )
        self.assertIn("model", task_props)
        self.assertEqual(task_props["model"]["type"], "string")
        self.assertIn("provider", task_props)
        self.assertEqual(task_props["provider"]["type"], "string")


class TestPerDispatchOverrideResolution(unittest.TestCase):
    """Per-task override resolves through the runtime-provider path and wins
    over the delegation config (and over parent inheritance)."""

    @patch("tools.delegate_tool._load_config")
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_override_wins_over_config(self, mock_resolve, mock_cfg):
        mock_resolve.side_effect = _fake_resolve
        # Config pins openrouter/config-model as the routing baseline.
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "openrouter",
        }
        parent = _make_mock_parent()

        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._run_single_child") as mock_run,
        ):
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            mock_run.return_value = _completed(0)
            delegate_task(
                tasks=[{"goal": "Resolve the override flags", "model": "override-model",
                        "provider": "override-prov"}],
                parent_agent=parent,
            )

        _, kwargs = MockAgent.call_args
        # The override beat config (and parent provider=openrouter).
        self.assertEqual(kwargs["model"], "override-model")
        self.assertEqual(kwargs["provider"], "override-prov")
        self.assertEqual(kwargs["base_url"], "https://override-prov.example/v1")
        self.assertEqual(kwargs["api_key"], "override-prov-key")

    @patch("tools.delegate_tool._load_config")
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_empty_or_missing_override_inherits_config(self, mock_resolve, mock_cfg):
        mock_resolve.side_effect = _fake_resolve
        # Config pins a DIFFERENT provider than the parent so inherit-config
        # (not inherit-parent) is observable.
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "nous",
        }
        parent = _make_mock_parent()  # parent provider=openrouter, model=anthropic/...

        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._run_single_child") as mock_run,
        ):
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            mock_run.return_value = _completed(0)
            delegate_task(
                tasks=[{"goal": "Resolve inherits when override is empty", "model": "", "provider": ""}],
                parent_agent=parent,
            )

        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["model"], "config-model")
        self.assertEqual(kwargs["provider"], "nous")

    @patch("tools.delegate_tool._load_config")
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_model_only_override_preserves_config_provider(self, mock_resolve, mock_cfg):
        mock_resolve.side_effect = _fake_resolve
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "openrouter",
        }
        parent = _make_mock_parent()

        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._run_single_child") as mock_run,
        ):
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            mock_run.return_value = _completed(0)
            delegate_task(
                tasks=[{"goal": "Override only the model", "model": "override-model"}],
                parent_agent=parent,
            )

        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["model"], "override-model")
        self.assertEqual(kwargs["provider"], "openrouter")

    @patch("tools.delegate_tool._load_config")
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_invalid_override_fails_that_dispatch_without_config_write(self, mock_resolve, mock_cfg):
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "openrouter",
        }

        def _resolve(**kwargs):
            if kwargs.get("requested") == "bogus-prov":
                raise RuntimeError("BOGUS_PROVIDER_API_KEY not set")
            return _fake_resolve(**kwargs)

        mock_resolve.side_effect = _resolve
        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            result = json.loads(
                delegate_task(
                    tasks=[{"goal": "Resolve with a bad provider", "provider": "bogus-prov"}],
                    parent_agent=parent,
                )
            )
            # The invalid override fails the dispatch loudly — no child is
            # spawned, so nothing runs on the wrong provider.
            MockAgent.assert_not_called()

        self.assertIn("error", result)
        self.assertIn("Task 0", result["error"])
        self.assertIn("bogus-prov", result["error"])
        # No config write happens: the override code path only READS config
        # (_load_config / load_config_readonly); there is no writer call site.
        self.assertEqual(mock_cfg.return_value["provider"], "openrouter")

    @patch("tools.delegate_tool._load_config")
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_mixed_overrides_resolve_each_child_independently(self, mock_resolve, mock_cfg):
        mock_resolve.side_effect = _fake_resolve
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "openrouter",
        }
        parent = _make_mock_parent()

        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._run_single_child") as mock_run,
        ):
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            mock_run.side_effect = [_completed(0), _completed(1)]
            result = json.loads(
                delegate_task(
                    tasks=[
                        {"goal": "First task with a pinned provider",
                         "model": "m0", "provider": "p0"},
                        {"goal": "Second task inherits the delegation config"},
                    ],
                    parent_agent=parent,
                )
            )

        self.assertNotIn("error", result)
        self.assertEqual(len(MockAgent.call_args_list), 2)
        # Children are constructed in task order; each resolves independently.
        _, kw0 = MockAgent.call_args_list[0]
        self.assertEqual(kw0["model"], "m0")
        self.assertEqual(kw0["provider"], "p0")
        _, kw1 = MockAgent.call_args_list[1]
        self.assertEqual(kw1["model"], "config-model")
        self.assertEqual(kw1["provider"], "openrouter")


class TestPerDispatchOverrideFailureIsolation(unittest.TestCase):
    """Errors during per-task override resolution must fail the dispatch
    cleanly and never orphan a prior constructed child or write config."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_non_valueerror_resolution_error_is_clean_tool_error(
        self, mock_resolve, mock_cfg
    ):
        """A non-ValueError raised while resolving a per-task override is
        attributed to the right task index (not a batch-crashing exception)."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "openrouter",
        }

        def _resolve(cfg, parent):
            if cfg.get("provider") == "bogus-prov":
                raise RuntimeError("RESOLUTION_EXPLODED")
            return {
                "model": cfg.get("model") or "config-model",
                "provider": cfg.get("provider") or "openrouter",
                "base_url": None,
                "api_key": None,
            }

        mock_resolve.side_effect = _resolve
        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            result = json.loads(
                delegate_task(
                    tasks=[{"goal": "g0", "provider": "bogus-prov"}],
                    parent_agent=parent,
                )
            )
            MockAgent.assert_not_called()

        self.assertIn("error", result)
        self.assertIn("Task 0", result["error"])
        self.assertIn("model/provider override failed", result["error"])

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_mixed_batch_invalid_override_never_constructs_prior_child(
        self, mock_resolve, mock_cfg
    ):
        """Task 0 with a valid override + task 1 with an invalid override must
        fail the whole dispatch BEFORE any child is constructed, so task 0's
        never-run child is not orphaned (SessionDB handle + active_children)."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "openrouter",
        }

        def _resolve(cfg, parent):
            if cfg.get("provider") == "bogus-prov":
                raise ValueError("bad override")
            return {
                "model": cfg.get("model") or "config-model",
                "provider": cfg.get("provider") or "openrouter",
                "base_url": None,
                "api_key": None,
            }

        mock_resolve.side_effect = _resolve
        parent = _make_mock_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            result = json.loads(
                delegate_task(
                    tasks=[
                        {"goal": "Resolve the valid override first", "provider": "good-prov"},
                        {"goal": "Resolve the invalid override second", "provider": "bogus-prov"},
                    ],
                    parent_agent=parent,
                )
            )
            # Two-pass: task 1's resolution fails in pass 1 before task 0's
            # child is even built — nothing is orphaned.
            MockAgent.assert_not_called()

        self.assertIn("error", result)
        self.assertIn("Task 1", result["error"])

    @patch("tools.delegate_tool._load_config")
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_override_to_provider_without_pool_is_graceful(
        self, mock_resolve, mock_cfg
    ):
        """Per-task override to a provider that is NOT the parent's pool
        provider degrades gracefully: no lease is taken from the wrong pool,
        and the child simply runs on the override provider. Pins that an
        override to a different provider cannot crash on pool leasing."""
        mock_resolve.side_effect = _fake_resolve
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "openrouter",
        }
        parent = _make_mock_parent()  # provider=openrouter, no _credential_pool

        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._run_single_child") as mock_run,
            patch("agent.credential_pool.load_pool", return_value=None) as mock_load,
        ):
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            mock_run.return_value = _completed(0)
            delegate_task(
                tasks=[{"goal": "g0", "provider": "override-prov"}],
                parent_agent=parent,
            )

        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["provider"], "override-prov")
        # The override provider ('override-prov') differs from the parent's
        # pool provider (openrouter) and has no pool of its own, so pool
        # resolution must return None (no lease, no child._credential_pool).
        self.assertIsNone(
            _resolve_child_credential_pool(
                "override-prov", parent, "https://override-prov.example/v1"
            )
        )

    @patch("tools.delegate_tool._load_config")
    @patch("hermes_cli.config.save_config")
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_override_dispatch_never_writes_config(self, mock_resolve, mock_save, mock_cfg):
        """Invariant: the per-task override path is read-only. If any code
        tried to persist config it would blow up (save_config patched to
        raise); we assert it is never even called."""
        mock_save.side_effect = AssertionError("override MUST NOT write config")
        mock_resolve.side_effect = _fake_resolve
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "config-model",
            "provider": "openrouter",
        }
        parent = _make_mock_parent()

        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._run_single_child") as mock_run,
        ):
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            mock_run.return_value = _completed(0)
            delegate_task(
                tasks=[{"goal": "g0", "model": "override-model", "provider": "override-prov"}],
                parent_agent=parent,
            )

        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["model"], "override-model")
        self.assertEqual(kwargs["provider"], "override-prov")
        mock_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
