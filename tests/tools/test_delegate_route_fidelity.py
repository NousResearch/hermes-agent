"""Regression tests for the per-task model/provider override route-fidelity bug.

Verified failure mode: per-task `model` / `provider` / `base_url` / `api_key`
overrides on `delegate_task(tasks=[{...}, ...])` were silently dropped — every
child built with `creds["model"]` from the top-level `delegation.*` config
regardless of what the per-task dict specified.

Impact: parallel-critique scatter "with cross-family signal" was actually
single-family; "MoA via delegate_task" routed all subagents to one model;
and any caller relying on per-task model selection got the orchestrator's
primary model regardless.

These tests pin the FIX (in `delegate_tool.py`'s per-task creds layering loop)
so it can't silently regress. Verified RED for the 6th time on 2026-05-13;
this fix lands on `feat/d1-kanban-worker-runtime` sibling branch
`fix/delegate-task-route-fidelity`.
"""
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _resolve_delegation_credentials, delegate_task


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "sk-parent"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-opus-4.7"
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


class TestPerTaskRouteFidelity(unittest.TestCase):
    """The per-task model/provider override must reach the child build."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_per_task_model_only_override_reaches_child(self, mock_creds, mock_cfg):
        """Single per-task `model` field overrides creds["model"] only.

        Other fields (provider, base_url, api_key, api_mode) are inherited
        from parent creds. This is the canonical scatter-gather case:
        same OpenRouter endpoint, different model slug per reviewer.
        """
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "anthropic/claude-opus-4.7",
            "provider": "openrouter",
        }
        mock_creds.return_value = {
            "model": "anthropic/claude-opus-4.7",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-parent",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)

        with patch("tools.delegate_tool._build_child_agent") as mock_build, \
             patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_build.return_value = MagicMock()
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
            }

            tasks = [
                {"goal": "Review with Gemini",  "model": "google/gemini-3.1-pro-preview"},
                {"goal": "Review with DeepSeek", "model": "deepseek/deepseek-v4-pro"},
                {"goal": "Review with Kimi",    "model": "moonshotai/kimi-k2.6"},
            ]
            delegate_task(tasks=tasks, parent_agent=parent)

            self.assertEqual(mock_build.call_count, 3)
            models_used = [c.kwargs["model"] for c in mock_build.call_args_list]
            self.assertEqual(
                models_used,
                ["google/gemini-3.1-pro-preview",
                 "deepseek/deepseek-v4-pro",
                 "moonshotai/kimi-k2.6"],
                "Per-task `model` override was silently dropped — "
                "route-fidelity bug regression",
            )
            # Provider / base / key / mode all inherited from parent creds
            for c in mock_build.call_args_list:
                self.assertEqual(c.kwargs["override_provider"], "openrouter")
                self.assertEqual(
                    c.kwargs["override_base_url"], "https://openrouter.ai/api/v1"
                )
                self.assertEqual(c.kwargs["override_api_key"], "sk-or-parent")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_per_task_no_model_override_uses_parent_creds(
        self, mock_creds, mock_cfg
    ):
        """Tasks without any model/provider keys keep parent creds verbatim."""
        mock_cfg.return_value = {"max_iterations": 45, "model": "x"}
        mock_creds.return_value = {
            "model": "anthropic/claude-opus-4.7",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-parent",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)

        with patch("tools.delegate_tool._build_child_agent") as mock_build, \
             patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_build.return_value = MagicMock()
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
            }
            tasks = [{"goal": "Task A"}, {"goal": "Task B"}]
            delegate_task(tasks=tasks, parent_agent=parent)
            self.assertEqual(mock_build.call_count, 2)
            for c in mock_build.call_args_list:
                self.assertEqual(c.kwargs["model"], "anthropic/claude-opus-4.7")
                self.assertEqual(c.kwargs["override_provider"], "openrouter")

    @patch("tools.delegate_tool._load_config")
    def test_per_task_provider_override_re_resolves_full_creds(
        self, mock_cfg
    ):
        """Per-task `provider` field triggers full credential re-resolution.

        We mock _resolve_delegation_credentials such that:
          - first call (top-level) returns parent creds (openrouter)
          - subsequent calls (per-task) return alternative creds based on the
            cfg dict that was passed in.
        Verify the re-resolved creds reach the child build, not the parent.
        """
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model": "anthropic/claude-opus-4.7",
            "provider": "openrouter",
        }
        parent = _make_mock_parent(depth=0)

        # Track resolution calls and return per-task-specific creds
        resolution_calls = []

        def fake_resolve(cfg, parent_agent):
            resolution_calls.append(dict(cfg))
            return {
                "model": cfg.get("model") or "anthropic/claude-opus-4.7",
                "provider": cfg.get("provider") or "openrouter",
                "base_url": cfg.get("base_url") or "https://openrouter.ai/api/v1",
                "api_key": cfg.get("api_key") or "sk-or-parent",
                "api_mode": "chat_completions",
            }

        with patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            side_effect=fake_resolve,
        ), patch("tools.delegate_tool._build_child_agent") as mock_build, \
           patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_build.return_value = MagicMock()
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
            }

            tasks = [
                {
                    "goal": "Anthropic-direct task",
                    "model": "claude-sonnet-4.6",
                    "provider": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "api_key": "sk-ant-direct",
                },
                {"goal": "OR task", "model": "google/gemini-3.1-pro-preview"},
            ]
            delegate_task(tasks=tasks, parent_agent=parent)

            self.assertEqual(mock_build.call_count, 2)
            # Task 0: anthropic-direct — provider override triggered re-resolution
            self.assertEqual(
                mock_build.call_args_list[0].kwargs["override_provider"],
                "anthropic",
            )
            self.assertEqual(
                mock_build.call_args_list[0].kwargs["override_base_url"],
                "https://api.anthropic.com",
            )
            self.assertEqual(
                mock_build.call_args_list[0].kwargs["override_api_key"],
                "sk-ant-direct",
            )
            self.assertEqual(
                mock_build.call_args_list[0].kwargs["model"], "claude-sonnet-4.6"
            )
            # Task 1: OR task — model-only override, parent provider preserved
            self.assertEqual(
                mock_build.call_args_list[1].kwargs["model"],
                "google/gemini-3.1-pro-preview",
            )
            self.assertEqual(
                mock_build.call_args_list[1].kwargs["override_provider"],
                "openrouter",
            )

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_per_task_credential_resolution_failure_falls_back_to_parent_creds(
        self, mock_creds, mock_cfg
    ):
        """If per-task credential re-resolution raises, fall back to parent creds
        with a WARNING log — surface-isolation invariant.
        """
        mock_cfg.return_value = {"max_iterations": 45}
        # First call returns parent creds; subsequent (per-task) calls raise
        parent_creds = {
            "model": "anthropic/claude-opus-4.7",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-parent",
            "api_mode": "chat_completions",
        }
        call_count = [0]

        def side_effect(cfg, parent_agent):
            call_count[0] += 1
            if call_count[0] == 1:
                return parent_creds
            raise ValueError("synthetic per-task resolution failure")

        mock_creds.side_effect = side_effect
        parent = _make_mock_parent(depth=0)

        with patch("tools.delegate_tool._build_child_agent") as mock_build, \
             patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_build.return_value = MagicMock()
            mock_run.return_value = {
                "task_index": 0, "status": "completed",
                "summary": "Done", "api_calls": 1, "duration_seconds": 1.0,
            }

            tasks = [
                {
                    "goal": "Task that triggers per-task resolution",
                    "provider": "broken-provider",  # triggers re-resolve which raises
                },
            ]
            delegate_task(tasks=tasks, parent_agent=parent)

            self.assertEqual(mock_build.call_count, 1)
            # Falls back to parent creds — no crash
            self.assertEqual(
                mock_build.call_args.kwargs["override_provider"], "openrouter"
            )
            self.assertEqual(
                mock_build.call_args.kwargs["model"], "anthropic/claude-opus-4.7"
            )


if __name__ == "__main__":
    unittest.main()
