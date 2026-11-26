"""Delegated children honour Codex usage admission when binding pooled credentials."""

import unittest
from unittest.mock import MagicMock, patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import (
    _build_child_agent,
    _resolve_child_credential_pool,
    _run_single_child,
)


class TestChildCredentialPoolUsageAdmission(unittest.TestCase):
    @patch(
        "agent.usage_admission.policy.usage_admission_credential_denied",
        return_value=False,
    )
    def test_unrelated_explicit_codex_key_does_not_attach_provider_pool(
        self, mock_denied
    ):
        parent = _make_mock_parent()
        parent._credential_pool = MagicMock(provider="openai-codex")

        result = _resolve_child_credential_pool(
            "openai-codex",
            parent,
            "https://chatgpt.com/backend-api/codex",
            explicit_api_key="unrelated-explicit-token",
        )

        self.assertIsNone(result)
        mock_denied.assert_called_once_with(
            "openai-codex", "unrelated-explicit-token"
        )

    @patch(
        "agent.usage_admission.policy.usage_admission_credential_denied",
        return_value=True,
    )
    def test_matching_policy_denied_explicit_codex_key_fails_closed(
        self, mock_denied
    ):
        parent = _make_mock_parent()

        with self.assertRaisesRegex(ValueError, "usage policy"):
            _resolve_child_credential_pool(
                "openai-codex",
                parent,
                "https://chatgpt.com/backend-api/codex",
                explicit_api_key="pooled-denied-token",
            )

        mock_denied.assert_called_once_with(
            "openai-codex", "pooled-denied-token"
        )

    @patch(
        "agent.usage_admission.policy.usage_admission_credential_denied",
        return_value=False,
    )
    @patch(
        "tools.delegate_tool_config._load_config",
        return_value={"api_key": "unrelated-explicit-token"},
    )
    def test_build_child_keeps_unrelated_explicit_codex_key_fixed(
        self, _mock_cfg, mock_denied
    ):
        parent = _make_mock_parent()
        parent.provider = "openai-codex"
        parent.base_url = "https://chatgpt.com/backend-api/codex"
        parent._credential_pool = MagicMock(provider="openai-codex")

        with patch("run_agent.AIAgent") as mock_agent_cls:
            child = MagicMock()
            child._credential_pool = None
            mock_agent_cls.return_value = child

            built = _build_child_agent(
                task_index=0,
                goal="Use the explicit credential",
                context=None,
                toolsets=None,
                model="gpt-5.5",
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                override_provider="openai-codex",
                override_base_url="https://chatgpt.com/backend-api/codex",
                override_api_key="unrelated-explicit-token",
                override_api_mode="codex_responses",
            )

        self.assertIs(built, child)
        self.assertIsNone(child._credential_pool)
        mock_denied.assert_called_once_with(
            "openai-codex", "unrelated-explicit-token"
        )


class TestChildCredentialLeaseUsageAdmission(unittest.TestCase):
    def test_run_single_child_binds_leased_entry_then_releases_lease_after_failure(self):
        child = MagicMock()
        child._credential_pool = MagicMock()
        child._credential_pool.acquire_lease.return_value = "cred-a"
        leased_entry = MagicMock(id="cred-a")
        child._credential_pool.current.return_value = leased_entry
        child._credential_pool.entries.return_value = [leased_entry]
        child._credential_pool_entry_id = None
        child._swap_credential.side_effect = lambda entry: setattr(
            child, "_credential_pool_entry_id", entry.id
        )
        child.run_conversation.side_effect = RuntimeError("boom")

        result = _run_single_child(
            task_index=1,
            goal="Trigger failure",
            child=child,
            parent_agent=_make_mock_parent(),
        )

        self.assertEqual(result["status"], "error")
        child._swap_credential.assert_called_once_with(leased_entry)
        child.run_conversation.assert_called_once()
        child._credential_pool.release_lease.assert_called_once_with("cred-a")

    @patch("agent.usage_admission.policy.get_pool_usage_limits", return_value={"cred-b": 80.0})
    def test_run_single_child_stops_when_leased_credential_binding_fails(self, _limits):
        from tools.delegate_tool import _run_single_child

        child = MagicMock()
        child._credential_pool = MagicMock(provider="openai-codex")
        child._credential_pool.acquire_lease.return_value = "cred-b"
        child._credential_pool.entries.return_value = [MagicMock(id="cred-b")]
        child._swap_credential.side_effect = RuntimeError("binding failed")

        result = _run_single_child(
            task_index=4,
            goal="Never use the inherited denied credential",
            child=child,
            parent_agent=_make_mock_parent(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_reason"], "credential_unavailable")
        child.run_conversation.assert_not_called()
        child._credential_pool.release_lease.assert_called_once_with("cred-b")

    def test_run_single_child_binds_exact_leased_id_not_mutable_pool_cursor(self):
        from tools.delegate_tool import _run_single_child

        leased_entry = MagicMock(id="cred-a")
        cursor_entry = MagicMock(id="cred-b")
        child = MagicMock()
        child._credential_pool = MagicMock(provider="openai-codex")
        child._credential_pool.acquire_lease.return_value = "cred-a"
        child._credential_pool.entries.return_value = [leased_entry, cursor_entry]
        child._credential_pool.current.return_value = cursor_entry

        def bind(entry):
            child._credential_pool_entry_id = entry.id

        child._swap_credential.side_effect = bind
        child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

        result = _run_single_child(
            task_index=5,
            goal="Bind the exact leased credential",
            child=child,
            parent_agent=_make_mock_parent(),
        )

        self.assertEqual(result["status"], "completed")
        child._swap_credential.assert_called_once_with(leased_entry)
        child._credential_pool.release_lease.assert_called_once_with("cred-a")

    @patch("agent.usage_admission.policy.get_pool_usage_limits", return_value={"cred-b": 80.0})
    def test_run_single_child_stops_when_lease_entry_cannot_be_resolved(self, _limits):
        from tools.delegate_tool import _run_single_child

        child = MagicMock()
        child._credential_pool = MagicMock(provider="openai-codex")
        child._credential_pool.acquire_lease.return_value = "cred-missing"
        child._credential_pool.entries.return_value = [MagicMock(id="cred-other")]

        result = _run_single_child(
            task_index=6,
            goal="Never run without the leased credential",
            child=child,
            parent_agent=_make_mock_parent(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_reason"], "credential_unavailable")
        child.run_conversation.assert_not_called()
        child._credential_pool.release_lease.assert_called_once_with("cred-missing")

    @patch("agent.usage_admission.policy.usage_admission_policy_denied", return_value=True)
    def test_run_single_child_does_not_use_inherited_policy_denied_credential(
        self, _policy_denied
    ):
        from tools.delegate_tool import _run_single_child

        child = MagicMock()
        child._credential_pool = MagicMock(provider="openai-codex")
        child._credential_pool.acquire_lease.return_value = None
        child._try_activate_fallback.return_value = False

        result = _run_single_child(
            task_index=2,
            goal="Respect the Codex reserve",
            child=child,
            parent_agent=_make_mock_parent(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_reason"], "credential_unavailable")
        child.run_conversation.assert_not_called()

    @patch(
        "agent.usage_admission.policy.usage_admission_policy_denied",
        side_effect=lambda provider: provider == "openai-codex",
    )
    def test_run_single_child_activates_fallback_after_policy_denial(
        self, _policy_denied
    ):
        from tools.delegate_tool import _run_single_child

        denied_pool = MagicMock(provider="openai-codex")
        denied_pool.acquire_lease.return_value = None
        fallback_pool = MagicMock(provider="anthropic")
        fallback_pool.acquire_lease.return_value = None

        child = MagicMock()
        child._credential_pool = denied_pool

        def activate_fallback():
            child._credential_pool = fallback_pool
            return True

        child._try_activate_fallback.side_effect = activate_fallback
        child.run_conversation.return_value = {
            "final_response": "done on fallback",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

        result = _run_single_child(
            task_index=3,
            goal="Use the configured fallback",
            child=child,
            parent_agent=_make_mock_parent(),
        )

        self.assertEqual(result["status"], "completed")
        child._try_activate_fallback.assert_called_once_with()
        child.run_conversation.assert_called_once()
