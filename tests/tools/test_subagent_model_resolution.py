#!/usr/bin/env python3
"""
Tests for subagent model resolution when delegation config is set
(issue #58298).

Regression for: ``Subagent delegation ignores config, always uses
credential-pool model (glm-4-flash)``.

The bug: when a user configures ``delegation.model`` (or the
``subagent.model`` alias) in ``config.yaml``, the spawned subagent still
inherited the parent's credential pool. ``acquire_lease()`` on that pool
could pick any provider's credential (e.g. zai's GLM key) and
``_swap_credential`` would then rewrite the child's ``base_url`` /
``api_key`` with the leased entry — silently routing the configured model
name to whatever endpoint the lease came from. The user-visible symptom
was every subagent landing on glm-4-flash regardless of the configured
model.

Fix:
1. ``_resolve_child_credential_pool`` now returns ``None`` (no pool
   sharing) whenever any of these delegation overrides are set:
     - ``delegation.provider``
     - ``delegation.base_url``
     - ``delegation.api_key``
     - ``delegation.model`` (or ``subagent.model``)
2. ``_resolve_delegation_credentials`` reads ``subagent.model`` as an
   alias for ``delegation.model`` so users can write either key.
3. ``_build_child_agent`` passes the override-set flags through so the
   pool resolver can short-circuit.

These tests assert the four required behaviours from the issue:
  - Default config: uses configured model (e.g. ``anthropic/claude-sonnet-4``)
  - ``subagent.model`` alias is honoured
  - No config: falls back to credential pool (no regression)
  - Parent model override: child uses parent's model when explicit
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _resolve_child_credential_pool,
    _resolve_delegation_credentials,
)


def _make_mock_parent(
    *,
    provider: str = "deepseek",
    base_url: str = "https://api.deepseek.com/v1",
    api_key: str = "deepseek-key-abc",
    model: str = "deepseek-v4-pro",
    pool=None,
):
    """Mock parent agent with the fields _resolve_child_credential_pool reads."""
    parent = MagicMock()
    parent.provider = provider
    parent.base_url = base_url
    parent.api_key = api_key
    parent.model = model
    parent._credential_pool = pool
    return parent


class TestDefaultConfigUsesConfiguredModel(unittest.TestCase):
    """Issue #58298: delegation.model must reach the child agent's model field."""

    def test_configured_model_resolves_through_resolve_delegation_credentials(self):
        """When delegation.model is set, _resolve_delegation_credentials returns it."""
        parent = _make_mock_parent()
        cfg = {"model": "anthropic/claude-sonnet-4", "provider": ""}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "anthropic/claude-sonnet-4")

    def test_configured_model_with_empty_provider_does_not_share_parent_pool(self):
        """When ONLY delegation.model is set (no provider), the child must NOT
        inherit the parent's pool — that's the exact path that caused the
        glm-4-flash leak in issue #58298."""
        parent = _make_mock_parent(
            provider="deepseek",
            pool=MagicMock(name="parent_deepseek_pool"),
        )
        result = _resolve_child_credential_pool(
            effective_provider="deepseek",
            parent_agent=parent,
            effective_base_url="https://api.deepseek.com/v1",
            delegation_model_explicit=True,
        )
        self.assertIsNone(result)

    def test_configured_model_does_not_share_parent_pool_even_when_providers_match(self):
        """Same provider, model pinned, no rotation — pool must be skipped."""
        parent = _make_mock_parent(
            provider="deepseek",
            pool=MagicMock(name="parent_pool_should_not_be_shared"),
        )
        result = _resolve_child_credential_pool(
            effective_provider="deepseek",
            parent_agent=parent,
            effective_base_url="https://api.deepseek.com/v1",
            delegation_model_explicit=True,
        )
        self.assertIsNot(result, parent._credential_pool)
        self.assertIsNone(result)


class TestSubagentModelAliasHonoured(unittest.TestCase):
    """``subagent.model`` is an alias for ``delegation.model``."""

    def test_subagent_model_dot_key_resolves(self):
        """Users who write ``subagent.model`` get the same resolution."""
        parent = _make_mock_parent()
        cfg = {"subagent.model": "anthropic/claude-sonnet-4"}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "anthropic/claude-sonnet-4")

    def test_subagent_model_underscore_key_resolves(self):
        """Defensive: accept ``subagent_model`` as well (some YAML flatteners)."""
        parent = _make_mock_parent()
        cfg = {"subagent_model": "anthropic/claude-sonnet-4"}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "anthropic/claude-sonnet-4")

    def test_canonical_model_wins_over_subagent_model(self):
        """When BOTH keys are set, ``delegation.model`` wins (canonical)."""
        parent = _make_mock_parent()
        cfg = {
            "model": "anthropic/claude-sonnet-4",
            "subagent.model": "different-model",
        }
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "anthropic/claude-sonnet-4")

    def test_subagent_model_with_empty_canonical_falls_back(self):
        """Empty canonical + set alias => alias is used."""
        parent = _make_mock_parent()
        cfg = {"model": "", "subagent.model": "anthropic/claude-sonnet-4"}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "anthropic/claude-sonnet-4")


class TestNoConfigFallsBackToCredentialPool(unittest.TestCase):
    """No delegation config => child inherits parent's pool (no regression)."""

    def test_no_overrides_shares_parent_pool_same_provider(self):
        """Pre-fix behaviour preserved: same provider, no overrides, share pool."""
        parent_pool = MagicMock(name="parent_pool")
        parent = _make_mock_parent(provider="openrouter", pool=parent_pool)
        result = _resolve_child_credential_pool(
            effective_provider="openrouter",
            parent_agent=parent,
            effective_base_url="https://openrouter.ai/api/v1",
        )
        self.assertIs(result, parent_pool)

    def test_no_overrides_inherits_parent_pool_when_provider_none(self):
        """When effective_provider is None and there are no overrides,
        inherit the parent's pool so existing behaviour is preserved."""
        parent_pool = MagicMock(name="parent_pool")
        parent = _make_mock_parent(pool=parent_pool)
        result = _resolve_child_credential_pool(
            effective_provider=None,
            parent_agent=parent,
        )
        self.assertIs(result, parent_pool)

    def test_no_overrides_different_provider_loads_own_pool(self):
        """Different provider + no overrides => load the provider-specific pool
        (existing behaviour preserved, no regression)."""
        parent = _make_mock_parent(provider="openrouter", pool=MagicMock())
        new_pool = MagicMock(name="openai_pool")
        new_pool.has_credentials.return_value = True
        with patch("agent.credential_pool.load_pool", return_value=new_pool):
            result = _resolve_child_credential_pool(
                effective_provider="openai",
                parent_agent=parent,
                effective_base_url="https://api.openai.com/v1",
            )
        self.assertIs(result, new_pool)


class TestParentModelOverrideUsedWhenExplicit(unittest.TestCase):
    """When the user passes a model override via delegate_task's creds flow,
    the child must end up with that model and not the parent's model."""

    def test_explicit_provider_override_does_not_share_parent_pool(self):
        """Override provider => no parent pool sharing."""
        parent_pool = MagicMock(name="parent_pool_must_not_leak")
        parent = _make_mock_parent(provider="deepseek", pool=parent_pool)
        result = _resolve_child_credential_pool(
            effective_provider="openrouter",
            parent_agent=parent,
            effective_base_url="https://openrouter.ai/api/v1",
            override_provider=True,
        )
        self.assertIsNone(result)
        self.assertIsNot(result, parent_pool)

    def test_explicit_base_url_override_does_not_share_parent_pool(self):
        parent_pool = MagicMock(name="parent_pool_must_not_leak")
        parent = _make_mock_parent(provider="custom", pool=parent_pool)
        result = _resolve_child_credential_pool(
            effective_provider="custom",
            parent_agent=parent,
            effective_base_url="https://custom-b.example.com/v1",
            override_base_url=True,
        )
        self.assertIsNone(result)

    def test_explicit_api_key_override_does_not_share_parent_pool(self):
        parent_pool = MagicMock(name="parent_pool_must_not_leak")
        parent = _make_mock_parent(pool=parent_pool)
        result = _resolve_child_credential_pool(
            effective_provider="openrouter",
            parent_agent=parent,
            effective_base_url="https://openrouter.ai/api/v1",
            override_api_key=True,
        )
        self.assertIsNone(result)


class TestBuildChildAgentPassesOverrideFlags(unittest.TestCase):
    """Integration: delegate_task → _build_child_agent wires the override
    flags through to the credential pool resolver so the configured model
    isn't silently overwritten by the parent's pool rotation."""

    def test_delegation_model_with_empty_provider_keeps_child_credentials(self):
        """The end-to-end regression for #58298: delegation.model set,
        delegation.provider empty, parent has a pool. The child's
        ``_credential_pool`` must NOT be set, so _swap_credential can't
        rewrite its base_url to a different provider's endpoint."""
        # Import here so the module-level import isn't pulled in for the
        # pure unit tests above.
        from tools.delegate_tool import _build_child_agent, _resolve_delegation_credentials

        parent = _make_mock_parent(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            api_key="deepseek-key",
            model="deepseek-v4-pro",
        )
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent._session_db = None
        parent._print_fn = None
        parent.tool_progress_callback = None
        parent.thinking_callback = None
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent.openrouter_min_coding_score = None

        # delegation.model set, delegation.provider empty (issue's control case)
        cfg = {"model": "deepseek-v4-pro", "provider": ""}
        creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["model"], "deepseek-v4-pro")
        self.assertIsNone(creds["provider"])

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child

            _build_child_agent(
                task_index=0,
                goal="Test glm-4-flash regression",
                context=None,
                toolsets=None,
                model=creds["model"],
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )

        # The child must NOT have a credential pool assigned — otherwise
        # _swap_credential would rewrite base_url/key to whichever
        # credential the parent pool happened to lease (the bug).
        # MagicMock returns a Mock for any attribute access, so we check
        # via the call to _resolve_child_credential_pool directly.
        from tools.delegate_tool import _resolve_child_credential_pool as _rccp

        with patch("tools.delegate_tool._resolve_child_credential_pool",
                   wraps=_rccp) as spy:
            mock_child2 = MagicMock()
            mock_child2._credential_pool = None  # ensure clean assignment surface
            with patch("run_agent.AIAgent", return_value=mock_child2):
                _build_child_agent(
                    task_index=0,
                    goal="Test glm-4-flash regression v2",
                    context=None,
                    toolsets=None,
                    model=creds["model"],
                    max_iterations=10,
                    parent_agent=parent,
                    task_count=1,
                )
            self.assertTrue(spy.called, "_resolve_child_credential_pool must be called")
            _, kwargs = spy.call_args
            # delegation_model_explicit must be True so the resolver returns None
            self.assertTrue(
                kwargs.get("delegation_model_explicit"),
                "delegation_model_explicit flag must be True when delegation.model is set",
            )
            # override_* flags all False (no provider/base_url/api_key override)
            self.assertFalse(kwargs.get("override_provider"))
            self.assertFalse(kwargs.get("override_base_url"))
            self.assertFalse(kwargs.get("override_api_key"))


if __name__ == "__main__":
    unittest.main()