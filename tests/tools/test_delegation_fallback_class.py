"""Executable class map: which fallback chain does a non-head agent get?

One test per open member of the delegation/fallback resolution class
(#80450 cross-PR map). Members with fixes in flight are xfail(strict=False)
so this suite is green today, documents the class, and flips to XPASS as
the member PRs land:

- pinned ``delegation.provider`` silently overridden by inherited parent
  chain — issue #80450, fix in flight PR #80465 (@teknium1)
- configured ``delegation.fallback_providers`` ignored — issue #65038
  (@mlahatte), fixes in flight PR #80438 (@wz-heng) / #80421 (@andrexibiza)
- pin + declared chain precedence — undecided, see the #80450 discussion

Chain-entry members (#80209 explicit api_mode, #79840 benched credentials)
and auxiliary-agent members (#79750 curator/background-review) live in
their own modules and carry suites in their PRs.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_child_agent
from tests.tools.test_delegate import _make_mock_parent


def _make_parent(chain=None):
    parent = _make_mock_parent(depth=0)
    parent._fallback_chain = chain
    return parent


PARENT_CHAIN = [
    {"provider": "openrouter", "model": "gpt-4o-mini", "api_key": "sk-or-parent"}
]
DELEGATION_CHAIN = [
    {"provider": "deepseek", "model": "deepseek-chat", "api_key": "sk-ds-child"}
]


def _spawn_child(parent, cfg=None, **overrides):
    # _load_config() returns the *delegation section* of the active config.
    with patch("tools.delegate_tool._load_config", return_value=cfg or {}):
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="class map",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
                **overrides,
            )
    return MockAgent.call_args[1]


class TestDelegationFallbackClass(unittest.TestCase):
    # --- member: pinned provider (#80450 / PR #80465) --------------------

    @pytest.mark.xfail(
        reason="#80450: pinned delegation.provider still inherits the parent "
        "fallback chain, so a mid-run failure silently reroutes the pin — "
        "fix in flight in PR #80465",
        strict=False,
    )
    def test_pinned_provider_child_does_not_inherit_parent_chain(self):
        parent = _make_parent(chain=list(PARENT_CHAIN))
        kwargs = _spawn_child(parent, override_provider="xai")
        self.assertIsNone(kwargs["fallback_model"])

    # --- member: configured delegation chain (#65038 / PRs #80438, #80421)

    @pytest.mark.xfail(
        reason="#65038: delegation.fallback_providers is ignored; children "
        "inherit the parent chain — fixes in flight in PR #80438 / #80421",
        strict=False,
    )
    def test_configured_delegation_chain_reaches_child(self):
        parent = _make_parent(chain=list(PARENT_CHAIN))
        kwargs = _spawn_child(
            parent,
            cfg={"fallback_providers": list(DELEGATION_CHAIN)},
        )
        chain = kwargs["fallback_model"] or []
        providers = {entry.get("provider") for entry in chain}
        self.assertIn("deepseek", providers)
        self.assertNotIn("openrouter", providers)

    @pytest.mark.xfail(
        reason="#65038: an explicit empty delegation.fallback_providers should "
        "disable inheritance (contract per PR #80438) — fix in flight",
        strict=False,
    )
    def test_explicit_empty_delegation_chain_disables_inheritance(self):
        parent = _make_parent(chain=list(PARENT_CHAIN))
        kwargs = _spawn_child(
            parent, cfg={"fallback_providers": []}
        )
        self.assertIsNone(kwargs["fallback_model"])

    # --- member: pin + declared chain composition (undecided) ------------

    @unittest.skip(
        "#80450 composition question: pin (override_provider) plus a declared "
        "delegation.fallback_providers — PR #80465 disables the chain on pin, "
        "PR #80438/#80421 install the configured chain; precedence is "
        "maintainer's call. Unskip once the contract is decided."
    )
    def test_pin_plus_declared_chain_precedence(self):
        parent = _make_parent(chain=list(PARENT_CHAIN))
        kwargs = _spawn_child(
            parent,
            cfg={"fallback_providers": list(DELEGATION_CHAIN)},
            override_provider="deepseek",
        )
        chain = kwargs["fallback_model"] or []
        providers = {entry.get("provider") for entry in chain}
        self.assertNotIn("openrouter", providers)

    # --- guarded baseline (green on main): default inheritance -----------

    def test_unpinned_unconfigured_child_inherits_parent_chain(self):
        parent = _make_parent(chain=list(PARENT_CHAIN))
        kwargs = _spawn_child(parent)
        self.assertEqual(kwargs["fallback_model"], PARENT_CHAIN)


if __name__ == "__main__":
    unittest.main()
