"""Full decision matrix for _resolve_child_fallback_chain (#80450 class).

Composes the pin semantics of PR #80465 (@teknium1), the
delegation.fallback_providers semantics of PRs #80438 (@wz-heng) /
#80421 (@andrexibiza) for issue #65038 (@mlahatte), and settles the
pin+declared-chain composition cell raised in the #80450 cross-PR map.
"""

import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _build_child_agent, _resolve_child_fallback_chain
from tests.tools.test_delegate import _make_mock_parent

PARENT_CHAIN = [
    {"provider": "openrouter", "model": "gpt-4o-mini", "api_key": "sk-or-parent"}
]
DECLARED_CHAIN = [
    {"provider": "deepseek", "model": "deepseek-chat", "api_key": "sk-ds-child"}
]


def _parent(chain=None):
    parent = _make_mock_parent(depth=0)
    parent._fallback_chain = chain
    return parent


class TestResolveChildFallbackChainMatrix(unittest.TestCase):
    """All six cells of the pin x declared-chain matrix."""

    # pin=yes ------------------------------------------------------------

    def test_pin_declared_nonempty_uses_declared_chain(self):
        chain = _resolve_child_fallback_chain(
            _parent(list(PARENT_CHAIN)),
            {"fallback_providers": list(DECLARED_CHAIN)},
            pinned=True,
        )
        providers = {e.get("provider") for e in (chain or [])}
        self.assertIn("deepseek", providers)
        self.assertNotIn("openrouter", providers)

    def test_pin_declared_empty_disables_chain(self):
        chain = _resolve_child_fallback_chain(
            _parent(list(PARENT_CHAIN)), {"fallback_providers": []}, pinned=True
        )
        self.assertIsNone(chain)

    def test_pin_absent_gets_no_chain(self):
        """#80450: a pinned child must fail loudly, not silently reroute."""
        chain = _resolve_child_fallback_chain(
            _parent(list(PARENT_CHAIN)), {}, pinned=True
        )
        self.assertIsNone(chain)

    # pin=no -------------------------------------------------------------

    def test_unpinned_declared_nonempty_uses_declared_chain(self):
        """#65038: delegation.fallback_providers reaches the child."""
        chain = _resolve_child_fallback_chain(
            _parent(list(PARENT_CHAIN)),
            {"fallback_providers": list(DECLARED_CHAIN)},
            pinned=False,
        )
        providers = {e.get("provider") for e in (chain or [])}
        self.assertIn("deepseek", providers)
        self.assertNotIn("openrouter", providers)

    def test_unpinned_declared_empty_disables_chain(self):
        chain = _resolve_child_fallback_chain(
            _parent(list(PARENT_CHAIN)), {"fallback_providers": []}, pinned=False
        )
        self.assertIsNone(chain)

    def test_unpinned_absent_inherits_parent_chain(self):
        """Historical default is preserved exactly."""
        chain = _resolve_child_fallback_chain(
            _parent(list(PARENT_CHAIN)), {}, pinned=False
        )
        self.assertEqual(chain, PARENT_CHAIN)

    # edges ----------------------------------------------------------------

    def test_unpinned_absent_with_empty_parent_chain_is_none(self):
        self.assertIsNone(_resolve_child_fallback_chain(_parent([]), {}, pinned=False))

    def test_declared_chain_is_normalized_and_deduped(self):
        """Entries route through the canonical get_fallback_chain normalizer."""
        chain = _resolve_child_fallback_chain(
            _parent(None),
            {"fallback_providers": list(DECLARED_CHAIN) + list(DECLARED_CHAIN)},
            pinned=False,
        )
        routes = [(e.get("provider"), e.get("model")) for e in (chain or [])]
        self.assertEqual(len(routes), len(set(routes)))

    def test_malformed_declared_value_pin_aware_fallback(self):
        """Malformed config logs and falls back pin-aware: None when pinned
        (never reintroduce the silent drag through the error path), parent
        chain otherwise — extends #80421's log-and-inherit contract."""
        with patch(
            "hermes_cli.fallback_config.get_fallback_chain",
            side_effect=TypeError("boom"),
        ):
            self.assertIsNone(
                _resolve_child_fallback_chain(
                    _parent(list(PARENT_CHAIN)),
                    {"fallback_providers": "not-a-list"},
                    pinned=True,
                )
            )
            self.assertEqual(
                _resolve_child_fallback_chain(
                    _parent(list(PARENT_CHAIN)),
                    {"fallback_providers": "not-a-list"},
                    pinned=False,
                ),
                PARENT_CHAIN,
            )

    def test_non_dict_config_uses_default(self):
        self.assertEqual(
            _resolve_child_fallback_chain(_parent(list(PARENT_CHAIN)), None, pinned=False),
            PARENT_CHAIN,
        )


class TestBuildChildAgentWiring(unittest.TestCase):
    """End-to-end through _build_child_agent: the resolver is actually wired."""

    def _spawn(self, parent, cfg, **overrides):
        model = overrides.pop("model", None)
        with (
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch("run_agent.AIAgent") as MockAgent,
        ):
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="matrix wiring",
                context=None,
                toolsets=None,
                model=model,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
                **overrides,
            )
        return MockAgent.call_args[1]

    def test_pinned_child_gets_no_parent_chain(self):
        kwargs = self._spawn(
            _parent(list(PARENT_CHAIN)), {}, override_provider="minimax",
            override_base_url="https://api.minimax.example/v1", override_api_key="sk-mm",
        )
        self.assertIsNone(kwargs["fallback_model"])

    def test_configured_delegation_chain_reaches_child(self):
        kwargs = self._spawn(
            _parent(list(PARENT_CHAIN)),
            {"fallback_providers": list(DECLARED_CHAIN)},
        )
        providers = {e.get("provider") for e in (kwargs["fallback_model"] or [])}
        self.assertIn("deepseek", providers)

    def test_pin_plus_declared_chain_uses_declared(self):
        kwargs = self._spawn(
            _parent(list(PARENT_CHAIN)),
            {"fallback_providers": list(DECLARED_CHAIN)},
            override_provider="deepseek",
            override_base_url="https://api.deepseek.example/v1",
            override_api_key="sk-ds",
        )
        providers = {e.get("provider") for e in (kwargs["fallback_model"] or [])}
        self.assertIn("deepseek", providers)
        self.assertNotIn("openrouter", providers)

    def test_model_only_pin_gets_no_parent_chain(self):
        """The model arm of #80450: delegation.model without delegation.provider
        must not inherit the parent chain — a mid-run failure would silently
        swap the pinned model."""
        kwargs = self._spawn(
            _parent(list(PARENT_CHAIN)), {}, model="deepseek-chat",
        )
        self.assertIsNone(kwargs["fallback_model"])

    def test_model_only_pin_with_declared_chain_uses_declared(self):
        kwargs = self._spawn(
            _parent(list(PARENT_CHAIN)),
            {"fallback_providers": list(DECLARED_CHAIN)},
            model="deepseek-chat",
        )
        providers = {e.get("provider") for e in (kwargs["fallback_model"] or [])}
        self.assertIn("deepseek", providers)
        self.assertNotIn("openrouter", providers)

    def test_default_inheritance_preserved(self):
        kwargs = self._spawn(_parent(list(PARENT_CHAIN)), {})
        self.assertEqual(kwargs["fallback_model"], PARENT_CHAIN)


if __name__ == "__main__":
    unittest.main()
