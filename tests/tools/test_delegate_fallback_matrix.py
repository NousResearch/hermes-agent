"""Full decision matrix for _resolve_child_fallback_chain (#80450 class).

Composes the pin semantics of PR #80465 (@teknium1), the
delegation.fallback_providers semantics of PRs #80438 (@wz-heng) /
#80421 (@andrexibiza) for issue #65038 (@mlahatte), and settles the
pin+declared-chain composition cell raised in the #80450 cross-PR map.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

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

    def test_valid_route_survives_invalid_neighbor(self):
        """One malformed entry must not discard a usable declared route."""
        declared = [
            {"provider": "deepseek", "model": "worker-fallback"},
            {"provider": "deepseek"},
        ]
        expected = [{"provider": "deepseek", "model": "worker-fallback"}]
        for pinned in (False, True):
            with self.subTest(pinned=pinned):
                self.assertEqual(
                    _resolve_child_fallback_chain(
                        _parent(list(PARENT_CHAIN)),
                        {"fallback_providers": declared},
                        pinned=pinned,
                    ),
                    expected,
                )

    def test_malformed_declared_value_pin_aware_fallback(self):
        """Malformed config logs and falls back pin-aware: None when pinned
        (never reintroduce the silent drag through the error path), parent
        chain otherwise — extends #80421's log-and-inherit contract."""
        for malformed in (
            "not-a-list",
            [{"provider": "deepseek"}],
            ["not-a-mapping"],
        ):
            with self.subTest(malformed=malformed):
                self.assertIsNone(
                    _resolve_child_fallback_chain(
                        _parent(list(PARENT_CHAIN)),
                        {"fallback_providers": malformed},
                        pinned=True,
                    )
                )
                self.assertEqual(
                    _resolve_child_fallback_chain(
                        _parent(list(PARENT_CHAIN)),
                        {"fallback_providers": malformed},
                        pinned=False,
                    ),
                    PARENT_CHAIN,
                )

    def test_normalizer_failure_uses_pin_aware_fallback(self):
        with patch(
            "hermes_cli.fallback_config.get_fallback_chain",
            side_effect=TypeError("boom"),
        ):
            self.assertIsNone(
                _resolve_child_fallback_chain(
                    _parent(list(PARENT_CHAIN)),
                    {"fallback_providers": list(DECLARED_CHAIN)},
                    pinned=True,
                )
            )
            self.assertEqual(
                _resolve_child_fallback_chain(
                    _parent(list(PARENT_CHAIN)),
                    {"fallback_providers": list(DECLARED_CHAIN)},
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


def test_declared_chain_flows_through_real_profile_config_loader(
    tmp_path, monkeypatch
):
    """The public key must survive DEFAULT_CONFIG/profile loading without
    patching ``_load_config`` and reach the child constructor."""
    import yaml

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    token = set_hermes_home_override(tmp_path)
    try:
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {"delegation": {"fallback_providers": list(DECLARED_CHAIN)}}
            ),
            encoding="utf-8",
        )
        with patch("run_agent.AIAgent") as mock_agent:
            mock_agent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="real config loader",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=_parent(list(PARENT_CHAIN)),
                task_count=1,
            )
    finally:
        reset_hermes_home_override(token)

    child_kwargs = mock_agent.call_args.kwargs
    assert child_kwargs["fallback_model"] == DECLARED_CHAIN


def test_explicit_empty_chain_survives_real_profile_config_loader(tmp_path, monkeypatch):
    """An explicit [] remains an authoritative disable after config loading."""
    import yaml

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    token = set_hermes_home_override(tmp_path)
    try:
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"delegation": {"fallback_providers": []}}),
            encoding="utf-8",
        )
        with patch("run_agent.AIAgent") as mock_agent:
            mock_agent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="explicit disable",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=_parent(list(PARENT_CHAIN)),
                task_count=1,
            )
    finally:
        reset_hermes_home_override(token)

    assert mock_agent.call_args.kwargs["fallback_model"] is None


def test_pinned_review_does_not_borrow_general_worker_chain(tmp_path, monkeypatch):
    """The public /review route owns its fallback policy as well as its model."""
    import yaml

    from agent.review_engine import start_review
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "delegation": {
                    "fallback_providers": [
                        {"provider": "deepseek", "model": "worker-fallback"}
                    ]
                },
                "auxiliary": {
                    "review": {
                        "provider": "custom",
                        "model": "review-model",
                        "base_url": "http://127.0.0.1:18479/v1",
                        "api_key": "test-only",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    parent = _parent(list(PARENT_CHAIN))
    parent.session_id = "review-80479-parent"
    captured = {}

    class ReachedConstructor(RuntimeError):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise ReachedConstructor()

    token = set_hermes_home_override(tmp_path)
    try:
        with patch("run_agent.AIAgent", side_effect=capture):
            with pytest.raises(ReachedConstructor):
                start_review(
                    parent,
                    [{"role": "user", "content": "Check the last result"}],
                )
    finally:
        reset_hermes_home_override(token)

    assert captured["model"] == "review-model"
    assert captured["base_url"] == "http://127.0.0.1:18479/v1"
    assert captured["fallback_model"] is None


def test_declared_child_chain_activates_on_primary_failure():
    """The selected chain is accepted by the real fallback activation rail."""
    from agent.error_classifier import FailoverReason
    from run_agent import AIAgent

    chain = _resolve_child_fallback_chain(
        _parent(list(PARENT_CHAIN)),
        {"fallback_providers": list(DECLARED_CHAIN)},
        pinned=True,
    )
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        child = AIAgent(
            api_key="primary-test-key",
            base_url="https://primary.example/v1",
            model="primary-model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=chain,
        )
    fallback_client = MagicMock()
    fallback_client.base_url = "https://fallback.example/v1"
    fallback_client.api_key = "fallback-test-key"
    with (
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "deepseek-chat"),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda model, _provider: model,
        ),
    ):
        assert child._try_activate_fallback(FailoverReason.rate_limit) is True

    assert child.model == "deepseek-chat"
    assert child.provider == "deepseek"


if __name__ == "__main__":
    unittest.main()
