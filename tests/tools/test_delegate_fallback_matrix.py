"""Pin x delegation.fallback_providers decision table for child agents (#80450, #65038)."""

from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_child_agent
from tools.delegate_tool_config import _resolve_child_fallback_chain
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



# (pinned, declared fallback_providers, expected) — the six cells plus the malformed edges.
@pytest.mark.parametrize(
    ("pinned", "declared", "expected"),
    [
        (True, "absent", None),                 # #80450: a pinned child fails loudly, never reroutes
        (True, None, None),
        (True, [], None),
        (True, DECLARED_CHAIN, DECLARED_CHAIN),
        (False, "absent", PARENT_CHAIN),        # historical default preserved
        (False, None, PARENT_CHAIN),
        (False, [], None),                      # explicit [] disables fallback
        (False, DECLARED_CHAIN, DECLARED_CHAIN),  # #65038: delegation.fallback_providers reaches the child
        (True, "not-a-list", None),
        (False, "not-a-list", PARENT_CHAIN),
        (True, [{"provider": "deepseek"}], None),
        (False, [{"provider": "deepseek"}], PARENT_CHAIN),
        (False, [{"provider": "deepseek", "model": "x"}, {"provider": "deepseek"}],
         [{"provider": "deepseek", "model": "x"}]),  # a valid route survives a malformed neighbour
    ],
)
def test_child_fallback_chain_matrix(pinned, declared, expected):
    cfg = {} if declared == "absent" else {"fallback_providers": declared}
    assert _resolve_child_fallback_chain(_parent(list(PARENT_CHAIN)), cfg, pinned=pinned) == expected


def _spawn_kwargs(parent, cfg, **overrides):
    model = overrides.pop("model", None)
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(task_index=0, goal="matrix wiring", context=None, toolsets=None, model=model,
                           max_iterations=10, parent_agent=parent, task_count=1, **overrides)
    return MockAgent.call_args[1]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, PARENT_CHAIN),                                                    # unpinned inherits
        ({"model": "deepseek-chat"}, None),                                    # model-only pin (#80450 model arm)
        ({"override_provider": "minimax", "override_base_url": "https://api.minimax.example/v1",
          "override_api_key": "sk-mm"}, None),                                 # provider pin
    ],
)
def test_pin_is_derived_from_provider_base_url_or_model(overrides, expected):
    assert _spawn_kwargs(_parent(list(PARENT_CHAIN)), {}, **overrides)["fallback_model"] == expected


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
