"""Delegated workers and /review inherit activation policy without acquiring a choice UI."""

import json
from unittest.mock import MagicMock, patch

import pytest

import hermes_yaml as yaml
from tools.delegate_tool import _build_child_agent

_PARENT_CHAIN = [{"provider": "deepseek", "model": "parent-backup", "api_key": "test-only"}]
_DECLARED_CHAIN = [{"provider": "deepseek", "model": "child-backup", "api_key": "test-only"}]


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    token = set_hermes_home_override(tmp_path)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("httpx.Client.send", side_effect=RuntimeError("offline test")),
    ):
        parent = AIAgent(
            api_key="test-only", base_url="https://primary.invalid/v1", provider="custom",
            model="primary-model", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=list(_PARENT_CHAIN),
        )
        parent._fallback_selection_interactive = True
        parent.clarify_callback = MagicMock()
        try:
            yield tmp_path, parent
        finally:
            for child in list(parent._active_children):
                child.close()
            parent.close()
            reset_hermes_home_override(token)


def _check_activation(child, expected_chain, auto):
    from agent.error_classifier import FailoverReason

    assert child._fallback_chain == (expected_chain or [])
    assert child._fallback_auto_activate is auto
    assert child._fallback_selection_interactive is False
    callback = MagicMock()
    # A synthetic callback must not turn an unattended child into an interactive session.
    child.clarify_callback = callback
    model = expected_chain[0]["model"] if expected_chain else "unused"
    with (
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(MagicMock(base_url="https://backup.invalid/v1", api_key="test-only"), model)) as resolve,
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, _p: m),
    ):
        activated = child._try_activate_fallback(FailoverReason.rate_limit)
    expected = bool(expected_chain) and auto
    assert activated is expected
    assert resolve.call_count == int(expected)
    callback.assert_not_called()


@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize(("cfg", "overrides", "expected_chain"), [
    pytest.param({}, {}, _PARENT_CHAIN, id="inherit"),
    pytest.param({}, {"model": "pinned-model"}, None, id="model-pin"),
    pytest.param({}, {"override_provider": "custom", "override_base_url": "https://pinned.invalid/v1"},
                 None, id="provider-pin"),
    pytest.param({}, {"override_base_url": "https://pinned.invalid/v1"}, None, id="endpoint-pin"),
    pytest.param({"fallback_providers": []}, {}, None, id="empty"),
    pytest.param({"fallback_providers": []}, {"model": "pinned-model"}, None, id="pinned-empty"),
    pytest.param({"fallback_providers": _DECLARED_CHAIN}, {"model": "pinned-model"},
                 _DECLARED_CHAIN, id="pinned-declared"),
])
def test_real_child_inherits_policy_without_changing_chain_ownership(agent_env, auto, cfg, overrides, expected_chain):
    home, parent = agent_env
    (home / "config.yaml").write_text(yaml.safe_dump({"delegation": cfg}), encoding="utf-8")
    parent._fallback_auto_activate = auto
    child = _build_child_agent(
        task_index=0, goal="check route", context=None, toolsets=None,
        max_iterations=2, parent_agent=parent, task_count=1, **{"model": None, **overrides})
    _check_activation(child, expected_chain, auto)
    parent.clarify_callback.assert_not_called()


@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("declared", [None, []], ids=["absent", "empty"])
def test_real_review_uses_its_owner_chain_and_parent_activation_policy(agent_env, auto, declared):
    from agent.review_engine import start_review

    home, parent = agent_env
    parent._fallback_auto_activate = auto
    review = {"provider": "custom", "model": "review-model", "base_url": "https://review.invalid/v1",
              "api_key": "test-only"}
    if declared is not None:
        review["fallback_providers"] = declared
    (home / "config.yaml").write_text(yaml.safe_dump({
        "delegation": {"fallback_providers": list(_PARENT_CHAIN)},
        "auxiliary": {"review": review},
    }), encoding="utf-8")
    captured = []

    def defer_run(batch, background):
        captured.extend(child for _index, _task, child in batch.children)
        return json.dumps({"status": "dispatched", "delegation_id": "test-review"})

    with patch("tools.delegate_tool._run_batch", side_effect=defer_run):
        result = start_review(parent, [{"role": "user", "content": "Review the last result"}])
    assert result["status"] == "dispatched"
    assert len(captured) == 1
    child = captured[0]
    assert (child.model, child.base_url) == ("review-model", "https://review.invalid/v1")
    _check_activation(child, declared, auto)
    parent.clarify_callback.assert_not_called()


@pytest.mark.parametrize("auto", [False, True])
def test_explicit_route_owner_chain_reaches_real_child(agent_env, auto):
    home, parent = agent_env
    parent._fallback_auto_activate = auto
    (home / "config.yaml").write_text(yaml.safe_dump({
        "delegation": {"fallback_providers": list(_PARENT_CHAIN)},
    }), encoding="utf-8")
    child = _build_child_agent(
        task_index=0, goal="review owner route", context=None, toolsets=None, model="review-model",
        max_iterations=2, parent_agent=parent, task_count=1,
        override_provider="custom", override_base_url="https://review.invalid/v1",
        routing_cfg={"fallback_providers": list(_DECLARED_CHAIN)},
    )
    _check_activation(child, _DECLARED_CHAIN, auto)
