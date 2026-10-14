"""Delegation under ``opusplan``: children run on the provider's exec model, through the existing resolver.

The main session runs the plan model; ``delegation.model`` stays optional. An explicit model id or a direct
``delegation.base_url`` is never overridden, and a provider without a plan/exec pair fails the spawn with an error
naming the provider instead of guessing a model.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.opusplan import delegation_model, mark_agent, session_opusplan_active
from tools.delegate_tool import _resolve_delegation_credentials

ROUTER_URL = "http://192.168.10.13:4000/v1"
PROVIDERS = {"ecc-router": {"base_url": ROUTER_URL, "api_key": "sk-x",
                            "opusplan": {"plan": "GLM-5.3-Flash-850K", "exec": "Qwen3.8FlashNext"}}}


def _parent(*, opusplan=None, model="GLM-5.3-Flash-850K", provider="ecc-router"):
    parent = SimpleNamespace(model=model, provider=provider, requested_provider=provider, base_url=ROUTER_URL,
                             api_key="sk-x", api_mode="chat_completions", acp_command=None, acp_args=[],
                             request_overrides=None)
    if opusplan is not None:
        parent.opusplan_active = opusplan
    return parent


@pytest.fixture
def configured(tmp_path):
    """Profile config with ``model.default: opusplan`` on the router, written under the isolated HERMES_HOME."""
    from hermes_constants import get_hermes_home
    def write(model_default="opusplan", **extra):
        cfg = {"model": {"default": model_default, "provider": "ecc-router"}, "providers": PROVIDERS, **extra}
        (get_hermes_home() / "config.yaml").write_text(json.dumps(cfg))  # JSON is valid YAML
    return write


class TestDelegationModel:
    def test_flagged_session_gets_the_exec_model_with_no_delegation_config(self, configured):
        configured()
        assert delegation_model(None, None, None, _parent(opusplan=True)) == "Qwen3.8FlashNext"

    def test_unflagged_session_is_left_alone(self, configured):
        configured()
        assert delegation_model(None, None, None, _parent(opusplan=False)) is None

    def test_explicit_delegation_model_is_never_overridden(self, configured):
        configured()
        assert delegation_model("glm-4.6", None, None, _parent(opusplan=True)) is None

    def test_delegation_model_keyword_forces_exec_even_without_the_session_flag(self, configured):
        configured("some-other-model")
        assert delegation_model("opusplan", None, None, _parent(opusplan=False)) == "Qwen3.8FlashNext"

    def test_direct_base_url_endpoint_is_not_second_guessed(self, configured):
        configured()
        assert delegation_model(None, None, "http://elsewhere/v1", _parent(opusplan=True)) is None

    def test_delegation_provider_picks_which_providers_pair_applies(self, configured):
        providers = {**PROVIDERS, "lab": {"base_url": "http://lab/v1", "opusplan": {"plan": "lb", "exec": "ls"}}}
        from hermes_constants import get_hermes_home
        (get_hermes_home() / "config.yaml").write_text(json.dumps(
            {"model": {"default": "opusplan", "provider": "ecc-router"}, "providers": providers}))
        assert delegation_model(None, "lab", None, _parent(opusplan=True)) == "ls"


class TestSessionDetection:
    def test_explicit_flag_wins_both_ways(self, configured):
        configured()
        assert session_opusplan_active(_parent(opusplan=True)) is True
        configured("opusplan")
        assert session_opusplan_active(_parent(opusplan=False)) is False

    def test_config_default_applies_while_the_agent_is_still_on_the_plan_model(self, configured):
        configured()
        assert session_opusplan_active(_parent()) is True

    def test_manual_switch_away_from_the_plan_model_ends_the_split(self, configured):
        configured()
        assert session_opusplan_active(_parent(model="some-other-model")) is False

    def test_plain_config_is_never_opusplan(self, configured):
        configured("qwen3")
        assert session_opusplan_active(_parent()) is False

    def test_mark_agent_tolerates_no_agent(self):
        mark_agent(None, True)
        agent = SimpleNamespace()
        mark_agent(agent, True)
        assert agent.opusplan_active is True


class TestCredentialResolution:
    def test_unset_delegation_model_resolves_to_exec_through_the_existing_inherit_branch(self, configured):
        configured()
        creds = _resolve_delegation_credentials({"model": "", "provider": ""}, _parent(opusplan=True))
        assert creds["model"] == "Qwen3.8FlashNext"
        assert creds["provider"] is None  # same provider/endpoint inherited from the parent, only the model differs

    def test_explicit_model_still_wins(self, configured):
        configured()
        creds = _resolve_delegation_credentials({"model": "pinned-model"}, _parent(opusplan=True))
        assert creds["model"] == "pinned-model"

    def test_opusplan_unset_behaves_exactly_as_before(self, configured):
        configured("qwen3")
        creds = _resolve_delegation_credentials({"model": "", "provider": ""}, _parent())
        assert creds["model"] is None

    def test_direct_endpoint_branch_keeps_its_pinned_model(self, configured):
        configured()
        cfg = {"model": "qwen2.5-coder", "base_url": "http://localhost:1234/v1", "api_key": "k"}
        assert _resolve_delegation_credentials(cfg, _parent(opusplan=True))["model"] == "qwen2.5-coder"

    def test_provider_branch_receives_the_exec_model_as_target_model(self, configured):
        providers = {**PROVIDERS, "lab": {"base_url": "http://lab/v1", "opusplan": {"plan": "lb", "exec": "ls"}}}
        from hermes_constants import get_hermes_home
        (get_hermes_home() / "config.yaml").write_text(json.dumps(
            {"model": {"default": "opusplan", "provider": "ecc-router"}, "providers": providers}))
        runtime = {"api_key": "k", "base_url": "http://lab/v1", "provider": "custom", "api_mode": "chat_completions"}
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime) as resolve:
            creds = _resolve_delegation_credentials({"provider": "lab"}, _parent(opusplan=True))
        assert resolve.call_args.kwargs["target_model"] == "ls"
        assert creds["model"] == "ls"

    def test_missing_pair_fails_the_spawn_with_a_provider_naming_error(self, configured):
        from hermes_constants import get_hermes_home
        (get_hermes_home() / "config.yaml").write_text(json.dumps(
            {"model": {"default": "opusplan", "provider": "acme-gw"},
             "providers": {"acme-gw": {"base_url": "http://acme/v1"}}}))
        parent = _parent(opusplan=True, provider="acme-gw")
        with patch("providers.get_provider_profile", return_value=None), \
             pytest.raises(ValueError, match="'acme-gw'.*opusplan"):
            _resolve_delegation_credentials({"model": "", "provider": ""}, parent)


def test_delegate_task_surfaces_the_missing_pair_as_a_tool_error(configured):
    """End to end through ``delegate_task``: the spawn is refused loudly, no child is built."""
    from hermes_constants import get_hermes_home
    from tools.delegate_tool import delegate_task
    (get_hermes_home() / "config.yaml").write_text(json.dumps(
        {"model": {"default": "opusplan", "provider": "acme-gw"}, "providers": {"acme-gw": {"base_url": "http://acme/v1"}}}))
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.opusplan_active = True
    parent.provider = parent.requested_provider = "acme-gw"
    parent.base_url = "http://acme/v1"
    with patch("providers.get_provider_profile", return_value=None):
        out = json.loads(delegate_task(goal="x", parent_agent=parent))
    assert "'acme-gw'" in out["error"] and "opusplan" in out["error"]
