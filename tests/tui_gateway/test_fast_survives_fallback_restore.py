"""The tier word is the session's pick: restoring the primary after a fallback must not undo a change."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tui_gateway import server

ANTHROPIC_URL = "https://api.anthropic.com"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
FALLBACK = {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}


@pytest.fixture
def build_patches():
    active = [
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ]
    for item in active:
        item.start()
    yield
    for item in active:
        item.stop()


def _agent(tier, overrides):
    return AIAgent(
        api_key="test-key-12345678", base_url=ANTHROPIC_URL, provider="anthropic", model="claude-opus-5-5",
        api_mode="anthropic_messages", service_tier=tier, quiet_mode=True, skip_context_files=True, skip_memory=True,
        request_overrides=overrides, fallback_model=FALLBACK,
    )


def _set_fast(agent, word):
    session = {"agent": agent, "session_key": "session-key"}
    with patch.dict(server._sessions, {"sid": session}):
        response = server._methods["config.set"]("rid", {"session_id": "sid", "key": "fast", "value": word})
    return response["result"]["value"]


def _activate_fallback(agent):
    client = MagicMock(base_url=OPENROUTER_URL, api_key="fallback-key-1234")
    with (
        patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network", return_value=None),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, None)),
    ):
        assert agent._try_activate_fallback() is True


def _speed_on_wire(agent):
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}], tools_for_api=[])
    return (kwargs.get("extra_body") or {}).get("speed")


@pytest.mark.parametrize("tier,overrides,word,expected", [
    ("priority", {"speed": "fast"}, "normal", None),
    (None, {}, "fast", "fast"),
])
def test_a_live_change_is_not_undone_by_a_fallback_and_restore(build_patches, tier, overrides, word, expected):
    agent = _agent(tier, overrides)
    assert _set_fast(agent, word) == word

    _activate_fallback(agent)
    assert agent._restore_primary_runtime() is True
    assert _speed_on_wire(agent) == expected


def test_fast_off_during_a_fallback_holds_after_the_restore(build_patches):
    agent = _agent("priority", {"speed": "fast"})
    _activate_fallback(agent)
    assert _set_fast(agent, "normal") == "normal"

    assert agent._restore_primary_runtime() is True
    assert _speed_on_wire(agent) is None


def test_other_overrides_of_the_primary_survive_a_live_fast_change(build_patches):
    agent = _agent("priority", {"speed": "fast", "extra_body": {"keep": 1}})
    _set_fast(agent, "normal")
    _activate_fallback(agent)
    assert agent._restore_primary_runtime() is True
    assert agent.request_overrides.get("extra_body", {}).get("keep") == 1


def test_fast_off_holds_when_a_temporary_model_restore_replaces_the_snapshot(build_patches):
    agent = _agent("priority", {"speed": "fast"})
    taken_before_the_change = copy.deepcopy(agent._primary_runtime)
    _set_fast(agent, "normal")
    _activate_fallback(agent)

    agent._primary_runtime = taken_before_the_change
    assert agent._restore_primary_runtime() is True
    assert _speed_on_wire(agent) is None


def test_a_tier_set_outside_slash_fast_is_honoured_through_the_gateway_merge_and_restore(build_patches):
    from gateway.run_turn_runner import TurnRunner

    agent = _agent("priority", {"speed": "fast"})
    _activate_fallback(agent)
    agent.service_tier = None
    TurnRunner._merge_turn_request_overrides(agent, {"request_overrides": {}})
    assert agent._restore_primary_runtime() is True
    assert _speed_on_wire(agent) is None


def test_an_unchanged_tier_restores_an_explicit_top_level_value(build_patches):
    agent = _agent(None, {"speed": "fast"})
    _activate_fallback(agent)
    assert agent._restore_primary_runtime() is True
    assert _speed_on_wire(agent) == "fast"


def test_a_tier_change_keeps_a_service_tier_that_is_not_a_fast_tier(build_patches):
    agent = _agent(None, {"service_tier": "flex"})
    agent.service_tier = "priority"
    agent.model = "unrelated"
    agent._primary_runtime.update(model="some-custom-model", provider="custom", base_url="https://proxy.example/v1")
    from agent.primary_runtime_snapshot import restored_request_overrides

    assert restored_request_overrides(agent, agent._primary_runtime) == {"service_tier": "flex"}


def _typed_fast(agent, word, session=None):
    session = session if session is not None else {"agent": agent, "session_key": "session-key"}
    with patch.dict(server._sessions, {"sid": session}), patch.object(server, "_emit", lambda *_: None):
        server._SLASH_MIRRORS["fast"]("sid", session, agent, word)


@pytest.mark.parametrize("command", ["off", "off --global", "--global off", "--session off"])
def test_typed_fast_off_stops_the_paid_tier_without_any_fallback(build_patches, command):
    agent = _agent("priority", {"speed": "fast"})
    _typed_fast(agent, command)
    assert agent.service_tier is None
    assert _speed_on_wire(agent) is None


def test_typed_fast_on_reaches_the_wire(build_patches):
    agent = _agent(None, {})
    _typed_fast(agent, "fast")
    assert _speed_on_wire(agent) == "fast"


@pytest.mark.parametrize("command", ["on foo", "fast typo", "toggle", "--global"])
def test_a_typed_command_the_worker_rejects_changes_nothing(build_patches, command):
    agent = _agent(None, {})
    _typed_fast(agent, command)
    assert agent.service_tier is None
    assert _speed_on_wire(agent) is None


def test_a_refused_typed_tier_changes_nothing_and_says_why(build_patches):
    agent = _agent("priority", {"speed": "fast"})
    session = {"agent": agent, "session_key": "session-key"}
    with patch.dict(server._sessions, {"sid": session}), patch.object(server, "_emit", lambda *_: None):
        warning = server._mirror_slash_side_effects("sid", session, "/fast ultrafast")
    assert warning
    assert agent.service_tier == "priority"
    assert _speed_on_wire(agent) == "fast"


def test_typed_fast_off_holds_through_a_model_switch_fallback_and_restore(build_patches):
    agent = _agent("priority", {"speed": "fast"})
    _typed_fast(agent, "off")
    agent.switch_model("claude-opus-5", "anthropic", api_key="switch-key-1234", base_url=ANTHROPIC_URL,
                       api_mode="anthropic_messages")
    _activate_fallback(agent)
    assert agent._restore_primary_runtime() is True
    assert _speed_on_wire(agent) is None


def test_fast_on_holds_when_the_session_was_built_with_the_tier_but_no_pin(build_patches):
    agent = _agent("priority", {})
    assert _set_fast(agent, "fast") == "fast"
    _activate_fallback(agent)
    assert agent._restore_primary_runtime() is True
    assert _speed_on_wire(agent) == "fast"


def test_typed_fast_off_replaces_the_tier_a_resumed_session_would_rebuild_with(build_patches):
    agent = _agent("priority", {"speed": "fast"})
    session = {"agent": agent, "session_key": "session-key", "create_service_tier_override": "priority"}
    _typed_fast(agent, "off", session)
    assert session["create_service_tier_override"] == ""
