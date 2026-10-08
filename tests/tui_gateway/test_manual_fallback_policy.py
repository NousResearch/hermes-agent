"""TUI/Desktop construction and turn admission honor manual fallback policy."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.auth import AuthError
from tui_gateway import server

CHAIN = [{"provider": "anthropic", "model": "chosen-fallback"}]


@pytest.fixture
def configured_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_fallback_settings_by_home", {})
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: primary\n  provider: openai\n"
        "fallback:\n  auto_activate: false\n"
        "fallback_providers:\n  - provider: anthropic\n    model: chosen-fallback\n")
    return tmp_path


def test_make_agent_blocks_pre_agent_manual_fallback(configured_home, monkeypatch):
    resolver = Mock(side_effect=AuthError("primary unavailable"))
    factory = Mock()
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)
    monkeypatch.setattr("run_agent.AIAgent", factory)
    with pytest.raises(AuthError, match="primary unavailable"):
        server._make_agent("sid", "key", context_cwd_is_launch_artifact=False)
    assert resolver.call_count == 1
    factory.assert_not_called()


def test_make_agent_propagates_manual_policy_and_structured_clarify(configured_home, monkeypatch):
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda **_: {"provider": "openai", "api_key": "test", "base_url": "https://example.invalid"})
    factory = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr("run_agent.AIAgent", factory)
    clarify = Mock(return_value={"outcome": "submitted", "answers": {"fallback_route": "chosen"}})
    monkeypatch.setattr(server, "_clarify_block", clarify)
    server._make_agent("sid", "key", context_cwd_is_launch_artifact=False)
    kwargs = factory.call_args.kwargs
    assert kwargs["fallback_model"] == CHAIN
    assert kwargs["fallback_auto_activate"] is False
    assert kwargs["fallback_selection_interactive"] is True
    questions = [{"qid": "fallback_route", "question": "Choose", "choices": ["chosen"], "multi_select": False}]
    assert kwargs["clarify_callback"](questions)["answers"] == {"fallback_route": "chosen"}
    clarify.assert_called_once_with("sid", questions)


def test_automatic_bootstrap_route_is_marked_for_manual_policy_expiry(configured_home, monkeypatch):
    from agent.manual_fallback import prepare_turn_runtime
    config_path = configured_home / "config.yaml"
    config_path.write_text(config_path.read_text().replace("auto_activate: false", "auto_activate: true"))
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("primary", "openai"))

    def resolve(**kwargs):
        if kwargs.get("requested") == "openai":
            raise AuthError("primary unavailable")
        return {"provider": "anthropic", "api_key": "test", "base_url": "https://backup.invalid"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    factory = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr("run_agent.AIAgent", factory)
    agent = server._make_agent("sid", "key", context_cwd_is_launch_artifact=False)
    assert agent._fallback_bootstrap_active is True
    agent._fallback_auto_activate = False
    with pytest.raises(RuntimeError, match="primary was unavailable at startup"):
        prepare_turn_runtime(agent, Mock())


def test_policy_only_edit_at_turn_admission_expires_auto_cooldown(configured_home, monkeypatch):
    from tests.tui_gateway.test_fallback_chain_hot_reload import _admit_turn
    import time
    agent = SimpleNamespace(_fallback_chain=CHAIN, _fallback_model=CHAIN[0], _fallback_index=1,
                            _fallback_auto_activate=True, _fallback_activated=True,
                            _rate_limited_until=time.monotonic() + 600)
    session = {"agent": agent, "session_key": "key"}
    config = (configured_home / "config.yaml").read_text()
    _admit_turn(monkeypatch, configured_home, session, config)
    assert agent._fallback_auto_activate is False
    assert agent._fallback_selection_interactive is True
    # Torn subsequent writes keep both fields rather than mixing a cached chain with auto=True.
    _admit_turn(monkeypatch, configured_home, session, "fallback: [\n")
    assert agent._fallback_chain == CHAIN
    assert agent._fallback_auto_activate is False


def test_tui_settings_keep_profiles_separate_on_failed_first_read(configured_home, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    assert server._load_fallback_settings() == (CHAIN, False)
    other = configured_home / "other"
    other.mkdir()
    (other / "config.yaml").write_text("fallback: [\n")
    token = set_hermes_home_override(str(other))
    try:
        assert server._load_fallback_settings() == (None, False)
        (other / "config.yaml").write_text("fallback:\n  auto_activate: true\n")
        assert server._load_fallback_settings() == (None, True)
    finally:
        reset_hermes_home_override(token)
    (configured_home / "config.yaml").write_text("fallback: [\n")
    assert server._load_fallback_settings() == (CHAIN, False)


@pytest.mark.parametrize("builder", ["_background_agent_kwargs", "_ephemeral_preview_agent_kwargs"])
def test_detached_agents_preserve_empty_chain_and_disable_selection(configured_home, builder):
    parent = SimpleNamespace(_fallback_chain=[], _fallback_auto_activate=False)
    kwargs = getattr(server, builder)(parent, "background")
    assert kwargs["fallback_model"] == []
    assert kwargs["fallback_auto_activate"] is False
    assert kwargs["fallback_selection_interactive"] is False


def test_disconnected_tui_clarify_bridge_authorizes_no_provider(configured_home, monkeypatch):
    from agent.manual_fallback import _selected_route
    from tui_gateway import server_requests
    monkeypatch.setattr(server_requests, "_answerable", lambda _: False)
    send = Mock()
    monkeypatch.setattr(server_requests, "_write", send)
    agent = SimpleNamespace(clarify_callback=server._agent_cbs("disconnected")["clarify_callback"])
    assert _selected_route(agent, [(0, CHAIN[0], "Continue with chosen-fallback via anthropic")]) is None
    assert agent._fallback_manual_cancelled is False
    send.assert_not_called()
