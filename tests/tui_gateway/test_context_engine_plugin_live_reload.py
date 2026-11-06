"""Live compression hot reload must respect a plugin engine's policy boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.context_engine import ContextEngine


class PluginEngine(ContextEngine):
    """A working ContextEngine with public state, not ContextCompressor internals."""

    def __init__(self):
        self.context_length = 160_000
        self.threshold_percent = 0.70
        self.threshold_tokens = 112_000
        self.protect_last_n = 6
        self.model_thresholds = {"plugin-model": 0.70}
        self.tail_mode = "plugin-tail"

    @property
    def name(self):
        return "plugin-engine"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return (prompt_tokens or self.last_prompt_tokens) >= self.threshold_tokens

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def update_model(self, model, context_length, base_url="", api_key="", provider="", api_mode=""):
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)


def host_agent(engine):
    return SimpleNamespace(
        model="plugin-model", provider="openai", base_url="", api_key="fixture-key",
        api_mode="chat_completions",
        _config_context_length=None,
        context_compressor=engine, compression_enabled=True,
        compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
    )


def edited_config():
    return {
        "compression": {
            "enabled": False,
            "codex_responses_native": True,
            "codex_responses_compact_threshold": 110_000,
            "idle_compact_after_seconds": 1800,
            "threshold": 0.65,
            "threshold_tokens": 90_000,
            "tail_mode": "legacy",
            "protect_last_n": 8,
            "proactive_prune_tokens": 30_000,
            "proactive_prune_min_result_chars": 12_000,
            "proactive_prune_min_reclaim_tokens": 5_000,
            "min_tail_user_messages": 3,
            "target_ratio": 0.35,
            "model_thresholds": {"plugin-model": 0.80},
        },
    }


def assert_generic_host_settings(agent):
    assert agent.compression_enabled is False
    assert agent.codex_responses_native_compaction is True
    assert agent.codex_responses_compact_threshold == 110_000
    assert agent.compression_idle_compact_after_seconds == 1800


def test_actual_live_adapter_does_not_require_or_mutate_compressor_private_state():
    from tui_gateway.session_compression import _apply_live_compression_config

    engine = PluginEngine()
    original_policy = dict(vars(engine))
    agent = host_agent(engine)

    # The adapter previously raised on _coerce_threshold_tokens_cap after overwriting plugin policy.
    _apply_live_compression_config(agent, edited_config())

    assert agent.context_compressor is engine
    assert vars(engine) == original_policy
    assert engine.should_compress(112_000) is True
    assert_generic_host_settings(agent)


def test_next_turn_hotreload_consumer_keeps_plugin_usable_without_warning(monkeypatch, caplog):
    from tui_gateway import server

    engine = PluginEngine()
    original_policy = dict(vars(engine))
    agent = host_agent(engine)
    session = {"agent": agent, "session_key": "plugin-hotreload"}
    cfg = edited_config()
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)

    server._sync_agent_compression_with_config("plugin-hotreload", session)

    assert session["agent"] is agent
    assert vars(engine) == original_policy
    assert engine.should_compress(112_000) is True
    assert_generic_host_settings(agent)
    assert "Could not apply live compression config" not in caplog.text


def test_removing_host_controls_restores_defaults_without_changing_plugin_policy():
    from tui_gateway.session_compression import _apply_live_compression_config

    engine = PluginEngine()
    original_policy = dict(vars(engine))
    agent = host_agent(engine)
    _apply_live_compression_config(agent, edited_config())
    _apply_live_compression_config(agent, {})

    assert agent.compression_enabled is True
    assert agent.codex_responses_native_compaction is False
    assert agent.codex_responses_compact_threshold == 200_000
    assert agent.compression_idle_compact_after_seconds == 0
    assert vars(engine) == original_policy


def test_builtin_compressor_still_adopts_live_policy_and_thresholds(monkeypatch):
    from agent.context_compressor import ContextCompressor
    from tui_gateway import server

    compressor = ContextCompressor(
        model="plugin-model", provider="openai", config_context_length=600_000,
        threshold_percent=0.50, quiet_mode=True,
    )
    agent = host_agent(compressor)
    session = {"agent": agent, "session_key": "builtin-hotreload"}
    cfg = edited_config()
    cfg["model"] = {"default": "plugin-model", "provider": "openai", "context_length": 600_000}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)

    server._sync_agent_compression_with_config("builtin-hotreload", session)

    assert agent.context_compressor is compressor
    assert compressor.context_length == 600_000
    assert compressor.threshold_tokens == 90_000
    assert compressor.threshold_percent == 0.80
    assert compressor.tail_mode == "legacy"
    assert compressor.protect_last_n == 8
    assert compressor.proactive_prune_tokens == 30_000
    assert compressor.proactive_prune_min_result_chars == 12_000
    assert compressor.proactive_prune_min_reclaim_tokens == 5_000
    assert compressor.min_tail_user_messages == 3
    assert compressor.summary_target_ratio == 0.35
    assert_generic_host_settings(agent)


def test_plugin_live_context_change_uses_public_update_on_unchanged_route(monkeypatch):
    from agent.conversation_loop import _apply_context_engine_selection
    from tui_gateway import server

    budgets = []

    class SelectingPlugin(PluginEngine):
        def select_context(self, request_messages, *, budget_tokens=0, **kwargs):
            budgets.append(budget_tokens)
            return request_messages

    engine = SelectingPlugin()
    agent = host_agent(engine)
    agent._config_context_length = 160_000
    cfg = edited_config()
    cfg["model"] = {"default": "plugin-model", "provider": "openai", "context_length": 320_000}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    session = {"agent": agent, "session_key": "plugin-context-reload"}

    server._sync_agent_compression_with_config("plugin-context-reload", session)

    assert agent._config_context_length == 320_000
    assert engine.context_length == 320_000
    assert engine.threshold_percent == 0.70
    assert engine.threshold_tokens == 224_000
    assert engine.tail_mode == "plugin-tail"
    assert engine.protect_last_n == 6
    assert (agent.model, agent.provider, agent.base_url, agent.api_key, agent.api_mode) == (
        "plugin-model", "openai", "", "fixture-key", "chat_completions",
    )
    history = [{"role": "user", "content": "current task"}]
    _apply_context_engine_selection(agent, history, history, history[-1], logger=server.logger)
    assert budgets == [320_000]


def test_plugin_context_pin_removal_reinfers_active_route_through_public_api(monkeypatch):
    from tui_gateway import server

    engine = PluginEngine()
    engine.context_length = 320_000
    agent = host_agent(engine)
    agent._config_context_length = 320_000

    def resolve(model, **kwargs):
        return 160_000

    monkeypatch.setattr("agent.model_metadata.get_model_context_length", resolve)
    server._apply_live_compression_config(agent, edited_config())

    assert agent._config_context_length is None
    assert engine.context_length == 160_000
    assert engine.threshold_tokens == 112_000
    assert (agent.model, agent.provider, agent.base_url) == ("plugin-model", "openai", "")


@pytest.mark.parametrize("configured_route", [
    {"default": "different-model", "provider": "openai"},
    {"default": "plugin-model", "provider": "anthropic"},
    {"default": "plugin-model", "provider": "openai", "base_url": "https://other.invalid/v1"},
])
def test_plugin_keeps_context_pin_scoped_to_configured_default_route(configured_route):
    from tui_gateway.session_compression import _apply_live_compression_config

    engine = PluginEngine()
    agent = host_agent(engine)
    cfg = edited_config()
    cfg["model"] = {**configured_route, "context_length": 320_000}

    _apply_live_compression_config(agent, cfg)

    assert agent._config_context_length is None
    assert engine.context_length == 160_000
