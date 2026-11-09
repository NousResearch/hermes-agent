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
        return (prompt_tokens or self.last_prompt_tokens) >= int(self.context_length * self.threshold_percent)

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


def test_next_turn_hotreload_preserves_plugin_compression_decisions(monkeypatch):
    from tui_gateway import server

    engine = PluginEngine()
    agent = host_agent(engine)
    session = {"agent": agent, "session_key": "plugin-hotreload"}
    cfg = edited_config()
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)

    server._sync_agent_compression_with_config("plugin-hotreload", session)

    assert agent.compression_enabled is False
    assert engine.should_compress(110_000) is False
    assert engine.should_compress(112_000) is True


def test_plugin_live_context_change_uses_public_update_on_unchanged_route(monkeypatch):
    from tui_gateway import server
    engine = PluginEngine()
    agent = host_agent(engine)

    agent._config_context_length = 160_000
    cfg = edited_config()
    cfg["model"] = {"default": "plugin-model", "provider": "openai", "context_length": 320_000}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    session = {"agent": agent, "session_key": "plugin-context-reload"}

    server._sync_agent_compression_with_config("plugin-context-reload", session)

    assert engine.context_length == 320_000
    assert engine.should_compress(200_000) is False
    assert engine.should_compress(224_000) is True
    assert (agent.model, agent.provider, agent.base_url, agent.api_key, agent.api_mode) == (
        "plugin-model", "openai", "", "fixture-key", "chat_completions",
    )


def test_plugin_context_pin_removal_reinfers_active_route_through_public_api(monkeypatch):
    from tui_gateway import server

    engine = PluginEngine()
    engine.context_length = 320_000
    agent = host_agent(engine)
    agent.base_url = "https://metadata.invalid/v1"
    agent._config_context_length = 320_000
    cfg = edited_config()
    # Exercise real inference through supported route metadata, not a mocked resolver.
    cfg["custom_providers"] = [{
        "name": "fixture", "base_url": agent.base_url, "context_length": 160_000,
    }]
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    server._sync_agent_compression_with_config("plugin-pin-removal", {"agent": agent})

    assert engine.context_length == 160_000
    assert engine.should_compress(100_000) is False
    assert engine.should_compress(112_000) is True
    assert (agent.model, agent.provider, agent.base_url) == (
        "plugin-model", "openai", "https://metadata.invalid/v1",
    )


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

    assert engine.context_length == 160_000
