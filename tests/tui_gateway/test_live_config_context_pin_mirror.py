"""Desktop/TUI config hot-reload must keep the agent's cached context pin in step with the compressor's.

``model.context_length`` is cached twice: on ``agent._config_context_length`` (read by the switch path,
fallback resolution and every display/`/usage` surface) and on
``context_compressor._config_context_length`` (read by the compressor's own re-resolution). The live
hot-reload updated only the compressor's copy, so an already-running Desktop session showed a pinned
ceiling while the compressor had already fallen back to provider metadata / the 256K fallback.
"""

from types import SimpleNamespace

import agent.context_compressor as cc_mod
from agent.context_compressor import ContextCompressor
from tui_gateway import server

PIN = 249_000


def _session(config_context_length=PIN):
    compressor = ContextCompressor(
        model="model-a", threshold_percent=0.50, config_context_length=config_context_length,
        quiet_mode=True,
    )
    agent = SimpleNamespace(
        model="model-a",
        provider="custom:acme",
        base_url="http://127.0.0.1:8123/v1",
        _config_context_length=config_context_length,
        context_compressor=compressor,
        compression_enabled=True,
        compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
    )
    return {"agent": agent, "session_key": "session-pin"}, compressor


def _sync(monkeypatch, session, cfg):
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    server._sync_agent_compression_with_config("sid-pin", session)


def test_removed_context_length_clears_the_agent_pin_too(monkeypatch):
    """Dropping ``model.context_length`` on disk must stop the agent claiming the pin."""
    session, compressor = _session()
    monkeypatch.setattr(cc_mod, "get_model_context_length", lambda *a, **k: 256_000)

    _sync(monkeypatch, session, {"model": {}, "compression": {}})

    assert compressor._config_context_length is None
    assert compressor.context_length == 256_000
    assert session["agent"]._config_context_length is None


def test_changed_context_length_refreshes_the_agent_pin_too(monkeypatch):
    """A new ``model.context_length`` must land on both cached copies."""
    session, compressor = _session()

    _sync(monkeypatch, session, {"model": {"default": "model-a", "provider": "custom:acme",
                                           "context_length": 400_000}, "compression": {}})

    assert compressor._config_context_length == 400_000
    assert compressor.context_length == 400_000
    assert session["agent"]._config_context_length == 400_000


def test_hot_reload_does_not_pin_a_session_on_another_route(monkeypatch):
    """The pin describes the configured default route; a session that switched elsewhere must not
    inherit it on the next config save (same scoping as the switch path)."""
    session, compressor = _session(config_context_length=None)
    session["agent"].base_url = "https://openrouter.ai/api/v1"
    session["agent"].provider = "openrouter"
    monkeypatch.setattr(cc_mod, "get_model_context_length", lambda *a, **k: 256_000)

    _sync(monkeypatch, session, {
        "model": {"default": "model-b", "provider": "custom:acme", "base_url": "http://127.0.0.1:8123/v1",
                  "context_length": PIN},
        "custom_providers": [{"name": "acme", "base_url": "http://127.0.0.1:8123/v1", "models": {}}],
        "compression": {},
    })

    assert compressor._config_context_length is None
    assert session["agent"]._config_context_length is None
