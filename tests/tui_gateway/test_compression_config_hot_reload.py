"""Desktop/TUI sessions must adopt live compression config on the next turn.

Regression for #95151: ``_sync_agent_model_with_config`` only compared the
model/provider. After ``hermes config set compression.threshold_tokens 100000``
the already-open session kept the computed threshold from agent creation.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from tui_gateway import server


def _session_with_compressor(**compression_ctor):
    compressor = ContextCompressor(
        model="gpt-5.6-sol",
        threshold_percent=0.85,
        config_context_length=272_000,
        quiet_mode=True,
        **compression_ctor,
    )
    agent = SimpleNamespace(
        model="gpt-5.6-sol",
        provider="openai-codex",
        context_compressor=compressor,
        compression_enabled=True,
        compression_idle_compact_after_seconds=0,
    )
    return {
        "agent": agent,
        "session_key": "session-95151",
    }, compressor


def test_live_threshold_tokens_applies_on_next_turn_without_rebuild(monkeypatch):
    session, compressor = _session_with_compressor()
    stale = compressor.threshold_tokens
    assert stale > 100_000

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {
                "default": "gpt-5.6-sol",
                "provider": "openai-codex",
                "context_length": 272_000,
            },
            "compression": {
                "threshold_tokens": 100_000,
                "proactive_prune_tokens": 48_000,
                "idle_compact_after_seconds": 1800,
                "tail_mode": "lean",
            },
        },
    )

    live_agent = session["agent"]
    server._sync_agent_compression_with_config("sid-95151", session)

    assert session["agent"] is live_agent
    assert compressor.threshold_tokens == 100_000
    assert compressor.proactive_prune_tokens == 48_000
    assert compressor.tail_mode == "lean"
    assert live_agent.compression_idle_compact_after_seconds == 1800


def test_unchanged_compression_config_is_noop(monkeypatch):
    session, compressor = _session_with_compressor(threshold_tokens_cap=100_000)
    cfg = {
        "model": {"context_length": 272_000},
        "compression": {"threshold_tokens": 100_000},
    }
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    session["config_compression_seen"] = server._tui_compression_config_signature(cfg)

    compressor.threshold_tokens = 99_999
    server._sync_agent_compression_with_config("sid-95151", session)

    assert compressor.threshold_tokens == 99_999


def test_clearing_threshold_tokens_restores_ratio_trigger(monkeypatch):
    session, compressor = _session_with_compressor(threshold_tokens_cap=100_000)
    assert compressor.threshold_tokens == 100_000

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {"context_length": 272_000},
            "compression": {},
        },
    )
    server._sync_agent_compression_with_config("sid-95151", session)

    assert compressor.threshold_tokens > 100_000
    assert compressor.threshold_tokens_cap is None


def test_prompt_submit_calls_compression_sync_after_model_sync():
    source = open(server.__file__, encoding="utf-8").read()
    model_idx = source.find("_sync_agent_model_with_config(sid, session)")
    compression_idx = source.find("_sync_agent_compression_with_config(sid, session)")
    assert model_idx != -1
    assert compression_idx != -1
    assert model_idx < compression_idx
