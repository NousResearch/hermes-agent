"""Desktop/TUI live sessions adopt global context-cap edits on the next turn."""

from __future__ import annotations

from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from tui_gateway import server


# Upstream already pins the Kimi Coding endpoint to 1,048,576 for kimi-k3
# (_ENDPOINT_SCOPED_CONTEXT), so this route resolves without network access.
KIMI_CODING = "https://api.kimi.com/coding"


def _kimi_session():
    compressor = ContextCompressor(
        model="kimi-k3", provider="kimi-coding", base_url=KIMI_CODING,
        config_context_length=None, quiet_mode=True,
    )
    agent = SimpleNamespace(
        model="kimi-k3", provider="kimi-coding", base_url=KIMI_CODING,
        context_compressor=compressor, compression_enabled=True,
        compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
    )
    return {"agent": agent, "session_key": "session-context-cap"}, compressor


def test_live_cap_changes_reresolve_without_route_pin(monkeypatch):
    session, compressor = _kimi_session()
    cfg = {
        "model": {
            "default": "kimi-k3", "provider": "kimi-coding", "base_url": KIMI_CODING,
        },
        "compression": {},
    }
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

    server._sync_agent_compression_with_config("sid-context-cap", session)
    assert compressor.context_length == 1_048_576

    cfg["model"]["context_cap"] = 350_000
    signature = server._tui_compression_config_signature(cfg)
    assert dict(signature)["model.context_cap"] == 350_000
    server._sync_agent_compression_with_config("sid-context-cap", session)
    assert compressor.context_length == 350_000

    cfg["model"].pop("context_cap")
    server._sync_agent_compression_with_config("sid-context-cap", session)
    assert compressor.context_length == 1_048_576
