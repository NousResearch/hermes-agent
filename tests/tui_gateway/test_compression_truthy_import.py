"""Direct split-module use of live compression config must not NameError (#115572).

tui_gateway/session_compression.py called is_truthy_value() without
importing it. The rebound copy on server masked this (server.py imports the
helper), but calling _apply_live_compression_config from the defining
module raised NameError.
"""

from __future__ import annotations

from types import SimpleNamespace

from tui_gateway.session_compression import _apply_live_compression_config


def _stub_agent():
    return SimpleNamespace(context_compressor=None)


def test_apply_live_config_coerces_truthy_strings_on_stub_agent():
    agent = _stub_agent()
    _apply_live_compression_config(agent, {"compression": {"codex_responses_native": "yes"}})
    assert agent.codex_responses_native_compaction is True


def test_apply_live_config_defaults_native_off_on_stub_agent():
    agent = _stub_agent()
    _apply_live_compression_config(agent, {"compression": {}})
    assert agent.codex_responses_native_compaction is False
