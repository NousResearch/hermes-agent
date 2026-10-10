"""The Desktop context RPC uses route-aware estimates (#134915)."""

from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from agent.usage_anchor import capture_usage_anchor
from tui_gateway import server


@pytest.mark.parametrize("mode", ["local", "anchor", "policy_error"])
def test_context_rpc_excludes_stale_reasoning_without_changing_history(monkeypatch, mode):
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer", "reasoning": "stored " * 1000},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "latest", "reasoning": "new thought"},
    ]
    if mode != "local":
        history[:0] = [
            {"role": "user", "content": "priced prompt"},
            {"role": "assistant", "content": "priced reply"},
        ]
    original = deepcopy(history)
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="custom", model="local-test",
        base_url="http://localhost:9999/v1", tools=[], _memory_store=None,
        _usage_anchor=None, _turn_base_usage_anchor=None,
        context_compressor=SimpleNamespace(context_length=200_000, last_prompt_tokens=0),
    )
    if mode != "local":
        agent._usage_anchor = capture_usage_anchor(1000, 30, history[:1])
    if mode == "policy_error":
        def unavailable(*args):
            raise RuntimeError("route unavailable")
        monkeypatch.setattr("agent.message_sanitization.stale_thinking_reaches_wire", unavailable)
    session = {"agent": agent, "history": history, "history_lock": Lock(),
               "session_key": "reasoning-test", "running": False}
    monkeypatch.setattr(server, "_sessions", {"reasoning-test": session})
    monkeypatch.setattr("agent.system_prompt.build_system_prompt_parts", lambda agent: {})
    monkeypatch.setattr("agent.context_file_sources.context_file_sources_for_agent", lambda agent: [])

    response = server._methods["session.context_breakdown"]("test", {"session_id": "reasoning-test"})

    assert "error" not in response, response
    payload = response["result"]
    expected = deepcopy(history)
    if mode != "policy_error":
        del expected[1 if mode == "local" else 3]["reasoning"]
    conversation = next(row["tokens"] for row in payload["categories"] if row["id"] == "conversation")
    assert conversation == estimate_messages_tokens_rough(expected)
    assert payload["context_used"] == (
        conversation if mode == "local" else 1030 + estimate_messages_tokens_rough(expected[2:])
    )
    assert payload["context_source"] == (
        "local_estimate" if mode == "local" else "provider_usage_plus_estimate"
    )
    assert history == original
