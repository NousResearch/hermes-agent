"""The Desktop context RPC uses route-aware estimates (#134915)."""

from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

from agent.model_metadata import estimate_messages_tokens_rough
from tui_gateway import server


def test_context_rpc_excludes_stale_reasoning_without_changing_history(monkeypatch):
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer", "reasoning": "stored " * 1000},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "latest", "reasoning": "new thought"},
    ]
    original = deepcopy(history)
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="custom", model="local-test",
        base_url="http://localhost:9999/v1", tools=[], _memory_store=None,
        _usage_anchor=None, _turn_base_usage_anchor=None,
        context_compressor=SimpleNamespace(context_length=200_000, last_prompt_tokens=0),
    )
    session = {"agent": agent, "history": history, "history_lock": Lock(),
               "session_key": "reasoning-test", "running": False}
    monkeypatch.setattr(server, "_sessions", {"reasoning-test": session})
    monkeypatch.setattr("agent.system_prompt.build_system_prompt_parts", lambda agent: {})
    monkeypatch.setattr("agent.context_file_sources.context_file_sources_for_agent", lambda agent: [])

    response = server._methods["session.context_breakdown"]("test", {"session_id": "reasoning-test"})

    assert "error" not in response, response
    payload = response["result"]
    expected = deepcopy(history)
    del expected[1]["reasoning"]
    conversation = next(row["tokens"] for row in payload["categories"] if row["id"] == "conversation")
    assert conversation == estimate_messages_tokens_rough(expected)
    assert payload["context_used"] == conversation
    assert payload["context_source"] == "local_estimate"
    assert history == original
