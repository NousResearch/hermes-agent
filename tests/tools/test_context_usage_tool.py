import json
from types import SimpleNamespace

import tools.context_usage_tool as context_usage_tool


class _Compressor(SimpleNamespace):
    def can_compress(self):
        return True


def _agent(**overrides):
    agent = SimpleNamespace(
        session_prompt_tokens=120,
        session_total_tokens=180,
        session_id="old-session",
        _cached_system_prompt="system prompt",
        context_compressor=_Compressor(
            context_length=1_000,
            last_prompt_tokens=400,
            last_real_prompt_tokens=350,
            last_compression_rough_tokens=0,
            awaiting_real_usage_after_compression=False,
        ),
        tools=None,
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def test_context_status_reports_post_compression_rough_estimate(monkeypatch):
    monkeypatch.setattr(
        context_usage_tool,
        "_read_compression_config",
        lambda _agent: {
            "enabled": True,
            "threshold": 0.5,
            "allow_agent_trigger": True,
            "agent_suggest_threshold": 0.75,
        },
    )
    agent = _agent(
        context_compressor=_Compressor(
            context_length=1_000,
            last_prompt_tokens=-1,
            last_real_prompt_tokens=900,
            last_compression_rough_tokens=123,
            awaiting_real_usage_after_compression=True,
        )
    )

    result = json.loads(context_usage_tool.context_status_handler({}, agent=agent))

    assert result["tokens_used"] == 123
    assert result["percent_used"] == 12.3
    assert result["compression_threshold_pct"] == 50.0
    assert result["allow_agent_trigger"] is True
    assert result["agent_suggest_threshold"] == 0.75


def test_request_compression_is_hidden_unless_config_allows(monkeypatch):
    monkeypatch.setattr(
        context_usage_tool,
        "_read_compression_config",
        lambda _agent: {"enabled": True, "allow_agent_trigger": False},
    )

    assert context_usage_tool._check_context_status_reqs() is True
    assert context_usage_tool._check_request_compression_reqs() is False

    result = json.loads(
        context_usage_tool.request_compression_handler(
            {"reason": "too long"},
            agent=_agent(),
            messages=[{"role": "user", "content": "hello"}],
        )
    )
    assert result["compressed"] is False
    assert "allow_agent_trigger" in result["error"]


def test_request_compression_mutates_live_messages_and_marks_loop_rebase(monkeypatch):
    monkeypatch.setattr(
        context_usage_tool,
        "_read_compression_config",
        lambda _agent: {
            "enabled": True,
            "allow_agent_trigger": True,
            "agent_suggest_threshold": 0.5,
        },
    )
    estimates = iter([1_000, 250])
    monkeypatch.setattr(
        context_usage_tool,
        "_estimate_request_tokens",
        lambda *_args, **_kwargs: next(estimates),
    )

    live_messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    compressed_messages = [{"role": "user", "content": "summary"}]

    def _compress_context(messages, system_message, **kwargs):
        assert messages is live_messages
        assert system_message is None
        assert kwargs["task_id"] == "task-1"
        assert kwargs["force"] is True
        assert kwargs["aggressive"] is True
        assert kwargs["focus_topic"] == "Agent-requested: preparing for poll loop"
        agent.session_id = "new-session"
        return compressed_messages, "new system prompt"

    agent = _agent()
    agent._compress_context = _compress_context

    result = json.loads(
        context_usage_tool.request_compression_handler(
            {"reason": "preparing for poll loop", "force": True},
            agent=agent,
            messages=live_messages,
            task_id="task-1",
        )
    )

    assert result["compressed"] is True
    assert result["applied"] is True
    assert result["tokens_saved"] == 750
    assert result["messages_before"] == 3
    assert result["messages_after"] == 1
    assert result["old_session_id"] == "old-session"
    assert result["new_session_id"] == "new-session"
    assert live_messages == compressed_messages
    assert agent._session_messages is live_messages
    assert agent.conversation_history is live_messages
    assert agent._cached_system_prompt == "new system prompt"
    assert agent._request_compression_applied is True


def test_request_compression_non_force_preserves_normal_compression_cooldown(monkeypatch):
    monkeypatch.setattr(
        context_usage_tool,
        "_read_compression_config",
        lambda _agent: {
            "enabled": True,
            "allow_agent_trigger": True,
            "agent_suggest_threshold": None,
        },
    )
    estimates = iter([1_000, 800])
    monkeypatch.setattr(
        context_usage_tool,
        "_estimate_request_tokens",
        lambda *_args, **_kwargs: next(estimates),
    )

    live_messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    compressed_messages = [{"role": "user", "content": "summary"}]

    def _compress_context(messages, system_message, **kwargs):
        assert messages is live_messages
        assert system_message is None
        assert kwargs["force"] is False
        assert kwargs["aggressive"] is False
        assert kwargs["focus_topic"] == "Agent-requested: routine cleanup"
        return compressed_messages, "system prompt"

    agent = _agent()
    agent._compress_context = _compress_context

    result = json.loads(
        context_usage_tool.request_compression_handler(
            {"reason": "routine cleanup"},
            agent=agent,
            messages=live_messages,
            task_id="task-1",
        )
    )

    assert result["compressed"] is True
    assert result["messages_before"] == 2
    assert result["messages_after"] == 1


def test_request_compression_preserves_inflight_tool_call_group(monkeypatch):
    monkeypatch.setattr(
        context_usage_tool,
        "_read_compression_config",
        lambda _agent: {
            "enabled": True,
            "allow_agent_trigger": True,
            "agent_suggest_threshold": None,
        },
    )
    estimates = iter([2_000, 500])
    monkeypatch.setattr(
        context_usage_tool,
        "_estimate_request_tokens",
        lambda *_args, **_kwargs: next(estimates),
    )

    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-status",
                "type": "function",
                "function": {"name": "context_status", "arguments": "{}"},
            },
            {
                "id": "call-compress",
                "type": "function",
                "function": {"name": "request_compression", "arguments": "{}"},
            },
        ],
    }
    prior_tool_result = {
        "role": "tool",
        "tool_call_id": "call-status",
        "name": "context_status",
        "content": "{}",
    }
    live_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "please work"},
        assistant_tool_call,
        prior_tool_result,
    ]
    compressed_prefix = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "summary"},
    ]

    def _compress_context(messages, system_message, **kwargs):
        assert messages == live_messages[:2]
        assert all(not message.get("tool_calls") for message in messages)
        assert system_message is None
        assert kwargs["force"] is False
        assert kwargs["aggressive"] is False
        agent.session_id = "new-session"
        return compressed_prefix, "new system prompt"

    agent = _agent()
    agent._compress_context = _compress_context

    result = json.loads(
        context_usage_tool.request_compression_handler(
            {"reason": "before long wait"},
            agent=agent,
            messages=live_messages,
            task_id="task-1",
            tool_call_id="call-compress",
        )
    )

    assert result["compressed"] is True
    assert result["protected_inflight_tool_messages"] == 2
    assert live_messages == compressed_prefix + [assistant_tool_call, prior_tool_result]
    assert not any(
        message.get("role") == "tool" and message.get("tool_call_id") == "call-compress"
        for message in live_messages
    )


def test_request_compression_error_response_escapes_exception_text(monkeypatch):
    monkeypatch.setattr(
        context_usage_tool,
        "_read_compression_config",
        lambda _agent: {"enabled": True, "allow_agent_trigger": True},
    )

    def _compress_context(*_args, **_kwargs):
        raise RuntimeError('bad "quote"\nand newline')

    agent = _agent()
    agent._compress_context = _compress_context

    result = json.loads(
        context_usage_tool.request_compression_handler(
            {"reason": "test escaping"},
            agent=agent,
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert result["compressed"] is False
    assert 'bad "quote"' in result["error"]
