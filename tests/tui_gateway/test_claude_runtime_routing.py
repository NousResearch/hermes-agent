"""Provider-neutral runtime routing contracts for TUI and Desktop sessions."""

from unittest.mock import MagicMock

from hermes_cli.auth import AuthError


class _Agent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_tui_auth_fallback_uses_fallback_runtime_and_model(monkeypatch):
    from hermes_cli import runtime_provider
    import run_agent
    from tui_gateway import server

    primary = {
        "default": "gpt-5.4",
        "provider": "openai-codex",
        "runtime": "codex_app_server",
    }
    fallback = {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
        "runtime": "claude_agent_sdk",
    }
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise AuthError("Codex subscription unavailable", provider="openai-codex")
        return {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "api_mode": "anthropic_messages",
            "runtime": "claude_agent_sdk",
            "base_url": "",
            "api_key": "",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"model": primary, "fallback_providers": [fallback]},
    )
    monkeypatch.setattr(
        server,
        "_resolve_startup_runtime",
        lambda: ("gpt-5.4", "openai-codex"),
    )
    monkeypatch.setattr(server, "_load_fallback_model", lambda: [fallback])
    monkeypatch.setattr(server, "_get_db", lambda: MagicMock())
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(run_agent, "AIAgent", _Agent)

    agent = server._make_agent("sid-claude", "key-claude")

    assert calls == [
        {
            "requested": "openai-codex",
            "target_model": "gpt-5.4",
            "route_config": primary,
        },
        {
            "requested": "anthropic",
            "target_model": "claude-opus-4-6",
            "route_config": fallback,
        },
    ]
    assert agent.kwargs["model"] == "claude-opus-4-6"
    assert agent.kwargs["runtime"] == "claude_agent_sdk"
    assert agent.kwargs["fallback_model"] == [fallback]


def test_tui_resumed_session_routes_with_its_persisted_runtime(monkeypatch):
    from hermes_cli import runtime_provider
    import run_agent
    from tui_gateway import server

    override = {
        "model": "claude-opus-4-6",
        "provider": "anthropic",
        "runtime": "claude_agent_sdk",
        "base_url": None,
        "api_mode": "anthropic_messages",
    }
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "api_mode": "anthropic_messages",
            "runtime": "claude_agent_sdk",
            "base_url": "",
            "api_key": "",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {}})
    monkeypatch.setattr(server, "_load_fallback_model", lambda: [])
    monkeypatch.setattr(server, "_get_db", lambda: MagicMock())
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(run_agent, "AIAgent", _Agent)

    agent = server._make_agent(
        "sid-resume", "key-resume", model_override=override
    )

    assert calls == [
        {
            "requested": "anthropic",
            "target_model": "claude-opus-4-6",
            "route_config": override,
        }
    ]
    assert agent.kwargs["runtime"] == "claude_agent_sdk"


def test_tui_session_derivatives_preserve_external_runtime(monkeypatch):
    from types import SimpleNamespace

    from tui_gateway import server

    agent = SimpleNamespace(
        model="claude-opus-4-6",
        provider="anthropic",
        runtime="claude_agent_sdk",
        base_url="",
        api_key="",
        api_mode="anthropic_messages",
        enabled_toolsets=["terminal", "file"],
    )

    persisted = server._runtime_model_config(agent)
    restored = server._stored_session_runtime_overrides(
        {"model": agent.model, "model_config": persisted}
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    background = server._background_agent_kwargs(agent, "background-1")

    assert persisted["runtime"] == "claude_agent_sdk"
    assert restored["model_override"]["runtime"] == "claude_agent_sdk"
    assert background["runtime"] == "claude_agent_sdk"


def test_tui_explicit_alternate_provider_does_not_inherit_claude_runtime(
    monkeypatch,
):
    from hermes_cli import runtime_provider
    import run_agent
    from tui_gateway import server

    primary = {
        "default": "claude-opus-4-6",
        "provider": "anthropic",
        "runtime": "claude_agent_sdk",
    }
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "runtime": "hermes",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "subscription-token",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": primary})
    monkeypatch.setattr(
        server,
        "_resolve_startup_runtime",
        lambda: ("claude-opus-4-6", "anthropic"),
    )
    monkeypatch.setattr(server, "_load_fallback_model", lambda: [])
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(run_agent, "AIAgent", _Agent)

    server._make_agent(
        "sid-codex",
        "key-codex",
        model_override="gpt-5.4",
        provider_override="openai-codex",
    )

    assert calls == [
        {
            "requested": "openai-codex",
            "target_model": "gpt-5.4",
            "route_config": None,
        }
    ]
