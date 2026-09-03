"""Build-time fast-tier bridging in ``tui_gateway`` ``_make_agent``.

`agent.service_tier: fast` in config.yaml (and session-pinned
``create_service_tier_override`` from ``session.create fast=true`` or the
``config.set fast`` RPC) only ever reached ``AIAgent(service_tier=...)``.
Every transport reads fast keys from ``agent.request_overrides`` —
``agent/chat_completion_helpers.py`` gates ``fast_mode`` on
``(agent.request_overrides or {}).get("speed") == "fast"`` and forwards
``request_overrides=agent.request_overrides`` — so the configured tier was
inert on every rebuilt session (deferred build, DB resume, ``/new``) while
``session.info`` still reported fast enabled. The classic CLI
(``hermes_cli/cli_agent_setup_mixin.py``) and the messaging gateway
(``gateway/run.py``) bridge the tier through ``resolve_fast_mode_overrides``
at build time; these tests pin the same contract for the desktop/TUI
backend.
"""

import types

import pytest

import tui_gateway.server as server


def _make_agent_capturing(monkeypatch, *, cfg, model, runtime, **make_kwargs):
    """Run ``server._make_agent`` with a FakeAgent capturing ctor kwargs."""
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    merged_runtime = {
        "provider": "openai",
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": [],
        "credential_pool": None,
    }
    merged_runtime.update(runtime or {})

    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: (model, merged_runtime["provider"]))
    monkeypatch.setattr(
        server,
        "_resolve_runtime_with_fallback",
        lambda _kw: types.SimpleNamespace(
            runtime=merged_runtime, used_fallback=False, selected_model=None
        ),
    )
    monkeypatch.setattr(server, "_load_reasoning_config", lambda _m=None: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda _p=None: set())
    monkeypatch.setattr(server, "_resolve_agent_platform", lambda _p=None: "cli")
    monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    monkeypatch.setattr(server, "_load_fallback_model", lambda: None)
    monkeypatch.setattr(server, "_parse_tui_skills_env", lambda: "")
    monkeypatch.setattr(
        "hermes_cli.config.resolve_ephemeral_system_prompt_from_config", lambda _cfg: None
    )
    monkeypatch.setattr(
        "tui_gateway.synthetic_turn.maybe_build_synthetic_agent", lambda _key, _mo: None
    )
    monkeypatch.setattr("hermes_cli.mcp_startup.wait_for_mcp_discovery", lambda: None)
    monkeypatch.setattr("tui_gateway.entry.wait_for_mcp_discovery", lambda: None)
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    agent = server._make_agent("sid", "key", context_cwd_is_launch_artifact=False, **make_kwargs)
    return agent, captured


def test_configured_fast_tier_reaches_request_overrides(monkeypatch):
    """config.yaml `agent.service_tier: fast` must bridge onto request_overrides."""
    agent, captured = _make_agent_capturing(
        monkeypatch,
        cfg={"agent": {"service_tier": "fast"}},
        model="gpt-5.6",
        runtime={"provider": "openai"},
    )
    assert captured["service_tier"] == "priority"
    assert captured["request_overrides"] == {"service_tier": "priority"}
    assert agent is not None


def test_session_pinned_priority_reaches_request_overrides(monkeypatch):
    """A pinned tier (session.create fast=true, stored row resume) bridges too."""
    _agent, captured = _make_agent_capturing(
        monkeypatch,
        cfg={},
        model="gpt-5.6",
        runtime={"provider": "openai"},
        service_tier_override="priority",
    )
    assert captured["service_tier"] == "priority"
    assert captured["request_overrides"] == {"service_tier": "priority"}


def test_pinned_normal_keeps_overrides_clear(monkeypatch):
    """The explicit normal pin (`""`) must not resurrect stale overrides."""
    _agent, captured = _make_agent_capturing(
        monkeypatch,
        cfg={"agent": {"service_tier": "fast"}},
        model="gpt-5.6",
        runtime={"provider": "openai"},
        service_tier_override="",
    )
    assert captured["service_tier"] == ""
    assert captured["request_overrides"] is None


def test_auto_tier_not_pinned_into_overrides(monkeypatch):
    """auto/cold windows are applied per request by agent.fast_mode, never pinned."""
    _agent, captured = _make_agent_capturing(
        monkeypatch,
        cfg={"agent": {"service_tier": "auto"}},
        model="gpt-5.6",
        runtime={"provider": "openai"},
    )
    assert captured["service_tier"] == "auto"
    assert captured["request_overrides"] is None


def test_normal_tier_keeps_overrides_clear(monkeypatch):
    _agent, captured = _make_agent_capturing(
        monkeypatch,
        cfg={"agent": {"service_tier": "normal"}},
        model="gpt-5.6",
        runtime={"provider": "openai"},
    )
    assert captured["service_tier"] is None
    assert captured["request_overrides"] is None


def test_anthropic_fast_model_gets_speed_override(monkeypatch):
    _agent, captured = _make_agent_capturing(
        monkeypatch,
        cfg={"agent": {"service_tier": "fast"}},
        model="claude-opus-4.8",
        runtime={
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
        },
    )
    assert captured["service_tier"] == "priority"
    assert captured["request_overrides"] == {"speed": "fast"}


def test_unsupported_route_fails_open(monkeypatch):
    """A route that must not see fast params (OpenRouter) builds the agent
    without overrides rather than failing the whole session build."""
    _agent, captured = _make_agent_capturing(
        monkeypatch,
        cfg={"agent": {"service_tier": "fast"}},
        model="gpt-5.6",
        runtime={
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
        },
    )
    assert captured["service_tier"] == "priority"
    assert captured["request_overrides"] is None


def test_resolve_failure_fails_open(monkeypatch):
    """A crashing resolve must not take the session build down with it."""
    import hermes_cli.models

    def _boom(*_a, **_k):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(hermes_cli.models, "resolve_fast_mode_overrides", _boom)
    _agent, captured = _make_agent_capturing(
        monkeypatch,
        cfg={"agent": {"service_tier": "fast"}},
        model="gpt-5.6",
        runtime={"provider": "openai"},
    )
    assert captured["service_tier"] == "priority"
    assert captured["request_overrides"] is None


if __name__ == "__main__":
    pytest.main([__file__])
