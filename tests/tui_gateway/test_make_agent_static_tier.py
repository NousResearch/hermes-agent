"""``tui_gateway.server._make_agent`` pins a static fast tier's request_overrides at build time.

Regression for #135672: a Desktop/TUI composer pick reached the agent as ``service_tier`` but the
matching fast-mode request override (Anthropic ``speed`` / OpenAI ``service_tier``) never reached
``agent.request_overrides``, so every request of the session went out at normal speed while the
session recorded fast. ``_make_agent`` is the one build every TUI/Desktop surface (session.create,
deferred build, compute host) goes through.
"""

from __future__ import annotations

import types

from tui_gateway import server


def _capture_build(monkeypatch, *, model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1",
                   request_overrides=None, **make_kwargs):
    """Drive the real ``_make_agent`` build and capture the AIAgent construction kwargs."""
    captured = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs.get("model"))

    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *_a: (
        model, {"provider": provider, "base_url": base_url, "request_overrides": request_overrides or {}}))
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})
    server._make_agent("sid", "session-key", **make_kwargs)
    return captured


def test_composer_pick_pins_the_static_tier_over_the_provider_body(monkeypatch):
    agent_kwargs = _capture_build(monkeypatch, request_overrides={"extra_body": {"keep": 1}},
                                  service_tier_override="priority")
    assert agent_kwargs["service_tier"] == "priority"
    assert agent_kwargs["request_overrides"] == {"extra_body": {"keep": 1}, "service_tier": "priority"}


def test_config_default_static_tier_pins_the_same_value_as_service_tier(monkeypatch):
    """No composer pick: the profile default is the tier, and both the agent and its request body
    carry it."""
    monkeypatch.setattr(server, "_load_service_tier", lambda: "priority")
    agent_kwargs = _capture_build(monkeypatch)
    assert agent_kwargs["service_tier"] == "priority"
    assert agent_kwargs["request_overrides"] == {"service_tier": "priority"}


def test_normal_and_bounded_tiers_leave_the_request_body_alone(monkeypatch):
    for tier in ("", "auto", "cold"):
        agent_kwargs = _capture_build(monkeypatch, request_overrides={"extra_body": {"keep": 1}},
                                      service_tier_override=tier)
        assert agent_kwargs["service_tier"] == tier
        assert agent_kwargs["request_overrides"] == {"extra_body": {"keep": 1}}


def test_static_tier_on_a_route_without_fast_capacity_pins_nothing(monkeypatch):
    agent_kwargs = _capture_build(monkeypatch, provider="openrouter", base_url="https://openrouter.ai/api/v1",
                                  service_tier_override="priority")
    assert agent_kwargs["service_tier"] == "priority"
    assert "service_tier" not in agent_kwargs["request_overrides"]


def test_fast_off_removes_build_time_pins_and_preserves_provider_body(monkeypatch):
    agent_kwargs = _capture_build(monkeypatch, request_overrides={"extra_body": {"keep": 1}},
                                  service_tier_override="priority")
    agent = types.SimpleNamespace(**agent_kwargs)
    session = {"agent": agent}
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda *_a: None)
    monkeypatch.setattr(server, "_emit_session_info", lambda *_a: None)
    server._set_fast(1, {"session_id": "sid"}, "fast", "off", session)
    assert agent.service_tier is None
    assert agent.request_overrides == {"extra_body": {"keep": 1}}
    assert session["create_service_tier_override"] == ""
