"""#119195: a Desktop/TUI session built on a pre-agent fallback returns to the configured primary.

When the configured primary cannot resolve at session start, ``_make_agent`` builds the agent on a
fallback entry. The per-turn config sync baselines on the configured model, so it never sees a
change, and the agent's own primary restore returns to the runtime it was built on (the fallback):
the session stayed on the fallback until the backend restarted, long after the primary's quota
reset. Turn start now switches back once the primary is usable again."""

import time
import types

import pytest

from agent.credential_pool import AUTH_TYPE_API_KEY, STATUS_EXHAUSTED, CredentialPool, PooledCredential
from hermes_cli.auth import AuthError
from tui_gateway import server

PRIMARY = ("claude-opus-5", "anthropic")
FALLBACK = {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}


def _pool(provider, *, benched_for=None):
    entry = PooledCredential(provider=provider, id=f"{provider}-1", label=provider, auth_type=AUTH_TYPE_API_KEY,
                             priority=0, source="manual", access_token=f"sk-{provider}-" + "x" * 32)
    if benched_for is not None:
        now = time.time()
        entry.last_status, entry.last_status_at = STATUS_EXHAUSTED, now
        entry.last_error_code, entry.last_error_reset_at = 429, now + benched_for
    return CredentialPool(provider, [entry])


@pytest.fixture
def desktop(monkeypatch):
    """Config primary anthropic/claude-opus-5 with one OpenRouter fallback; ``state`` flips the primary."""
    state = {"primary_down": True, "switches": []}
    for var in ("HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER", "HERMES_DESKTOP",
                "HERMES_DESKTOP_TERMINAL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {
        "model": {"default": PRIMARY[0], "provider": PRIMARY[1]}, "fallback_providers": [FALLBACK]})

    def resolve(requested=None, target_model=None, **_kw):
        if requested in (None, "anthropic"):  # None = the configured provider
            if state["primary_down"]:
                raise AuthError("No Anthropic credentials found.", provider="anthropic")
            return {"provider": "anthropic", "api_key": "sk-ant", "base_url": "https://api.anthropic.com",
                    "api_mode": "anthropic_messages"}
        return {"provider": requested, "api_key": "sk-or", "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: CredentialPool(provider, []))
    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: types.SimpleNamespace(
        model=kw.get("model"), provider=kw.get("provider")))
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    def fake_switch(sid, session, raw, **kw):
        state["switches"].append((raw, kw))
        model, _, provider = raw.partition(" --provider ")
        session["agent"].model, session["agent"].provider = model, provider

    monkeypatch.setattr(server, "_apply_model_switch", fake_switch)
    return state


def _open_session():
    agent = server._make_agent("sid", "session-key")
    # _finish_agent_build's baseline for the per-turn sync: the CONFIGURED model, not the agent's.
    return {"agent": agent, "session_key": "session-key", "config_model_seen": server._config_model_target()}


def _turn_start(session):
    server._sync_agent_model_with_config("sid", session)
    server._return_to_configured_primary("sid", session)


def test_session_returns_to_the_primary_once_it_resolves_again(desktop):
    session = _open_session()
    assert (session["agent"].provider, session["agent"].model) == ("openrouter", "deepseek/deepseek-v4-pro")
    assert session["agent"]._built_on_pre_agent_fallback is True

    _turn_start(session)  # primary still benched: stay on the fallback, quietly
    assert desktop["switches"] == []

    desktop["primary_down"] = False  # the quota window reset
    _turn_start(session)
    assert desktop["switches"] == [("claude-opus-5 --provider anthropic", {
        "confirm_expensive_model": True, "pin_session_override": False, "persist_override": False})]
    assert (session["agent"].provider, session["agent"].model) == ("anthropic", "claude-opus-5")
    assert session["agent"]._built_on_pre_agent_fallback is False

    _turn_start(session)
    assert len(desktop["switches"]) == 1


def test_config_sync_alone_never_leaves_the_fallback(desktop):
    """The gap: the sync's baseline is the configured model, so it never switches back."""
    session = _open_session()
    desktop["primary_down"] = False
    server._sync_agent_model_with_config("sid", session)
    assert desktop["switches"] == []


def test_primary_resolving_but_pool_benched_is_not_ready(desktop, monkeypatch):
    """A benched credential can still resolve; switching to it would only fail the turn again."""
    session = _open_session()
    desktop["primary_down"] = False
    pools = {"anthropic": _pool("anthropic", benched_for=3600)}
    monkeypatch.setattr("agent.credential_pool.load_pool",
                        lambda provider: pools.get(provider) or CredentialPool(provider, []))
    _turn_start(session)
    assert desktop["switches"] == []

    pools["anthropic"] = _pool("anthropic")
    _turn_start(session)
    assert len(desktop["switches"]) == 1


def test_model_pin_wins(desktop):
    session = _open_session()
    session["model_override"] = {"model": "deepseek/deepseek-v4-pro", "provider": "openrouter"}
    desktop["primary_down"] = False
    _turn_start(session)
    assert desktop["switches"] == []


def test_session_built_on_the_primary_is_untouched(desktop):
    desktop["primary_down"] = False
    session = _open_session()
    assert getattr(session["agent"], "_built_on_pre_agent_fallback", False) is False
    session["agent"].model = "some/other-model"  # e.g. mid-session in-turn fallback; the agent owns that
    _turn_start(session)
    assert desktop["switches"] == []


def test_failed_switch_keeps_the_session_and_retries_next_turn(desktop, monkeypatch):
    session = _open_session()
    desktop["primary_down"] = False
    calls = []

    def failing(sid, sess, raw, **kw):
        calls.append(raw)
        raise ValueError("model switch failed")

    monkeypatch.setattr(server, "_apply_model_switch", failing)
    _turn_start(session)
    _turn_start(session)
    assert len(calls) == 2
    assert session["agent"].provider == "openrouter" and session["agent"]._built_on_pre_agent_fallback is True


def test_turn_start_runs_the_return_after_the_config_sync():
    import inspect

    from tui_gateway import prompt_turn
    source = inspect.getsource(prompt_turn._prepare_turn_input)
    assert 0 < source.index("_sync_agent_model_with_config(sid, session)") < source.index(
        "_return_to_configured_primary(sid, session)")
