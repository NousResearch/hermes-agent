"""In-process TUI gateway: a mid-turn route change re-emits ``session.info`` for the session (#78613).

A ``hermes serve`` / dashboard backend runs the agent in this process, and its status bar reads
``session.info.model``. ``_make_agent`` installs ``route_change_callback`` so a fallback switch or
primary restore refreshes the label for that session's sid right away.
"""

import types

from agent.route_binding import notify_route_change
from tui_gateway import server


def _built_agent(monkeypatch):
    def fake_agent(**kwargs):
        return types.SimpleNamespace(model=kwargs.get("model"), provider=kwargs.get("provider"))

    for name in ("HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER", "HERMES_DESKTOP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "claude-opus-5-5", "provider": "anthropic"}})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": "anthropic", "base_url": "https://api.anthropic.com", "api_key": "token",
            "api_mode": "anthropic_messages", "credential_pool": None,
        },
    )
    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform=None: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)
    return server._make_agent("sid-1", "session-key")


def test_a_route_change_emits_one_session_info_for_the_sid_on_the_new_model(monkeypatch):
    agent = _built_agent(monkeypatch)
    session = {"agent": agent, "session_key": "session-key"}
    monkeypatch.setitem(server._sessions, "sid-1", session)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    monkeypatch.setattr(server, "_session_info", lambda a, s=None: {"model": a.model, "same_session": s is session})
    agent.model, agent.provider = "global.anthropic.claude-opus-5-5", "bedrock"
    assert notify_route_change(agent) is True
    assert emitted == [("session.info", "sid-1", {"model": "global.anthropic.claude-opus-5-5", "same_session": True})]
