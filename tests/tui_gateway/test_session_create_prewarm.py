"""``session.create`` with ``prewarm: false`` skips the agent pre-warm; first use still builds it.

The dashboard Chat sidebar's sidecar never runs a turn, yet every page load and chat switch pre-warmed a
full AIAgent for it, including external memory providers (with Honcho, a dialectic LLM call each time).
"""

from tui_gateway import server


def test_prewarm_false_defers_the_agent_to_first_use(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: claude-opus-5\n  provider: anthropic\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_profile_home", lambda *a: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *a: None)
    prewarmed, built = [], []
    monkeypatch.setattr(
        server, "_schedule_agent_build", lambda sid, *a: prewarmed.append(sid)
    )
    monkeypatch.setattr(
        server, "_start_agent_build", lambda sid, session: built.append(sid)
    )

    response = server._methods["session.create"](
        "r1",
        {"cols": 80, "close_on_disconnect": True, "source": "tool", "prewarm": False},
    )
    sid = response["result"]["session_id"]
    assert prewarmed == [] and built == []

    session, err = server._sess_building({"session_id": sid}, "r2")
    assert err is None and session is server._sessions[sid]
    assert built == [sid]
