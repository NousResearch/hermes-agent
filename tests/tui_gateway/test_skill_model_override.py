"""Skill model routing through the TUI/Desktop gateway."""

from unittest.mock import patch


def test_command_dispatch_returns_and_queues_skill_model_config(monkeypatch):
    import agent.skill_commands as skills
    import tui_gateway.server as server

    session = {"session_key": "session-key"}
    monkeypatch.setitem(server._sessions, "sid", session)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_resolve_name", lambda name: name)
    monkeypatch.setattr(server, "_skill_scaffold_projection", lambda _: "/work do it")
    monkeypatch.setattr(
        skills, "scan_skill_commands", lambda: {"/work": {"name": "work"}}
    )
    monkeypatch.setattr(
        skills, "build_skill_invocation_message", lambda *args, **kwargs: "expanded"
    )
    monkeypatch.setattr(
        skills,
        "get_skill_model_config",
        lambda key: {"provider": "openrouter", "model": "openai/gpt-5.5"},
    )

    response = server._methods["command.dispatch"](
        "rid", {"session_id": "sid", "name": "work", "arg": "do it"}
    )

    assert response["result"]["model_config"] == {
        "provider": "openrouter",
        "model": "openai/gpt-5.5",
    }
    assert session["pending_skill_model_override"] == response["result"]["model_config"]


def test_pending_skill_model_override_is_one_turn(monkeypatch):
    import tui_gateway.server as server

    session = {
        "agent": object(),
        "pending_skill_model_override": {
            "provider": "openrouter",
            "model": "openai/gpt-5.5",
        },
    }

    with patch.object(
        server, "_apply_model_switch", return_value={"confirm_required": False}
    ) as apply:
        server._apply_pending_skill_model_override("sid", session)

    assert "pending_skill_model_override" not in session
    apply.assert_called_once_with(
        "sid",
        session,
        "openai/gpt-5.5 --once --provider openrouter",
        confirm_expensive_model=True,
        pin_session_override=False,
        persist_override=False,
    )
