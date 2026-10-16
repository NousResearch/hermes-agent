"""TUI launch skills have the same stable prompt placement as CLI launch skills."""

from tui_gateway import server


def test_startup_separates_personality_and_skill_bytes(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.resolve_ephemeral_system_prompt_from_config", lambda cfg: "PERSONALITY")
    monkeypatch.setenv("HERMES_TUI_SKILLS", "retained")
    monkeypatch.setattr("agent.skill_commands.build_preloaded_skills_prompt",
                        lambda names, task_id: ("EXACT SKILL BODY", names, []))
    assert server._startup_system_prompt({}, "session") == ("PERSONALITY", "EXACT SKILL BODY")


def test_startup_without_skills_keeps_personality(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.resolve_ephemeral_system_prompt_from_config", lambda cfg: "PERSONALITY")
    monkeypatch.delenv("HERMES_TUI_SKILLS", raising=False)
    assert server._startup_system_prompt({}, "session") == ("PERSONALITY", None)


def test_personality_change_keeps_real_tui_skill_in_stable_prefix(monkeypatch):
    import threading
    from agent.system_prompt import build_system_prompt
    from agent.turn_context import build_api_messages
    from hermes_constants import get_hermes_home
    from tests.agent.test_system_prompt import _make_agent

    skill = get_hermes_home() / "skills" / "launch-probe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: launch-probe\ndescription: Test startup skill.\n---\n\nTUI STARTUP GUIDANCE", encoding="utf-8")
    monkeypatch.setenv("HERMES_TUI_SKILLS", "launch-probe,launch-probe")
    personality, preload = server._startup_system_prompt({"agent": {"system_prompt": "OLD PERSONA"}}, "tui-session")
    agent = _make_agent(preloaded_skills_prompt=preload, ephemeral_system_prompt=personality,
                        _current_turn_timestamp=0, platform="tui")
    prompt = build_system_prompt(agent)
    session = {"agent": agent, "history": [], "history_lock": threading.RLock()}
    monkeypatch.setattr(server, "_session_info", lambda _agent: {})
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    server._apply_personality_to_session("tui-session", session, "NEW PERSONA", "new")
    messages, effective = build_api_messages(
        agent, [], current_turn_user_idx=None, ext_prefetch_cache=None,
        plugin_user_context=None, moa_config=None, active_system_prompt=prompt,
    )
    assert effective.count("TUI STARTUP GUIDANCE") == 1
    assert "TUI STARTUP GUIDANCE" in agent._cached_system_prompt_static
    assert "OLD PERSONA" not in effective
    assert effective.endswith("NEW PERSONA")
    assert messages[0]["content"] == effective
