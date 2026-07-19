import tui_gateway.server as srv


def test_tui_startup_skills_merge_profile_defaults_before_env(monkeypatch):
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    monkeypatch.setenv("HERMES_TUI_SKILLS", "extra-skill,profile-skill")

    startup_skills, explicit_skills = srv._resolve_tui_startup_skills(
        {"skills": {"defaults": ["profile-skill", "profile-helper"]}}
    )

    assert startup_skills == ["profile-skill", "profile-helper", "extra-skill"]
    assert explicit_skills == ["extra-skill", "profile-skill"]


def test_tui_ignore_rules_suppresses_profile_default_skills(monkeypatch):
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    monkeypatch.setenv("HERMES_TUI_SKILLS", "extra-skill")

    startup_skills, explicit_skills = srv._resolve_tui_startup_skills(
        {"skills": {"defaults": ["profile-skill"]}}
    )

    assert startup_skills == ["extra-skill"]
    assert explicit_skills == ["extra-skill"]
