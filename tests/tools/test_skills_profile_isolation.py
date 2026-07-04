import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from hermes_constants import get_hermes_home, get_skills_dir
from tools.skills_tool import get_skills_dir as tool_get_skills_dir
from tools.skill_manager_tool import get_skills_dir as manager_get_skills_dir
from agent.skill_commands import get_skill_commands, scan_skill_commands

@pytest.fixture
def clean_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    
    # Create profiles directory and files
    profiles_dir = hermes_home / "profiles"
    profiles_dir.mkdir()
    
    for name in ("profileA", "profileB"):
        pdir = profiles_dir / name
        pdir.mkdir()
        (pdir / "skills").mkdir()
        (pdir / "skills" / f"skill_{name}").mkdir(parents=True)
        (pdir / "skills" / f"skill_{name}" / "SKILL.md").write_text(
            f"---\nname: skill_{name}\ndescription: I am {name}\n---\nbody of {name}", 
            encoding="utf-8"
        )

    # Set up global skill
    global_skills = hermes_home / "skills"
    global_skills.mkdir()
    (global_skills / "skill_main").mkdir()
    (global_skills / "skill_main" / "SKILL.md").write_text(
        "---\nname: skill_main\ndescription: I am main\n---\nbody of main", 
        encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    with patch("gateway.run._hermes_home", hermes_home):
        yield hermes_home

def test_skills_dir_resolves_dynamically(clean_hermes_home):
    """Verify that get_skills_dir and tool-specific skills dirs resolve dynamically."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    
    # Check default path
    assert get_skills_dir().resolve() == (clean_hermes_home / "skills").resolve()
    assert tool_get_skills_dir().resolve() == (clean_hermes_home / "skills").resolve()
    assert manager_get_skills_dir().resolve() == (clean_hermes_home / "skills").resolve()
    
    # Override
    profile_home = clean_hermes_home / "profiles" / "profileA"
    token = set_hermes_home_override(profile_home)
    try:
        assert get_skills_dir().resolve() == (profile_home / "skills").resolve()
        assert tool_get_skills_dir().resolve() == (profile_home / "skills").resolve()
        assert manager_get_skills_dir().resolve() == (profile_home / "skills").resolve()
    finally:
        reset_hermes_home_override(token)
        
    assert get_skills_dir().resolve() == (clean_hermes_home / "skills").resolve()

def test_skill_commands_isolation_across_profiles(clean_hermes_home):
    """Verify that skill commands dynamically refresh and isolate based on active profile."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    
    # Initially main skills
    cmds = get_skill_commands()
    assert "/skill-main" in cmds
    assert "/skill-profilea" not in cmds
    assert "/skill-profileb" not in cmds

    # Switch to profileA
    profile_home_a = clean_hermes_home / "profiles" / "profileA"
    token_a = set_hermes_home_override(profile_home_a)
    try:
        cmds_a = get_skill_commands()
        assert "/skill-profilea" in cmds_a
        assert "/skill-main" not in cmds_a
        assert "/skill-profileb" not in cmds_a
    finally:
        reset_hermes_home_override(token_a)

    # Switch to profileB
    profile_home_b = clean_hermes_home / "profiles" / "profileB"
    token_b = set_hermes_home_override(profile_home_b)
    try:
        cmds_b = get_skill_commands()
        assert "/skill-profileb" in cmds_b
        assert "/skill-main" not in cmds_b
        assert "/skill-profilea" not in cmds_b
    finally:
        reset_hermes_home_override(token_b)

    # Back to main
    cmds_final = get_skill_commands()
    assert "/skill-main" in cmds_final
    assert "/skill-profilea" not in cmds_final
    assert "/skill-profileb" not in cmds_final
