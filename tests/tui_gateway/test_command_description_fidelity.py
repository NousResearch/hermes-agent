"""Command descriptions survive both catalog and completion transport."""

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document


def test_full_descriptions_survive_catalog_and_completion(monkeypatch):
    from agent import skill_commands
    from hermes_cli import plugins
    from hermes_cli.commands_completion import SlashCommandCompleter
    from tui_gateway import server

    description = "Read the entire description before selecting a command. " * 8
    skills = {"/proof-skill": {"name": "proof-skill", "description": description}}
    monkeypatch.setattr(skill_commands, "scan_skill_commands", lambda: skills)
    monkeypatch.setattr(plugins, "get_plugin_commands", lambda: {"proof-plugin": {"description": description}})
    monkeypatch.setattr(server, "_load_cfg", lambda: {"quick_commands": {"proof-quick": {"description": description}}})
    catalog = server._methods["commands.catalog"](1, {})["result"]
    pairs = dict(catalog["pairs"])
    for name in ("/proof-skill", "/proof-plugin", "/proof-quick"):
        assert pairs[name] == description
    completer = SlashCommandCompleter(skill_commands_provider=lambda: skills)
    completions = list(completer.get_completions(Document("/proof"), CompleteEvent()))
    assert len(completions) == 2
    assert all(description in completion.display_meta_text for completion in completions)


def test_exact_skill_is_canonical_alongside_longer_quick_alias(tmp_path, monkeypatch):
    from agent import skill_commands
    from tools import skills_tool
    from tui_gateway import server

    home = tmp_path / "profile"
    skill = home / "skills" / "proof-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: proof-skill\ndescription: Catalog regression probe.\n---\n\n# Probe\n"
    )
    (home / "config.yaml").write_text(
        "quick_commands:\n  proof-skill-alias:\n    type: alias\n    target: /proof-skill\n"
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    sid = "skill-canonical-probe"
    monkeypatch.setitem(server._sessions, sid, {
        "session_key": sid, "agent": None, "profile_home": str(home), "cwd": str(tmp_path),
    })

    catalog = server._methods["commands.catalog"](1, {"session_id": sid})["result"]
    assert "/proof-skill" in catalog["skills"]
    assert catalog["canon"].get("/proof-skill") == "/proof-skill"
    assert catalog["canon"]["/proof-skill-alias"] == "/proof-skill-alias"
    dispatched = server._methods["command.dispatch"](
        2, {"session_id": sid, "name": "proof-skill", "arg": "probe"}
    )["result"]
    assert dispatched["type"] == "skill"
    assert "# Probe" in dispatched["message"]


@pytest.mark.parametrize("owner", ["registry", "quick", "plugin"])
def test_skill_canon_preserves_existing_command_owner(owner, monkeypatch):
    from agent import skill_commands
    from hermes_cli import plugins
    from tui_gateway import server

    key = "/reset" if owner == "registry" else "/proof-skill"
    expected = "/new" if owner == "registry" else "/Proof-Skill"
    monkeypatch.setattr(skill_commands, "get_interactive_skill_commands", lambda: {
        key: {"name": key[1:], "description": "Collision probe"},
    })
    monkeypatch.setattr(server, "_load_cfg", lambda: {
        "quick_commands": {"Proof-Skill": {"type": "alias", "target": "/help"}}
        if owner == "quick" else {},
    })
    monkeypatch.setattr(plugins, "get_plugin_commands", lambda: {
        "Proof-Skill": {"description": "Plugin probe"},
    } if owner == "plugin" else {})

    catalog = server._methods["commands.catalog"](1, {})["result"]
    assert catalog["canon"][key] == expected


def test_skill_canon_follows_session_profile_a_b_a(tmp_path, monkeypatch):
    from agent import skill_commands
    from tools import skills_tool
    from tui_gateway import server

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path / "no-local-skills")
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    for name in ("proof-a", "proof-b"):
        home = tmp_path / name
        skill = home / "external" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Profile isolation probe.\n---\n\n# Probe\n"
        )
        (home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {skill.parent}\n")
        monkeypatch.setitem(server._sessions, name, {
            "session_key": name, "agent": None, "profile_home": str(home), "cwd": str(tmp_path),
        })

    for own, other in (("proof-a", "proof-b"), ("proof-b", "proof-a"), ("proof-a", "proof-b")):
        catalog = server._methods["commands.catalog"](1, {"session_id": own})["result"]
        assert catalog["canon"][f"/{own}"] == f"/{own}"
        assert f"/{other}" not in catalog["canon"]
