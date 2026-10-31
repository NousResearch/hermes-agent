"""A plugin skill loads by its bare name unless an on-disk skill or another plugin claims it (#119976).

Real plugins under a temp ``HERMES_HOME`` go through the real loader, so these cover the
registry, ``skill_view``, the interactive slash table and ``hermes skills list`` together.
"""
import json
from io import StringIO

from rich.console import Console


def _write_plugin(home, plugin, skill):
    plugin_dir = home / "plugins" / plugin
    skill_dir = plugin_dir / "skills" / skill
    skill_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(f"name: {plugin}\nversion: 0.1.0\ndescription: probe\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill}\ndescription: {plugin} {skill}\n---\n\n{plugin} plugin body.\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "def register(ctx):\n"
        f"    ctx.register_skill({skill!r}, Path(__file__).parent / 'skills' / {skill!r} / 'SKILL.md', 'probe')\n",
        encoding="utf-8")


def _home(*plugins, local=None):
    """``plugins`` = ``(plugin, skill)`` pairs, all enabled; ``local`` = an on-disk skill name."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    for plugin, skill in plugins:
        _write_plugin(home, plugin, skill)
    enabled = "".join(f"    - {plugin}\n" for plugin, _ in plugins)
    (home / "config.yaml").write_text(f"plugins:\n  enabled:\n{enabled}", encoding="utf-8")
    if local:
        skill_dir = home / "skills" / local
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {local}\ndescription: on-disk {local}\n---\n\nOn-disk body.\n", encoding="utf-8")
    return home


def _view(name):
    from tools.skills_tool import skill_view

    return json.loads(skill_view(name))


def test_bare_name_loads_the_plugin_skill_and_its_slash_command():
    from agent.skill_commands import build_skill_invocation_message, get_interactive_skill_commands

    _home(("probe", "visible"))

    viewed = _view("visible")
    assert viewed["success"] is True, viewed
    assert viewed["name"] == "probe:visible"
    assert "probe plugin body." in viewed["content"]
    commands = get_interactive_skill_commands()
    assert commands["/visible"]["skill_identifier"] == "probe:visible"
    assert "/probe:visible" in commands  # the qualified spelling keeps working
    assert "probe plugin body." in build_skill_invocation_message("/visible")


def test_an_on_disk_skill_wins_the_bare_name_and_the_listing_says_so():
    from agent.skill_commands import get_interactive_skill_commands
    from hermes_cli.skills_hub import do_list

    _home(("probe", "visible"), local="visible")

    assert "On-disk body." in _view("visible")["content"]
    assert "probe plugin body." in _view("probe:visible")["content"]
    commands = get_interactive_skill_commands()
    assert commands["/visible"].get("source") != "plugin"
    assert commands["/probe:visible"]["source"] == "plugin"
    sink = StringIO()
    do_list(console=Console(file=sink, force_terminal=False, color_system=None, width=300))
    assert "'visible' loads the on-disk skill; use probe:visible" in sink.getvalue()


def test_two_plugins_sharing_a_bare_name_are_never_guessed_between():
    from agent.skill_commands import get_interactive_skill_commands

    _home(("first", "shared"), ("second", "shared"))

    refused = _view("shared")
    assert refused["success"] is False
    assert refused["available_skills"] == ["first:shared", "second:shared"]
    commands = get_interactive_skill_commands()
    assert "/shared" not in commands
    assert {"/first:shared", "/second:shared"} <= set(commands)
