"""Plugin-registered skills in the TUI/Desktop slash menu (``/plugin:skill``).

``tui_gateway.server`` is imported at module level, not through a mocked import window: the
profile scope must bind the real ``hermes_constants`` home override for A->B->A to mean anything.
"""

from tui_gateway import server


def test_plugin_skills_reach_tui_slash_menu_and_stack_per_profile(tmp_path, monkeypatch):
    """``ctx.register_skill`` skills are offered and dispatched as ``/plugin:skill`` in the
    TUI/Desktop slash menu, scoped to the session's profile, and a stacked
    ``/plugin:a /plugin:b`` loads both bodies. Native skill menus stay filesystem-only."""
    from agent import skill_commands
    from hermes_cli import plugins

    homes = {"alpha": tmp_path / "profile-a", "beta": tmp_path / "profile-b"}
    for label, home in homes.items():
        plugin = home / "plugins" / f"probe-{label}"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(f"name: probe-{label}\nversion: 0.1.0\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(
            "from pathlib import Path\ndef register(ctx):\n"
            "    for name in ('guide', 'review'):\n"
            "        ctx.register_skill(name, Path(__file__).parent / 'skills' / name / 'SKILL.md')\n", encoding="utf-8")
        for name in ("guide", "review"):
            md = plugin / "skills" / name / "SKILL.md"
            md.parent.mkdir(parents=True)
            md.write_text(f"---\nname: {name}\ndescription: {label} {name}.\n---\n\n{label}-{name}-body\n", encoding="utf-8")
        (home / "config.yaml").write_text(f"plugins:\n  enabled: [probe-{label}]\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(homes["alpha"]))
    plugins._reset_plugin_managers_for_tests()
    try:
        for label in ("alpha", "beta", "alpha"):
            sid, other = f"skill-{label}", "beta" if label == "alpha" else "alpha"
            server._sessions[sid] = {"session_key": sid, "profile_home": str(homes[label]), "agent": None}
            guide, review = f"/probe-{label}:guide", f"/probe-{label}:review"
            catalog = server.handle_request({"id": "c", "method": "commands.catalog", "params": {"session_id": sid}})
            completed = server.handle_request({"id": "t", "method": "complete.slash",
                                               "params": {"session_id": sid, "text": f"/probe-{label}:"}})
            stacked = server.handle_request({"id": "d", "method": "command.dispatch", "params": {
                "session_id": sid, "name": guide[1:], "arg": f"{review} apply it"}})
            assert guide in catalog["result"]["skills"], catalog
            assert f"/probe-{other}:guide" not in catalog["result"]["skills"]
            assert {i["text"] for i in completed["result"]["items"] if i["kind"] == "skill"} >= {
                guide[1:], review[1:]}, completed
            result = stacked["result"]
            assert result["type"] == "skill", stacked
            assert f"{label}-guide-body" in result["message"] and f"{label}-review-body" in result["message"]
            assert "apply it" in result["message"]
            assert result.get("notice", "").startswith("⚡ Loading 2 stacked skills"), result
            assert guide not in skill_commands.get_skill_commands()
    finally:
        for label in homes:
            server._sessions.pop(f"skill-{label}", None)
        plugins._reset_plugin_managers_for_tests()
