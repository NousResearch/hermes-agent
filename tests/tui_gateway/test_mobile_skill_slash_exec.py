"""Tests for mobile slash.exec skill routing and prompt.submit skill expansion (CF-48)."""

from tui_gateway import server


def test_mobile_slash_exec_skill_returns_send_directive_while_desktop_keeps_4018(tmp_path, monkeypatch):
    from agent import skill_commands, skill_utils
    from tools import skills_tool

    skills_dir = tmp_path / "skills"
    (skills_dir / "wrap").mkdir(parents=True)
    (skills_dir / "wrap" / "SKILL.md").write_text(
        "---\nname: wrap\ndescription: Close out the session\n---\n\n# Wrap\n\nRUN THE WRAP CHECKLIST.\n"
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda *a, **k: [])
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    skill_commands.scan_skill_commands()

    monkeypatch.setattr(
        server,
        "_sessions",
        {
            "sid-mobile": {"session_key": "k-mobile", "source": "mobile"},
            "sid-desktop": {"session_key": "k-desktop", "source": "desktop"},
        },
    )

    # Desktop/TUI session keeps the 4018 gate so its client calls command.dispatch
    desktop_res = server._methods["slash.exec"]("r1", {"session_id": "sid-desktop", "command": "/wrap"})
    assert "error" in desktop_res
    assert desktop_res["error"]["code"] == 4018
    assert "use command.dispatch for /wrap" in desktop_res["error"]["message"]

    # Mobile session gets {type: "send", message: "/wrap"} so HermesViewModel.applySlashResult
    # submits "/wrap" cleanly (keeping the bubble "/wrap") and prompt.submit expands it on the server.
    mobile_res = server._methods["slash.exec"]("r2", {"session_id": "sid-mobile", "command": "/wrap now"})
    assert "result" in mobile_res, mobile_res
    assert mobile_res["result"]["type"] == "send"
    assert mobile_res["result"]["message"] == "/wrap now"
    assert "wrap" in mobile_res["result"]["notice"]

    expanded = server._expand_skill_invocation_for_replay(mobile_res["result"]["message"], "k-mobile")
    assert "RUN THE WRAP CHECKLIST." in expanded
    assert server._skill_scaffold_projection(expanded) == "/wrap now"
