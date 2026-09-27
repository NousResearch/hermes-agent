"""A live completion RPC observes skill mutations without sending a turn."""
from pathlib import Path

import pytest

from tui_gateway import server


def _write_skill(home, name):
    path = home / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: Local fixture.\n---\nFixture body.\n", encoding="utf-8")
    return path


@pytest.fixture
def completion(tmp_path, monkeypatch):
    import agent.skill_commands as commands
    import agent.prompt_builder as prompts

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("skills:\n  external_dirs: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "_sessions", {"fixture": {"cwd": str(tmp_path), "profile_home": str(home)}})
    monkeypatch.setattr(commands, "_skill_commands", {})

    def forbidden(**kwargs):
        pytest.fail("completion must not invalidate system-prompt snapshots")

    monkeypatch.setattr(prompts, "clear_skills_system_prompt_cache", forbidden)

    def skills(sid="fixture"):
        reply = server.handle_request({"id": "r", "method": "complete.slash", "params": {
            "text": "/fixture", "session_id": sid}})
        assert "result" in reply, reply
        return {row["text"] for row in reply["result"]["items"] if row["kind"] == "skill"}

    return home, skills


def test_warm_slash_completion_observes_new_skill(completion):
    home, skills = completion
    _write_skill(home, "fixture-before")
    assert skills() == {"fixture-before"}
    _write_skill(home, "fixture-after")
    assert skills() == {"fixture-before", "fixture-after"}
    catalog = server._methods["commands.catalog"]("c", {"session_id": "fixture"})
    assert set(catalog["result"]["skills"]) == {"/fixture-before", "/fixture-after"}


def test_warm_slash_completion_drops_deleted_and_renamed_skills(completion):
    home, skills = completion
    before = _write_skill(home, "fixture-before")
    assert skills() == {"fixture-before"}
    before.write_text("---\nname: fixture-renamed\n---\nNew name.\n", encoding="utf-8")
    assert skills() == {"fixture-renamed"}
    before.unlink()
    assert skills() == set()
    _write_skill(home, "fixture-after")
    assert skills() == {"fixture-after"}


def test_completion_refresh_preserves_profile_isolation(completion, tmp_path, monkeypatch):
    home, skills = completion
    other = tmp_path / "other"
    _write_skill(other, "fixture-other")
    (other / "config.yaml").write_text("skills:\n  external_dirs: []\n", encoding="utf-8")
    monkeypatch.setitem(server._sessions, "other", {"cwd": str(tmp_path), "profile_home": str(other)})
    _write_skill(home, "fixture-own")
    assert skills() == {"fixture-own"}
    assert skills("other") == {"fixture-other"}
    _write_skill(home, "fixture-new")
    assert skills() == {"fixture-own", "fixture-new"}


def test_unreadable_skill_does_not_hide_healthy_siblings(completion):
    home, skills = completion
    _write_skill(home, "fixture-good")
    bad = _write_skill(home, "fixture-bad")
    assert skills() == {"fixture-good", "fixture-bad"}
    bad.write_bytes(b"\xff\xff")
    assert skills() == {"fixture-good"}
