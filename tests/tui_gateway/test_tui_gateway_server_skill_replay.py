"""Rewind/regenerate re-expands a skill turn the way ``command.dispatch`` built it."""

from pathlib import Path

import pytest

from tui_gateway import server


def _write_replay_skill(home: Path, name: str) -> None:
    (home / "skills" / name).mkdir(parents=True)
    (home / "skills" / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name}\n---\n\n# {name}\n\n{name.upper()}-BODY\n",
        encoding="utf-8")


@pytest.mark.parametrize("typed", ["/alpha /beta do X", "/duo do X"], ids=["stacked", "bundle"])
def test_expand_skill_invocation_for_replay_round_trips_stacked_skills_and_bundles(
    monkeypatch, typed
):
    # command.dispatch expands ``/alpha /beta …`` into one stacked invocation and ``/duo …`` into a
    # bundle; both project to what the user typed. Re-expanding only the first token as a single
    # skill sent ``/beta do X`` to the model as the instruction, and a bundle as literal text.
    import agent.skill_utils as skill_utils
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda *a, **k: [])
    home = get_hermes_home()
    _write_replay_skill(home, "alpha")
    _write_replay_skill(home, "beta")
    (home / "skill-bundles").mkdir()
    (home / "skill-bundles" / "duo.yaml").write_text(
        "name: duo\ndescription: both\nskills: [alpha, beta]\n", encoding="utf-8")
    session = {"session_key": "task-1"}

    name, arg = typed[1:].split(" ", 1)
    original = server._methods["command.dispatch"]("r1", {"name": name, "arg": arg})["result"]["message"]
    shown = server._skill_scaffold_projection(original)
    expanded = server._expand_skill_invocation_for_replay(shown, session)

    assert shown == typed
    assert "ALPHA-BODY" in expanded and "BETA-BODY" in expanded
    assert server._skill_scaffold_projection(expanded) == typed


def test_expand_skill_invocation_for_replay_resolves_in_the_sessions_profile(tmp_path, monkeypatch):
    # A session bound to another profile dispatched a skill only that profile has; the rewind runs on
    # the socket thread with no home bound, so it must re-bind the session's home like dispatch does.
    import agent.skill_utils as skill_utils

    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda *a, **k: [])
    profile_home = tmp_path / "profiles" / "other"
    _write_replay_skill(profile_home, "gamma")
    session = {"session_key": "task-1", "profile_home": str(profile_home)}

    expanded = server._expand_skill_invocation_for_replay("/gamma do Y", session)

    assert "GAMMA-BODY" in expanded
    assert server._skill_scaffold_projection(expanded) == "/gamma do Y"
    # The launch profile does not have it: without the session's scope it stays literal.
    assert server._expand_skill_invocation_for_replay("/gamma do Y", {}) == "/gamma do Y"


def test_expand_skill_invocation_for_replay_keeps_the_text_when_dispatch_fails(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("skill scan failed")

    monkeypatch.setattr(server, "_dispatch_bundle", boom)
    assert server._expand_skill_invocation_for_replay("/work fix it", {}) == "/work fix it"

