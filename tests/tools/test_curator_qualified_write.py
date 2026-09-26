"""Public skill_manage regression for category-qualified curator writes."""
import json

import pytest


@pytest.mark.parametrize("case", ["allowed", "bare-pin", "alias-pin", "unmanaged", "ambiguous", "unread", "lookup-error"])
def test_category_qualified_curator_patch_uses_existing_ownership(tmp_path, monkeypatch, case):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.skill_manager_tool import skill_manage
    from tools.skills_tool import skill_view
    from tools.skill_usage import mark_agent_created, bump_view
    from tools.skill_provenance import (
        BACKGROUND_REVIEW, set_current_write_origin, reset_current_write_origin,
    )

    skill = tmp_path / "skills" / "lessons" / "learned"
    skill.mkdir(parents=True)
    target = skill / "SKILL.md"
    target.write_text(
        "---\nname: learned\ndescription: Remember verified lessons.\n---\n"
        "# Lessons\n\nCheck the input.\n", encoding="utf-8",
    )
    if case != "unmanaged":
        mark_agent_created("learned")
    bump_view("lessons/learned")
    if case in {"bare-pin", "alias-pin"}:
        from tools.skill_usage import set_pinned
        set_pinned("learned" if case == "bare-pin" else "lessons/learned", True)
    if case == "ambiguous":
        other = tmp_path / "skills" / "other" / "learned"
        other.mkdir(parents=True)
        (other / "SKILL.md").write_bytes(target.read_bytes())
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        if case != "unread":
            viewed = json.loads(skill_view("lessons/learned"))
            assert viewed["success"], viewed
        if case == "lookup-error":
            def fail():
                raise OSError("unreadable provenance")
            monkeypatch.setattr("tools.skill_usage.load_usage", fail)
        result = json.loads(skill_manage(
            action="patch", name="lessons/learned",
            old_string="Check the input.", new_string="Check both inputs.",
        ))
    finally:
        reset_current_write_origin(token)
    assert result["success"] is (case == "allowed"), result
    assert ("Check both inputs." in target.read_text(encoding="utf-8")) is (case == "allowed")
