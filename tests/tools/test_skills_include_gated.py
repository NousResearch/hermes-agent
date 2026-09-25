"""Regression: category toggling must see environment-gated skills.

The Capabilities "Devops" section rendered kanban-* skills whose SKILL.md
declares ``environments:`` — then ``PUT /api/skills/toggle-category`` computed
membership through the same environment-filtered discovery, found nothing once
the Kanban environment went inactive, and answered 400 "Unknown skill category".
``_find_all_skills(include_gated=True)`` is the toggle's membership source:
category membership is independent of platform/environment/app gating.
"""

from pathlib import Path

from tools import skills_tool


def _write_skill(root: Path, rel: str | Path, category: str | None, extra: str = "") -> None:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True)
    lines = ["---", "name: " + skill_dir.name, "description: test skill."]
    if category is not None:
        lines.insert(2, "category: " + category)
    lines += [extra, "---", "# body"] if extra else ["---", "# body"]
    (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


def test_include_gated_lists_environment_gated_skill(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    _write_skill(
        root,
        Path("devops") / "kanban-orchestrator",
        "devops",
        extra="environments: [kanban]",
    )
    monkeypatch.setattr(skills_tool, "_skills_dir", lambda: root)
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})

    gated = {s["name"] for s in skills_tool._find_all_skills(skip_disabled=True, include_gated=True)}
    plain = {s["name"] for s in skills_tool._find_all_skills(skip_disabled=True)}

    assert "kanban-orchestrator" in gated  # membership ignores environment gating
    assert "kanban-orchestrator" not in plain  # default listing stays filtered
