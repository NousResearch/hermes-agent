"""Regression coverage for the bundled Kanban Codex lane skill."""

import json
from pathlib import Path

from tools import skills_tool
from tools.skill_manager_tool import _validate_frontmatter


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "autonomous-ai-agents" / "kanban-codex-lane"
SKILL_MD = SKILL_DIR / "SKILL.md"
TEMPLATE = SKILL_DIR / "templates" / "pmb-codex-lane-prompt.md"


def _skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_kanban_codex_lane_skill_frontmatter_is_valid():
    content = _skill_text()

    assert _validate_frontmatter(content) is None
    assert "name: kanban-codex-lane" in content
    assert "description: Use when" in content


def test_kanban_codex_lane_skill_is_discoverable_with_template(monkeypatch, tmp_path):
    local_skills = tmp_path / "skills"
    local_skills.mkdir()
    bundled_skills = REPO_ROOT / "skills"

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", local_skills)
    monkeypatch.setattr(
        "agent.skill_utils.get_external_skills_dirs",
        lambda: [bundled_skills],
    )

    listed = json.loads(skills_tool.skills_list("autonomous-ai-agents"))
    assert listed["success"] is True
    assert any(skill["name"] == "kanban-codex-lane" for skill in listed["skills"])

    viewed = json.loads(skills_tool.skill_view("kanban-codex-lane"))
    assert viewed["success"] is True
    assert viewed["path"].endswith("kanban-codex-lane/SKILL.md")
    assert viewed["linked_files"]["templates"] == ["templates/pmb-codex-lane-prompt.md"]

    template = json.loads(
        skills_tool.skill_view(
            "kanban-codex-lane",
            file_path="templates/pmb-codex-lane-prompt.md",
        )
    )
    assert template["success"] is True
    assert "PMB safety constraints" in template["content"]


def test_kanban_codex_lane_documents_required_contracts():
    content = _skill_text()
    template = TEMPLATE.read_text(encoding="utf-8")

    required_skill_phrases = [
        "Hermes is always the task owner",
        "Codex is an input lane only",
        "git -C \"$REPO\" worktree add -b \"$BRANCH\" \"$WORKTREE\" \"$BASE\"",
        "codex --version",
        "codex features list | grep -i goals || true",
        "codex exec --full-auto",
        "/goal Work in this repository only",
        "process(action=\"kill\", session_id=session_id)",
        "scripts/run_tests.sh",
        '"codex_lane"',
        '"used"',
        '"mode"',
        '"worktree"',
        '"branch"',
        '"command"',
        '"result"',
        '"accepted_commits"',
        '"rejected_reason"',
        '"tests_run"',
        '"artifacts"',
        "accepted | rejected | partial | timed_out",
    ]
    for phrase in required_skill_phrases:
        assert phrase in content

    required_safety_phrases = [
        "live-SIM is paper-only; do not add or enable live REST order entry",
        "Never use market orders",
        "Do not add execution crossing",
        "Do not fake passive fills",
        "Do not weaken risk gates",
        "Do not read, print, write, or require secrets/tokens/credentials",
    ]
    for phrase in required_safety_phrases:
        assert phrase in content
        assert phrase in template
