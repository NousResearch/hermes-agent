"""Tests for SkillOpt optimization candidate reporting."""

from __future__ import annotations

import json
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _write_skill(path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(f"---\nname: {name}\n---\n\n# {name}\n", encoding="utf-8")


def _with_home(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_skillopt_candidate_report_ranks_active_agent_skills(tmp_path):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    _write_skill(skills / "alpha", "alpha")
    _write_skill(skills / "beta", "beta")
    _write_skill(skills / "gamma", "gamma")
    (skills / ".usage.json").write_text(
        json.dumps(
            {
                "alpha": {"created_by": "agent", "use_count": 5, "view_count": 2, "patch_count": 0},
                "beta": {"created_by": "agent", "use_count": 1, "view_count": 0, "patch_count": 4},
                "gamma": {"created_by": "agent", "use_count": 0, "view_count": 0, "patch_count": 0, "pinned": True},
            }
        ),
        encoding="utf-8",
    )

    def run():
        from tools.skill_usage import skillopt_candidate_report

        return skillopt_candidate_report()

    rows = _with_home(home, run)

    assert [row["name"] for row in rows] == ["beta", "alpha"]
    assert rows[0]["skillopt_score"] > rows[1]["skillopt_score"]
    assert rows[0]["reasons"] == ["patched-often", "used"]
    assert rows[1]["reasons"] == ["used"]


def test_skillopt_candidate_report_excludes_unpersisted_and_non_agent_skills(tmp_path):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    _write_skill(skills / "local", "local")
    _write_skill(skills / "hubbed", "hubbed")
    (skills / ".hub").mkdir(parents=True)
    (skills / ".hub" / "lock.json").write_text(
        json.dumps({"installed": {"hubbed": {"install_path": "hubbed"}}}),
        encoding="utf-8",
    )
    (skills / ".usage.json").write_text(
        json.dumps({"hubbed": {"use_count": 99}, "local": {"use_count": 0}}),
        encoding="utf-8",
    )

    def run():
        from tools.skill_usage import skillopt_candidate_report

        return skillopt_candidate_report()

    assert _with_home(home, run) == []
