"""Tests for stack-aware skill recommendation helpers."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


class _StubSource:
    def __init__(self, items):
        self._items = items

    def inspect(self, identifier):
        return self._items.get(identifier)


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_detect_project_finds_nextjs_react_typescript(tmp_path):
    project = tmp_path / "webapp"
    project.mkdir()
    _write_json(
        project / "package.json",
        {
            "dependencies": {
                "next": "15.0.0",
                "react": "19.0.0",
                "react-dom": "19.0.0",
            },
            "devDependencies": {"typescript": "5.0.0"},
        },
    )
    (project / "tsconfig.json").write_text("{}", encoding="utf-8")

    from tools.skills_recommend import detect_project

    result = detect_project(project)

    assert result.root == project.resolve()
    assert {"nextjs", "react", "typescript"}.issubset(result.technologies)


def test_detect_project_finds_python_and_fastapi(tmp_path):
    project = tmp_path / "api"
    project.mkdir()
    (project / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")

    from tools.skills_recommend import detect_project

    result = detect_project(project)

    assert {"python", "fastapi"}.issubset(result.technologies)


def test_detect_project_collects_project_local_skills(tmp_path):
    project = tmp_path / "repo"
    skill_dir = project / ".hermes" / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (project / ".git").mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Project skill\n---\n\n# My Skill\n",
        encoding="utf-8",
    )

    from tools.skills_recommend import detect_project

    result = detect_project(project)

    assert [s["name"] for s in result.available_project_skills] == ["my-skill"]
    assert result.available_project_skills[0]["source"] == "project-local-hermes"


def test_recommend_skills_dedupes_and_prefers_combo_matches(tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    _write_json(
        project / "package.json",
        {
            "dependencies": {
                "next": "15.0.0",
                "react": "19.0.0",
                "react-dom": "19.0.0",
            }
        },
    )

    fake_meta = {
        "hub/react": type(
            "Meta",
            (),
            {
                "name": "react-skill",
                "description": "React skill",
                "source": "github",
                "trust_level": "trusted",
                "identifier": "hub/react",
            },
        )(),
        "hub/next": type(
            "Meta",
            (),
            {
                "name": "next-skill",
                "description": "Next skill",
                "source": "github",
                "trust_level": "trusted",
                "identifier": "hub/next",
            },
        )(),
    }

    with (
        patch(
            "tools.skills_recommend.RECOMMENDATION_MAP",
            {
                "react": {"hub": ["hub/react"]},
                "nextjs": {"hub": ["hub/next"]},
            },
        ),
        patch(
            "tools.skills_recommend.COMBO_RECOMMENDATION_MAP",
            {("nextjs", "react"): {"hub": ["hub/next"]}},
        ),
        patch(
            "tools.skills_recommend.create_source_router",
            return_value=[_StubSource(fake_meta)],
        ),
    ):
        from tools.skills_recommend import recommend_skills

        result = recommend_skills(project)

    third_party = result["third_party"]
    assert [r["identifier"] for r in third_party] == ["hub/next", "hub/react"]
    assert third_party[0]["score"] > third_party[1]["score"]
    assert "nextjs+react" in third_party[0]["matched_on"]


def test_recommend_skills_respects_official_source_filter(tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    (project / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")

    with patch(
        "tools.skills_recommend.RECOMMENDATION_MAP",
        {"docker": {"official": ["official/devops/docker-management"], "hub": ["hub/docker"]}},
    ):
        from tools.skills_recommend import recommend_skills

        result = recommend_skills(project, source_filter="official")

    assert [r["identifier"] for r in result["official"]] == ["official/devops/docker-management"]
    assert result["third_party"] == []
