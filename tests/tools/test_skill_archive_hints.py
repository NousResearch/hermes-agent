"""skill_manage create surfaces archived skills on a similar topic, and stays quiet otherwise (#58975)."""

import json

import pytest

from hermes_constants import get_hermes_home
from tools.skill_manager_tool import skill_manage


def _skill_md(name: str, description: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nStep 1: Do the thing.\n"


@pytest.fixture
def archive():
    """Real archive under the test's isolated HERMES_HOME, holding one archived skill."""
    skills = get_hermes_home() / "skills"
    old = skills / ".archive" / "postgres-backup-restore"
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text(_skill_md(
        "postgres-backup-restore", "Use when backing up a Postgres database with pg_dump."))
    return skills


def _create(name: str, description: str) -> dict:
    return json.loads(skill_manage(action="create", name=name, content=_skill_md(name, description)))


def test_create_on_archived_topic_points_at_the_archived_skill(archive):
    result = _create("pg-dump-nightly", "Use when scheduling nightly Postgres backups.")

    assert result["success"] is True
    assert [h["name"] for h in result["similar_archived_skills"]] == ["postgres-backup-restore"]
    assert "hermes curator restore" in result["archived_hint"]
    # A hint only: the archived skill stays archived and the new skill is created.
    assert (archive / ".archive" / "postgres-backup-restore" / "SKILL.md").exists()
    assert not (archive / "postgres-backup-restore").exists()


def test_create_on_unrelated_topic_adds_nothing(archive):
    result = _create("tennis-drills", "Use when planning tennis serve drills.")

    assert result["success"] is True
    assert "similar_archived_skills" not in result
    assert "archived_hint" not in result
