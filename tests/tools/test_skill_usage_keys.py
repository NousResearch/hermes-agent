"""Key-canonicalization contracts for skill usage telemetry (#103542).

One canonical .usage.json key per skill: separators normalized, dir paths
resolved to frontmatter names, pre-existing splits merged with counters
summed. Uses an isolated HERMES_HOME; no network, no GPU.
"""

import json

import pytest

from tools.skill_usage import (
    bump_use,
    bump_view,
    canonical_skill_key,
    load_usage,
    seed_record_if_missing,
)


@pytest.fixture()
def skills_home(tmp_path, monkeypatch):
    """Isolated skills tree with one nested skill (dir != frontmatter name)."""
    home = tmp_path / ".hermes"
    skills = home / "skills" / "devops" / "skill-router"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: sr\ndescription: Routes things.\n---\n\nBody.\n", encoding="utf-8")
    plain = home / "skills" / "plain-skill"
    plain.mkdir(parents=True)
    (plain / "SKILL.md").write_text(
        "---\nname: plain-skill\ndescription: Plain.\n---\n\nBody.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_canonical_key_normalizes_separators(skills_home):
    assert canonical_skill_key("devops\\plain-skill-not-a-dir") == "devops/plain-skill-not-a-dir"
    assert canonical_skill_key("  devops/skill-router/ ") == "sr"  # resolves via the real dir
    assert canonical_skill_key("sr") == "sr"


def test_canonical_key_resolves_dir_to_frontmatter_name(skills_home):
    assert canonical_skill_key("devops/skill-router") == "sr"
    assert canonical_skill_key("devops\\skill-router") == "sr"


def test_canonical_key_never_raises_and_degrades(skills_home):
    assert canonical_skill_key("") == ""
    assert canonical_skill_key(None) == ""
    assert canonical_skill_key(42) == "42"
    assert canonical_skill_key("no/such/dir") == "no/such/dir"
    assert canonical_skill_key(object()) != ""


def test_canonical_key_survives_hostile_input(skills_home):
    """The never-raises contract holds for hostile __str__ and null bytes."""

    class Hostile:
        def __bool__(self):
            return True

        def __str__(self):
            raise ValueError("boom")

    assert canonical_skill_key(Hostile()) == ""
    assert canonical_skill_key("a\0b/c") == "a\0b/c"


def test_separator_split_aggregates_under_one_key(skills_home):
    """Issue split #1: `/` vs `\\` writes land on a single record."""
    bump_use("devops\\plain-skill")
    bump_use("devops/plain-skill")
    data = load_usage()
    assert list(data) == ["devops/plain-skill"]
    assert data["devops/plain-skill"]["use_count"] == 2


def test_dirname_frontmatter_split_aggregates(skills_home):
    """Issue split #2: dir-path writes resolve to the frontmatter name."""
    for _ in range(3):
        bump_use("devops/skill-router")
    data = load_usage()
    assert list(data) == ["sr"]
    assert data["sr"]["use_count"] == 3


def test_preexisting_split_heals_on_next_write(skills_home):
    """Counters from both spellings sum; the duplicate key disappears."""
    home = skills_home
    usage_file = home / "skills" / ".usage.json"
    usage_file.write_text(json.dumps({
        "devops\\plain-skill": {"use_count": 1, "view_count": 0, "patch_count": 0,
                                "last_used_at": "2026-01-01T00:00:00",
                                "last_viewed_at": None, "last_patched_at": None,
                                "created_at": None, "created_by": None, "pinned": False},
        "devops/plain-skill": {"use_count": 2, "view_count": 0, "patch_count": 0,
                               "last_used_at": "2026-09-01T00:00:00",
                               "last_viewed_at": None, "last_patched_at": None,
                               "created_at": None, "created_by": None, "pinned": False},
    }), encoding="utf-8")
    bump_use("devops/plain-skill")
    data = load_usage()
    assert list(data) == ["devops/plain-skill"]
    assert data["devops/plain-skill"]["use_count"] == 4
    assert data["devops/plain-skill"]["last_used_at"] >= "2026-09-01T00:00:00"


def test_seed_uses_canonical_key(skills_home):
    seed_record_if_missing("devops\\skill-router")
    data = load_usage()
    assert "sr" in data
    assert "devops\\skill-router" not in data


def test_view_and_use_share_the_record(skills_home):
    bump_view("sr")
    bump_use("sr")
    data = load_usage()
    assert data["sr"]["view_count"] == 1
    assert data["sr"]["use_count"] == 1
