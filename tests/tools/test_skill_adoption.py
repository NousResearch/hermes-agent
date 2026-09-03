"""Tests for tools.skill_adoption (Wisdom v1, M3)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skill_adoption import (
    adopt_skill,
    decline_share,
    list_org_skills,
    load_state,
    pending_shares,
    save_state,
)

NOW = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
ORG = "org-test-1"


def _make_org_skill(tmp_path, org_id, category, name, content="# Test Skill"):
    """Create a skill in the org mirror."""
    skill_dir = tmp_path / "skills" / "_org" / org_id / category / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def _mock_home(tmp_path, monkeypatch, state=None):
    """Point the state file and dirs at a temp home."""
    saved = []

    monkeypatch.setattr(
        "tools.skill_adoption.load_state",
        lambda: state or {"adopted": {}, "declined": {}},
    )
    monkeypatch.setattr("tools.skill_adoption.save_state", lambda d: saved.append(d))
    monkeypatch.setattr(
        "tools.skill_adoption._org_dir", lambda: tmp_path / "skills" / "_org"
    )
    monkeypatch.setattr("tools.skill_adoption._skills_dir", lambda: tmp_path / "skills")
    return saved


class TestListOrgSkills:
    def test_empty_org(self, tmp_path, monkeypatch):
        _mock_home(tmp_path, monkeypatch)
        assert list_org_skills(ORG) == []

    def test_lists_skills_with_status(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "code-review")
        _make_org_skill(tmp_path, ORG, "dev", "debugging")
        _mock_home(tmp_path, monkeypatch)

        skills = list_org_skills(ORG)
        assert len(skills) == 2
        assert all(not s["adopted"] for s in skills)
        assert all(not s["declined"] for s in skills)
        assert all(not s["has_local_copy"] for s in skills)

    def test_skips_non_skill_dirs(self, tmp_path, monkeypatch):
        org_root = tmp_path / "skills" / "_org" / ORG
        (org_root / ".hidden").mkdir(parents=True)
        (org_root / "dev" / "no-skill-md").mkdir(parents=True)
        _make_org_skill(tmp_path, ORG, "dev", "real-skill")
        _mock_home(tmp_path, monkeypatch)

        skills = list_org_skills(ORG)
        assert len(skills) == 1
        assert skills[0]["name"] == "real-skill"


class TestPendingShares:
    def test_all_pending_when_no_decisions(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "skill-a")
        _make_org_skill(tmp_path, ORG, "dev", "skill-b")
        _mock_home(tmp_path, monkeypatch)

        pending = pending_shares(ORG)
        assert len(pending) == 2

    def test_adopted_excluded(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "skill-a")
        _make_org_skill(tmp_path, ORG, "dev", "skill-b")
        state = {
            "adopted": {ORG: {"dev/skill-a": {"adopted_at": "2026-08-13"}}},
            "declined": {},
        }
        _mock_home(tmp_path, monkeypatch, state)

        pending = pending_shares(ORG)
        assert len(pending) == 1
        assert pending[0]["name"] == "skill-b"

    def test_declined_excluded(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "skill-a")
        _make_org_skill(tmp_path, ORG, "dev", "skill-b")
        state = {
            "adopted": {},
            "declined": {ORG: ["dev/skill-a"]},
        }
        _mock_home(tmp_path, monkeypatch, state)

        pending = pending_shares(ORG)
        assert len(pending) == 1
        assert pending[0]["name"] == "skill-b"


class TestAdoptSkill:
    def test_copies_skill_to_personal(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "code-review")
        _mock_home(tmp_path, monkeypatch)

        result = adopt_skill(ORG, "dev/code-review")
        assert result["ok"] is True
        assert result["skill_name"] == "code-review"

        dest = tmp_path / "skills" / "dev" / "code-review"
        assert dest.exists()
        assert (dest / "SKILL.md").exists()
        assert (dest / ".adoption-provenance.json").exists()

    def test_provenance_recorded(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "code-review")
        _mock_home(tmp_path, monkeypatch)

        adopt_skill(ORG, "dev/code-review", source_commit="sha256:abc", author="user-1")

        prov_path = (
            tmp_path / "skills" / "dev" / "code-review" / ".adoption-provenance.json"
        )
        prov = json.loads(prov_path.read_text())
        assert prov["origin"] == "adopted"
        assert prov["org_id"] == ORG
        assert prov["source_commit"] == "sha256:abc"
        assert prov["author"] == "user-1"

    def test_state_updated(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "code-review")
        saved = _mock_home(tmp_path, monkeypatch)

        adopt_skill(ORG, "dev/code-review")
        assert len(saved) == 1
        assert "dev/code-review" in saved[0]["adopted"][ORG]

    def test_refuses_duplicate(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "code-review")
        (tmp_path / "skills" / "dev" / "code-review").mkdir(parents=True)
        _mock_home(tmp_path, monkeypatch)

        result = adopt_skill(ORG, "dev/code-review")
        assert result["ok"] is False
        assert "already exists" in result["error"]

    def test_refuses_nonexistent(self, tmp_path, monkeypatch):
        _mock_home(tmp_path, monkeypatch)
        result = adopt_skill(ORG, "dev/nonexistent")
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_removes_from_declined(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "code-review")
        state = {
            "adopted": {},
            "declined": {ORG: ["dev/code-review"]},
        }
        saved = _mock_home(tmp_path, monkeypatch, state)

        adopt_skill(ORG, "dev/code-review")
        assert "dev/code-review" not in saved[0]["declined"].get(ORG, [])


class TestDeclineShare:
    def test_persists_decline(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "code-review")
        saved = _mock_home(tmp_path, monkeypatch)

        result = decline_share(ORG, "dev/code-review")
        assert result["ok"] is True
        assert len(saved) == 1
        assert "dev/code-review" in saved[0]["declined"][ORG]

    def test_idempotent(self, tmp_path, monkeypatch):
        _make_org_skill(tmp_path, ORG, "dev", "code-review")
        state = {"adopted": {}, "declined": {ORG: ["dev/code-review"]}}
        saved = _mock_home(tmp_path, monkeypatch, state)

        result = decline_share(ORG, "dev/code-review")
        assert result["ok"] is True
        assert len(saved) == 0  # no save when already declined

    def test_refuses_nonexistent(self, tmp_path, monkeypatch):
        _mock_home(tmp_path, monkeypatch)
        result = decline_share(ORG, "dev/nonexistent")
        assert result["ok"] is False
