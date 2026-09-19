"""Opt-in ``skills.required_author`` provenance policy for new local skills.

Rebuild of upstream PR #102022. When ``skills.required_author`` is configured,
every local creation surface (flat ``skill_manage``, the atomic batch, and the
dashboard/web create — all of which route through ``_create_skill``) injects
that author when the new skill's frontmatter omits it and REJECTS a conflicting
explicit author BEFORE the write-approval gate, so a staged diff shows exactly
the metadata approval will write. Existing skills and hub installs are
untouched, and a config without the key behaves exactly as before.

These tests drive the REAL functions against a temp ``HERMES_HOME`` with a real
``config.yaml`` (root AGENTS.md: E2E with real imports, not mocks of the seam
under test) and read staged payloads back from the real pending store.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.skill_utils import parse_frontmatter
from tools import write_approval as wa
from tools.skill_manager_tool import _create_skill, skill_manage

REQUIRED_AUTHOR = "Example Author"

NO_AUTHOR_CONTENT = """\
---
name: test-skill
description: A test skill for unit testing.
---

# Test Skill

Step 1: Do the thing.
"""


def _authored_content(author_line: str) -> str:
    return NO_AUTHOR_CONTENT.replace(
        "description: A test skill for unit testing.\n",
        f"description: A test skill for unit testing.\n{author_line}\n")


def _write_config(home: Path, skills_yaml: str) -> None:
    (home / "config.yaml").write_text("skills:\n" + skills_yaml)


def _make_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes-home"
    (home / "skills").mkdir(parents=True)
    return home


@pytest.fixture
def policy_home(tmp_path, monkeypatch):
    """Real temp HERMES_HOME whose config.yaml sets skills.required_author."""
    home = _make_home(tmp_path)
    _write_config(home, f'  required_author: "{REQUIRED_AUTHOR}"\n')
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def plain_home(tmp_path, monkeypatch):
    """Real temp HERMES_HOME with NO required_author key (policy disabled)."""
    home = _make_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@contextmanager
def _skill_dir(home: Path):
    """Confine _find_skill/_resolve_skill_dir to the fixture home's skills dir —
    never the real ~/.hermes/skills/."""
    skills = home / "skills"
    skills.mkdir(exist_ok=True)
    with patch("tools.skill_manager_tool.SKILLS_DIR", skills), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills]):
        yield skills


def _staged_payloads() -> list:
    return wa.list_pending("skills")


# ---------------------------------------------------------------------------
# Policy disabled (empty/absent) — behavior must be exactly as before
# ---------------------------------------------------------------------------


class TestPolicyDisabled:
    def test_create_without_key_does_not_inject_author_e2e(self, plain_home):
        """Invariant: absent skills.required_author → creation is untouched."""
        with _skill_dir(plain_home):
            result = _create_skill("my-skill", NO_AUTHOR_CONTENT)
        assert result["success"] is True
        frontmatter, _ = parse_frontmatter(
            (plain_home / "skills" / "my-skill" / "SKILL.md").read_text())
        assert "author" not in frontmatter

    def test_conflicting_author_allowed_without_key_e2e(self, plain_home):
        content = _authored_content("author: Someone Else\n")
        with _skill_dir(plain_home):
            result = _create_skill("my-skill", content)
        assert result["success"] is True
        frontmatter, _ = parse_frontmatter(
            (plain_home / "skills" / "my-skill" / "SKILL.md").read_text())
        assert frontmatter["author"] == "Someone Else"


# ---------------------------------------------------------------------------
# Creation through _create_skill (flat skill_manage + dashboard/web router)
# ---------------------------------------------------------------------------


class TestCreateSkillPolicy:
    def test_create_injects_configured_required_author_e2e(self, policy_home):
        with _skill_dir(policy_home):
            result = _create_skill("my-skill", NO_AUTHOR_CONTENT)
        assert result["success"] is True
        content = (policy_home / "skills" / "my-skill" / "SKILL.md").read_text()
        frontmatter, _ = parse_frontmatter(content)
        assert frontmatter["author"] == REQUIRED_AUTHOR
        assert content.count("\nauthor:") == 1

    def test_create_accepts_matching_explicit_author(self, policy_home):
        content = _authored_content(f"author: {REQUIRED_AUTHOR}\n")
        with _skill_dir(policy_home):
            result = _create_skill("my-skill", content)
        assert result["success"] is True
        written = (policy_home / "skills" / "my-skill" / "SKILL.md").read_text()
        frontmatter, _ = parse_frontmatter(written)
        assert frontmatter["author"] == REQUIRED_AUTHOR
        assert written.count("\nauthor:") == 1

    def test_create_rejects_conflicting_explicit_author(self, policy_home):
        content = _authored_content("author: Someone Else\n")
        with _skill_dir(policy_home):
            result = _create_skill("my-skill", content)
        assert result["success"] is False
        assert f"must use author '{REQUIRED_AUTHOR}'" in result["error"]
        assert not (policy_home / "skills" / "my-skill").exists()

    def test_stripped_policy_value_is_normalized(self, policy_home):
        """A padded config value must not forge '  Example Author  ' attributions."""
        _write_config(policy_home, f'  required_author: "  {REQUIRED_AUTHOR}  "\n')
        with _skill_dir(policy_home):
            result = _create_skill("my-skill", NO_AUTHOR_CONTENT)
        assert result["success"] is True
        frontmatter, _ = parse_frontmatter(
            (policy_home / "skills" / "my-skill" / "SKILL.md").read_text())
        assert frontmatter["author"] == REQUIRED_AUTHOR

    def test_create_tolerates_utf8_bom_like_the_validator(self, policy_home):
        """_validate_frontmatter strips a Windows UTF-8 BOM; the author policy must
        too, or a BOM-prefixed skill silently escapes injection/rejection."""
        content = "\ufeff" + NO_AUTHOR_CONTENT
        with _skill_dir(policy_home):
            result = _create_skill("my-skill", content)
        assert result["success"] is True
        frontmatter, _ = parse_frontmatter(
            (policy_home / "skills" / "my-skill" / "SKILL.md").read_text())
        assert frontmatter["author"] == REQUIRED_AUTHOR


# ---------------------------------------------------------------------------
# Flat skill_manage: normalize BEFORE the write-approval gate
# ---------------------------------------------------------------------------


class TestFlatManageStaging:
    @pytest.fixture(autouse=True)
    def gate_on(self, policy_home):
        _write_config(policy_home,
                      f'  required_author: "{REQUIRED_AUTHOR}"\n  write_approval: true\n')

    def test_staged_payload_carries_normalized_author_e2e(self, policy_home):
        with _skill_dir(policy_home):
            raw = skill_manage("create", "my-skill", content=NO_AUTHOR_CONTENT)
        result = json.loads(raw)
        assert result["staged"] is True
        pending = _staged_payloads()
        assert len(pending) == 1
        staged_content = pending[0]["payload"]["content"]
        frontmatter, _ = parse_frontmatter(staged_content)
        assert frontmatter["author"] == REQUIRED_AUTHOR

    def test_conflicting_author_rejected_before_staging(self, policy_home):
        content = _authored_content("author: Someone Else\n")
        with _skill_dir(policy_home):
            raw = skill_manage("create", "my-skill", content=content)
        result = json.loads(raw)
        assert result["success"] is False
        assert f"must use author '{REQUIRED_AUTHOR}'" in result["error"]
        assert _staged_payloads() == []


# ---------------------------------------------------------------------------
# Atomic batch: normalize staged ops without mutating the caller's dicts
# ---------------------------------------------------------------------------


def _batch(operations):
    return json.loads(skill_manage("", "", operations=operations))


class TestBatchStaging:
    @pytest.fixture(autouse=True)
    def gate_on(self, policy_home):
        _write_config(policy_home,
                      f'  required_author: "{REQUIRED_AUTHOR}"\n  write_approval: true\n')

    def test_batch_stages_normalized_author(self, policy_home):
        operations = [{"action": "create", "name": "my-skill",
                       "content": NO_AUTHOR_CONTENT}]
        with _skill_dir(policy_home):
            result = _batch(operations)
        assert result["staged"] is True
        pending = _staged_payloads()
        assert len(pending) == 1
        staged_ops = pending[0]["payload"]["operations"]
        frontmatter, _ = parse_frontmatter(staged_ops[0]["content"])
        assert frontmatter["author"] == REQUIRED_AUTHOR

    def test_batch_does_not_mutate_caller_operations(self, policy_home):
        operations = [{"action": "create", "name": "my-skill",
                       "content": NO_AUTHOR_CONTENT}]
        with _skill_dir(policy_home):
            _batch(operations)
        # The caller's dict is reused (transcript logs, staged replay); the
        # normalization belongs to the staged copy only.
        assert "\nauthor:" not in operations[0]["content"]

    def test_batch_rejects_conflicting_author_before_staging(self, policy_home):
        operations = [{"action": "create", "name": "my-skill",
                       "content": _authored_content("author: Someone Else\n")}]
        with _skill_dir(policy_home):
            result = _batch(operations)
        assert result["success"] is False
        assert f"must use author '{REQUIRED_AUTHOR}'" in result["error"]
        assert _staged_payloads() == []


# ---------------------------------------------------------------------------
# Existing skills stay untouched
# ---------------------------------------------------------------------------


class TestExistingSkillsUntouched:
    def test_edit_of_existing_skill_is_not_normalized(self, policy_home):
        """The policy governs creation only: a full rewrite of an existing skill
        keeps its content as authored (hub-installed/team skills included)."""
        with _skill_dir(policy_home):
            assert _create_skill("my-skill", NO_AUTHOR_CONTENT)["success"] is True
            raw = skill_manage("edit", "my-skill", content=NO_AUTHOR_CONTENT)
        result = json.loads(raw)
        assert result["success"] is True
        frontmatter, _ = parse_frontmatter(
            (policy_home / "skills" / "my-skill" / "SKILL.md").read_text())
        assert "author" not in frontmatter
