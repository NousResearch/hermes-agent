"""Tests for tools/skills_validator.py — static skill validation."""

from pathlib import Path

import pytest

from tools.skills_validator import (
    BUILTIN_RULESETS,
    SEVERITY_BLOCKING,
    SEVERITY_SUGGEST,
    Finding,
    ValidationReport,
    find_skills,
    load_ruleset,
    validate_skill,
)


# ─── ruleset loading ─────────────────────────────────────────────────


def test_load_builtin_hermes_ruleset():
    rules = load_ruleset("hermes")
    assert rules["frontmatter.name.present"] == SEVERITY_BLOCKING
    # Hermes ruleset is lenient — should not include section.* rules
    assert "section.when_to_use.present" not in rules


def test_load_builtin_agentskills_ruleset():
    rules = load_ruleset("agentskills")
    assert rules["frontmatter.name.kebab_case"] == SEVERITY_BLOCKING
    assert rules["section.when_to_use.present"] == SEVERITY_SUGGEST


def test_load_unknown_ruleset_raises():
    with pytest.raises(ValueError, match="not a built-in"):
        load_ruleset("does-not-exist-as-name-or-path")


def test_load_custom_ruleset_from_yaml(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "rules:\n"
        "  frontmatter.present: BLOCKING\n"
        "  body.non_empty: SUGGEST\n",
        encoding="utf-8",
    )
    rules = load_ruleset(str(custom))
    assert rules == {
        "frontmatter.present": "BLOCKING",
        "body.non_empty": "SUGGEST",
    }


def test_load_custom_ruleset_invalid_severity(tmp_path):
    custom = tmp_path / "bad.yaml"
    custom.write_text(
        "rules:\n  frontmatter.present: TOTALLY_INVALID\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid severity"):
        load_ruleset(str(custom))


# ─── validate_skill — valid case ─────────────────────────────────────


def _write_skill(tmp_path: Path, name: str, content: str) -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


VALID_AGENTSKILLS_SKILL = """\
---
name: example-skill
description: An example skill that validates against the full agentskills.io spec.
license: MIT
allowed-tools: Read Grep
metadata:
  author: Test Author
  version: "1.0.0"
  domain: testing
  complexity: basic
  tags: example, test, validation
---

# Example Skill

## When to Use

Use this when validating the validator.

## Procedure

1. Read the file.
2. Check the structure.
"""


def test_validate_clean_skill_against_agentskills(tmp_path):
    skill_dir = _write_skill(tmp_path, "example-skill", VALID_AGENTSKILLS_SKILL)
    rules = load_ruleset("agentskills")
    report = validate_skill(skill_dir, rules, ruleset_name="agentskills")
    assert report.has_blocking is False, [f.message for f in report.findings]
    assert report.findings == []


# ─── validate_skill — failure modes ─────────────────────────────────


MINIMAL_HERMES_SKILL = """\
---
name: minimal-skill
description: Only the bare minimum fields.
---

# Minimal Skill

Body content.
"""


def test_validate_minimal_skill_against_hermes_passes(tmp_path):
    skill_dir = _write_skill(tmp_path, "minimal-skill", MINIMAL_HERMES_SKILL)
    rules = load_ruleset("hermes")
    report = validate_skill(skill_dir, rules, ruleset_name="hermes")
    assert report.has_blocking is False


def test_validate_minimal_skill_against_agentskills_reports_suggestions(tmp_path):
    skill_dir = _write_skill(tmp_path, "minimal-skill", MINIMAL_HERMES_SKILL)
    rules = load_ruleset("agentskills")
    report = validate_skill(skill_dir, rules, ruleset_name="agentskills")
    assert report.has_blocking is False  # name/description present
    rule_ids = {f.rule for f in report.findings}
    # Should flag missing recommended fields
    assert "frontmatter.license.present" in rule_ids
    assert "frontmatter.allowed_tools.present" in rule_ids
    assert "frontmatter.metadata.version.present" in rule_ids
    assert "section.when_to_use.present" in rule_ids


def test_validate_missing_name_is_blocking(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        "nameless",
        "---\ndescription: A skill without a name.\n---\n\nBody.\n",
    )
    rules = load_ruleset("hermes")
    report = validate_skill(skill_dir, rules, ruleset_name="hermes")
    assert report.has_blocking is True
    blocking_rules = {f.rule for f in report.blocking}
    assert "frontmatter.name.present" in blocking_rules


def test_validate_name_must_match_directory(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        "directory-name",
        "---\nname: different-name\ndescription: x\n---\n\nBody.\n",
    )
    rules = load_ruleset("agentskills")
    report = validate_skill(skill_dir, rules, ruleset_name="agentskills")
    rule_ids = {f.rule for f in report.findings}
    assert "frontmatter.name.matches_dir" in rule_ids


def test_validate_name_must_be_kebab_case(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        "BadName_Skill",
        "---\nname: BadName_Skill\ndescription: x\n---\n\nBody.\n",
    )
    rules = load_ruleset("agentskills")
    report = validate_skill(skill_dir, rules, ruleset_name="agentskills")
    rule_ids = {f.rule for f in report.findings}
    assert "frontmatter.name.kebab_case" in rule_ids


def test_validate_long_description_is_blocking(tmp_path):
    long_desc = "x" * 1100
    skill_dir = _write_skill(
        tmp_path,
        "long-desc",
        f"---\nname: long-desc\ndescription: {long_desc}\n---\n\nBody.\n",
    )
    rules = load_ruleset("hermes")
    report = validate_skill(skill_dir, rules, ruleset_name="hermes")
    assert report.has_blocking is True
    assert "frontmatter.description.length" in {f.rule for f in report.blocking}


def test_validate_empty_body_is_blocking(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        "empty-body",
        "---\nname: empty-body\ndescription: A skill with no body.\n---\n\n   \n",
    )
    rules = load_ruleset("hermes")
    report = validate_skill(skill_dir, rules, ruleset_name="hermes")
    assert report.has_blocking is True
    assert "body.non_empty" in {f.rule for f in report.blocking}


def test_validate_missing_frontmatter_is_blocking(tmp_path):
    skill_dir = _write_skill(
        tmp_path, "no-fm", "# Just a heading, no frontmatter.\n"
    )
    rules = load_ruleset("hermes")
    report = validate_skill(skill_dir, rules, ruleset_name="hermes")
    assert report.has_blocking is True


def test_validate_invalid_semver_is_suggest(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        "bad-version",
        "---\nname: bad-version\ndescription: x\nmetadata:\n"
        "  version: not-a-version\n---\n\nBody.\n",
    )
    rules = load_ruleset("agentskills")
    report = validate_skill(skill_dir, rules, ruleset_name="agentskills")
    rule_ids = {f.rule for f in report.findings}
    assert "frontmatter.metadata.version.semver" in rule_ids


# ─── find_skills ────────────────────────────────────────────────────


def test_find_skills_excludes_archive(tmp_path):
    _write_skill(tmp_path, "active-skill", MINIMAL_HERMES_SKILL)
    archive_skill = tmp_path / ".archive" / "old-skill"
    archive_skill.mkdir(parents=True)
    (archive_skill / "SKILL.md").write_text(MINIMAL_HERMES_SKILL, encoding="utf-8")

    found = find_skills(tmp_path)
    names = {p.name for p in found}
    assert names == {"active-skill"}


def test_find_skills_recursive(tmp_path):
    nested = tmp_path / "category-a" / "nested-skill"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(MINIMAL_HERMES_SKILL, encoding="utf-8")
    _write_skill(tmp_path, "top-level", MINIMAL_HERMES_SKILL)

    found = find_skills(tmp_path)
    names = {p.name for p in found}
    assert names == {"top-level", "nested-skill"}
