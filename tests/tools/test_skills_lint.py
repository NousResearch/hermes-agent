"""Tests for tools/skills_lint.py — per-rule positive and negative cases."""

import re

import pytest

from agent.skill_utils import parse_frontmatter, parse_frontmatter_strict
from tools.skills_lint import (
    RULES,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    format_lint_report,
    lint_skill,
)


CLEAN_FRONTMATTER = """\
---
name: {name}
description: A perfectly fine skill for testing.
version: 1.0.0
author: tests
license: MIT
platforms: [linux, macos, windows]
---
"""


def make_skill(tmp_path, *, dirname="clean-skill", frontmatter=None, body="# Usage\n"):
    """Write a SKILL.md into tmp_path/<dirname> and return the skill dir."""
    skill_dir = tmp_path / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = (
        frontmatter
        if frontmatter is not None
        else CLEAN_FRONTMATTER.format(name=dirname)
    ) + body
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def rule_ids(result):
    return sorted({f.rule_id for f in result.findings})


# ---------------------------------------------------------------------------
# Registry meta-test
# ---------------------------------------------------------------------------

def test_rule_registry_is_well_formed():
    ids = [rule_id for rule_id, _, _ in RULES]
    assert len(ids) == len(set(ids)), "duplicate rule IDs"
    for rule_id, severity, fn in RULES:
        assert re.fullmatch(r"HSL\d{3}", rule_id), rule_id
        assert severity in (SEVERITY_ERROR, SEVERITY_WARNING), rule_id
        assert callable(fn)


# ---------------------------------------------------------------------------
# Clean skill / structural rules
# ---------------------------------------------------------------------------

def test_clean_skill_has_no_findings(tmp_path):
    result = lint_skill(make_skill(tmp_path))
    assert result.findings == []
    assert result.skill_name == "clean-skill"
    assert "OK" in format_lint_report(result)


def test_missing_skill_md_is_hsl001(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = lint_skill(empty)
    assert rule_ids(result) == ["HSL001"]


def test_no_frontmatter_is_hsl001(tmp_path):
    result = lint_skill(make_skill(tmp_path, frontmatter="", body="# Just markdown\n"))
    assert "HSL001" in rule_ids(result)


def test_unterminated_fence_is_hsl001(tmp_path):
    result = lint_skill(
        make_skill(tmp_path, frontmatter="---\nname: x\ndescription: y\n", body="")
    )
    assert rule_ids(result) == ["HSL001"]


def test_broken_yaml_is_hsl002_and_runtime_would_not_error(tmp_path):
    broken = "---\nname: [unclosed\ndescription: \"also: broken\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=broken, body="body\n"))
    assert rule_ids(result) == ["HSL002"]

    # Pin the gap this rule exists for: the lenient runtime parser swallows
    # the same content without raising, silently degrading to naive parsing.
    fm, _ = parse_frontmatter(broken + "body\n")
    assert isinstance(fm, dict)
    _, _, yaml_error = parse_frontmatter_strict(broken + "body\n")
    assert yaml_error is not None


def test_non_mapping_frontmatter_is_hsl003(tmp_path):
    result = lint_skill(
        make_skill(tmp_path, frontmatter="---\n- just\n- a\n- list\n---\n")
    )
    assert rule_ids(result) == ["HSL003"]


def test_structural_failure_suppresses_field_rules(tmp_path):
    broken = "---\nname: [unclosed\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=broken))
    # No misleading "name/description missing" cascade on top of HSL002.
    assert rule_ids(result) == ["HSL002"]


# ---------------------------------------------------------------------------
# name / version / description
# ---------------------------------------------------------------------------

def test_missing_name_is_hsl010(tmp_path):
    fm = "---\ndescription: fine\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL010" in rule_ids(result)


def test_long_name_is_hsl011(tmp_path):
    fm = f"---\nname: {'x' * 70}\ndescription: fine\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL011" in rule_ids(result)


def test_unsafe_name_is_hsl012(tmp_path):
    fm = "---\nname: ../escape\ndescription: fine\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL012" in rule_ids(result)


def test_name_dir_mismatch_is_hsl013(tmp_path):
    fm = "---\nname: other-name\ndescription: fine\n---\n"
    result = lint_skill(make_skill(tmp_path, dirname="this-dir", frontmatter=fm))
    assert "HSL013" in rule_ids(result)
    assert result.skill_name == "other-name"


def test_lint_cwd_dot_resolves_real_dirname(tmp_path, monkeypatch):
    """Path('.') must compare against the resolved directory name, not ''."""
    from pathlib import Path

    skill = make_skill(tmp_path)  # name matches dirname "clean-skill"
    monkeypatch.chdir(skill)
    result = lint_skill(Path("."))
    assert "HSL013" not in rule_ids(result)
    assert result.findings == []


def test_bad_version_is_hsl014(tmp_path):
    fm = "---\nname: clean-skill\ndescription: fine\nversion: not.a.version!\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL014" in rule_ids(result)


def test_missing_description_is_hsl015(tmp_path):
    fm = "---\nname: clean-skill\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL015" in rule_ids(result)


def test_long_description_is_hsl016(tmp_path):
    fm = f"---\nname: clean-skill\ndescription: {'d' * 1100}\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL016" in rule_ids(result)


# ---------------------------------------------------------------------------
# platforms / unknown keys
# ---------------------------------------------------------------------------

def test_unknown_platform_is_hsl020(tmp_path):
    fm = "---\nname: clean-skill\ndescription: fine\nplatforms: [linux, solaris]\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL020" in rule_ids(result)


def test_platforms_wrong_type_is_hsl020(tmp_path):
    fm = "---\nname: clean-skill\ndescription: fine\nplatforms:\n  os: linux\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL020" in rule_ids(result)


def test_single_platform_string_is_valid(tmp_path):
    fm = "---\nname: clean-skill\ndescription: fine\nplatforms: macos\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL020" not in rule_ids(result)


def test_unknown_key_is_hsl021(tmp_path):
    fm = "---\nname: clean-skill\ndescription: fine\ncolour: red\n---\n"
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL021" in rule_ids(result)


# ---------------------------------------------------------------------------
# requirement declarations
# ---------------------------------------------------------------------------

def test_env_entry_without_name_is_hsl030(tmp_path):
    fm = (
        "---\nname: clean-skill\ndescription: fine\n"
        "required_environment_variables:\n  - prompt: enter the token\n---\n"
    )
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL030" in rule_ids(result)


def test_env_block_wrong_type_is_hsl030(tmp_path):
    fm = (
        "---\nname: clean-skill\ndescription: fine\n"
        "required_environment_variables: API_KEY\n---\n"
    )
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL030" in rule_ids(result)


def test_valid_env_entries_pass(tmp_path):
    fm = (
        "---\nname: clean-skill\ndescription: fine\n"
        "required_environment_variables:\n"
        "  - name: API_KEY\n    prompt: enter key\n"
        "  - SECOND_VAR\n---\n"
    )
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL030" not in rule_ids(result)
    assert "HSL031" not in rule_ids(result)


def test_invalid_env_var_name_is_hsl031(tmp_path):
    fm = (
        "---\nname: clean-skill\ndescription: fine\n"
        "required_environment_variables:\n  - name: 1BAD-NAME\n---\n"
    )
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL031" in rule_ids(result)


def test_legacy_prerequisites_is_hsl032(tmp_path):
    fm = (
        "---\nname: clean-skill\ndescription: fine\n"
        "prerequisites:\n  env_vars: [MY_TOKEN]\n  commands: [curl]\n---\n"
    )
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    assert "HSL032" in rule_ids(result)
    assert all(f.severity == SEVERITY_WARNING for f in result.findings)


def test_invalid_dependency_spec_is_hsl040(tmp_path):
    fm = (
        "---\nname: clean-skill\ndescription: fine\n"
        "dependencies: ['requests>=2.0', 'not a valid spec!!']\n---\n"
    )
    result = lint_skill(make_skill(tmp_path, frontmatter=fm))
    findings = [f for f in result.findings if f.rule_id == "HSL040"]
    assert len(findings) == 1
    assert "not a valid spec!!" in findings[0].message


# ---------------------------------------------------------------------------
# cross-references
# ---------------------------------------------------------------------------

def _related_fm(related):
    return (
        "---\nname: clean-skill\ndescription: fine\n"
        "metadata:\n  hermes:\n    related_skills: [" + ", ".join(related) + "]\n---\n"
    )


def test_unknown_related_skill_is_hsl050(tmp_path):
    skill = make_skill(tmp_path, frontmatter=_related_fm(["exists", "ghost"]))
    result = lint_skill(skill, known_skill_names=frozenset({"exists"}))
    findings = [f for f in result.findings if f.rule_id == "HSL050"]
    assert len(findings) == 1
    assert "ghost" in findings[0].message


def test_related_skills_skipped_without_known_names(tmp_path):
    skill = make_skill(tmp_path, frontmatter=_related_fm(["ghost"]))
    result = lint_skill(skill, known_skill_names=None)
    assert "HSL050" not in rule_ids(result)


def test_missing_script_reference_is_hsl060(tmp_path):
    skill = make_skill(tmp_path, body="Run `python scripts/run.py` to start.\n")
    (skill / "scripts").mkdir()
    result = lint_skill(skill)
    findings = [f for f in result.findings if f.rule_id == "HSL060"]
    assert len(findings) == 1
    assert "scripts/run.py" in findings[0].field


def test_existing_script_reference_passes(tmp_path):
    skill = make_skill(tmp_path, body="Run `python scripts/run.py` to start.\n")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    result = lint_skill(skill)
    assert "HSL060" not in rule_ids(result)


def test_unanchored_path_reference_is_ignored(tmp_path):
    # docs/ is neither a resource dir nor an existing directory — too likely
    # to be prose, so HSL060 stays quiet.
    skill = make_skill(tmp_path, body="See docs/guide.md upstream.\n")
    result = lint_skill(skill)
    assert "HSL060" not in rule_ids(result)


# ---------------------------------------------------------------------------
# parse_frontmatter_strict
# ---------------------------------------------------------------------------

def test_parse_frontmatter_strict_valid():
    fm, body, err = parse_frontmatter_strict("---\nname: x\n---\nbody\n")
    assert fm == {"name": "x"}
    assert body == "body\n"
    assert err is None


def test_parse_frontmatter_strict_matches_lenient_on_valid_input():
    content = CLEAN_FRONTMATTER.format(name="x") + "body\n"
    strict_fm, strict_body, err = parse_frontmatter_strict(content)
    lenient_fm, lenient_body = parse_frontmatter(content)
    assert err is None
    assert strict_fm == lenient_fm
    assert strict_body == lenient_body
