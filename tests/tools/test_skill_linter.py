"""Tests for tools/skill_linter.py — the advisory SKILL.md convention linter."""

from pathlib import Path

import pytest

from tools.skill_linter import (
    ERROR,
    WARNING,
    format_findings,
    has_errors,
    lint_content,
    lint_skill,
)

# A clean, peer-shaped SKILL.md that should produce zero findings.
CLEAN = """---
name: my-skill
description: Search arXiv papers by keyword, author, or ID.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [arxiv, research]
    related_skills: []
---

# My Skill

## Overview
Does a thing.

## When to Use
- When the user wants X.

## Procedure
1. Use `read_file` to load it.
"""


def _rules(findings):
    return {f.rule for f in findings}


def test_clean_skill_has_no_findings():
    assert lint_content(CLEAN) == []


def test_description_too_long_is_warning():
    long_desc = "x" * 80
    content = CLEAN.replace(
        "Search arXiv papers by keyword, author, or ID.", long_desc
    )
    findings = lint_content(content)
    assert "description-length" in _rules(findings)
    assert all(f.severity == WARNING for f in findings)


def test_marketing_words_flagged():
    content = CLEAN.replace(
        "Search arXiv papers by keyword, author, or ID.",
        "A powerful comprehensive tool.",
    )
    findings = lint_content(content)
    assert "description-marketing" in _rules(findings)


def test_shell_utility_reference_in_prose_flagged():
    content = CLEAN.replace("Use `read_file` to load it.", "Use `grep` to find it.")
    findings = lint_content(content)
    assert "shell-utility-reference" in _rules(findings)


def test_shell_utility_inside_code_block_not_flagged():
    # A fenced code block legitimately shows grep; prose check must skip it.
    content = CLEAN + "\n```bash\ngrep -r foo .\n```\n"
    findings = lint_content(content)
    assert "shell-utility-reference" not in _rules(findings)


def test_missing_metadata_block_warns():
    content = """---
name: bare-skill
description: Does a thing briefly.
---

# Bare Skill

## When to Use
- now
"""
    findings = lint_content(content)
    rules = _rules(findings)
    assert "missing-metadata" in rules


def test_missing_when_to_use_section_warns():
    content = CLEAN.replace("## When to Use\n- When the user wants X.\n", "")
    findings = lint_content(content)
    assert "missing-section" in _rules(findings)


def test_bad_name_format_is_error():
    content = CLEAN.replace("name: my-skill", "name: My_Skill!")
    findings = lint_content(content)
    assert "name-format" in _rules(findings)
    assert has_errors(findings)


def test_name_dir_mismatch_is_error(tmp_path):
    skill_dir = tmp_path / "actual-dir"
    skill_dir.mkdir()
    findings = lint_content(CLEAN, skill_dir=skill_dir)  # name is my-skill
    assert "name-dir-mismatch" in _rules(findings)
    assert has_errors(findings)


def test_dangling_reference_link_flagged(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    content = CLEAN + "\nSee references/missing.md for detail.\n"
    findings = lint_content(content, skill_dir=skill_dir)
    assert "dangling-reference" in _rules(findings)


def test_present_reference_link_not_flagged(tmp_path):
    skill_dir = tmp_path / "my-skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "references" / "detail.md").write_text("x")
    content = CLEAN + "\nSee references/detail.md for detail.\n"
    findings = lint_content(content, skill_dir=skill_dir)
    assert "dangling-reference" not in _rules(findings)


def test_posix_primitive_without_platforms_warns(tmp_path):
    skill_dir = tmp_path / "my-skill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "run.py").write_text("import fcntl\nfcntl.flock(1, 2)\n")
    findings = lint_content(CLEAN, skill_dir=skill_dir)
    assert "platforms-gating" in _rules(findings)


def test_posix_primitive_with_platforms_ok(tmp_path):
    skill_dir = tmp_path / "my-skill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "run.py").write_text("import fcntl\n")
    content = CLEAN.replace(
        "version: 1.0.0", "version: 1.0.0\nplatforms: [linux, macos]"
    )
    findings = lint_content(content, skill_dir=skill_dir)
    assert "platforms-gating" not in _rules(findings)


def test_forbidden_file_flagged(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "README.md").write_text("# readme")
    findings = lint_content(CLEAN, skill_dir=skill_dir)
    assert "forbidden-file" in _rules(findings)


def test_invalid_platforms_value_warns():
    content = CLEAN.replace(
        "version: 1.0.0", "version: 1.0.0\nplatforms: [linux, solaris]"
    )
    findings = lint_content(content)
    assert "platforms-value" in _rules(findings)


def test_lint_skill_reads_from_disk(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(CLEAN)
    findings = lint_skill(skill_md)
    assert findings == []


def test_author_caps_warned():
    content = CLEAN.replace("author: Hermes Agent", "author: hermes agent")
    findings = lint_content(content)
    assert "author-caps" in _rules(findings)


def test_format_findings_renders():
    findings = lint_content(CLEAN.replace("name: my-skill", "name: BAD"))
    out = format_findings(findings)
    assert "name-format" in out
