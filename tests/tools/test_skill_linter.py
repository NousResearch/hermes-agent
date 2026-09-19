"""Tests for tools/skill_linter.py — the advisory SKILL.md convention linter."""

from pathlib import Path

import pytest

from tools.skill_linter import (
    ERROR,
    WARNING,
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
    assert any(f.severity == ERROR for f in findings)


def test_name_dir_mismatch_is_error(tmp_path):
    skill_dir = tmp_path / "actual-dir"
    skill_dir.mkdir()
    findings = lint_content(CLEAN, skill_dir=skill_dir)  # name is my-skill
    assert "name-dir-mismatch" in _rules(findings)
    assert any(f.severity == ERROR for f in findings)


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


def test_findings_carry_rule_and_severity():
    findings = lint_content(CLEAN.replace("name: my-skill", "name: BAD"))
    assert any(f.rule == "name-format" and f.severity == ERROR for f in findings)


def test_incident_log_shape_flagged_and_rule_shape_not():
    # A body narrating incidents by PR number is a log, not a lesson; the same lesson stated as a
    # rule + why with no numbers passes. Density-gated so one citation in a long body is fine.
    log = CLEAN.replace(
        "1. Use `read_file` to load it.",
        "In #12345 the watcher died; #23456 was the same; see PR #34567 and issue #45678 for the fix.",
    )
    rule = CLEAN.replace(
        "1. Use `read_file` to load it.",
        "Launch the watcher from a directory that outlives the watch; a deleted cwd reads as a stall.",
    )
    assert "incident-log-shape" in _rules(lint_content(log))
    assert "incident-log-shape" not in _rules(lint_content(rule))


def test_references_sprawl_flagged_above_cap(tmp_path):
    from tools.skill_linter import _MAX_REFERENCE_FILES
    skill_dir = tmp_path / "my-skill"
    refs = skill_dir / "references"
    refs.mkdir(parents=True)
    for i in range(_MAX_REFERENCE_FILES + 1):
        (refs / f"note-{i}.md").write_text("x")
    (skill_dir / "SKILL.md").write_text(CLEAN)
    assert "references-sprawl" in _rules(lint_skill(skill_dir / "SKILL.md"))
    (refs / f"note-{_MAX_REFERENCE_FILES}.md").unlink()
    assert "references-sprawl" not in _rules(lint_skill(skill_dir / "SKILL.md"))


# ---------------------------------------------------------------------------
# scripts/ anchors and collection-level rules (lint_collection)
# ---------------------------------------------------------------------------

def _skill(root, name, description, related=(), tags=("x",), body_extra=""):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    content = (CLEAN.replace("name: my-skill", f"name: {name}")
               .replace("Search arXiv papers by keyword, author, or ID.", description)
               .replace("tags: [arxiv, research]", f"tags: [{', '.join(tags)}]")
               .replace("related_skills: []", f"related_skills: [{', '.join(related)}]"))
    (skill_dir / "SKILL.md").write_text(content + body_extra, encoding="utf-8")
    return skill_dir


def test_scripts_anchor_dangling_when_skill_ships_scripts_dir(tmp_path):
    skill_dir = tmp_path / "my-skill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "present.py").write_text("print(1)\n", encoding="utf-8")
    content = CLEAN + "\nRun scripts/present.py, then scripts/renamed.py.\n"
    findings = [f for f in lint_content(content, skill_dir=skill_dir) if f.rule == "dangling-reference"]
    assert len(findings) == 1 and "scripts/renamed.py" in findings[0].message


def test_scripts_anchor_ignored_without_scripts_dir(tmp_path):
    # Dev skills legitimately cite repo-root scripts/; only a skill that ships its own
    # scripts/ owns the paths it names.
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    content = CLEAN + "\nRun scripts/run_tests.sh from the repo root.\n"
    assert "dangling-reference" not in _rules(lint_content(content, skill_dir=skill_dir))


def test_collection_clean_pair_is_silent(tmp_path):
    from tools.skill_linter import lint_collection
    a = _skill(tmp_path, "arxiv-search", "Search arXiv papers by keyword, author, or ID.", related=("pdf-reader",))
    b = _skill(tmp_path, "pdf-reader", "Extract text and tables from PDF files.")
    assert lint_collection([a, b]) == []


def test_related_skill_missing_flagged_with_attribution(tmp_path):
    from tools.skill_linter import lint_collection
    a = _skill(tmp_path, "arxiv-search", "Search arXiv papers by keyword, author, or ID.",
               related=("pdf-reader", "vllm", "gguf"))
    b = _skill(tmp_path, "pdf-reader", "Extract text and tables from PDF files.")
    findings = [f for f in lint_collection([a, b]) if f.rule == "related-skill-missing"]
    assert len(findings) == 1
    assert findings[0].skill == "arxiv-search" and findings[0].severity == WARNING
    assert "'vllm'" in findings[0].message and "'gguf'" in findings[0].message
    assert "pdf-reader" not in findings[0].message


def test_related_skill_accepts_namespace_qualified_and_dir_names(tmp_path):
    from tools.skill_linter import lint_collection
    a = _skill(tmp_path, "arxiv-search", "Search arXiv papers by keyword, author, or ID.",
               related=("plugin:pdf-reader",))
    b = _skill(tmp_path, "pdf-reader", "Extract text and tables from PDF files.")
    assert "related-skill-missing" not in _rules(lint_collection([a, b]))


def test_alias_overlap_flags_near_duplicate_once(tmp_path):
    from tools.skill_linter import lint_collection
    a = _skill(tmp_path, "arxiv-search", "Search arXiv papers by keyword, author, or ID.")
    b = _skill(tmp_path, "paper-finder", "Search arXiv papers by author, keyword, or ID.")
    c = _skill(tmp_path, "pdf-reader", "Extract text and tables from PDF files.")
    findings = [f for f in lint_collection([a, b, c]) if f.rule == "alias-overlap"]
    assert len(findings) == 1  # the pair is reported once, on the first skill
    assert findings[0].skill == "arxiv-search" and "'paper-finder'" in findings[0].message
    assert findings[0].severity == WARNING


def test_alias_overlap_same_tags_lowers_threshold(tmp_path):
    from tools.skill_linter import lint_collection
    # 50% shared trigger words: below the general bar, above the same-lane bar.
    a = _skill(tmp_path, "deploy-k8s", "Deploy manifests to a Kubernetes cluster.", tags=("k8s", "deploy"))
    b = _skill(tmp_path, "k8s-push", "Push Kubernetes manifests to a cluster with canaries.", tags=("k8s", "deploy"))
    c = _skill(tmp_path, "k8s-canary", "Push Kubernetes manifests to a cluster with canaries.", tags=("k8s", "canary"))
    assert "alias-overlap" in _rules(lint_collection([a, b]))
    assert "alias-overlap" not in _rules(lint_collection([a, c]))


def test_alias_overlap_keeps_the_skill_name_as_a_distinction(tmp_path):
    # Same operation delegated to different products is a family, not an alias: the product
    # name in each description is what routes, so it must count as a distinguishing token.
    from tools.skill_linter import lint_collection
    a = _skill(tmp_path, "claude-code", "Delegate coding to Claude Code CLI (features, PRs).", tags=("claude",))
    b = _skill(tmp_path, "codex", "Delegate coding to OpenAI Codex CLI (features, PRs).", tags=("codex",))
    assert "alias-overlap" not in _rules(lint_collection([a, b]))


def test_bland_trigger_flags_generic_descriptions_only(tmp_path):
    from tools.skill_linter import lint_collection
    a = _skill(tmp_path, "helper", "Helps with tasks.")
    b = _skill(tmp_path, "utils", "General utilities.")
    c = _skill(tmp_path, "arxiv-search", "Search arXiv papers by keyword, author, or ID.")
    findings = [f for f in lint_collection([a, b, c]) if f.rule == "bland-trigger"]
    assert {f.skill for f in findings} == {"helper", "utils"}
    assert all(f.severity == WARNING for f in findings)


def test_bland_trigger_discounts_the_skill_name_itself(tmp_path):
    # "Kubernetes tools." on a skill named kubernetes-tools carries no trigger beyond its name.
    from tools.skill_linter import lint_collection
    a = _skill(tmp_path, "kubernetes-tools", "Kubernetes tools.")
    b = _skill(tmp_path, "kubernetes-deploy", "Deploy manifests to a Kubernetes cluster.")
    findings = [f for f in lint_collection([a, b]) if f.rule == "bland-trigger"]
    assert [f.skill for f in findings] == ["kubernetes-tools"]


def test_lint_collection_skips_unreadable_and_empty(tmp_path):
    from tools.skill_linter import lint_collection
    assert lint_collection([]) == []
    assert lint_collection([tmp_path / "does-not-exist"]) == []
