"""Tests for the owner-qualified cross-skill reference resolution in skill_linter.

A SKILL.md body may cite a reference owned by ANOTHER skill, spelled as
``other-skill/references/topic.md`` (or ``category/other-skill/references/...``).
The linter must resolve that mention instead of reporting it as dangling.

Run: pytest tests/tools/test_skill_linter_owner_prefix.py -q
"""

from pathlib import Path

from tools.skill_linter import lint_content

HEAD = """---
name: {name}
description: {desc}
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [test]
    related_skills: []
---

# {title}

## Overview
Body.

## When to Use
- When the user wants X.
"""


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _dangling(findings) -> list[str]:
    return [f.message for f in findings if f.rule == "dangling-reference"]


def test_owner_qualified_reference_resolves(tmp_path):
    """A reference owned by another skill is not dangling."""
    owner = tmp_path / "other-skill"
    _write(owner, "references/topic.md", "# Topic\n")
    consumer = tmp_path / "consumer-skill"
    consumer.mkdir(parents=True)
    body = HEAD.format(
        name="consumer-skill",
        desc="Consumer skill.",
        title="Consumer",
    ) + "\nSee `other-skill/references/topic.md` for details.\n"
    (consumer / "SKILL.md").write_text(body, encoding="utf-8")

    findings = lint_content(body, skill_dir=consumer)
    assert _dangling(findings) == [], f"owner-qualified ref wrongly dangling: {_dangling(findings)}"


def test_owner_qualified_reference_in_category_layout_resolves(tmp_path):
    """``category/other-skill/references/x.md`` resolves against the skills root."""
    _write(tmp_path, "devops/other-skill/references/topic.md", "# Topic\n")
    consumer = tmp_path / "devops" / "consumer-skill"
    consumer.mkdir(parents=True)
    body = HEAD.format(
        name="consumer-skill",
        desc="Consumer skill.",
        title="Consumer",
    ) + "\nSee `devops/other-skill/references/topic.md` for details.\n"
    (consumer / "SKILL.md").write_text(body, encoding="utf-8")

    findings = lint_content(body, skill_dir=consumer)
    assert _dangling(findings) == [], f"category owner ref wrongly dangling: {_dangling(findings)}"


def test_genuinely_missing_reference_is_still_dangling(tmp_path):
    """Owner-prefix resolution must not silence a real missing file."""
    consumer = tmp_path / "consumer-skill"
    consumer.mkdir(parents=True)
    body = HEAD.format(
        name="consumer-skill",
        desc="Consumer skill.",
        title="Consumer",
    ) + "\nSee `other-skill/references/missing.md` for details.\n"
    (consumer / "SKILL.md").write_text(body, encoding="utf-8")

    findings = lint_content(body, skill_dir=consumer)
    assert len(_dangling(findings)) == 1, "a truly missing reference must still be reported"


def test_own_missing_reference_is_still_dangling(tmp_path):
    """A skill's own missing reference is still reported."""
    consumer = tmp_path / "consumer-skill"
    consumer.mkdir(parents=True)
    body = HEAD.format(
        name="consumer-skill",
        desc="Consumer skill.",
        title="Consumer",
    ) + "\nSee `references/missing.md` for details.\n"
    (consumer / "SKILL.md").write_text(body, encoding="utf-8")

    findings = lint_content(body, skill_dir=consumer)
    assert len(_dangling(findings)) == 1, "a missing own reference must still be reported"


def test_fenced_example_is_not_a_link(tmp_path):
    """A fenced code block spelling out `references/file.md` is documentation."""
    consumer = tmp_path / "consumer-skill"
    consumer.mkdir(parents=True)
    body = HEAD.format(
        name="consumer-skill",
        desc="Consumer skill.",
        title="Consumer",
    ) + "\nWrite links like this:\n\n```markdown\nreferences/file.md\n```\n"
    (consumer / "SKILL.md").write_text(body, encoding="utf-8")

    findings = lint_content(body, skill_dir=consumer)
    assert _dangling(findings) == [], f"fenced example wrongly dangling: {_dangling(findings)}"


def test_url_path_is_not_a_local_link(tmp_path):
    """A remote URL containing an assets/ path is not a local skill file."""
    consumer = tmp_path / "consumer-skill"
    consumer.mkdir(parents=True)
    body = HEAD.format(
        name="consumer-skill",
        desc="Consumer skill.",
        title="Consumer",
    ) + "\nSee https://example.com/docs/assets/schema.json for details.\n"
    (consumer / "SKILL.md").write_text(body, encoding="utf-8")

    findings = lint_content(body, skill_dir=consumer)
    assert _dangling(findings) == [], f"URL path wrongly dangling: {_dangling(findings)}"


def test_directory_mention_is_not_a_link(tmp_path):
    """A bare directory mention (`references/`) is not a file link."""
    consumer = tmp_path / "consumer-skill"
    consumer.mkdir(parents=True)
    body = HEAD.format(
        name="consumer-skill",
        desc="Consumer skill.",
        title="Consumer",
    ) + "\nPut topic files under references/ and assets/.\n"
    (consumer / "SKILL.md").write_text(body, encoding="utf-8")

    findings = lint_content(body, skill_dir=consumer)
    assert _dangling(findings) == [], f"directory mention wrongly dangling: {_dangling(findings)}"


def test_existing_own_reference_still_resolves(tmp_path):
    """The common case (own reference present) is unaffected."""
    consumer = tmp_path / "consumer-skill"
    _write(consumer, "references/topic.md", "# Topic\n")
    body = HEAD.format(
        name="consumer-skill",
        desc="Consumer skill.",
        title="Consumer",
    ) + "\nSee `references/topic.md` for details.\n"
    (consumer / "SKILL.md").write_text(body, encoding="utf-8")

    findings = lint_content(body, skill_dir=consumer)
    assert _dangling(findings) == [], f"own present ref wrongly dangling: {_dangling(findings)}"
