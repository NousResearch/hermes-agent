from pathlib import Path

import pytest

from tools.bundled_skill_validator import (
    available_bundled_skill_checks,
    validate_bundled_skills,
)


def _write_skill(root: Path, rel: str, body: str) -> Path:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return skill_dir


def _issue_codes(root: Path) -> set[str]:
    return {issue.code for issue in validate_bundled_skills(root)}


def test_validate_bundled_skills_flags_missing_local_file_references(tmp_path):
    _write_skill(
        tmp_path,
        "productivity/slides",
        """---
name: slides
---

Run `python scripts/build_deck.py`.
See `references/style.md` before writing.
""",
    )
    (tmp_path / "productivity/slides/scripts").mkdir()
    (tmp_path / "productivity/slides/scripts/build_deck.py").write_text(
        "print('ok')\n",
        encoding="utf-8",
    )

    issues = validate_bundled_skills(tmp_path)

    assert [(issue.code, issue.target) for issue in issues] == [
        ("missing-file-reference", "references/style.md")
    ]


def test_validate_bundled_skills_flags_unknown_related_skills(tmp_path):
    _write_skill(
        tmp_path,
        "research/wiki",
        """---
name: wiki
related_skills: [known-skill, missing-skill]
---

Use the related skills when useful.
""",
    )
    _write_skill(
        tmp_path,
        "research/known",
        """---
name: known-skill
---

Known helper.
""",
    )

    issues = validate_bundled_skills(tmp_path)

    assert [(issue.code, issue.target) for issue in issues] == [
        ("unknown-related-skill", "missing-skill")
    ]


def test_validate_bundled_skills_reads_nested_hermes_related_skills(tmp_path):
    _write_skill(
        tmp_path,
        "research/wiki",
        """---
name: wiki
metadata:
  hermes:
    related_skills: [known-skill, missing-skill]
---

Use the related skills when useful.
""",
    )
    _write_skill(
        tmp_path,
        "research/known",
        """---
name: known-skill
---

Known helper.
""",
    )

    issues = validate_bundled_skills(tmp_path)

    assert [(issue.code, issue.target) for issue in issues] == [
        ("unknown-related-skill", "missing-skill")
    ]


def test_validate_bundled_skills_flags_unknown_skill_tool_references(tmp_path):
    _write_skill(
        tmp_path,
        "research/paper",
        """---
name: paper
---

Call `skill_view("known-skill")` first.
Call `skill_view("missing-view")` for diagrams.
Call `skill_manage("install", "missing-install")` when needed.
""",
    )
    _write_skill(
        tmp_path,
        "research/known",
        """---
name: known-skill
---

Known helper.
""",
    )

    assert _issue_codes(tmp_path) == {
        "unknown-skill-view-reference",
        "unknown-skill-manage-reference",
    }


def test_validate_bundled_skills_flags_hermes_tools_in_bash_fences(tmp_path):
    _write_skill(
        tmp_path,
        "research/wiki",
        """---
name: wiki
---

```bash
read_file "$WIKI/SCHEMA.md"
search_files "$WIKI" "topic"
```
""",
    )

    issues = validate_bundled_skills(tmp_path)

    assert [(issue.code, issue.target) for issue in issues] == [
        ("hermes-tool-in-bash-fence", "read_file"),
        ("hermes-tool-in-bash-fence", "search_files"),
    ]


def test_validate_bundled_skills_can_run_a_named_check_subset(tmp_path):
    _write_skill(
        tmp_path,
        "productivity/slides",
        """---
name: slides
related_skills: [missing-helper]
---

Run `python scripts/missing.py`.
""",
    )

    issues = validate_bundled_skills(tmp_path, checks=["related-skills"])

    assert available_bundled_skill_checks() == (
        "local-file-references",
        "related-skills",
        "skill-tool-references",
        "bash-fence-tools",
    )
    assert [(issue.code, issue.target) for issue in issues] == [
        ("unknown-related-skill", "missing-helper")
    ]


def test_validate_bundled_skills_rejects_unknown_check_names(tmp_path):
    with pytest.raises(ValueError, match="Unknown bundled skill validation check"):
        validate_bundled_skills(tmp_path, checks=["freshness"])
