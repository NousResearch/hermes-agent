"""Tests for deterministic Hermes skill quality grading."""

import json
from pathlib import Path

import pytest

from tools.skill_quality import (
    GRADE_A,
    GRADE_F,
    GRADE_S,
    grade_skill,
    render_quality_report,
)


def _write_skill(path: Path, content: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    skill_md = path / "SKILL.md"
    skill_md.write_text(content, encoding="utf-8")
    return path


def test_valid_workflow_skill_grades_s(tmp_path):
    skill_dir = _write_skill(
        tmp_path / "weekly-report",
        """---
name: weekly-report
description: Use when preparing recurring weekly reports from chat notes and email snippets.
---

# Weekly Report

1. Collect chat and email evidence.
2. Summarize by MBO axis.
3. Verify blockers and next plans.
""",
    )

    report = grade_skill(skill_dir)

    assert report.grade == GRADE_S
    assert report.failed_counts == {"BLOCKER": 0, "MAJOR": 0, "MINOR": 0}


def test_body_only_trigger_is_blocker(tmp_path):
    skill_dir = _write_skill(
        tmp_path / "hidden-trigger",
        """---
name: hidden-trigger
description: A helper for reports.
---

# Hidden Trigger

Use when preparing recurring weekly reports from Telegram and email.
""",
    )

    report = grade_skill(skill_dir)

    assert report.grade == GRADE_F
    finding = next(item for item in report.findings if item.id == "3.4")
    assert finding.status == "fail"
    assert finding.severity == "BLOCKER"
    assert "description" in finding.how_to_fix


def test_major_count_drives_non_blocking_grade(tmp_path):
    skill_dir = _write_skill(
        tmp_path / "thin-skill",
        """---
name: thin-skill
description: Use when doing any quality task.
---

# Thin Skill

Read carefully and be helpful.
""",
    )

    report = grade_skill(skill_dir)

    assert report.grade == GRADE_A
    assert report.failed_counts["BLOCKER"] == 0
    assert report.failed_counts["MAJOR"] in {1, 2}


def test_script_syntax_failure_is_major(tmp_path):
    skill_dir = _write_skill(
        tmp_path / "scripted-skill",
        """---
name: scripted-skill
description: Use when running a repeatable scripted workflow.
---

# Scripted Skill

Run `scripts/broken.py` to execute the deterministic workflow.
""",
    )
    scripts = skill_dir / "scripts"
    scripts.mkdir()
    (scripts / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    report = grade_skill(skill_dir)

    finding = next(item for item in report.findings if item.id == "5.6")
    assert finding.status == "fail"
    assert finding.severity == "MAJOR"
    assert report.grade == GRADE_A


def test_reference_symlink_escape_is_not_read_for_quality_residue(tmp_path):
    skill_dir = _write_skill(
        tmp_path / "symlink-skill",
        """---
name: symlink-skill
description: Use when checking symlink handling in skill quality reports.
---

# Symlink Skill

Step 1: Inspect only files inside the skill directory.
""",
    )
    refs = skill_dir / "references"
    refs.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("TODO outside marker should not be read by residue checks\n", encoding="utf-8")
    try:
        (refs / "leak.md").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is not available on this platform")

    report = grade_skill(skill_dir)

    residue = next(item for item in report.findings if item.id == "5.8")
    safety = next(item for item in report.findings if item.id == "6.1")
    assert residue.status == "pass"
    assert safety.status == "fail"
    assert "symlink" in safety.why.lower()


def test_render_quality_report_includes_actionable_failures(tmp_path):
    skill_dir = _write_skill(
        tmp_path / "hidden-trigger",
        """---
name: hidden-trigger
description: A helper for reports.
---

# Hidden Trigger

Use when preparing recurring weekly reports from Telegram and email.
""",
    )

    markdown = render_quality_report(grade_skill(skill_dir))

    assert "grade F" in markdown
    assert "Blocking issues" in markdown
    assert "why:" in markdown
    assert "how_to_fix:" in markdown
    assert "3.4" in markdown


def test_report_json_shape_is_stable(tmp_path):
    skill_dir = _write_skill(
        tmp_path / "weekly-report",
        """---
name: weekly-report
description: Use when preparing recurring weekly reports from chat notes and email snippets.
---

# Weekly Report

Step 1. Collect evidence.
""",
    )

    data = grade_skill(skill_dir).to_dict()

    assert json.loads(json.dumps(data))["grade"] == GRADE_S
    assert data["skill_name"] == "weekly-report"
    assert isinstance(data["findings"], list)
