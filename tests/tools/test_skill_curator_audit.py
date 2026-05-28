"""Tests for deterministic skill lifecycle curator audits."""

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "skill_curator_audit.py"
SPEC = importlib.util.spec_from_file_location("skill_curator_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
skill_curator_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(skill_curator_audit)

audit_skills = skill_curator_audit.audit_skills
render_report = skill_curator_audit.render_report


def write_skill(root: Path, rel: str, frontmatter: str, body: str = "# Skill\n\nBody.\n") -> Path:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\n{frontmatter}---\n\n{body}", encoding="utf-8")
    return skill_dir


def test_flags_stale_learning_skill(tmp_path):
    write_skill(
        tmp_path,
        "research/stale-learning",
        """name: stale-learning
description: A stale learning skill.
metadata:
  hermes:
    lifecycle:
      status: learning
      last_validated: '2026-04-01'
      validation_level: observed
""",
        body="# Stale\n\n## Validation & Retention\n\n- Falsifier: update when contradicted.\n",
    )

    report = audit_skills(tmp_path, today="2026-05-27", stale_learning_days=30)

    assert report["summary"]["skills_audited"] == 1
    assert report["summary"]["findings"] == 1
    assert report["findings"][0]["check"] == "stale_learning"
    assert report["findings"][0]["skill"] == "stale-learning"
    assert "56 days" in report["findings"][0]["message"]


def test_flags_deprecated_skill_missing_replacement(tmp_path):
    write_skill(
        tmp_path,
        "devops/old-skill",
        """name: old-skill
description: A deprecated skill.
metadata:
  hermes:
    lifecycle:
      status: deprecated
""",
        body="# Old\n\n## Validation & Retention\n\n- Falsifier: superseded.\n",
    )

    report = audit_skills(tmp_path, today="2026-05-27")

    assert [f["check"] for f in report["findings"]] == ["deprecated_missing_replacement"]
    assert report["findings"][0]["action"] == "Add metadata.hermes.lifecycle.superseded_by."


def test_flags_bloated_references_folder_without_raw_content(tmp_path):
    skill_dir = write_skill(
        tmp_path,
        "ops/bloated",
        """name: bloated
description: A skill with too much retained evidence.
metadata:
  hermes:
    lifecycle:
      status: candidate
      validation_level: repeated
""",
        body="# Bloated\n\n## Validation & Retention\n\n- Falsifier: if receipts stop reproducing.\n",
    )
    refs = skill_dir / "references"
    refs.mkdir()
    retained = "SECRET_TOKEN=***" + ("x" * 2048)
    (refs / "raw-transcript.txt").write_text(retained, encoding="utf-8")

    report = audit_skills(tmp_path, today="2026-05-27", max_references_bytes=1024)
    rendered = render_report(report)

    assert report["findings"][0]["check"] == "bloated_references"
    assert "raw-transcript.txt" not in rendered
    assert "SECRET_TOKEN" not in rendered
    assert f"references/ is {len(retained)} bytes" in rendered


def test_report_order_is_deterministic(tmp_path):
    write_skill(
        tmp_path,
        "z/zeta",
        """name: zeta
description: Z skill.
metadata:
  hermes:
    lifecycle:
      status: deprecated
""",
    )
    write_skill(
        tmp_path,
        "a/alpha",
        """name: alpha
description: A skill.
metadata:
  hermes:
    lifecycle:
      status: deprecated
""",
    )

    report = audit_skills(tmp_path, today="2026-05-27")

    assert [finding["skill"] for finding in report["findings"]] == ["alpha", "zeta"]
