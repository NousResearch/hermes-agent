import json
from pathlib import Path

from tools.skill_cleaner import audit_skills, main, report_to_dict, write_report_files


def _write_skill(root: Path, rel: str, *, name: str, description: str, body: str) -> Path:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n"
        f"# {name}\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _healthy_body(topic: str) -> str:
    return f"""
Use when auditing {topic} skills for reusable operating guidance.

## Workflow
1. Inspect the card frontmatter and body.
2. Compare the procedure against related skills.
3. Record report-only findings for Glen approval.

## Verification
Validate the output artifact and do not mutate skills during the audit.
"""


def test_audit_uses_active_profile_home_and_writes_report(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    skills_root = tmp_path / "profile-a" / "skills"
    _write_skill(
        skills_root,
        "ops/test-audit",
        name="test-audit",
        description="Audit skill for report-only cleaner tests.",
        body=_healthy_body("Hermes"),
    )

    report = audit_skills()
    data = report_to_dict(report)

    assert report.hermes_home == str(tmp_path / "profile-a")
    assert report.scanned_roots == [str(skills_root)]
    assert data["summary"]["skill_count"] == 1
    assert report.skills[0].name == "test-audit"
    assert report.skills[0].source == "active-profile"

    md_path, json_path = write_report_files(report)
    assert md_path.is_file()
    assert json_path.is_file()
    assert str(tmp_path / "profile-a" / "reports" / "skill-cleaner") in str(md_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["skill_count"] == 1


def test_verified_card_contract_flags_thin_missing_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
    skill_dir = tmp_path / "profile-b" / "skills" / "bad-card"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Bad\n\nToo short.\n", encoding="utf-8")

    report = audit_skills()
    codes = {finding.code for finding in report.skills[0].card_findings}

    assert "missing_frontmatter" in codes
    assert "missing_description" in codes
    assert "thin_body" in codes
    assert "missing_skill_card" in codes
    assert report.finding_counts["error"] >= 2


def test_audit_includes_configured_external_skill_dirs(tmp_path, monkeypatch):
    home = tmp_path / "profile-external"
    external_root = tmp_path / "external-skills"
    monkeypatch.setenv("HERMES_HOME", str(home))
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "skills:\n  external_dirs:\n    - " + str(external_root) + "\n",
        encoding="utf-8",
    )
    _write_skill(
        home / "skills",
        "ops/local-audit",
        name="local-audit",
        description="Local audit skill.",
        body=_healthy_body("local"),
    )
    _write_skill(
        external_root,
        "ops/external-audit",
        name="external-audit",
        description="External audit skill.",
        body=_healthy_body("external"),
    )

    report = audit_skills()

    assert str(home / "skills") in report.scanned_roots
    assert str(external_root) in report.scanned_roots
    assert {skill.source for skill in report.skills} == {"active-profile", "external"}


def test_duplicate_detection_can_include_bundled_root(tmp_path, monkeypatch):
    active_home = tmp_path / "profile-c"
    bundled_root = tmp_path / "bundled-skills"
    monkeypatch.setenv("HERMES_HOME", str(active_home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled_root))

    duplicate_body = _healthy_body("duplicate skill overlap") + "\nshared terraform dns deployment audit approval gate " * 8
    _write_skill(
        active_home / "skills",
        "operations/dns-audit",
        name="dns-audit",
        description="Audit DNS deployment gates and approvals.",
        body=duplicate_body,
    )
    _write_skill(
        bundled_root,
        "operations/dns-audit-copy",
        name="dns-audit-copy",
        description="Audit DNS deployment gates and approvals.",
        body=duplicate_body,
    )

    without_bundled = audit_skills(include_bundled=False, similarity_threshold=0.8)
    with_bundled = audit_skills(include_bundled=True, similarity_threshold=0.8)

    assert len(without_bundled.skills) == 1
    assert len(with_bundled.skills) == 2
    assert with_bundled.duplicates
    assert with_bundled.duplicates[0].similarity >= 0.8


def test_session_artifact_detection_ignores_numeric_date_references(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-dates"))
    _write_skill(
        tmp_path / "profile-dates" / "skills",
        "ops/date-reference",
        name="date-reference",
        description="Audit dated reference names without false SHA matches.",
        body=_healthy_body("dated references") + "\nOpen references/report-20260525.md when needed.",
    )
    _write_skill(
        tmp_path / "profile-dates" / "skills",
        "ops/sha-reference",
        name="sha-reference",
        description="Audit real SHA-like artifacts.",
        body=_healthy_body("SHA references") + "\nMove stale detail for deadbee when found.",
    )

    report = audit_skills()
    findings_by_name = {skill.name: {finding.code for finding in skill.card_findings} for skill in report.skills}

    assert "session_artifact" not in findings_by_name["date-reference"]
    assert "session_artifact" in findings_by_name["sha-reference"]


def test_cli_no_write_json_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-d"))
    _write_skill(
        tmp_path / "profile-d" / "skills",
        "ops/test-audit",
        name="test-audit",
        description="Audit skill for CLI smoke tests.",
        body=_healthy_body("CLI"),
    )

    assert main(["--no-write", "--json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["summary"]["skill_count"] == 1
    assert "artifacts" not in payload
