import hashlib
import json
from pathlib import Path

from tools.skill_cleaner import (
    _canonical_skill_content_for_hash,
    audit_skills,
    main,
    report_to_dict,
    write_report_files,
)


def _write_skill(
    root: Path, rel: str, *, name: str, description: str, body: str
) -> Path:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n\n{body}\n",
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


def test_documented_frontmatter_contract_flags_thin_missing_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
    skill_dir = tmp_path / "profile-b" / "skills" / "bad-card"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Bad\n\nToo short.\n", encoding="utf-8")

    report = audit_skills()
    codes = {finding.code for finding in report.skills[0].card_findings}

    assert "missing_frontmatter" in codes
    assert "missing_description" in codes
    assert "thin_body" in codes
    assert "missing_skill_card" not in codes
    assert report.finding_counts["error"] >= 2


def test_optional_skill_card_hash_excludes_its_own_field(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-hash"))
    skill_dir = tmp_path / "profile-hash" / "skills" / "hash-audit"
    skill_dir.mkdir(parents=True)
    template = (
        "---\n"
        "name: hash-audit\n"
        "description: Verify optional content hashes.\n"
        "skill_card:\n"
        "  verification:\n"
        "    content_sha256: HASH_PLACEHOLDER\n"
        "---\n"
        "# hash-audit\n\n" + _healthy_body("content hashes") + "\n"
    )
    digest = hashlib.sha256(
        _canonical_skill_content_for_hash(template).encode("utf-8")
    ).hexdigest()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(template.replace("HASH_PLACEHOLDER", digest), encoding="utf-8")

    report = audit_skills()
    codes = {finding.code for finding in report.skills[0].card_findings}
    assert "content_hash_mismatch" not in codes

    skill_md.write_text(
        skill_md.read_text(encoding="utf-8") + "Changed after review.\n",
        encoding="utf-8",
    )
    changed_report = audit_skills()
    changed_codes = {finding.code for finding in changed_report.skills[0].card_findings}
    assert "content_hash_mismatch" in changed_codes


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

    duplicate_body = (
        _healthy_body("duplicate skill overlap")
        + "\nshared terraform dns deployment audit approval gate " * 8
    )
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


def test_session_artifact_detection_ignores_dates_and_durable_sha_pins(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-dates"))
    _write_skill(
        tmp_path / "profile-dates" / "skills",
        "ops/date-reference",
        name="date-reference",
        description="Audit dated reference names without false SHA matches.",
        body=_healthy_body("dated references")
        + "\nOpen references/report-20260525.md when needed.",
    )
    _write_skill(
        tmp_path / "profile-dates" / "skills",
        "ops/sha-reference",
        name="sha-reference",
        description="Audit durable SHA-pinned dependencies.",
        body=_healthy_body("SHA references")
        + "\nUse actions/checkout@deadbeef1234567890 for the pinned dependency.",
    )
    _write_skill(
        tmp_path / "profile-dates" / "skills",
        "ops/pr-reference",
        name="pr-reference",
        description="Audit stale pull request references.",
        body=_healthy_body("PR references")
        + "\nMove stale detail from PR #1234 to references.",
    )

    report = audit_skills()
    findings_by_name = {
        skill.name: {finding.code for finding in skill.card_findings}
        for skill in report.skills
    }

    assert "session_artifact" not in findings_by_name["date-reference"]
    assert "session_artifact" not in findings_by_name["sha-reference"]
    assert "session_artifact" in findings_by_name["pr-reference"]


def test_prompt_footprint_measures_rendered_index_not_full_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-index"))
    _write_skill(
        tmp_path / "profile-index" / "skills",
        "ops/large-reference",
        name="large-reference",
        description="A deliberately long description that the real prompt builder truncates before rendering.",
        body=_healthy_body("large references") + ("\nDetailed procedure text." * 1000),
    )

    report = audit_skills()
    skill = report.skills[0]
    codes = {finding.code for finding in skill.card_findings}

    assert skill.char_count > 20_000
    assert skill.estimated_tokens < 30
    assert "prompt_bloat" not in codes
    assert "large_card" not in codes


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
