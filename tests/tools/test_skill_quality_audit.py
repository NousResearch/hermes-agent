"""Tests for deterministic, static skill quality auditing."""

from pathlib import Path

from tools.skill_quality_audit import audit_skill_quality, save_verification_receipt


def _write_skill(
    root: Path,
    *,
    name: str = "demo-skill",
    metadata: str = "",
    body: str = "# Demo\n\n## Verification Checklist\n- [ ] Check output\n",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: A test skill.\n"
        f"{metadata}"
        "---\n\n"
        f"{body}",
        encoding="utf-8",
    )
    return root


def test_valid_static_skill_passes(tmp_path):
    skill = _write_skill(
        tmp_path / "demo-skill",
        metadata="metadata:\n  hermes:\n    verification:\n      level: static\n",
    )
    (skill / "references").mkdir()
    (skill / "references" / "guide.md").write_text("Guide", encoding="utf-8")
    skill_md = skill / "SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8") + "\nRead [the guide](references/guide.md).\n",
        encoding="utf-8",
    )

    result = audit_skill_quality(skill, skill_name="demo-skill", source="local")

    assert result.status == "pass"
    assert all(finding.severity == "pass" for finding in result.findings)


def test_missing_verification_metadata_is_warning(tmp_path):
    skill = _write_skill(tmp_path / "demo-skill")

    result = audit_skill_quality(skill, skill_name="demo-skill", source="local")

    assert result.status == "warning"
    assert any(finding.check == "verification_metadata" and finding.severity == "warning"
               for finding in result.findings)


def test_invalid_verification_level_fails(tmp_path):
    skill = _write_skill(
        tmp_path / "demo-skill",
        metadata="metadata:\n  hermes:\n    verification:\n      level: execute\n",
    )

    result = audit_skill_quality(skill, skill_name="demo-skill", source="local")

    assert result.status == "fail"
    assert any(finding.check == "verification_metadata" and finding.severity == "fail"
               for finding in result.findings)


def test_missing_local_reference_fails(tmp_path):
    skill = _write_skill(tmp_path / "demo-skill")
    skill_md = skill / "SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8") + "\nRead [the guide](references/missing.md).\n",
        encoding="utf-8",
    )

    result = audit_skill_quality(skill, skill_name="demo-skill", source="local")

    assert result.status == "fail"
    assert any(finding.check == "local_references" and finding.severity == "fail"
               for finding in result.findings)


def test_reference_path_escape_fails_without_reading_outside_skill(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    skill = _write_skill(tmp_path / "demo-skill")
    skill_md = skill / "SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8") + "\nRead [outside](../outside.md).\n",
        encoding="utf-8",
    )

    result = audit_skill_quality(skill, skill_name="demo-skill", source="local")

    assert result.status == "fail"
    assert any(finding.check == "local_references" and "outside" in finding.message
               for finding in result.findings)


def test_name_mismatch_fails(tmp_path):
    skill = _write_skill(tmp_path / "demo-skill", name="different-skill")

    result = audit_skill_quality(skill, skill_name="demo-skill", source="local")

    assert result.status == "fail"
    assert any(finding.check == "skill_name" and finding.severity == "fail"
               for finding in result.findings)


def test_missing_verification_checklist_is_warning(tmp_path):
    skill = _write_skill(tmp_path / "demo-skill", body="# Demo\n\nInstructions.\n")

    result = audit_skill_quality(skill, skill_name="demo-skill", source="local")

    assert result.status == "warning"
    assert any(finding.check == "verification_checklist" and finding.severity == "warning"
               for finding in result.findings)


def test_save_verification_receipt_is_local_and_replaces_same_skill(tmp_path):
    skill = _write_skill(tmp_path / "demo-skill")
    result = audit_skill_quality(skill, skill_name="demo-skill", source="local")
    receipt = tmp_path / "skills" / ".verification.json"

    save_verification_receipt(receipt, result)
    save_verification_receipt(receipt, result)

    import json

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["receipts"]) == 1
    saved = next(iter(payload["receipts"].values()))
    assert saved["status"] == result.status
    assert "content_hash" in saved


def test_receipts_do_not_collide_for_distinct_same_named_skills(tmp_path):
    first = _write_skill(tmp_path / "one", name="same")
    second = _write_skill(tmp_path / "two", name="same")
    receipt = tmp_path / "skills" / ".verification.json"

    save_verification_receipt(receipt, audit_skill_quality(first, skill_name="same", source="local"))
    save_verification_receipt(receipt, audit_skill_quality(second, skill_name="same", source="local"))

    import json

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert len(payload["receipts"]) == 2
    assert {entry["path"] for entry in payload["receipts"].values()} == {str(first), str(second)}
