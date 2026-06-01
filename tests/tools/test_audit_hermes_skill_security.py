"""Tests for tools.audit_hermes_skill_security — operator skill security audit."""

from __future__ import annotations

from pathlib import Path

from tools.audit_hermes_skill_security import (
    AUDIT_CATEGORIES,
    audit_skill_security,
    format_audit_report,
)
from tools.skills_guard import content_hash

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skills_audit"
SAFE_SKILL = FIXTURES / "safe-skill"
UNSAFE_SKILL = FIXTURES / "unsafe-skill"
MIRROR_SAFE = FIXTURES / "mirror-safe-skill"


class TestAuditSkillSecurity:
    def test_safe_fixture_passes(self):
        result = audit_skill_security(SAFE_SKILL)
        assert result.verdict == "pass"
        assert result.findings == []
        assert result.drift_detected is False
        assert result.content_hash.startswith("sha256:")

    def test_unsafe_fixture_fails_with_expected_categories(self):
        result = audit_skill_security(UNSAFE_SKILL)
        assert result.verdict == "fail"
        assert result.drift_detected is False

        categories = {f.category for f in result.findings}
        assert "unsafe_instruction" in categories
        assert "credential_use" in categories
        assert "hidden_background" in categories

        check_ids = {f.check_id for f in result.findings}
        assert "prompt_injection_ignore" in check_ids
        assert "terminal_background" in check_ids
        assert "env_exfil_curl" in check_ids
        assert "python_getenv_secret" in check_ids or "python_os_environ" in check_ids

    def test_mirror_match_no_drift(self):
        result = audit_skill_security(SAFE_SKILL, mirror_path=MIRROR_SAFE)
        assert result.verdict == "pass"
        assert result.drift_detected is False
        assert result.mirror_hash == result.content_hash
        assert not any(f.category == "source_mirror_drift" for f in result.findings)

    def test_mirror_mismatch_reports_drift(self, tmp_path: Path):
        drift_mirror = tmp_path / "drift-mirror"
        drift_mirror.mkdir()
        (drift_mirror / "SKILL.md").write_text(
            "---\nname: changed\n---\n# Different content\n",
            encoding="utf-8",
        )
        result = audit_skill_security(SAFE_SKILL, mirror_path=drift_mirror)
        assert result.drift_detected is True
        assert result.verdict == "fail"
        drift = [f for f in result.findings if f.category == "source_mirror_drift"]
        assert any(f.check_id == "mirror_hash_mismatch" for f in drift)

    def test_expected_hash_mismatch(self):
        result = audit_skill_security(SAFE_SKILL, expected_hash="sha256:0000000000000000")
        assert result.drift_detected is True
        assert result.verdict == "fail"
        assert any(f.check_id == "expected_hash_mismatch" for f in result.findings)

    def test_format_report_includes_verdict_and_categories(self):
        result = audit_skill_security(UNSAFE_SKILL)
        report = format_audit_report(result)
        assert "Operator security audit" in report
        assert "FAIL" in report
        for cat in ("unsafe_instruction", "credential_use", "hidden_background"):
            assert cat in report

    def test_all_audit_categories_defined(self):
        assert len(AUDIT_CATEGORIES) == 5
        assert "source_mirror_drift" in AUDIT_CATEGORIES

    def test_content_hash_stable_for_safe_fixture(self):
        h1 = content_hash(SAFE_SKILL)
        h2 = content_hash(SAFE_SKILL)
        assert h1 == h2
