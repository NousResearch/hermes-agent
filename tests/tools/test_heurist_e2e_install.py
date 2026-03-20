#!/usr/bin/env python3
"""
End-to-end install pipeline tests for HeuristSource.

Validates the full install flow (fetch -> quarantine -> scan -> policy check -> install -> lock)
for real Heurist skills against the live API, using a temporary directory as HERMES_HOME.

These tests hit the real API and are marked with @pytest.mark.integration.
Run with: pytest tests/tools/test_heurist_e2e_install.py -v -m integration
"""

import shutil
import pytest

from tools.skills_hub import HeuristSource, quarantine_bundle, install_from_quarantine, HubLockFile
from tools.skills_guard import scan_skill, should_allow_install, format_scan_report


pytestmark = pytest.mark.integration

TEST_SKILLS = [
    "heurist:query-onchain-data",
    "heurist:pay-for-service",
    "heurist:search-for-service",
]


@pytest.fixture(scope="module")
def source():
    return HeuristSource()


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    """Redirect all hub paths to a temp directory."""
    import tools.skills_hub as hub_mod
    skills_dir = tmp_path / "skills"
    hub_dir = skills_dir / ".hub"
    monkeypatch.setattr(hub_mod, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(hub_mod, "HUB_DIR", hub_dir)
    monkeypatch.setattr(hub_mod, "LOCK_FILE", hub_dir / "lock.json")
    monkeypatch.setattr(hub_mod, "QUARANTINE_DIR", hub_dir / "quarantine")
    monkeypatch.setattr(hub_mod, "AUDIT_LOG", hub_dir / "audit.log")
    monkeypatch.setattr(hub_mod, "INDEX_CACHE_DIR", hub_dir / "index-cache")
    hub_mod.ensure_hub_dirs()
    return tmp_path


class TestHeuristE2EInstallPipeline:
    @pytest.mark.parametrize("identifier", TEST_SKILLS)
    def test_full_install_pipeline(self, source, hermes_home, identifier):
        slug = identifier.split(":", 1)[-1]

        # Step 1: Fetch
        bundle = source.fetch(identifier)
        if bundle is None:
            pytest.skip(f"Fetch returned None for {identifier} (possible SHA256 mismatch)")
        assert "SKILL.md" in bundle.files

        # Step 2: Risk warnings
        warnings = HeuristSource.format_risk_warnings(bundle.metadata)
        assert isinstance(warnings, list)

        # Step 3: Quarantine
        q_path = quarantine_bundle(bundle)
        assert q_path.exists()
        assert (q_path / "SKILL.md").exists()

        # Step 4: Scan
        scan_result = scan_skill(q_path, source=identifier)
        assert scan_result.verdict in ("safe", "caution", "dangerous")

        # Step 5: Policy check
        allowed, reason = should_allow_install(scan_result)
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)

        # Step 6: Install (if allowed) or verify correct blocking
        if allowed:
            install_dir = install_from_quarantine(q_path, slug, "", bundle, scan_result)
            assert install_dir.exists()
            assert (install_dir / "SKILL.md").exists()

            # Verify lock file
            lock = HubLockFile()
            entry = lock.get_installed(slug)
            assert entry is not None
            assert entry["source"] == "heurist"
            assert entry["identifier"] == identifier
        else:
            # Correctly blocked by community + caution/dangerous policy — this is expected
            assert "community" in reason.lower() or "block" in reason.lower()
            # Clean up quarantine
            shutil.rmtree(q_path, ignore_errors=True)

    @pytest.mark.parametrize("identifier", TEST_SKILLS)
    def test_scan_report_is_valid(self, source, hermes_home, identifier):
        bundle = source.fetch(identifier)
        if bundle is None:
            pytest.skip(f"Fetch returned None for {identifier}")
        q_path = quarantine_bundle(bundle)
        scan_result = scan_skill(q_path, source=identifier)
        report = format_scan_report(scan_result)
        assert isinstance(report, str)
        assert scan_result.skill_name in report
        shutil.rmtree(q_path, ignore_errors=True)
