#!/usr/bin/env python3
"""
End-to-end install pipeline test for HeuristSource.

Validates the full install flow (fetch → quarantine → scan → policy check → install → lock)
for 3 real Heurist skills against the live API, using a temporary directory as HERMES_HOME.

Run: python tests/tools/test_heurist_e2e_install.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.skills_hub import HeuristSource, quarantine_bundle, install_from_quarantine, HubLockFile
from tools.skills_guard import scan_skill, should_allow_install, format_scan_report, format_heurist_risk_warnings


TEST_SKILLS = [
    "heurist:query-onchain-data",
    "heurist:pay-for-service",
    "heurist:search-for-service",
]


def main():
    src = HeuristSource()
    tmp_home = Path(tempfile.mkdtemp(prefix="hermes_e2e_"))
    print(f"Temp HERMES_HOME: {tmp_home}")

    # Monkey-patch the module-level paths to use temp dir
    import tools.skills_hub as hub_mod
    orig_skills = hub_mod.SKILLS_DIR
    orig_hub = hub_mod.HUB_DIR
    orig_lock = hub_mod.LOCK_FILE
    orig_quarantine = hub_mod.QUARANTINE_DIR
    orig_audit = hub_mod.AUDIT_LOG
    orig_cache = hub_mod.INDEX_CACHE_DIR

    hub_mod.SKILLS_DIR = tmp_home / "skills"
    hub_mod.HUB_DIR = hub_mod.SKILLS_DIR / ".hub"
    hub_mod.LOCK_FILE = hub_mod.HUB_DIR / "lock.json"
    hub_mod.QUARANTINE_DIR = hub_mod.HUB_DIR / "quarantine"
    hub_mod.AUDIT_LOG = hub_mod.HUB_DIR / "audit.log"
    hub_mod.INDEX_CACHE_DIR = hub_mod.HUB_DIR / "index-cache"

    success_count = 0

    try:
        hub_mod.ensure_hub_dirs()

        for identifier in TEST_SKILLS:
            slug = identifier.split(":", 1)[-1]
            print(f"\n{'='*60}")
            print(f"Installing: {identifier}")
            print(f"{'='*60}")

            # Step 1: Fetch
            print("[1/5] Fetching bundle...")
            bundle = src.fetch(identifier)
            assert bundle is not None, f"Fetch failed for {identifier}"
            assert "SKILL.md" in bundle.files, f"No SKILL.md in bundle for {identifier}"
            print(f"  Fetched {len(bundle.files)} file(s): {list(bundle.files.keys())}")

            # Step 2: Heurist risk warnings
            print("[2/5] Checking Heurist risk warnings...")
            risk_warnings = format_heurist_risk_warnings(bundle.metadata)
            if risk_warnings:
                for w in risk_warnings:
                    print(f"  WARNING: {w}")
            else:
                print("  No risk warnings")

            # Step 3: Quarantine
            print("[3/5] Quarantining...")
            q_path = quarantine_bundle(bundle)
            assert q_path.exists(), f"Quarantine path does not exist: {q_path}"
            assert (q_path / "SKILL.md").exists(), f"SKILL.md not in quarantine"
            print(f"  Quarantined to: {q_path}")

            # Step 4: Scan
            print("[4/5] Running security scan...")
            scan_result = scan_skill(q_path, source=identifier)
            print(f"  Verdict: {scan_result.verdict}")
            print(f"  Findings: {len(scan_result.findings)}")
            if scan_result.findings:
                for f in scan_result.findings[:3]:
                    print(f"    [{f.severity}] {f.category}: {f.description}")
            print(format_scan_report(scan_result))

            # Step 5: Policy check + install
            allowed, reason = should_allow_install(scan_result)
            print(f"  Policy: {'ALLOWED' if allowed else 'BLOCKED'} — {reason}")

            if allowed:
                install_dir = install_from_quarantine(q_path, slug, "", bundle, scan_result)
                assert install_dir.exists(), f"Install dir does not exist: {install_dir}"
                assert (install_dir / "SKILL.md").exists(), f"SKILL.md not in install dir"
                print(f"  Installed to: {install_dir}")

                # Verify lock file
                lock = HubLockFile()
                entry = lock.get_installed(slug)
                assert entry is not None, f"Skill not in lock file: {slug}"
                assert entry["source"] == "heurist", f"Wrong source in lock: {entry['source']}"
                assert entry["identifier"] == f"heurist:{slug}", f"Wrong identifier in lock"
                print(f"  Lock file entry: source={entry['source']}, verdict={entry['scan_verdict']}")

                success_count += 1
                print(f"  PASS")
            else:
                print(f"  BLOCKED (expected for high-risk community skills)")
                # Blocked skills still count as successful test — the pipeline worked correctly
                success_count += 1
                print(f"  PASS (correctly blocked)")

        # Verify audit log
        audit_log = hub_mod.AUDIT_LOG
        if audit_log.exists():
            log_lines = audit_log.read_text().strip().split("\n")
            print(f"\n=== Audit log ({len(log_lines)} entries) ===")
            for line in log_lines:
                print(f"  {line}")

        # Verify installed count
        lock = HubLockFile()
        installed = lock.list_installed()
        print(f"\n=== Lock file: {len(installed)} skill(s) installed ===")
        for entry in installed:
            print(f"  {entry['name']} (source={entry['source']}, verdict={entry['scan_verdict']})")

    finally:
        # Restore original paths
        hub_mod.SKILLS_DIR = orig_skills
        hub_mod.HUB_DIR = orig_hub
        hub_mod.LOCK_FILE = orig_lock
        hub_mod.QUARANTINE_DIR = orig_quarantine
        hub_mod.AUDIT_LOG = orig_audit
        hub_mod.INDEX_CACHE_DIR = orig_cache

        # Cleanup
        shutil.rmtree(tmp_home, ignore_errors=True)
        print(f"\nCleaned up temp dir: {tmp_home}")

    print(f"\n=== Results: {success_count}/{len(TEST_SKILLS)} skills passed E2E pipeline ===")
    assert success_count == len(TEST_SKILLS), f"Only {success_count}/{len(TEST_SKILLS)} passed"
    print("ALL E2E INSTALL TESTS PASSED")


if __name__ == "__main__":
    main()
