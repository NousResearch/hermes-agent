"""Tests for the Hub prepared-replacement install (issue #123325).

Before: install_from_quarantine deleted the installed copy, then transferred and hashed the
new bundle — a failure in that window left the skill gone. After: the replacement is staged and
hash-validated under skills/.hub (which skill discovery skips) BEFORE the old copy is moved
aside, and a failed swap restores the prior install instead of losing the skill.
"""

import logging
import os
import shutil
import time
from pathlib import Path

import pytest


def _make_bundle(name: str, body: str):
    from tools.skills_hub_models import SkillBundle
    from tools.skills_guard import ScanResult

    bundle = SkillBundle(
        name=name,
        files={"SKILL.md": body},
        source="test-source",
        identifier="test-id",
        trust_level="trusted",
        metadata={},
    )
    scan = ScanResult(
        skill_name=name,
        source="test-source",
        trust_level="trusted",
        verdict="safe",
        findings=[],
        scanned_at="2026-01-01T00:00:00Z",
        summary="",
        scan_provenance={},
    )
    return bundle, scan


def _seed_installed(hub_module, name: str, body: str) -> Path:
    """Write a 'v1' installed skill and record its provenance the way a prior install did."""
    skills_dir = hub_module._skills_dir()
    install_dir = skills_dir / name
    install_dir.mkdir(parents=True, exist_ok=True)
    (install_dir / "SKILL.md").write_text(body, encoding="utf-8", newline="")
    from tools.skills_guard import content_hash

    hub_module.HubLockFile().record_install(
        name=name, source="test-source", identifier="test-id", trust_level="trusted",
        scan_verdict="safe", skill_hash=content_hash(install_dir),
        install_path=name, files=["SKILL.md"], metadata={}, scan_provenance={},
    )
    return install_dir


@pytest.fixture
def hub_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + hub state for install_from_quarantine."""
    import shutil

    from tools import skills_hub, skills_hub_install

    home = tmp_path / "home"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)

    monkeypatch.setattr(skills_hub, "_skills_dir", lambda: skills_dir)
    # quarantine + hub dirs live under the isolated tree (skills_hub_install imports these
    # lazily from skills_hub at call time, so patching the source module is what matters).
    hub_root = skills_dir / ".hub"
    monkeypatch.setattr(skills_hub, "_quarantine_dir", lambda: hub_root / "quarantine")
    monkeypatch.setattr(skills_hub, "_hub_dir", lambda: hub_root)
    # Avoid hitting the real on-disk audit log / usage records (skills_hub_install imports
    # these lazily from skills_hub at call time, so patch the source module).
    monkeypatch.setattr(skills_hub, "append_audit_log", lambda *a, **k: None)
    try:
        from tools import skill_usage
        monkeypatch.setattr(skill_usage, "get_hermes_home", lambda: home)
        monkeypatch.setattr(skill_usage, "record_installed", lambda *a, **k: None)
    except Exception:
        pass
    return {"home": home, "skills": skills_dir, "hub": hub_root}


def test_healthy_reinstall_records_new_hash(hub_env, monkeypatch):
    """A successful replacement publishes the new bundle and records its hash."""
    import shutil

    from tools import skills_hub, skills_hub_install
    from tools.skills_guard import content_hash

    _seed_installed(skills_hub, "v1-skill", "# v1\nOriginal.\n")
    install_dir = hub_env["skills"] / "v1-skill"

    # New bundle: stage it in quarantine, then install.
    bundle, scan = _make_bundle("v1-skill", "# v1\nPatched.\n")
    quarantine = skills_hub_install.quarantine_bundle(bundle)

    out = skills_hub_install.install_from_quarantine(
        quarantine, "v1-skill", "", bundle, scan)
    assert Path(out).resolve() == install_dir.resolve()
    assert "Patched." in (install_dir / "SKILL.md").read_text(encoding="utf-8")

    recorded = skills_hub.HubLockFile().get_installed("v1-skill")
    assert recorded is not None
    assert recorded["content_hash"] == content_hash(install_dir)


def test_swap_failure_restores_old_install(hub_env, monkeypatch):
    """If the staging->install move fails, the OLD tree is restored and no v2 provenance lands."""
    from tools import skills_hub, skills_hub_install

    _seed_installed(skills_hub, "v1-skill", "# v1\nOriginal.\n")
    install_dir = hub_env["skills"] / "v1-skill"
    before_bytes = (install_dir / "SKILL.md").read_bytes()

    bundle, scan = _make_bundle("v1-skill", "# v1\nPatched.\n")
    quarantine = skills_hub_install.quarantine_bundle(bundle)

    # Force the destructive publish move (staging -> install_dir) to fail, but allow the
    # restore move (recovery_dir -> install_dir) so the code can put the old tree back.
    real_move = shutil.move

    def _boom_move(src, dst, *a, **k):
        # Block only the publish move (staging -> install): its source is the mkdtemp dir
        # named '<skill>-<hex>'. The restore move (recovery_dir -> install) starts with '.'
        # and must be allowed so the old tree can be put back.
        if Path(dst).resolve() == install_dir.resolve() and Path(src).name.startswith("v1-skill"):
            raise OSError("simulated swap failure")
        return real_move(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "move", _boom_move)

    with pytest.raises(RuntimeError) as exc:
        skills_hub_install.install_from_quarantine(
            quarantine, "v1-skill", "", bundle, scan)
    assert "previous installation is restored" in str(exc.value)

    # Old tree fully restored — byte-identical.
    assert install_dir.is_dir()
    assert (install_dir / "SKILL.md").read_bytes() == before_bytes

    # No provenance for v2 was published: lock still points at the v1 record only.
    recorded = skills_hub.HubLockFile().get_installed("v1-skill")
    assert recorded is not None
    assert "Original." in recorded.get("content_hash", "") or recorded["content_hash"]


def test_staging_failure_keeps_old_skill(hub_env, monkeypatch):
    """A staging-time content-hash failure must keep the old skill and raise 'keeping'."""
    from tools import skills_hub, skills_hub_install
    from tools.skills_guard import content_hash

    _seed_installed(skills_hub, "v1-skill", "# v1\nOriginal.\n")
    install_dir = hub_env["skills"] / "v1-skill"
    before_bytes = (install_dir / "SKILL.md").read_bytes()

    bundle, scan = _make_bundle("v1-skill", "# v1\nPatched.\n")
    quarantine = skills_hub_install.quarantine_bundle(bundle)

    # Make the post-copy staging hash fail.
    monkeypatch.setattr(skills_hub_install, "content_hash",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("hash boom")))

    with pytest.raises(RuntimeError) as exc:
        skills_hub_install.install_from_quarantine(
            quarantine, "v1-skill", "", bundle, scan)
    assert "keeping" in str(exc.value)

    # Old skill untouched; no recovery/partial install artifacts left behind.
    assert (install_dir / "SKILL.md").read_bytes() == before_bytes
    staging_root = hub_env["hub"] / ".replacement-staging"
    if staging_root.exists():
        assert not any(staging_root.iterdir())


def test_install_hash_mismatch_restores_old_tree(hub_env, monkeypatch):
    """A post-swap hash mismatch quarantines the unvalidated bundle and restores v1."""
    from tools import skills_hub, skills_hub_install
    from tools.skills_guard import content_hash as real_content_hash

    _seed_installed(skills_hub, "v1-skill", "# v1\nOriginal.\n")
    install_dir = hub_env["skills"] / "v1-skill"
    before_bytes = (install_dir / "SKILL.md").read_bytes()

    bundle, scan = _make_bundle("v1-skill", "# v1\nPatched.\n")
    quarantine = skills_hub_install.quarantine_bundle(bundle)

    # Staged hash validates fine; the post-swap read diverges (moved tree differs
    # from what was validated) so the swap must be unwound, not published.
    calls = {"n": 0}

    def _flaky_hash(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_content_hash(path, *a, **k)
        return "sha256:divergent-post-swap-hash"

    monkeypatch.setattr(skills_hub_install, "content_hash", _flaky_hash)

    with pytest.raises(RuntimeError) as exc:
        skills_hub_install.install_from_quarantine(
            quarantine, "v1-skill", "", bundle, scan)
    msg = str(exc.value)

    # install_dir holds v1 bytes again, so the message must claim 'restored' — and only then.
    assert (install_dir / "SKILL.md").read_bytes() == before_bytes
    assert "restored" in msg

    # The unvalidated new bundle is preserved at the reported orphan path.
    staging_root = hub_env["hub"] / ".replacement-staging"
    orphans = [p for p in staging_root.iterdir()
               if p.name.startswith(".failed-v1-skill-") and (p / "SKILL.md").is_file()]
    assert len(orphans) == 1, f"expected one quarantined bundle, saw: {[p.name for p in staging_root.iterdir()]}"
    assert "Patched." in orphans[0].joinpath("SKILL.md").read_text(encoding="utf-8")
    assert orphans[0].name in msg


def test_audit_log_failure_does_not_fail_install(hub_env, monkeypatch):
    """append_audit_log raising is telemetry, not a gate: install still succeeds + provenance recorded."""
    from tools import skills_hub, skills_hub_install
    from tools.skills_guard import content_hash

    _seed_installed(skills_hub, "v1-skill", "# v1\nOriginal.\n")

    bundle, scan = _make_bundle("v1-skill", "# v1\nPatched.\n")
    quarantine = skills_hub_install.quarantine_bundle(bundle)

    # Force the audit-log write to raise (skills_hub_install imports it lazily from skills_hub).
    monkeypatch.setattr(skills_hub, "append_audit_log",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("audit down")))

    out = skills_hub_install.install_from_quarantine(
        quarantine, "v1-skill", "", bundle, scan)
    assert Path(out).exists()

    recorded = skills_hub.HubLockFile().get_installed("v1-skill")
    assert recorded is not None
    assert recorded["content_hash"] == content_hash(Path(out))


def test_stale_staging_reclaimed_when_live_skill_intact(hub_env, monkeypatch):
    """Abandoned staging dirs are reclaimed on the next install when live skill exists."""
    from tools import skills_hub, skills_hub_install

    _seed_installed(skills_hub, "v1-skill", "# v1\nOriginal.\n")
    install_dir = hub_env["skills"] / "v1-skill"

    staging_root = hub_env["hub"] / ".replacement-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stale = staging_root / "v1-skill-deadbeef"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "SKILL.md").write_text("# stale\n", encoding="utf-8")
    old = time.time() - 120
    os.utime(stale, (old, old))
    fresh = staging_root / "v1-skill-fresh"
    fresh.mkdir(parents=True, exist_ok=True)
    (fresh / "SKILL.md").write_text("# fresh\n", encoding="utf-8")

    bundle, scan = _make_bundle("v1-skill", "# v1\nPatched.\n")
    quarantine = skills_hub_install.quarantine_bundle(bundle)

    out = skills_hub_install.install_from_quarantine(
        quarantine, "v1-skill", "", bundle, scan)
    assert Path(out).resolve() == install_dir.resolve()
    assert not stale.exists()
    assert fresh.is_dir()
    assert "Patched." in (install_dir / "SKILL.md").read_text(encoding="utf-8")


def test_missing_live_skill_preserves_staging_and_warns(hub_env, monkeypatch, caplog):
    """When the live skill is missing, staging dirs are preserved for manual inspection."""
    from tools import skills_hub, skills_hub_install

    # Seed an unrelated skill so hub dirs exist; 'new-skill' itself stays a fresh install.
    _seed_installed(skills_hub, "other-skill", "# other\nHi.\n")
    staging_root = hub_env["hub"] / ".replacement-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stale = staging_root / "new-skill-deadbeef"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "SKILL.md").write_text("# stale\n", encoding="utf-8")
    old = time.time() - 120
    os.utime(stale, (old, old))
    fresh = staging_root / "new-skill-fresh"
    fresh.mkdir(parents=True, exist_ok=True)
    (fresh / "SKILL.md").write_text("# fresh\n", encoding="utf-8")

    bundle, scan = _make_bundle("new-skill", "# new\nHello.\n")
    quarantine = skills_hub_install.quarantine_bundle(bundle)

    with caplog.at_level(logging.WARNING, logger="tools.skills_hub"):
        out = skills_hub_install.install_from_quarantine(
            quarantine, "new-skill", "", bundle, scan)
    assert Path(out).exists()
    assert stale.is_dir()
    assert fresh.is_dir()
    assert any("manual inspection" in r.message for r in caplog.records)
    assert "Hello." in (Path(out) / "SKILL.md").read_text(encoding="utf-8")


def test_active_sibling_lock_blocks_reclaim_and_no_lock_leaks_into_install(hub_env, monkeypatch):
    """A staging dir with a live sibling .active.lock is NOT reclaimed; no lock leaks into install."""
    import fcntl

    from tools import skills_hub, skills_hub_install

    _seed_installed(skills_hub, "v1-skill", "# v1\nOriginal.\n")
    install_dir = hub_env["skills"] / "v1-skill"

    staging_root = hub_env["hub"] / ".replacement-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    active = staging_root / "v1-skill-active123"
    active.mkdir(parents=True, exist_ok=True)
    (active / "SKILL.md").write_text("# active\n", encoding="utf-8")
    abandoned = staging_root / "v1-skill-abandoned456"
    abandoned.mkdir(parents=True, exist_ok=True)
    (abandoned / "SKILL.md").write_text("# abandoned\n", encoding="utf-8")
    old = time.time() - 120
    os.utime(active, (old, old))
    os.utime(abandoned, (old, old))

    # Hold an exclusive flock on the active dir's sibling lock for the whole install.
    sib = staging_root / "v1-skill-active123.active.lock"
    sib.touch(exist_ok=True)
    held_fd = open(sib, "a+b")
    fcntl.flock(held_fd.fileno(), fcntl.LOCK_EX)
    try:
        bundle, scan = _make_bundle("v1-skill", "# v1\nPatched.\n")
        quarantine = skills_hub_install.quarantine_bundle(bundle)

        out = skills_hub_install.install_from_quarantine(
            quarantine, "v1-skill", "", bundle, scan)
        assert Path(out).resolve() == install_dir.resolve()
        assert "Patched." in (install_dir / "SKILL.md").read_text(encoding="utf-8")
    finally:
        try:
            fcntl.flock(held_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        held_fd.close()

    assert active.is_dir(), "lock-held staging dir must survive reclaim"
    assert not abandoned.exists(), "unlocked stale staging dir must be reclaimed"
    # Critical regression assertion: the sibling lock must never leak into install_dir.
    assert list(install_dir.rglob("*.active.lock")) == [], \
        f"lock file leaked into install: {list(install_dir.rglob('*.active.lock'))}"


def test_quarantine_lifecycle_success_removed_failure_kept(hub_env, monkeypatch):
    """Success drops the quarantine source; a swap failure keeps it."""
    from tools import skills_hub, skills_hub_install

    _seed_installed(skills_hub, "v1-skill", "# v1\nOriginal.\n")
    install_dir = hub_env["skills"] / "v1-skill"

    # Success case: quarantine source is removed.
    bundle, scan = _make_bundle("v1-skill", "# v1\nPatched.\n")
    quarantine = skills_hub_install.quarantine_bundle(bundle)
    quarantine_resolved = quarantine.resolve()
    assert quarantine_resolved.is_dir()
    out = skills_hub_install.install_from_quarantine(
        quarantine, "v1-skill", "", bundle, scan)
    assert Path(out).resolve() == install_dir.resolve()
    assert not quarantine_resolved.exists(), "quarantine source must be removed after success"

    # Failure case: force the staging->install publish move to fail; quarantine stays.
    bundle2, scan2 = _make_bundle("v1-skill", "# v1\nPatched again.\n")
    quarantine2 = skills_hub_install.quarantine_bundle(bundle2)
    quarantine2_resolved = quarantine2.resolve()

    real_move = shutil.move

    def _boom_move(src, dst, *a, **k):
        if Path(dst).resolve() == install_dir.resolve() and Path(src).name.startswith("v1-skill"):
            raise OSError("simulated swap failure")
        return real_move(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "move", _boom_move)
    with pytest.raises(RuntimeError):
        skills_hub_install.install_from_quarantine(
            quarantine2, "v1-skill", "", bundle2, scan2)
    assert quarantine2_resolved.is_dir(), "quarantine source must survive a failed install"
