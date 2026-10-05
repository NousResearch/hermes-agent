"""Tests for the Hub prepared-replacement install (issue #123325).

Before: install_from_quarantine deleted the installed copy, then transferred and hashed the
new bundle — a failure in that window left the skill gone. After: the replacement is staged and
hash-validated under skills/.hub (which skill discovery skips) BEFORE the old copy is moved
aside, and a failed swap restores the prior install instead of losing the skill.
"""

import shutil
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
