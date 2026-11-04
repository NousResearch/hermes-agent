"""A failed Hub replacement must not destroy the installed skill."""

import errno
from io import StringIO
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from rich.console import Console

from agent.skill_utils import iter_skill_index_files

import tools.skills_hub as hub
import tools.skills_hub_install as installer
from tools.skills_guard import ScanResult, content_hash
from tools.skills_hub_models import SkillBundle, SkillMeta


@pytest.fixture
def installation(tmp_path, monkeypatch):
    from hermes_cli import skills_hub as cli_hub

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text("skills:\n  tier1_advisory: false\n", encoding="utf-8")
    old = SkillBundle("demo", {"SKILL.md": "Original skill", "original.txt": b"original data"},
                      "github", "example/skills/demo", "community")
    scan = ScanResult("demo", "github", "community", "safe")
    target = installer.install_from_quarantine(installer.quarantine_bundle(old), "demo", "", old, scan)
    bundle = SkillBundle("demo", {"SKILL.md": "Updated skill", "new.txt": b"new data"},
                         "github", old.identifier, "community")
    meta = SkillMeta("demo", "Demo skill", "github", bundle.identifier, "community")
    source = SimpleNamespace(inspect=lambda _: meta, fetch=lambda _: bundle)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [source])
    output = StringIO()

    def install():
        cli_hub.do_install(bundle.identifier, force=True, skip_confirm=True,
                           invalidate_cache=False, console=Console(file=output))

    return target, bundle, install, output


@pytest.mark.parametrize("failure", ["copy", "hash", "activate", "rollback"])
def test_failed_reinstall_keeps_previous_content(installation, monkeypatch, failure):
    target, _, install, _ = installation
    original_record = hub._lock_file().read_bytes()
    quarantine = hub._quarantine_dir() / "demo"
    real_rename, real_copytree = os.rename, shutil.copytree

    def rename(src, dst, *args, **kwargs):
        if Path(dst) == target:
            if failure == "copy" and Path(src) == quarantine:
                raise OSError(errno.EXDEV, "cross-device move")
            if failure in {"activate", "rollback"}:
                is_new = (Path(src) / "SKILL.md").read_text(encoding="utf-8") == "Updated skill"
                if is_new or failure == "rollback":
                    raise OSError(errno.EACCES, "injected rename failure")
        return real_rename(src, dst, *args, **kwargs)

    def copytree(src, dst, *args, **kwargs):
        if Path(src) == quarantine and (failure == "copy" or Path(dst) == target):
            Path(dst).mkdir(parents=True, exist_ok=True)
            (Path(dst) / "partial.bin").write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "injected partial copy")
        return real_copytree(src, dst, *args, **kwargs)

    def fail_hash(_path):
        raise OSError(errno.EIO, "injected hash read failure")

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(shutil, "copytree", copytree)
    if failure == "hash":
        monkeypatch.setattr(installer, "content_hash", fail_hash)
    with pytest.raises(OSError) as error:
        install()

    if failure == "rollback":
        # A second filesystem failure can prevent automatic recovery; retain
        # the complete old tree for manual recovery instead of cleaning it up.
        originals = list(target.parent.rglob("original.txt"))
        assert len(originals) == 1
        restored = originals[0].parent
        assert str(restored) in str(error.value)
        assert restored / "SKILL.md" not in list(iter_skill_index_files(target.parent, "SKILL.md"))
    else:
        restored = target
    assert (restored / "SKILL.md").read_text(encoding="utf-8") == "Original skill"
    assert (restored / "original.txt").read_bytes() == b"original data"
    assert hub._lock_file().read_bytes() == original_record
    assert (quarantine / "new.txt").read_bytes() == b"new data"


@pytest.mark.parametrize("existing", [True, False])
def test_install_publishes_complete_bundle_and_provenance(installation, existing):
    target, bundle, install, output = installation
    siblings = {p.name for p in target.parent.iterdir() if p != target}
    if not existing:
        assert installer.uninstall_skill("demo")[0]
    install()
    assert "Installed:" in output.getvalue()
    assert {p.relative_to(target).as_posix(): p.read_bytes()
            for p in target.rglob("*") if p.is_file()} == {
        key: value.encode("utf-8") if isinstance(value, str) else value
        for key, value in bundle.files.items()
    }
    record = hub.HubLockFile().get_installed("demo")
    assert record is not None
    assert record["content_hash"] == content_hash(target)
    assert record["install_path"] == "demo"
    assert not (hub._quarantine_dir() / "demo").exists()
    assert {p.name for p in target.parent.iterdir()} == siblings | {target.name}
