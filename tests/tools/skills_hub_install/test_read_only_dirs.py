"""Read-only directories in an installed skill tree (#159440).

``shutil.rmtree`` never clears a directory's read-only attribute, and on Windows a directory carrying
``FILE_ATTRIBUTE_READONLY`` cannot be removed by ``os.rmdir`` at all, so both the hub update and the uninstall used
to die with WinError 5 — after the new bundle was fetched, and (because rmtree deletes depth-first) after the
installed skill had already been emptied.
"""

import json
import os
import stat
from contextlib import suppress
from unittest.mock import patch

import pytest

from tools.skills_guard import ScanResult
from tools.skills_hub import HubLockFile
from tools.skills_hub_install import install_from_quarantine, uninstall_skill
from tools.skills_hub_models import SkillBundle

READ_ONLY = stat.S_IREAD | stat.S_IEXEC  # 0o500: traversable, not writable


def _lock_entry(install_path: str) -> dict:
    return {"installed": {"my-skill": {
        "source": "github", "identifier": "x", "trust_level": "trusted", "scan_verdict": "pass",
        "content_hash": "h", "install_path": install_path, "files": [], "metadata": {},
        "installed_at": "now", "updated_at": "now"}}}


def _patch_lock_file(tmp_path) -> None:
    """HubLockFile captures LOCK_FILE as a default arg at class definition time, so patch __defaults__."""
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(_lock_entry("my-skill")))
    HubLockFile.__init__.__defaults__ = (lock_path,)


def test_install_from_quarantine_replaces_read_only_directories(tmp_path):
    import tools.skills_hub as hub

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    existing = skills_dir / "my-skill"
    (existing / "renderers" / "empty-bucket").mkdir(parents=True)
    (existing / "SKILL.md").write_text("old content")
    (existing / "renderers" / "guide.md").write_text("old guide")  # removing it needs a writable parent

    quarantine_root = skills_dir / ".hub" / "quarantine"
    q_dir = quarantine_root / "pending"
    q_dir.mkdir(parents=True)
    (q_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nnew")
    bundle = SkillBundle(name="my-skill", files={"SKILL.md": "---\nname: my-skill\n---\nnew"},
                         source="community", identifier="x", trust_level="community")
    scan_result = ScanResult(skill_name="my-skill", source="community", trust_level="community", verdict="safe")

    read_only = (existing, existing / "renderers", existing / "renderers" / "empty-bucket")
    for d in read_only:
        os.chmod(d, READ_ONLY)
    try:
        with patch.object(hub, "SKILLS_DIR", skills_dir), patch.object(hub, "QUARANTINE_DIR", quarantine_root):
            installed = install_from_quarantine(q_dir, "my-skill", "", bundle, scan_result)
    finally:
        for d in read_only:
            with suppress(OSError):
                os.chmod(d, stat.S_IRWXU)

    assert (installed / "SKILL.md").read_text().strip() == "---\nname: my-skill\n---\nnew"
    assert not (installed / "renderers" / "guide.md").exists()  # fully removed, not half-deleted


def test_uninstall_removes_read_only_directories(tmp_path):
    import tools.skills_hub as hub

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skill = skills_dir / "my-skill"
    (skill / "renderers").mkdir(parents=True)
    (skill / "renderers" / "guide.md").write_text("g")  # removing it needs a writable parent
    (skill / "SKILL.md").write_text("x")
    os.chmod(skill / "renderers", READ_ONLY)

    hub.SKILLS_DIR = skills_dir
    try:
        _patch_lock_file(tmp_path)
        try:
            ok, msg = uninstall_skill("my-skill")
        finally:
            for d in (skill, skill / "renderers"):
                with suppress(OSError):
                    os.chmod(d, stat.S_IRWXU)
    finally:
        with suppress(AttributeError):
            delattr(hub, "SKILLS_DIR")

    assert ok is True, msg
    assert not skill.exists()


def test_removal_clears_the_read_only_attribute_on_a_file_and_keeps_shared_bits(tmp_path):
    """A read-only *file* is unlinked by the same retry — the handler chmods the path it failed on and its
    parent. The permissions the tree already granted group/other survive: it ORs into the mode rather than
    replacing it with 0o700."""
    import tools.skills_hub as hub

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skill = skills_dir / "my-skill"
    skill.mkdir()
    (skill / "run.sh").write_text("#!/bin/sh\n")
    os.chmod(skill / "run.sh", stat.S_IREAD)
    os.chmod(skill, 0o550)  # r-x: unlinking the child, then rmdir-ing the skill, both need the retry
    os.chmod(skills_dir, 0o550)

    hub.SKILLS_DIR = skills_dir
    _patch_lock_file(tmp_path)
    try:
        ok, msg = uninstall_skill("my-skill")
        mode = stat.S_IMODE(skills_dir.stat().st_mode)
    finally:
        with suppress(AttributeError):
            delattr(hub, "SKILLS_DIR")
        with suppress(OSError):
            os.chmod(skills_dir, stat.S_IRWXU)

    assert ok is True, msg
    assert not skill.exists()
    # 0o550 | owner write+execute — not a hard 0o700, which would have dropped group/other r-x
    assert mode == 0o750


@pytest.mark.require_symlinks
def test_a_symlinked_target_keeps_stdout_open(tmp_path):
    """rmtree's retry for a symlinked directory is ``os.path.islink``, which answers a bool — and a bool IS an
    int, so closing "the fd it returned" closes fd 1 (stdout) and the next print dies with EBADF."""
    import tools.skills_sync as sync

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    real = skills_dir / "real-skill"
    real.mkdir()
    (real / "SKILL.md").write_text("x\n")
    link = skills_dir / "linked-skill"
    link.symlink_to(real, target_is_directory=True)

    sync._rmtree_writable(link, root=skills_dir)

    os.fstat(1)


def test_guard_refuses_a_target_outside_the_root(tmp_path):
    """The hub passes its own root, so the guard still has to refuse a path that is not under it."""
    from tools.skills_sync import _rmtree_writable

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    outside = tmp_path / "victim"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")

    with pytest.raises(ValueError, match="scope guard"):
        _rmtree_writable(outside, root=skills_dir)
    assert (outside / "keep.txt").exists()
