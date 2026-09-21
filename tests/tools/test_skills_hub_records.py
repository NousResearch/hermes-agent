"""Installed-skill provenance survives concurrent updates and failed writes."""

import multiprocessing
import os
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from tools.skills_hub import HubLockFile, ensure_hub_dirs


def _install(lock, name):
    lock.record_install(
        name=name,
        source="github",
        identifier=f"example/skills/{name}",
        trust_level="community",
        scan_verdict="pass",
        skill_hash=name,
        install_path=name,
        files=["SKILL.md"],
    )


def _record_in_process(home, name, read, release=None, uninstall=False, started=None):
    os.environ["HERMES_HOME"] = str(home)
    ensure_hub_dirs()
    lock = HubLockFile()
    original_load = lock.load

    def paused_load():
        data = original_load()
        read.set()
        if release is not None:
            assert release.wait(15)
        return data

    lock.load = paused_load
    if started is not None:
        started.set()
    if uninstall:
        lock.record_uninstall("existing")
    else:
        _install(lock, name)


@pytest.mark.parametrize("uninstall", [False, True])
def test_concurrent_record_updates_compose(uninstall):
    home = get_hermes_home()
    ensure_hub_dirs()
    _install(HubLockFile(), "existing")
    ctx = multiprocessing.get_context("spawn")
    read_a, read_b, release, started_b = [ctx.Event() for _ in range(4)]
    first = ctx.Process(target=_record_in_process, args=(home, "first", read_a, release))
    second = ctx.Process(
        target=_record_in_process,
        args=(home, "second", read_b, None, uninstall, started_b),
    )
    children = []
    try:
        first.start()
        children.append(first)
        assert read_a.wait(10)
        second.start()
        children.append(second)
        assert started_b.wait(10)
        # An unlocked writer reaches load(); a locked writer waits for process A.
        assert not read_b.wait(3)
        release.set()
        for child in children:
            child.join(10)
            assert child.exitcode == 0
        expected = {"first"} if uninstall else {"existing", "first", "second"}
        assert {entry["name"] for entry in HubLockFile().list_installed()} == expected
    finally:
        release.set()
        for child in children:
            child.join(5)
            if child.is_alive():
                child.terminate()
                child.join(5)


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        '{"version": 1, "installed": {"broken": []}}',
        '{"version": 1, "installed": {"broken": {}}}',
        '{"version": 1, "installed": {"broken": {"install_path": []}}}',
        (
            '{"version": 1, "installed": {"broken": {"source": "github", '
            '"trust_level": "community", "install_path": "broken", "name": []}}}'
        ),
    ],
)
def test_invalid_record_file_is_not_overwritten(tmp_path, payload):
    path = tmp_path / "lock.json"
    path.write_text(payload, encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ValueError, match="Invalid skills hub lock file"):
        _install(HubLockFile(path), "failed")

    assert path.read_bytes() == before


def test_invalid_utf8_record_file_is_not_overwritten(tmp_path):
    path = tmp_path / "lock.json"
    path.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="Invalid skills hub lock file"):
        _install(HubLockFile(path), "failed")

    assert path.read_bytes() == b"\xff"


def test_unreadable_record_file_is_reported_with_context(tmp_path, monkeypatch):
    path = tmp_path / "lock.json"

    def deny_read(self, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", deny_read)
    with pytest.raises(ValueError, match="Invalid skills hub lock file.*denied"):
        HubLockFile(path).load()


def test_optional_backfill_preserves_concurrent_record(tmp_path, monkeypatch):
    from tools import skills_sync_optional

    skills_dir = tmp_path / "skills"
    optional_dir = tmp_path / "optional-skills"
    for root in (skills_dir, optional_dir):
        skill = root / "official-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: official-skill\n---\n", encoding="utf-8")

    class FakeSync:
        @staticmethod
        def _skills_dir():
            return skills_dir

        @staticmethod
        def _get_optional_dir():
            return optional_dir

        @staticmethod
        def _dir_hash(_path):
            return "same"

    monkeypatch.setattr(skills_sync_optional, "_ss", lambda: FakeSync)

    def concurrent_install(_path):
        _install(HubLockFile(skills_dir / ".hub" / "lock.json"), "concurrent")
        return "official-hash"

    monkeypatch.setattr(skills_sync_optional, "_content_hash", concurrent_install)

    assert skills_sync_optional._backfill_optional_provenance(quiet=True) == ["official-skill"]
    assert {
        entry["name"] for entry in HubLockFile(skills_dir / ".hub" / "lock.json").list_installed()
    } == {"concurrent", "official-skill"}


def test_optional_backfill_preserves_corrupt_record(tmp_path, monkeypatch):
    from tools import skills_sync_optional

    skills_dir = tmp_path / "skills"
    optional_dir = tmp_path / "optional-skills"
    optional_dir.mkdir()
    lock_path = skills_dir / ".hub" / "lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("{", encoding="utf-8")

    class FakeSync:
        @staticmethod
        def _skills_dir():
            return skills_dir

        @staticmethod
        def _get_optional_dir():
            return optional_dir

    monkeypatch.setattr(skills_sync_optional, "_ss", lambda: FakeSync)

    with pytest.raises(ValueError, match="Invalid skills hub lock file"):
        skills_sync_optional._backfill_optional_provenance(quiet=True)
    assert lock_path.read_text(encoding="utf-8") == "{"


def test_failed_atomic_publication_preserves_record_file(tmp_path, monkeypatch):
    lock = HubLockFile(tmp_path / "lock.json")
    _install(lock, "existing")
    before = lock.path.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("injected publication failure")

    with monkeypatch.context() as patch:
        patch.setattr("utils.atomic_replace", fail_replace)
        with pytest.raises(OSError, match="injected publication failure"):
            _install(lock, "failed")

    assert lock.path.read_bytes() == before
    _install(lock, "recovered")
    lock.record_uninstall("existing")
    assert {entry["name"] for entry in lock.list_installed()} == {"recovered"}
