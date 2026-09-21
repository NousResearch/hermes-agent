"""Installed-skill provenance survives concurrent updates and failed writes."""

import multiprocessing
import os

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


def _record_in_process(home, name, read, release=None, uninstall=False):
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
    read_a, read_b, release = [ctx.Event() for _ in range(3)]
    first = ctx.Process(target=_record_in_process, args=(home, "first", read_a, release))
    second = ctx.Process(
        target=_record_in_process,
        args=(home, "second", read_b, None, uninstall),
    )
    children = []
    try:
        first.start()
        children.append(first)
        assert read_a.wait(10)
        second.start()
        children.append(second)
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


@pytest.mark.parametrize("payload", ["{", "[]"])
def test_invalid_record_file_is_not_overwritten(tmp_path, payload):
    path = tmp_path / "lock.json"
    path.write_text(payload, encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ValueError, match="Invalid skills hub lock file"):
        _install(HubLockFile(path), "failed")

    assert path.read_bytes() == before


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