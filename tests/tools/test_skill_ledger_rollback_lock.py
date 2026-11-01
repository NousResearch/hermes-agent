"""Ledger rollback and ordinary skill writes share one ownership window."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


_CONTENT = "---\nname: rollback-probe\ndescription: Use when testing rollback.\n---\nValue one.\n"


@pytest.fixture
def rollback_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("skills:\n  ledger: true\n", encoding="utf-8")
    from tools import skill_ledger, skill_manager_tool

    def mutate(**op):
        return json.loads(skill_manager_tool.registry.get_entry("skill_manage").handler(
            {"operations": [{"name": "rollback-probe", **op}]}))

    assert mutate(action="create", category="testing", content=_CONTENT)["success"]
    assert mutate(action="patch", old_string="Value one.", new_string="Value two.")["success"]
    entry = skill_ledger.list_entries("rollback-probe")[0]
    target = home / "skills" / "testing" / "rollback-probe" / "SKILL.md"
    return skill_ledger, mutate, entry, target


@pytest.mark.parametrize("pause_action", ["pre-rollback", "rollback"])
def test_rollback_serializes_safety_restore_and_completion(rollback_env, monkeypatch, pause_action):
    """A writer arriving after safety capture cannot land before rollback's final row.

    Pausing the actual append forces the unsafe window; the second writer uses
    the real batch/patch path and the OS lock, not a mocked mutation.
    """
    ledger, mutate, entry, target = rollback_env
    from tools import skill_usage

    paused = threading.Event()
    writer_ready = threading.Event()
    release = threading.Event()
    ownership_guard = threading.Lock()
    rollback_locks = set()
    real_append = ledger.append_entry
    real_flock = skill_usage._flock

    def paused_append(action, *args, **kwargs):
        if action == pause_action:
            paused.set()
            assert release.wait(10), "rollback was not released"
        return real_append(action, *args, **kwargs)

    def observed_flock(fd, lock):
        if lock and threading.current_thread().name.startswith("writer"):
            with ownership_guard:
                if fd.name in rollback_locks:
                    writer_ready.set()  # about to contend on a real held OS lock
        result = real_flock(fd, lock)
        if threading.current_thread().name.startswith("rollback"):
            with ownership_guard:
                if lock:
                    rollback_locks.add(fd.name)
                else:
                    rollback_locks.discard(fd.name)
        return result

    def writer():
        try:
            # A full rewrite is valid both before and after rollback, so failure
            # cannot be explained by a stale patch match.
            # Note: content-form patch records as 'edit', not 'patch'.
            return mutate(action="patch", name="testing/rollback-probe",
                          content=_CONTENT.replace("Value one.", "Value three."))
        finally:
            writer_ready.set()  # on base, the write finishes before we release rollback

    monkeypatch.setattr(ledger, "append_entry", paused_append)
    monkeypatch.setattr(skill_usage, "_flock", observed_flock)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollback") as rollback_pool, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="writer") as writer_pool:
        rollback = rollback_pool.submit(ledger.rollback_entry, entry["id"])
        try:
            assert paused.wait(10), "rollback did not reach its safety/commit window"
            pending = writer_pool.submit(writer)
            # Resume only once the writer either reaches a lock actually held
            # by rollback or completes its write. No sleep-based race window.
            assert writer_ready.wait(10), "writer neither contended nor completed"
        finally:
            release.set()
        ok, message = rollback.result(timeout=10)
        result = pending.result(timeout=10)

    assert ok, message
    assert result["success"], result
    assert "Value three." in target.read_text(encoding="utf-8"), "rollback erased a successful edit"
    rows = ledger.list_entries()
    assert [row["action"] for row in rows[:3]] == ["edit", "rollback", "pre-rollback"]
    safety = rows[2]
    assert b"Value two." in ledger.read_blob(safety["before"][0]["sha256"])
    assert b"Value one." in ledger.read_blob(rows[0]["before"][0]["sha256"])


@pytest.mark.parametrize("failure", ["capture", "append", "restore"])
def test_rollback_failure_releases_lock_for_next_writer(rollback_env, monkeypatch, failure):
    """Both fail-closed returns and raised restore errors release ownership."""
    ledger, mutate, entry, target = rollback_env
    current = target.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    with monkeypatch.context() as patch:
        if failure == "capture":
            patch.setattr(ledger, "_store_blob", fail)
        elif failure == "append":
            patch.setattr(ledger, "append_entry", lambda *args, **kwargs: None)
        else:
            real_write = Path.write_bytes

            def fail_restore(path, data):
                if path == target:
                    fail()
                return real_write(path, data)

            patch.setattr(Path, "write_bytes", fail_restore)
        if failure == "restore":
            with pytest.raises(OSError, match="injected failure"):
                ledger.rollback_entry(entry["id"])
        else:
            ok, message = ledger.rollback_entry(entry["id"])
            assert not ok and "safety capture failed" in message
        assert target.read_bytes() == current

    # Another thread must acquire it: same-thread reentrancy would hide a leak.
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(mutate, action="patch", old_string="Value two.",
                             new_string="Value three.").result(timeout=10)
    assert result["success"], result
    assert "Value three." in target.read_text(encoding="utf-8")
