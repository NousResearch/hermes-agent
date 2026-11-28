"""A stop that had to SIGNAL the browser removes the ``Singleton*`` files that browser left behind.

Chromium deletes them itself on a normal exit; one Hermes had to tear down (``_terminate``: SIGTERM, then SIGKILL)
leaves them, and the next Hermes then reads that lock as a browser another environment holds and refuses the profile
(``_foreign_holder`` → ``_foreign_unreachable``: "… delete …/SingletonLock"). ``_reclaim_owned_singletons()`` runs in
both stop paths, under the profile's records lock, after the browser is gone and before its launch record is deleted.

Everything here is fabricated state (profile dir + launch record + registry entry): no browser is spawned.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tools import browser_launcher as launcher
from tools.browser_launcher_holders import ns_identity

SINGLETONS = ("SingletonLock", "SingletonCookie", "SingletonSocket")


@pytest.fixture()
def shared_root(tmp_path, monkeypatch):
    """Keep the profile's own state dir (records, leases, lock) out of the real home."""
    root = tmp_path / "shared"
    monkeypatch.setattr(launcher, "_shared_profiles_root", lambda: root)
    return root


@pytest.fixture()
def profile(tmp_path):
    """A persistent profile as a browser that was torn down leaves it: the three files plus business data."""
    udd = tmp_path / "profile"
    (udd / "Default").mkdir(parents=True)
    (udd / "Default" / "Preferences").write_text('{"session": {"restore_on_startup": 1}}', encoding="utf-8")
    (udd / "Cookies").write_text("business data\n", encoding="utf-8")
    # Chromium writes both as symlinks: the cookie to its 64-bit token, the socket to its socket path in $TMPDIR.
    (udd / "SingletonCookie").symlink_to("11303760953199789830")
    (udd / "SingletonSocket").symlink_to(tmp_path / "elsewhere" / "SingletonSocket")
    return udd


def _lock_to(udd: Path, pid: int, host: str | None = None) -> None:
    (udd / "SingletonLock").unlink(missing_ok=True)
    (udd / "SingletonLock").symlink_to(f"{host or socket.gethostname()}-{pid}")


def _dead_pid() -> int:
    """A pid that certainly is not running: a child that has been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    return proc.pid


def _browser(udd: Path, pid: int, *, key: str = "k") -> launcher.LaunchedBrowser:
    return launcher.LaunchedBrowser(key=key, pid=pid, start_time=None,
                                    cdp_url="ws://127.0.0.1:1/devtools/browser/x", http_url="",
                                    user_data_dir=str(udd), owned=True, ephemeral=False, home_key="h")


def _write_record(browser: launcher.LaunchedBrowser, **extra) -> Path:
    record = launcher._record_path(browser)
    assert record is not None
    record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record.write_text(json.dumps({"pid": browser.pid, "ns": ns_identity(), "owner_pid": os.getpid(),
                                  "user_data_dir": browser.user_data_dir, "start_time": None, **extra}),
                      encoding="utf-8")
    return record


# ── the helper itself ───────────────────────────────────────────────────────────────────────────────

def test_the_instances_own_singletons_go_cookie_and_socket_first_then_the_lock(profile, monkeypatch):
    pid = _dead_pid()
    _lock_to(profile, pid)
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)

    removed = launcher._reclaim_owned_singletons(str(profile), pid)

    assert removed == ["SingletonCookie", "SingletonSocket", "SingletonLock"]
    assert not [n for n in SINGLETONS if os.path.lexists(profile / n)]


def test_a_live_pid_keeps_its_lock(profile, monkeypatch):
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _lock_to(profile, holder.pid)
        monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: True)

        assert launcher._reclaim_owned_singletons(str(profile), holder.pid) == []
        assert os.path.lexists(profile / "SingletonLock")
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_another_instances_lock_is_left_alone(profile, monkeypatch):
    _lock_to(profile, 4242)                       # the lock names a DIFFERENT browser than the one stopped
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)

    assert launcher._reclaim_owned_singletons(str(profile), 1234) == []
    assert sorted(n for n in SINGLETONS if os.path.lexists(profile / n)) == sorted(SINGLETONS)


def test_another_environments_lock_is_left_alone(profile, monkeypatch):
    _lock_to(profile, 4242, host="another-container")
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)

    assert launcher._reclaim_owned_singletons(str(profile), 4242) == []
    assert os.readlink(profile / "SingletonLock") == "another-container-4242"


def test_the_symlinks_target_is_not_touched_and_business_data_survives(profile, monkeypatch, tmp_path):
    pid = _dead_pid()
    outside = tmp_path / "outside-target"
    outside.write_text("keep me\n", encoding="utf-8")
    (profile / "SingletonLock").unlink(missing_ok=True)
    (profile / "SingletonLock").symlink_to(outside)
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)
    # the lock names the outside target, so it is not this instance's: nothing is removed, nothing is followed
    assert launcher._reclaim_owned_singletons(str(profile), pid) == []
    assert outside.read_text(encoding="utf-8") == "keep me\n"

    _lock_to(profile, pid)
    launcher._reclaim_owned_singletons(str(profile), pid)
    assert outside.exists() and outside.read_text(encoding="utf-8") == "keep me\n"
    assert (profile / "Default" / "Preferences").exists()
    assert not os.path.lexists(profile / "SingletonCookie") and not os.path.lexists(profile / "SingletonSocket")
    assert (profile / "Cookies").read_text(encoding="utf-8") == "business data\n"


def test_no_lock_means_nothing_to_do(profile, monkeypatch):
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)
    assert launcher._reclaim_owned_singletons(str(profile), 4242) == []


def test_a_failed_cookie_removal_stops_the_cleanup_and_keeps_the_lock(profile, monkeypatch):
    pid = _dead_pid()
    _lock_to(profile, pid)
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)
    real_unlink = os.unlink

    def fail_on_cookie(path, *a, **kw):
        if str(path).endswith("SingletonCookie"):
            raise PermissionError(13, "denied", str(path))
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(launcher.os, "unlink", fail_on_cookie)
    assert launcher._reclaim_owned_singletons(str(profile), pid) == []
    assert os.path.lexists(profile / "SingletonLock")          # the lock stays: never hand out half a profile
    assert os.path.lexists(profile / "SingletonSocket")


def test_a_failed_socket_removal_keeps_the_lock(profile, monkeypatch):
    pid = _dead_pid()
    _lock_to(profile, pid)
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)
    real_unlink = os.unlink

    def fail_on_socket(path, *a, **kw):
        if str(path).endswith("SingletonSocket"):
            raise PermissionError(13, "denied", str(path))
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(launcher.os, "unlink", fail_on_socket)
    assert launcher._reclaim_owned_singletons(str(profile), pid) == ["SingletonCookie"]
    assert os.path.lexists(profile / "SingletonLock")


def test_a_regular_file_of_that_name_is_not_removed(profile, monkeypatch):
    pid = _dead_pid()
    _lock_to(profile, pid)
    (profile / "SingletonCookie").unlink()                     # a plain file, NOT the symlink Chromium writes
    (profile / "SingletonCookie").write_text("someone's data\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)

    assert launcher._reclaim_owned_singletons(str(profile), pid) == []
    assert (profile / "SingletonCookie").read_text(encoding="utf-8") == "someone's data\n"
    assert os.path.lexists(profile / "SingletonLock")


def test_another_browser_still_using_the_dir_keeps_the_lock(profile, monkeypatch):
    pid = _dead_pid()
    _lock_to(profile, pid)                                     # the lock names our stopped instance ...
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(launcher, "_browser_running_on", lambda _udd: True)   # ... but another Chromium is up here

    assert launcher._reclaim_owned_singletons(str(profile), pid) == []
    assert sorted(n for n in SINGLETONS if os.path.lexists(profile / n)) == sorted(SINGLETONS)


# ── the stop path ──────────────────────────────────────────────────────────────────────────────────

def test_stop_reclaims_before_the_record_is_deleted(profile, shared_root, monkeypatch):
    pid = _dead_pid()
    _lock_to(profile, pid)
    browser = _browser(profile, pid)
    record = _write_record(browser)
    launcher._browsers[browser.key] = browser
    terminated = []
    monkeypatch.setattr(launcher, "_terminate", lambda *a: terminated.append(a))
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)
    try:
        launcher._stop_locked(browser.key)
    finally:
        launcher._browsers.pop(browser.key, None)

    assert terminated == [(pid, None, None)]
    assert not [n for n in SINGLETONS if os.path.lexists(profile / n)]
    assert not record.exists()


def test_stop_keeps_the_lock_when_another_holder_is_still_in(profile, shared_root, monkeypatch):
    pid = _dead_pid()
    _lock_to(profile, pid)
    browser = _browser(profile, pid)
    record = _write_record(browser)
    launcher._browsers[browser.key] = browser
    terminated = []
    monkeypatch.setattr(launcher, "_terminate", lambda *a: terminated.append(a))
    monkeypatch.setattr(launcher, "_last_holder_out", lambda *_a: False)      # another live holder exists
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)
    try:
        launcher._stop_locked(browser.key)
    finally:
        launcher._browsers.pop(browser.key, None)

    assert terminated == []                                                   # nothing was stopped
    assert sorted(n for n in SINGLETONS if os.path.lexists(profile / n)) == sorted(SINGLETONS)
    assert record.exists()


def test_a_failing_lock_removal_does_not_fail_the_stop(profile, shared_root, monkeypatch):
    pid = _dead_pid()
    _lock_to(profile, pid)
    browser = _browser(profile, pid)
    record = _write_record(browser)
    launcher._browsers[browser.key] = browser
    monkeypatch.setattr(launcher, "_terminate", lambda *a: None)
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)

    real_unlink = os.unlink

    def refuse(path, *a, **kw):                  # only the profile's own files fail: the records dir still works
        if str(path).startswith(str(profile)):
            raise OSError(1, "operation not permitted")
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(launcher.os, "unlink", refuse)
    try:
        launcher._stop_locked(browser.key)                                    # must not raise
    finally:
        launcher._browsers.pop(browser.key, None)
    monkeypatch.undo()

    assert not record.exists()                       # the stop completed even though nothing could be removed
    assert os.path.lexists(profile / "SingletonLock")


# ── the reaper path ────────────────────────────────────────────────────────────────────────────────

def test_the_reaper_reclaims_too(profile, shared_root, monkeypatch, tmp_path):
    pid = _dead_pid()
    _lock_to(profile, pid)
    record = shared_root / "digest" / f"{pid}.json"
    record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record.write_text(json.dumps({"pid": pid, "ns": ns_identity(), "owner_pid": os.getpid(),
                                  "user_data_dir": str(profile), "start_time": None}), encoding="utf-8")
    home = str(tmp_path)
    terminated = []
    monkeypatch.setattr(launcher, "_terminate", lambda *a: terminated.append(a))
    monkeypatch.setattr(launcher, "_is_recorded_browser", lambda *_a: True)
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: False)

    assert launcher._reap_record(record, home) is True

    assert terminated == [(pid, None, None)]
    assert not [n for n in SINGLETONS if os.path.lexists(profile / n)]
    assert not record.exists()


def test_the_reaper_leaves_another_holders_lock(profile, shared_root, monkeypatch, tmp_path):
    pid = _dead_pid()
    _lock_to(profile, pid)
    record = shared_root / "digest" / f"{pid}.json"
    record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record.write_text(json.dumps({"pid": pid, "ns": ns_identity(), "owner_pid": os.getpid(),
                                  "user_data_dir": str(profile), "start_time": None}), encoding="utf-8")
    monkeypatch.setattr(launcher, "_terminate", lambda *a: None)
    monkeypatch.setattr(launcher, "_is_recorded_browser", lambda *_a: True)
    monkeypatch.setattr(launcher, "_pid_alive", lambda _pid: True)            # the browser is still running

    assert launcher._reap_record(record, str(tmp_path)) is True
    assert sorted(n for n in SINGLETONS if os.path.lexists(profile / n)) == sorted(SINGLETONS)
