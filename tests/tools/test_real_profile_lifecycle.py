"""Tests for persistent ownership of directly-launched real-profile Chrome."""

import json
import os
from unittest.mock import Mock, patch


def _setup_home(tmp_path, monkeypatch):
    import tools.real_profile_lifecycle as lifecycle

    home = tmp_path / "hermes-home"
    monkeypatch.setattr(lifecycle, "_hermes_home", lambda: home)
    return lifecycle, home, home / "browser-profile" / "chrome"


def _record(lifecycle, home, *, pid=4321, target_start=99, owners=None):
    profile = home / "browser-profile" / "chrome"
    binary = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    record = lifecycle._base_record(
        "chrome", profile.resolve(), lifecycle._canonical_path(binary), pid, target_start
    )
    record = lifecycle._with_owner_fields(record, owners or [])
    assert lifecycle._write_record("chrome", record)
    return record


def test_register_persists_exact_target_and_owner(tmp_path, monkeypatch):
    lifecycle, home, profile = _setup_home(tmp_path, monkeypatch)
    binary = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    target_pid = 4321
    monkeypatch.setattr(
        lifecycle,
        "_safe_start_time",
        lambda pid: 222 if pid == target_pid else 333,
    )

    record = lifecycle.register_real_profile_chrome(
        "chrome", str(profile), binary, target_pid
    )

    assert record is not None
    stored = json.loads(
        (home / "cache" / "browser-use" / "real-profile" / "chrome.json").read_text()
    )
    assert stored["pid"] == target_pid
    assert stored["target_start_time"] == 222
    assert stored["profile_dir"] == str(profile.resolve())
    assert stored["binary"] == binary
    assert stored["owners"] == [{"pid": os.getpid(), "start_time": 333}]
    if os.name != "nt":
        assert (home / "cache" / "browser-use" / "real-profile").stat().st_mode & 0o077 == 0


def test_register_does_not_replace_another_live_target(tmp_path, monkeypatch):
    lifecycle, home, profile = _setup_home(tmp_path, monkeypatch)
    binary = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    _record(lifecycle, home, pid=1111, target_start=10)
    monkeypatch.setattr(lifecycle, "_safe_start_time", lambda pid: 20)
    monkeypatch.setattr(lifecycle, "_target_matches", lambda *args: True)

    result = lifecycle.register_real_profile_chrome("chrome", str(profile), binary, 2222)

    assert result is None
    stored = json.loads(
        (home / "cache" / "browser-use" / "real-profile" / "chrome.json").read_text()
    )
    assert stored["pid"] == 1111


def test_claim_adds_a_second_live_owner(tmp_path, monkeypatch):
    lifecycle, home, profile = _setup_home(tmp_path, monkeypatch)
    binary = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    other_owner = {"pid": 7777, "start_time": 70}
    _record(lifecycle, home, pid=4321, target_start=99, owners=[other_owner])
    monkeypatch.setattr(lifecycle, "_target_matches", lambda *args: True)
    monkeypatch.setattr(
        lifecycle,
        "_safe_start_time",
        lambda pid: 88 if pid == os.getpid() else 70,
    )
    monkeypatch.setattr(
        lifecycle,
        "_owner_is_alive",
        lambda owner: owner == other_owner or owner["pid"] == os.getpid(),
    )

    result = lifecycle.claim_real_profile_chrome("chrome", str(profile), binary)

    assert result and result["pid"] == 4321
    stored = json.loads(
        (home / "cache" / "browser-use" / "real-profile" / "chrome.json").read_text()
    )
    assert stored["owners"] == [other_owner, {"pid": os.getpid(), "start_time": 88}]


def test_retire_releases_only_current_owner_when_browser_is_shared(tmp_path, monkeypatch):
    lifecycle, home, profile = _setup_home(tmp_path, monkeypatch)
    binary = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    current_owner = {"pid": os.getpid(), "start_time": 88}
    other_owner = {"pid": 7777, "start_time": 70}
    _record(lifecycle, home, pid=4321, target_start=99, owners=[other_owner, current_owner])
    monkeypatch.setattr(lifecycle, "_safe_start_time", lambda pid: 88)
    monkeypatch.setattr(
        lifecycle,
        "_owner_is_alive",
        lambda owner: owner == other_owner,
    )
    terminate = Mock()

    with patch(
        "tools.process_registry.ProcessRegistry._terminate_host_pid",
        terminate,
    ):
        result = lifecycle.retire_real_profile_chrome(
            "chrome", str(profile), binary, 4321
        )

    assert result is False
    terminate.assert_not_called()
    stored = json.loads(
        (home / "cache" / "browser-use" / "real-profile" / "chrome.json").read_text()
    )
    assert stored["owners"] == [other_owner]


def test_retire_last_owner_tree_kills_exact_target(tmp_path, monkeypatch):
    lifecycle, home, profile = _setup_home(tmp_path, monkeypatch)
    binary = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    current_owner = {"pid": os.getpid(), "start_time": 88}
    _record(lifecycle, home, pid=4321, target_start=99, owners=[current_owner])
    monkeypatch.setattr(lifecycle, "_safe_start_time", lambda pid: 88)
    monkeypatch.setattr(lifecycle, "_target_matches", lambda *args: True)
    terminate = Mock()

    with patch(
        "tools.process_registry.ProcessRegistry._terminate_host_pid",
        terminate,
    ):
        result = lifecycle.retire_real_profile_chrome(
            "chrome", str(profile), binary, 4321
        )

    assert result is True
    terminate.assert_called_once_with(4321, expected_start=99)
    assert not (home / "cache" / "browser-use" / "real-profile" / "chrome.json").exists()


def test_reaper_kills_exact_orphan_and_removes_record(tmp_path, monkeypatch):
    lifecycle, home, _profile = _setup_home(tmp_path, monkeypatch)
    _record(lifecycle, home, owners=[])
    monkeypatch.setattr(lifecycle, "_live_owners", lambda record: [])
    monkeypatch.setattr(lifecycle, "_target_matches", side_effect := Mock(side_effect=[True, False]))
    terminate = Mock()

    with patch(
        "tools.process_registry.ProcessRegistry._terminate_host_pid",
        terminate,
    ):
        reaped = lifecycle.reap_orphaned_real_profile_chrome()

    assert reaped == 1
    terminate.assert_called_once_with(4321, expected_start=99)
    assert not (home / "cache" / "browser-use" / "real-profile" / "chrome.json").exists()
    assert side_effect.call_count == 2


def test_reaper_spares_live_owner(tmp_path, monkeypatch):
    lifecycle, home, _profile = _setup_home(tmp_path, monkeypatch)
    _record(lifecycle, home, owners=[{"pid": os.getpid(), "start_time": 1}])
    monkeypatch.setattr(lifecycle, "_live_owners", lambda record: [{"pid": os.getpid()}])
    matches = Mock(return_value=True)
    monkeypatch.setattr(lifecycle, "_target_matches", matches)
    terminate = Mock()

    with patch(
        "tools.process_registry.ProcessRegistry._terminate_host_pid",
        terminate,
    ):
        assert lifecycle.reap_orphaned_real_profile_chrome() == 0

    matches.assert_not_called()
    terminate.assert_not_called()


def test_reaper_removes_recycled_pid_without_signalling(tmp_path, monkeypatch):
    lifecycle, home, _profile = _setup_home(tmp_path, monkeypatch)
    _record(lifecycle, home, owners=[])
    monkeypatch.setattr(lifecycle, "_live_owners", lambda record: [])
    monkeypatch.setattr(lifecycle, "_target_matches", lambda *args: False)
    terminate = Mock()

    with patch(
        "tools.process_registry.ProcessRegistry._terminate_host_pid",
        terminate,
    ):
        assert lifecycle.reap_orphaned_real_profile_chrome() == 0

    terminate.assert_not_called()
    assert not (home / "cache" / "browser-use" / "real-profile" / "chrome.json").exists()


def test_reaper_leaves_malformed_record_without_signalling(tmp_path, monkeypatch):
    lifecycle, home, _profile = _setup_home(tmp_path, monkeypatch)
    state_dir = home / "cache" / "browser-use" / "real-profile"
    state_dir.mkdir(parents=True)
    record_path = state_dir / "chrome.json"
    record_path.write_text("{not-json")
    terminate = Mock()

    with patch(
        "tools.process_registry.ProcessRegistry._terminate_host_pid",
        terminate,
    ):
        assert lifecycle.reap_orphaned_real_profile_chrome() == 0

    terminate.assert_not_called()
    assert record_path.exists()
