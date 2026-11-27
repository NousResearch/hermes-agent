"""``profile_command_allowlist``: an admin-set, per-profile, immutable allowlist.

Distinct from ``command_allowlist`` (user-approved via `[a]lways`, revocable by
hand-editing config.yaml): profile_command_allowlist is set by whoever deploys a
profile (e.g. a fleet operator's profile template) to pre-authorize a fixed set
of commands for that profile specifically, without exposing them as something a
user's own `[a]lways` / on-disk edit can accidentally revoke, and without ever
being written into the user-editable ``command_allowlist`` list (which would
make them look user-revocable, and would resurrect a pattern an admin later
removes from ``profile_command_allowlist``).
"""

import pytest

import tools.approval as approval
from tools.approval import _read_permanent_allowlist


@pytest.fixture
def fake_config(monkeypatch):
    """A dict standing in for config.yaml, plus a clean module baseline."""
    store = {"command_allowlist": [], "profile_command_allowlist": []}

    def _load():
        # Return the stored values as-is (never coerce through list(...) -- a test
        # simulating a malformed string value must see that exact string, not its
        # exploded characters).
        return {
            "command_allowlist": store["command_allowlist"],
            "profile_command_allowlist": store["profile_command_allowlist"],
        }

    def _save(config):
        store["command_allowlist"] = list(config.get("command_allowlist", []))
        # profile_command_allowlist is admin-set — never written by save_permanent_allowlist,
        # but a test asserting that must see it untouched, so still reflect it if present.
        if "profile_command_allowlist" in config:
            store["profile_command_allowlist"] = list(config["profile_command_allowlist"])

    monkeypatch.setattr("hermes_cli.config.load_config", _load, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _load, raising=False)
    monkeypatch.setattr("hermes_cli.config.save_config", _save, raising=False)

    saved_approved = set(approval._permanent_approved)
    saved_baseline = dict(approval._permanent_baseline_by_home)
    approval._permanent_approved.clear()
    approval._permanent_baseline_by_home.clear()
    try:
        yield store
    finally:
        approval._permanent_approved.clear()
        approval._permanent_approved.update(saved_approved)
        approval._permanent_baseline_by_home.clear()
        approval._permanent_baseline_by_home.update(saved_baseline)


def _start_process_with(store, *, user=(), profile=()):
    """Simulate import-time load against the current file contents."""
    store["command_allowlist"] = list(user)
    store["profile_command_allowlist"] = list(profile)
    merged = set(user) | set(profile)
    approval.load_permanent(merged)
    approval._permanent_baseline_by_home[""] = merged


class TestReadMergesProfileAllowlist:
    def test_profile_patterns_are_merged_into_the_read_set(self, fake_config):
        fake_config["command_allowlist"] = ["git status"]
        fake_config["profile_command_allowlist"] = ["git log", "npm test"]

        result = _read_permanent_allowlist()

        assert result == {"git status", "git log", "npm test"}

    def test_malformed_profile_allowlist_is_ignored_not_fatal(self, fake_config):
        fake_config["command_allowlist"] = ["git status"]
        fake_config["profile_command_allowlist"] = "not-a-list"

        result = _read_permanent_allowlist()

        assert result == {"git status"}

    def test_malformed_user_allowlist_does_not_disable_profile_allowlist(self, fake_config):
        """A broken user-editable list must not take down the admin-set one."""
        fake_config["command_allowlist"] = "not-a-list"
        fake_config["profile_command_allowlist"] = ["git log"]

        result = _read_permanent_allowlist()

        assert result == {"git log"}


class TestProfileAllowlistNeverPersistedBack:
    def test_profile_pattern_is_not_written_to_command_allowlist(self, fake_config):
        _start_process_with(fake_config, user=["ls *"], profile=["git log"])

        approval.approve_permanent("docker *")
        approval.save_permanent_allowlist(approval._permanent_approved)

        assert "git log" not in fake_config["command_allowlist"]
        assert sorted(fake_config["command_allowlist"]) == ["docker *", "ls *"]

    def test_profile_pattern_added_after_baseline_is_still_excluded_from_disk(self, fake_config):
        """A profile pattern that shows up AFTER this process's baseline was captured
        (e.g. an admin edited the profile template and the process re-read config)
        must still never land in command_allowlist -- the exclusion in
        save_permanent_allowlist is explicit, not just an artifact of baseline timing."""
        _start_process_with(fake_config, user=["ls *"], profile=[])
        # Admin adds a profile pattern after this process's baseline was captured.
        fake_config["profile_command_allowlist"] = ["git log"]
        # The governing set now reflects it too (as a real caller's flow would,
        # e.g. via a config-reload path re-running _read_permanent_allowlist()).
        approval._permanent_approved.add("git log")

        approval.approve_permanent("docker *")
        approval.save_permanent_allowlist(approval._permanent_approved)

        assert "git log" not in fake_config["command_allowlist"]
        assert sorted(fake_config["command_allowlist"]) == ["docker *", "ls *"]

    def test_profile_pattern_stays_honoured_in_memory_after_save(self, fake_config):
        """Excluded from the disk write, but is_approved() must still honour it."""
        _start_process_with(fake_config, user=["ls *"], profile=["git log"])

        approval.approve_permanent("docker *")
        approval.save_permanent_allowlist(approval._permanent_approved)

        assert "git log" in approval._permanent_approved
        assert approval.is_approved("some-session", "git log")

    def test_admin_removing_a_profile_pattern_is_not_resurrected_by_a_save(self, fake_config):
        """Mirrors the existing revoked-command_allowlist-entry guarantee, for the
        profile list: once an admin removes a pattern from profile_command_allowlist,
        a save from a process that still remembers it in memory must not write it
        anywhere that would keep it alive."""
        _start_process_with(fake_config, user=["ls *"], profile=["git log"])
        fake_config["profile_command_allowlist"] = []  # admin revokes it

        approval.approve_permanent("docker *")
        approval.save_permanent_allowlist(approval._permanent_approved)

        assert "git log" not in fake_config["command_allowlist"]
        assert sorted(fake_config["command_allowlist"]) == ["docker *", "ls *"]
