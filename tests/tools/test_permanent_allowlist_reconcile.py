"""`command_allowlist` is a file the operator edits; a save must not clobber it.

`load_permanent_allowlist()` runs once, at module import (tools/approval.py, the
call at the bottom of the module), and `load_permanent()` only ever unions into
`_permanent_approved` — nothing removes. `save_permanent_allowlist()` then wrote
that in-memory set straight back over `config["command_allowlist"]`.

So a hand edit made while a Hermes process is live was undone by the next
`[a]lways`, in both directions at once: an entry the operator ADDED on disk was
deleted, and an entry they REMOVED — the documented way to withdraw a standing
approval — came back.

Reconciling at write time fixes both. The file is re-read and the result is
what is on disk now, plus what this process approved since its own baseline.
"""

import logging

import pytest

import tools.approval as approval


@pytest.fixture
def fake_config(monkeypatch):
    """A dict standing in for config.yaml, plus a clean module baseline."""
    store = {"command_allowlist": []}

    def _load():
        return {"command_allowlist": list(store["command_allowlist"])}

    def _save(config):
        store["command_allowlist"] = list(config.get("command_allowlist", []))

    monkeypatch.setattr("hermes_cli.config.load_config", _load, raising=False)
    monkeypatch.setattr("hermes_cli.config.save_config", _save, raising=False)

    saved_approved = set(approval._permanent_approved)
    saved_baseline = set(approval._permanent_baseline)
    approval._permanent_approved.clear()
    approval._permanent_baseline = set()
    try:
        yield store
    finally:
        approval._permanent_approved.clear()
        approval._permanent_approved.update(saved_approved)
        approval._permanent_baseline = saved_baseline


def _start_process_with(store, entries):
    """Simulate import-time load against the current file contents."""
    store["command_allowlist"] = list(entries)
    approval.load_permanent(set(entries))
    approval._permanent_baseline = set(entries)


# ── the two halves of the bug ─────────────────────────────────────────


def test_an_entry_the_operator_added_on_disk_survives_a_save(fake_config):
    _start_process_with(fake_config, ["git status", "ls *"])
    fake_config["command_allowlist"] = ["git status", "ls *", "npm test"]

    approval.approve_permanent("docker *")
    approval.save_permanent_allowlist(approval._permanent_approved)

    assert "npm test" in fake_config["command_allowlist"], (
        "a hand-added entry was deleted by an unrelated [a]lways"
    )
    assert "docker *" in fake_config["command_allowlist"]


def test_an_entry_the_operator_revoked_on_disk_is_not_resurrected(fake_config):
    _start_process_with(fake_config, ["git status", "ls *"])
    fake_config["command_allowlist"] = ["ls *"]          # operator revokes it

    approval.approve_permanent("docker *")
    approval.save_permanent_allowlist(approval._permanent_approved)

    assert "git status" not in fake_config["command_allowlist"], (
        "a revoked standing approval came back on the next save"
    )
    assert sorted(fake_config["command_allowlist"]) == ["docker *", "ls *"]


def test_a_revoked_entry_stops_being_honoured_in_memory_after_the_save(fake_config):
    _start_process_with(fake_config, ["git status", "ls *"])
    fake_config["command_allowlist"] = ["ls *"]

    approval.approve_permanent("docker *")
    approval.save_permanent_allowlist(approval._permanent_approved)

    assert "git status" not in approval._permanent_approved
    assert "ls *" in approval._permanent_approved
    assert "docker *" in approval._permanent_approved


# ── the ordinary path must not move ───────────────────────────────────


def test_an_untouched_file_round_trips_unchanged(fake_config):
    _start_process_with(fake_config, ["git status", "ls *"])

    approval.approve_permanent("docker *")
    approval.save_permanent_allowlist(approval._permanent_approved)

    assert sorted(fake_config["command_allowlist"]) == ["docker *", "git status", "ls *"]


def test_repeated_saves_are_idempotent(fake_config):
    _start_process_with(fake_config, ["ls *"])
    approval.approve_permanent("docker *")

    approval.save_permanent_allowlist(approval._permanent_approved)
    first = sorted(fake_config["command_allowlist"])
    approval.save_permanent_allowlist(approval._permanent_approved)

    assert sorted(fake_config["command_allowlist"]) == first == ["docker *", "ls *"]


def test_a_second_process_writing_first_does_not_lose_this_ones_approval(fake_config):
    """Two live Hermes processes. Whoever writes second must not drop the first."""
    _start_process_with(fake_config, ["ls *"])
    # The other process approved something and wrote it out.
    fake_config["command_allowlist"] = ["ls *", "cargo *"]

    approval.approve_permanent("docker *")
    approval.save_permanent_allowlist(approval._permanent_approved)

    assert sorted(fake_config["command_allowlist"]) == ["cargo *", "docker *", "ls *"]


def test_empty_start_and_first_approval(fake_config):
    _start_process_with(fake_config, [])
    approval.approve_permanent("ls *")
    approval.save_permanent_allowlist(approval._permanent_approved)
    assert fake_config["command_allowlist"] == ["ls *"]


def test_save_failure_is_logged_not_raised(fake_config, monkeypatch, caplog):
    """The existing contract: a config write failure must not break approval."""
    _start_process_with(fake_config, ["ls *"])

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom, raising=False)
    approval.approve_permanent("docker *")
    with caplog.at_level(logging.WARNING, logger=approval.logger.name):
        approval.save_permanent_allowlist(approval._permanent_approved)   # must not raise
    assert "Could not save allowlist" in caplog.text
    assert "docker *" in approval._permanent_approved


def test_a_caller_that_passes_a_smaller_set_does_not_remove(fake_config):
    """Documented consequence of reconciling: ``patterns`` may only add.

    A future `allowlist remove` built on this function would silently no-op.
    The docstring says so; this pins it so the next reader finds out here
    rather than in production.
    """
    _start_process_with(fake_config, ["ls *", "docker *"])

    approval.save_permanent_allowlist({"ls *"})          # tries to drop docker *

    assert "docker *" in fake_config["command_allowlist"]
