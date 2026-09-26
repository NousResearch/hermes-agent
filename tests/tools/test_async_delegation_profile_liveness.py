"""Delegation replay must not resurrect a removed profile (#123265)."""

from pathlib import Path
from queue import Queue
import time

import pytest

from hermes_constants import (
    mark_named_profile_deleted,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import async_delegation as ad


@pytest.mark.parametrize("tombstoned", [False, True])
def test_replay_refuses_missing_named_profile(tmp_path, monkeypatch, tombstoned):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    profile = root / "profiles" / "gone"
    if tombstoned:
        mark_named_profile_deleted(profile)
    token = set_hermes_home_override(profile)
    try:
        try:
            ad.restore_undelivered_completions(Queue())
        except FileNotFoundError:
            pass  # the caller owns how a removed profile is reported
        assert not profile.exists(), "delegation replay recreated a missing profile"
    finally:
        reset_hermes_home_override(token)


def test_replay_keeps_live_profile_ledgers_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    homes = [root / "profiles" / name for name in ("alpha", "beta")]
    for home in homes:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        token = set_hermes_home_override(home)
        try:
            ad._persist_dispatch({"delegation_id": home.name, "goal": home.name, "dispatched_at": time.time()})
            ad._persist_completion(
                {"delegation_id": home.name, "status": "completed"},
                {"summary": home.name},
            )
        finally:
            reset_hermes_home_override(token)
    for home in [*homes, homes[0]]:
        token = set_hermes_home_override(home)
        try:
            queue = Queue()
            assert ad.restore_undelivered_completions(queue) == 1
            assert queue.get_nowait()["delegation_id"] == home.name
            assert queue.empty()
            assert ad.get_durable_delegation(home.name)["result"]["summary"] == home.name
        finally:
            reset_hermes_home_override(token)
