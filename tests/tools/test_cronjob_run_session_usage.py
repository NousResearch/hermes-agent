"""A manual cron run's usage line: the API calls its run session recorded over the compression chain and the newest
model, or nothing when there is no run session to read."""

import sqlite3

import pytest

from hermes_constants import get_hermes_home
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import UsageDelta
from persistence_assembly import open_persistence, profile_scope
from tools.cronjob_tools import _run_session_usage


def _calls(store, session_id, count):
    store.usage.queue_usage(session_id, UsageDelta(api_call_count=count))
    assert store.usage.flush_usage()


@pytest.fixture()
def home():
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture()
def fired(home):
    """Job ``job-1`` fired twice: execution ``exec-1`` ran ``run-1`` and compressed into ``run-2``; ``exec-2`` ran ``other``."""
    store = open_persistence(profile_scope(home=home), eager=True)
    try:
        epoch = store.runtime.begin_epoch(instance_id="test")
        store.sessions.create(NewSession(session_id="run-1", source="cron", model="model-a"))
        store.sessions.create(NewSession(session_id="other", source="cron", model="model-other"))
        for session_id, execution in (("run-1", "exec-1"), ("other", "exec-2")):
            store.admissions.admit(epoch=epoch, principal_id="cron-owner", session_id=session_id,
                                   request_id=f"cron:job-1:{execution}", payload={"text": "go"})
        _calls(store, "run-1", 2)
        _calls(store, "other", 7)
        store.sessions.end("run-1", "compression")
        store.sessions.create(NewSession(session_id="run-2", source="cron", model="model-b", parent_session_id="run-1"))
        _calls(store, "run-2", 3)
    finally:
        store.close()
    return home


def test_the_run_sessions_calls_sum_over_its_compression_chain_with_the_newest_model(fired):
    assert _run_session_usage("job-1", "exec-1") == {"api_calls": 5, "model": "model-b"}
    assert _run_session_usage("job-1", "exec-2") == {"api_calls": 7, "model": "model-other"}


def test_no_execution_or_no_run_session_reads_nothing(fired):
    assert _run_session_usage("job-1", None) == {}
    assert _run_session_usage("job-1", "") == {}
    assert _run_session_usage("job-1", "exec-missing") == {}
    assert _run_session_usage("job-2", "exec-1") == {}


def test_a_missing_state_db_reads_nothing_and_creates_none(home):
    assert not (home / "state.db").exists()
    assert _run_session_usage("job-1", "exec-1") == {}
    assert not (home / "state.db").exists()


def test_a_locked_state_db_reads_nothing(fired):
    conn = sqlite3.connect(fired / "state.db", timeout=0)
    try:
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("BEGIN EXCLUSIVE")
        assert _run_session_usage("job-1", "exec-1") == {}
    finally:
        conn.rollback()
        conn.close()
    assert _run_session_usage("job-1", "exec-1") == {"api_calls": 5, "model": "model-b"}
