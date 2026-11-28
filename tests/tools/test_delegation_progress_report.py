"""Background-delegation progress reports: per-delegation gating, event shape, i18n.

The configurable cadence (``delegation.progress_report_interval``) rides the existing
stale-delegation monitor tick, so the thing worth pinning is that a tick NEVER re-sends a
line for a unit whose own interval has not elapsed, and that concurrent units keep separate
clocks.
"""

import time
from pathlib import Path

import pytest

import hermes_yaml as yaml

from tools import async_delegation as ad
from tools.process_registry import process_registry
from tools.async_delegation import DELEGATION_PROGRESS_EVENT

NOW = 1_700_000_000.0
INTERVAL = 300.0

EN_LOCALES = Path(__file__).resolve().parents[2] / "locales"


@pytest.fixture(autouse=True)
def _clean_records(monkeypatch):
    monkeypatch.setattr(ad, "_records", {})
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _record(delegation_id="d1", *, started_ago=600.0, now=NOW, **over):
    record = {
        "delegation_id": delegation_id,
        "goal": "Investigate the flaky retry test",
        "status": "running",
        "progress_fn": lambda: ((), False),
        "_started": True,
        "_started_at": now - started_ago,
        "dispatched_at": now - started_ago,
        "session_key": "agent:main:telegram:dm:42",
        "origin_ui_session_id": "",
        "origin_session_id": "",
        "parent_session_id": "sess-42",
        "scope_id": "scope-7",
        "user_id": "u-9",
        "user_name": "ana",
        "_progress_token": ((3, "browser_exec", now - 5),),
        "is_batch": True,
        "goals": ["one", "two", "three"],
        "task_indexes": [0, 1, 2],
        "_finished_children": {0: {"task_index": 0, "status": "completed"}},
    }
    record.update(over)
    return record


# ── Gating ──────────────────────────────────────────────────────────────────
def test_not_due_before_interval_elapsed():
    ad._records["d1"] = rec = _record(started_ago=60.0)

    assert ad._collect_due_progress_locked(NOW, INTERVAL) == []
    assert "_progress_report_ts" not in rec


def test_due_emits_exactly_one_event_and_records_its_own_clock():
    ad._records["d1"] = rec = _record(started_ago=600.0)

    events = ad._collect_due_progress_locked(NOW, INTERVAL)

    assert [r["delegation_id"] for r in events] == ["d1"]
    assert rec["_progress_report_ts"] == NOW


def test_repeated_ticks_do_not_repeat_a_report_inside_the_interval():
    ad._records["d1"] = _record(started_ago=600.0)

    assert len(ad._collect_due_progress_locked(NOW, INTERVAL)) == 1
    # A second monitor tick (30 s later) must stay quiet: this is the anti-spam invariant.
    assert ad._collect_due_progress_locked(NOW + 30, INTERVAL) == []
    assert len(ad._collect_due_progress_locked(NOW + INTERVAL, INTERVAL)) == 1


def test_interval_zero_disables_reports():
    ad._records["d1"] = rec = _record(started_ago=10_000.0)

    assert ad._collect_due_progress_locked(NOW, 0) == []
    assert ad._collect_due_progress_locked(NOW, -5) == []
    assert "_progress_report_ts" not in rec


def test_concurrent_delegations_keep_independent_clocks():
    ad._records["old"] = _record("old", started_ago=900.0)
    ad._records["new"] = _record("new", started_ago=10.0)

    assert [r["delegation_id"] for r in ad._collect_due_progress_locked(NOW, INTERVAL)] == ["old"]
    # "new" is still inside its first window and must not inherit "old"'s timestamp.
    assert "_progress_report_ts" not in ad._records["new"]
    assert ad._collect_due_progress_locked(NOW + 10, INTERVAL) == []


def test_queued_and_finished_units_are_silent():
    ad._records["queued"] = _record("queued", started_ago=10_000.0, _started=False, _started_at=None)
    ad._records["done"] = _record("done", started_ago=10_000.0, status="completed")

    assert ad._collect_due_progress_locked(NOW, INTERVAL) == []


# ── Event shape ─────────────────────────────────────────────────────────────
def test_event_carries_progress_counts_latest_tool_and_routing():
    rec = _record(started_ago=12 * 60)
    rec["_progress_token"] = ((1, "read_file", NOW - 60), (2, "browser_exec", NOW - 5))

    evt = ad.progress_event(rec, NOW, language="en")

    assert evt["type"] == DELEGATION_PROGRESS_EVENT
    assert evt["type"] != "async_delegation"  # completion consumers key on the exact name
    assert (evt["minutes"], evt["done"], evt["total"]) == (12, 1, 3)
    assert evt["detail"] == "browser_exec"  # most recent tool wins
    assert "12 min" in evt["text"] and "1/3" in evt["text"] and "browser_exec" in evt["text"]
    assert evt["session_key"] == rec["session_key"]
    assert (evt["scope_id"], evt["user_id"], evt["user_name"]) == ("scope-7", "u-9", "ana")


def test_single_delegation_counts_as_zero_of_one():
    rec = _record(is_batch=False, task_indexes=None, goals=None, _finished_children=None)

    evt = ad.progress_event(rec, NOW, language="en")

    assert (evt["done"], evt["total"]) == (0, 1)


def test_event_falls_back_to_waiting_when_no_child_is_in_a_tool():
    rec = _record()
    rec["_progress_token"] = ((4, None, NOW - 1),)

    evt = ad.progress_event(rec, NOW, language="en")

    assert evt["detail"] == "waiting on the model"
    assert "waiting on the model" in evt["text"]


def test_long_goal_is_truncated_on_a_word_boundary():
    rec = _record(goal="Audit the retry logic in gateway/run_turn.py and propose a targeted fix for it")

    evt = ad.progress_event(rec, NOW, language="en")

    assert evt["text"].count("…") == 1
    assert "gateway/run_turn.py" in evt["text"]  # the useful part survives
    assert not evt["goal"].endswith("…f")


def test_english_catalog_defines_the_key_with_every_placeholder():
    catalog = yaml.safe_load((EN_LOCALES / "en.yaml").read_text(encoding="utf-8"))
    heartbeat = catalog["delegation"]["progress"]["heartbeat"]
    for field in ("{goal}", "{minutes}", "{done}", "{total}", "{detail}"):
        assert field in heartbeat
    assert catalog["delegation"]["progress"]["waiting"]


def test_unknown_language_falls_back_to_english():
    rec = _record(started_ago=180)

    english = ad.progress_event(rec, NOW, language="en")
    fallback = ad.progress_event(rec, NOW, language="zz")

    assert fallback["text"] == english["text"]
    assert "Background task" in fallback["text"]


# ── Config resolution ───────────────────────────────────────────────────────
def test_interval_reads_config_then_env(monkeypatch):
    from tools import delegate_tool_config as cfg

    monkeypatch.setattr(cfg, "_load_config", lambda: {"progress_report_interval": 120})
    assert cfg._get_progress_report_interval() == 120.0

    monkeypatch.setattr(cfg, "_load_config", dict)
    monkeypatch.setenv("HERMES_DELEGATION_PROGRESS_INTERVAL", "90")
    assert cfg._get_progress_report_interval() == 90.0

    monkeypatch.delenv("HERMES_DELEGATION_PROGRESS_INTERVAL")
    assert cfg._get_progress_report_interval() == 0.0  # default off


def test_interval_rejects_garbage_and_negatives(monkeypatch):
    from tools import delegate_tool_config as cfg

    monkeypatch.setattr(cfg, "_load_config", lambda: {"progress_report_interval": "soon"})
    assert cfg._get_progress_report_interval() == 0.0

    monkeypatch.setattr(cfg, "_load_config", lambda: {"progress_report_interval": -30})
    assert cfg._get_progress_report_interval() == 0.0


# ── Monitor wiring ──────────────────────────────────────────────────────────
def test_monitor_tick_enqueues_a_due_report(monkeypatch):
    """The loop must push the line onto the shared queue the gateway/CLI drains."""
    ad._records["d1"] = _record(started_ago=INTERVAL + 60)
    monkeypatch.setattr("tools.delegate_tool_config._get_progress_report_interval", lambda: INTERVAL)

    waits = iter([False, False, True])

    class _Stop:
        def wait(self, _timeout):
            return next(waits, True)

    monkeypatch.setattr(ad, "_monitor_stop", _Stop())

    ad._stale_monitor_loop()

    events = []
    while not process_registry.completion_queue.empty():
        events.append(process_registry.completion_queue.get_nowait())
    progress = [e for e in events if e.get("type") == DELEGATION_PROGRESS_EVENT]
    assert len(progress) == 1
    assert progress[0]["delegation_id"] == "d1"
    assert progress[0]["text"]
