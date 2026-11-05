"""Enrolled results survive readiness disable without weakening the scanner.

These cases enroll finite work and feed its framed result through the real
ingestion seam. Ordinary strings remain silent. Native admission and
real subprocess coverage live in tests/gateway/test_process_work_results.py.
"""
from unittest.mock import patch

import pytest

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    WATCH_LIFETIME_MAX_HITS,
    WATCH_MIN_INTERVAL_SECONDS,
    WATCH_STRIKE_LIMIT,
)


def _drain(registry):
    events = []
    while not registry.completion_queue.empty():
        events.append(registry.completion_queue.get_nowait())
    return events


def _disabled_source(reason):
    registry = ProcessRegistry()
    source = ProcessSession(
        id="proc_fixture", command="fixture persistent subscription",
        task_id="fixture-owner", owner_task_id="fixture-owner",
        session_key="agent:main:discord:dm:fixture", parent_session_id="fixture-parent",
        started_at=1000.0, watch_patterns=["TASK_UPDATE"],
    )
    now = 1000.0
    with patch("tools.process_registry.time.time", side_effect=lambda: now):
        count = WATCH_STRIKE_LIMIT if reason == "strikes" else WATCH_LIFETIME_MAX_HITS
        for revision in range(count):
            registry._check_watch_patterns(source, f"TASK_UPDATE revision={revision} running\n")
            if reason == "strikes":
                now += 1
                registry._check_watch_patterns(source, f"TASK_UPDATE revision={revision} busy\n")
            now += WATCH_MIN_INTERVAL_SECONDS + 1
    events = _drain(registry)
    assert source._watch_disabled
    assert source.notify_on_complete
    assert not source.exited
    assert sum(event["type"] == "watch_disabled" for event in events) == 1
    return registry, source, now


@pytest.mark.parametrize("reason", ["strikes", "lifetime"])
def test_progress_flood_remains_silent_after_disable(reason):
    registry, source, now = _disabled_source(reason)
    with patch("tools.process_registry.time.time", return_value=now):
        for _ in range(100):
            registry._check_watch_patterns(source, "TASK_UPDATE still running\n")
    assert registry.completion_queue.empty()
    assert not source.exited


@pytest.mark.parametrize("reason", ["strikes", "lifetime"])
def test_terminal_result_obligation_survives_readiness_disable(reason):
    from contextlib import closing
    from hermes_state import SessionDB
    from tools.process_work_results import enroll, records
    from tests.gateway.test_process_work_results import frame
    registry, source, _now = _disabled_source(reason)
    with closing(SessionDB()) as db:
        db.create_session(source.parent_session_id, source='discord')
        enroll(source, [{'work_id': 'build', 'incarnation': '1'}])
        registry._ingest_output(source, 'TASK_UPDATE terminal=completed ordinary-string\n')
        assert records(db)[0]['disposition'] == 'enrolled'
        registry._ingest_output(source, frame())
        assert records(db)[0]['disposition'] == 'pending'
        assert 'fixture-result' in records(db)[0]['payload']
    assert not source.exited
    assert registry.completion_queue.empty(), 'semantic results do not bypass readiness queue caps'


def test_global_watch_overflow_cannot_drop_explicit_result():
    from contextlib import closing
    from hermes_state import SessionDB
    from tools.process_registry import WATCH_GLOBAL_MAX_PER_WINDOW
    from tools.process_work_results import enroll, records
    from tests.gateway.test_process_work_results import frame
    registry = ProcessRegistry()
    with patch('tools.process_registry.time.time', return_value=1000):
        for i in range(WATCH_GLOBAL_MAX_PER_WINDOW + 1):
            source = ProcessSession(id=f'proc_{i}', command='fixture', watch_patterns=['READY'])
            registry._check_watch_patterns(source, 'READY\n')
        assert registry._global_watch_tripped_until > 1000
        _drain(registry)
        source = ProcessSession(id='proc_result', command='fixture', watch_patterns=['READY'],
            parent_session_id='parent', session_key='agent:main:discord:dm:fixture')
        with closing(SessionDB()) as db:
            db.create_session('parent', source='discord')
            enroll(source, [{'work_id': 'build', 'incarnation': '1'}])
            registry._ingest_output(source, 'READY\n' + frame())
            assert records(db)[0]['disposition'] == 'pending'
        assert registry.completion_queue.empty()
