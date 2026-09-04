import json
import sys

import pytest

from tui_gateway.observer import ObserverPlane, ObserverJournal, ObserverRecord, logical_session_id


def _plane(transcript=None):
    return ObserverPlane(
        lambda: [{"id": "runtime-1", "title": "Safe title", "running": True}],
        lambda _sid: transcript or [{"durable_item_id": "durable-1", "role": "user", "text": "hello"}],
    )


def test_projection_allowlist_drops_nested_sensitive_source_fields():
    record = _plane().capture("tool.phase", "runtime-1", {"activity_id": "a1", "name": "Read file", "phase": "complete", "duration_ms": 10, "args": {"token": "MUST-NOT-LEAK"}, "headers": {"authorization": "MUST-NOT-LEAK"}, "url": "https://MUST-NOT-LEAK"})
    assert record is not None
    assert "MUST-NOT-LEAK" not in json.dumps(record.as_dict())
    assert record.payload == {"activity_id": "a1", "name": "Read file", "phase": "complete", "duration_ms": 10}


def test_snapshot_watermark_and_subsequent_event_are_distinct():
    plane, lid = _plane(), logical_session_id("runtime-1")
    plane.capture("message.start", "runtime-1", {"text": "partial"})
    snapshot = plane.snapshot(lid)
    later = plane.capture("message.complete", "runtime-1", {"text": "complete"})
    assert snapshot["stream"]["through_seq"] == 1
    assert later.seq == 2


def test_subscribe_registers_atomically_against_capture():
    plane, lid = _plane(), logical_session_id("runtime-1")
    plane.capture("status.current", "runtime-1", {"text": "before", "kind": "working"})
    q, replay, gap = plane.subscribe(lid, 0)
    assert gap is None and [r.seq for r in replay] == [1]
    plane.capture("status.current", "runtime-1", {"text": "after", "kind": "working"})
    assert q.get_nowait().seq == 2


def test_unknown_and_hidden_events_advance_cursor_without_payload():
    plane = _plane()
    hidden = plane.capture("secret.request", "runtime-1", {"secret": "MUST-NOT-LEAK", "request_id": "r1"})
    unknown = plane.capture("future.event", "runtime-1", {"body": "MUST-NOT-LEAK"})
    assert hidden.kind == "cursor.advance" and unknown.kind == "unknown_activity"
    assert hidden.payload == unknown.payload == {}
    assert hidden.seq == 1 and unknown.seq == 2


def test_journal_floor_reports_retention_gap():
    journal = ObserverJournal(max_events=2, max_age_seconds=900)
    for _ in range(3):
        journal.append(lambda stream, seq: ObserverRecord("ls", stream, seq, f"{stream}:{seq}", "now", "control", "cursor.advance", "hidden", {}))
    replay, gap = journal.replay(0)
    assert replay == [] and gap == "retention_exceeded"


def test_stream_epoch_mismatch_is_explicit_gap():
    plane, lid = _plane(), logical_session_id("runtime-1")
    plane.capture("status.current", "runtime-1", {"text": "working", "kind": "working"})
    _q, replay, gap = plane.subscribe(lid, 0, stream_id="st_old")
    assert replay == [] and gap == "stream_restarted"
    assert plane.rotate_stream(lid).stream_id != "st_old"


def test_lineage_change_rotates_epoch_and_notifies_existing_observer():
    plane, lid = _plane(), logical_session_id("runtime-1")
    q, _, _ = plane.subscribe(lid)
    first = plane.capture("status.current", "runtime-1", {"text": "working", "kind": "working"})
    changed = plane.capture("lineage.changed", "runtime-1", {"lineage_generation": 2, "reason": "compression"})
    assert first.stream_id != changed.stream_id
    assert q.get_nowait().kind == "status.current"
    assert q.get_nowait().kind == "gap"


def test_observer_router_uses_distinct_bearer(monkeypatch):
    monkeypatch.setenv("HERMES_OBSERVER_SOURCE_TOKEN", "observer-secret")
    from tui_gateway.observer_router import create_observer_router, observer_token_is_valid
    router = create_observer_router(_plane())
    assert observer_token_is_valid("observer-secret")
    assert not observer_token_is_valid("controller-secret")
    assert {route.path for route in router.routes} >= {"/observer/v1/me", "/observer/v1/sessions"}
    me = next(route.endpoint for route in router.routes if route.path == "/observer/v1/me")
    with pytest.raises(Exception):
        me(authorization="Bearer controller-secret")


def test_observer_router_import_does_not_initialize_controller_dispatcher():
    # The source router must be importable in an observer-only process without
    # loading the controller method registry as an import-time side effect.
    sys.modules.pop("tui_gateway.server", None)
    import tui_gateway.observer_router  # noqa: F401

    assert "tui_gateway.server" not in sys.modules
