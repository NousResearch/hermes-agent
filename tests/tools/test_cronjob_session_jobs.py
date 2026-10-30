"""`cronjob_manage` turn/step branch: grammar, storage, and the truthful response contract."""

import json

import pytest

from hermes_state_registry import acquire, release_or_close
from tools.cronjob_session_jobs import parse_logical_schedule


@pytest.fixture
def state_db():
    db = acquire()
    db.create_session("s", "cli", model="m")
    try:
        yield db
    finally:
        release_or_close(db)


def _tool(**kwargs):
    from tools.cronjob_tools import cronjob

    return json.loads(cronjob(**kwargs))


def test_logical_grammar_parses_before_durations():
    assert parse_logical_schedule("every 3 turns") == {"kind": "turn", "every": 3, "once": False}
    assert parse_logical_schedule("every 2 steps") == {"kind": "step", "every": 2, "once": False}
    assert parse_logical_schedule("every turn") == {"kind": "turn", "every": 1, "once": False}
    assert parse_logical_schedule("each step") == {"kind": "step", "every": 1, "once": False}
    assert parse_logical_schedule("in 5 steps") == {"kind": "step", "every": 5, "once": True}
    assert parse_logical_schedule("in 2 turns") == {"kind": "turn", "every": 2, "once": True}
    for other in ("every 30m", "0 9 * * *", "every monday 9am", "2026-06-01T09:00:00", ""):
        assert parse_logical_schedule(other) is None


def test_create_binds_the_session_and_never_touches_jobs_json(state_db):
    from cron.jobs import list_jobs

    payload = _tool(action="create", schedule="every 2 steps", prompt="audit caches",
                    name="cache audit", session_id="s")
    assert payload["success"] is True, payload
    assert payload["next_due_step"] == 2
    assert "next_run_at" not in payload
    assert payload["activation"] == "this session (in-process; no gateway)"
    assert payload["lifetime"] == "ends with this session; pauses when it ends"
    job = state_db.get_session_job(payload["job_id"])
    assert (job["origin_session_id"], job["kind"], job["every"], job["next_due_seq"]) == ("s", "step", 2, 2)
    assert list_jobs() == []  # never a cron job


def test_once_schedule_sets_remaining_and_turn_due(state_db):
    payload = _tool(action="create", schedule="in 3 turns", prompt="wrap up", session_id="s")
    assert payload["next_due_turn"] == 3
    job = state_db.get_session_job(payload["job_id"])
    assert (job["kind"], job["every"], job["remaining"], job["next_due_seq"]) == ("turn", 3, 1, 3)


def test_create_requires_an_active_session(state_db):
    payload = _tool(action="create", schedule="every 2 steps", prompt="x")
    assert payload["success"] is False
    assert "session-bound" in payload["error"]


def test_list_surfaces_session_jobs_without_a_gateway_warning(state_db):
    created = _tool(action="create", schedule="each turn", prompt="x", session_id="s")
    payload = _tool(action="list")
    assert payload["success"] is True
    assert [job["job_id"] for job in payload["session_jobs"]] == [created["job_id"]]
    assert "lifetime" in payload["session_jobs"][0]
    assert "gateway_running" not in payload


def test_update_rebases_the_logical_due_point(state_db):
    created = _tool(action="create", schedule="every 2 steps", prompt="x", session_id="s")
    state_db._execute_write(lambda conn: conn.execute("UPDATE sessions SET step_seq = 10 WHERE id = 's'"))
    payload = _tool(action="update", job_id=created["job_id"], schedule="every 5 steps")
    assert payload["success"] is True
    job = state_db.get_session_job(created["job_id"])
    assert (job["every"], job["next_due_seq"]) == (5, 15)


def test_pause_resume_remove_route_on_the_session_branch(state_db):
    created = _tool(action="create", schedule="every 2 steps", prompt="x", session_id="s")
    job_id = created["job_id"]
    assert _tool(action="pause", job_id=job_id, reason="hold")["success"] is True
    assert state_db.get_session_job(job_id)["enabled"] == 0
    assert _tool(action="resume", job_id=job_id)["success"] is True
    assert state_db.get_session_job(job_id)["enabled"] == 1
    assert _tool(action="remove", job_id=job_id)["success"] is True
    assert state_db.get_session_job(job_id) is None


def test_run_fires_without_moving_the_next_due_point(state_db, monkeypatch):
    created = _tool(action="create", schedule="every 2 steps", prompt="x", session_id="s")
    seen = []
    import scheduler.session_jobs as sj

    monkeypatch.setattr(sj, "run_job", lambda job, seq: seen.append((job["id"], seq)))
    payload = _tool(action="run", job_id=created["job_id"])
    assert payload["success"] is True
    assert seen == [(created["job_id"], 2)]
    assert state_db.get_session_job(created["job_id"])["next_due_seq"] == 2


def test_chain_depth_cap_rejects_the_fourth_job(state_db):
    parent = state_db.create_session_job(
        origin_session_id="s", kind="step", every=1, prompt="p", next_due_seq=1, origin_depth=2)
    run_session = f"cron_{parent}_20260101_000000"
    with pytest.raises(ValueError, match="chain depth"):
        state_db.create_session_job(
            origin_session_id=run_session, kind="step", every=1, prompt="p",
            next_due_seq=1, origin_depth=3)
