"""Human request settlement, unlike desktop read/act, is observable without content."""
import os
from types import SimpleNamespace
import pytest
from hermes_state import SessionDB
from tui_gateway import server_requests as requests


@pytest.fixture
def owner(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "desktop", profile_name="default")
        holder = f"pid={os.getpid()}:turn=synthetic"
        db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        yield db, "s", turn, holder
        requests.cancel()


def test_human_wait_closes_at_response_timeout_cancel_and_late_answer(owner):
    db, sid, turn, holder = owner
    assert "observation_owner" in __import__("inspect").signature(requests.bind_sinks).parameters, "request producer not wired"
    frames = []
    def write(frame):
        frames.append(frame)
        row = db.read_session_observations([sid], profile="default")[0]
        assert row["attention"]["kind"] == "approval"
    requests.bind_sinks(write, lambda *a: None, lambda sid: True, observation_owner=lambda sid: owner)
    settle = requests.send_async("approval", "ui", {"command": "synthetic", "description": "synthetic"}, lambda result: None)
    request_id = frames[-1]["id"]
    assert requests.resolve_response({"id": request_id, "result": {"choice": "once"}})
    assert db.read_session_observations([sid], profile="default")[0]["attention"]["kind"] == "none"
    settle("timeout")
    requests.send("approval", "ui", {"command": "synthetic", "description": "synthetic"}, timeout=0)
    assert db.read_session_observations([sid], profile="default")[0]["attention"]["kind"] == "none"
    requests.send_async("approval", "ui", {"command": "synthetic", "description": "synthetic"}, lambda result: None)
    assert requests.cancel("ui") == 1
    assert not requests.resolve_response({"id": frames[-1]["id"], "result": {"choice": "once"}})
    assert db.read_session_observations([sid], profile="default")[0]["attention"]["kind"] == "none"


def test_long_question_is_not_technical_bridge_or_stale_generation(owner):
    db, sid, turn, holder = owner
    assert "observation_owner" in __import__("inspect").signature(requests.bind_sinks).parameters, "request producer not wired"
    frames = []
    requests.bind_sinks(frames.append, lambda *a: None, lambda sid: True, observation_owner=lambda sid: owner)
    settle = requests.send_async("clarify", "ui", {"questions": [{"qid": "q", "question": "synthetic", "choices": [], "multi_select": False}]}, lambda result: None)
    # Attention has no age-based deadline; only the actual request owns settlement.
    row = db.read_session_observations([sid], profile="default")[0]
    assert row["attention"]["kind"] == "question"
    db._write_sql("UPDATE session_observations SET updated_at = 1 WHERE conversation_id = ?", (sid,))
    assert db.read_session_observations([sid], profile="default")[0]["attention"]["kind"] == "question"
    requests.send_async("window.read", "ui", {}, lambda result: None)
    assert db.read_session_observations([sid], profile="default")[0]["attention"] == row["attention"]
    settle("cancelled")
    assert db.read_session_observations([sid], profile="default")[0]["attention"]["kind"] == "none"
