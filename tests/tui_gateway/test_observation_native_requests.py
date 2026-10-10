"""Current TUI/Desktop requests publish through the real per-session server binding."""
import json
import os
from types import SimpleNamespace

import pytest
from hermes_state import SessionDB


@pytest.mark.parametrize("source", ["desktop", "tui"])
@pytest.mark.parametrize("method,params,kind", [
    ("sudo", {"command": "synthetic-private-sentinel"}, "approval"),
    ("setup_choose", {"kind": "question", "question": "synthetic-private-sentinel"}, "question"),
    ("window.read", {}, "none"),
])
def test_real_server_binding_uses_owner_db_not_launch_profile(tmp_path, monkeypatch, source, method, params, kind):
    from hermes_cli import banner
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    from tui_gateway import server, server_requests as requests
    requests.reset_for_tests()
    with SessionDB(tmp_path / "launch" / "state.db") as launch, SessionDB(tmp_path / "owner" / "state.db") as db:
        for store, profile in [(launch, "default"), (db, "secondary")]:
            store.create_session("same", source, profile_name=profile)
        holder = f"pid={os.getpid()}:turn=owner"
        assert db.try_acquire_session_turn_lease("same", holder)
        turn = db.begin_session_observation("same", holder)
        agent = SimpleNamespace(session_id="same", _session_db=db,
                                _active_session_turn_lease_holder=holder, _observation_turn_id=turn)
        monkeypatch.setitem(server._sessions, "native-ui", {"agent": agent, "session_key": "same", "source": source})
        frames = []
        def write(frame):
            frames.append(frame)
            row = db.read_session_observations(["same"], profile="secondary")[0]
            assert row["attention"]["kind"] == kind, "current native human request was reported absent"
            assert "synthetic-private-sentinel" not in json.dumps(row)
            assert holder not in json.dumps(row)
            assert launch.read_session_observations(["same"], profile="default")[0]["execution"] == "unknown"
        monkeypatch.setattr(server, "write_json", write)
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        try:
            results = []
            settle = requests.send_async(method, "native-ui", params, results.append)
            answer = {"picked": "synthetic"} if method == "setup_choose" else {"value": "synthetic"}
            assert requests.resolve_response({"id": frames[-1]["id"], "result": answer})
            assert results == [answer]
            assert db.read_session_observations(["same"], profile="secondary")[0]["attention"]["kind"] == "none"
            settle("timeout")  # duplicate settlement must not change the generation
            assert db.finish_session_observation("same", holder, turn, "complete")
        finally:
            requests.cancel("native-ui")
            requests.reset_for_tests()


def test_real_clarify_partial_lock_frame_failure_and_late_response(tmp_path, monkeypatch):
    from hermes_cli import banner
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    from tui_gateway import server, server_requests as requests
    requests.reset_for_tests()
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "desktop", profile_name="default")
        holder = f"pid={os.getpid()}:turn=clarify"
        assert db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        agent = SimpleNamespace(session_id="s", _session_db=db,
                                _active_session_turn_lease_holder=holder, _observation_turn_id=turn)
        monkeypatch.setitem(server._sessions, "native-ui", {"agent": agent, "session_key": "s", "source": "desktop"})
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        def attention():
            return db.read_session_observations(["s"], profile="default")[0]["attention"]
        def lock_questions(frame):
            rid = frame["id"]
            assert attention()["request_id"] == rid
            assert requests.lock_answer(rid, "a", "synthetic") == ["b"]
            assert attention()["kind"] == "question"
            assert requests.lock_answer(rid, "b", None) == []
            assert attention()["kind"] == "none"
        monkeypatch.setattr(server, "write_json", lock_questions)
        params = {"questions": [{"qid": qid, "question": "synthetic", "choices": [], "multi_select": False} for qid in ["a", "b"]]}
        assert requests.send("clarify", "native-ui", params, timeout=0, qids=["a", "b"])["outcome"] == "submitted"
        frames = []
        def fail_write(frame):
            frames.append(frame)
            assert attention()["kind"] == "approval"
            raise OSError("synthetic transport failure")
        monkeypatch.setattr(server, "write_json", fail_write)
        with pytest.raises(OSError, match="synthetic"):
            server._ask("sudo", "native-ui", {}, timeout=0)
        assert attention()["kind"] == "none" and requests.open_requests("native-ui") == []
        monkeypatch.setattr(server, "write_json", frames.append)
        requests.send_async("sudo", "native-ui", {}, lambda result: None)
        old_request = frames[-1]["id"]
        db.release_session_turn_lease("s", holder)
        successor = holder + "-new"
        assert db.try_acquire_session_turn_lease("s", successor)
        current = db.begin_session_observation("s", successor)
        agent._active_session_turn_lease_holder, agent._observation_turn_id = successor, current
        requests.send_async("sudo", "native-ui", {}, lambda result: None)
        current_attention = attention()
        assert current_attention["turn_id"] == current
        assert requests.resolve_response({"id": old_request, "result": {"value": "late"}})
        assert attention() == current_attention, "late answer resolved the successor's request"
        requests.cancel("native-ui")
        assert attention()["kind"] == "none"
        requests.reset_for_tests()
