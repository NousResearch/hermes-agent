"""Profile-scoped review RPC exercises real stores, not slash output parsing."""
from pathlib import Path
from tools import write_approval as wa
from tools.memory_tool import MemoryStore
import pytest


@pytest.mark.parametrize("session", [None, {}])
def test_explicit_unknown_or_unowned_session_never_reads_launch_home(tmp_path, monkeypatch, session):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from tui_gateway import server
    record = wa.stage_write(wa.MEMORY, {"action": "add", "content": "Generic entry."},
                            summary="Generic", origin="foreground")
    monkeypatch.setattr(server, "_sessions", {} if session is None else {"unknown": session})
    response = server._methods["memory.pending"](1, {"session_id": "unknown"})
    assert "error" in response
    response = server._methods["memory.decide"](2, {"session_id": "unknown", "id": record["id"],
                                                    "decision": "reject", "revision": ""})
    assert "error" in response
    assert wa.get_pending(wa.MEMORY, record["id"]) is not None


def test_live_launch_session_can_review_and_decide(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from tui_gateway import server
    record = wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "Generic launch entry."},
                            summary="Generic launch", origin="foreground")
    monkeypatch.setattr(server, "_sessions", {
        "live-launch": {"profile_home": None, "session_key": "synthetic-key", "source": "desktop"}})
    response = server._methods["memory.pending"](1, {"session_id": "live-launch"})
    assert "result" in response
    batch = response["result"]["batches"][0]
    assert batch["id"] == record["id"]
    response = server._methods["memory.decide"](2, {"session_id": "live-launch", "id": batch["id"],
                                                  "decision": "approve", "revision": batch["revision"]})
    assert response["result"]["success"]
    assert (tmp_path / "memories/MEMORY.md").read_text() == "Generic launch entry."
    assert wa.get_pending(wa.MEMORY, record["id"]) is None


def test_rpc_review_follows_session_homes_a_b_a(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    from tui_gateway import server
    homes = [tmp_path / "a", tmp_path / "b"]
    ids = []
    for home, label in zip(homes, ["Generic A", "Generic B"]):
        monkeypatch.setenv("HERMES_HOME", str(home))
        ids.append(wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": label},
                                 summary=label, origin="foreground")["id"])
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setattr(server, "_sessions", {
        "review-a": {"profile_home": str(homes[0])}, "review-b": {"profile_home": str(homes[1])}})
    for sid, expected in [("review-a", ids[0]), ("review-b", ids[1]), ("review-a", ids[0])]:
        response = server._methods["memory.pending"](1, {"session_id": sid})
        assert response["result"]["batches"][0]["id"] == expected
    batch = server._methods["memory.pending"](2, {"session_id": "review-b"})["result"]["batches"][0]
    result = server._methods["memory.decide"](3, {"session_id": "review-b", "id": batch["id"],
                                                "decision": "approve", "revision": batch["revision"]})
    assert result["result"]["success"]
    assert (homes[1] / "memories/MEMORY.md").read_text() == "Generic B"
    assert not (homes[0] / "memories/MEMORY.md").exists()
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    assert wa.get_pending(wa.MEMORY, ids[0]) is not None
