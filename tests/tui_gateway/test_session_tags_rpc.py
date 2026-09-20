"""Tag RPCs exercise the real dispatcher and independent on-disk profile stores."""
from pathlib import Path

import pytest

from hermes_state import SessionDB
import hermes_state_registry as registry
from tui_gateway import server


def call(method, **params):
    return server.handle_request({"id": "tags", "method": method, "params": params})


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    other = launch / "profiles" / "work"
    other.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", str(launch))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_served_profile_homes", set())
    for home in (launch, other):
        db = SessionDB(home / "state.db")
        db.create_session("shared-id", source="desktop")
        db.set_session_title("shared-id", "Example")
        db.append_message("shared-id", "user", "Hello")
        db.close()
    try:
        yield launch, other
    finally:
        registry.close_all()


def test_profile_a_b_a_catalogue_and_rows_survive_reconnect(homes, monkeypatch):
    launch, other = homes
    for profile, tag in [(None, "A"), ("work", "B"), (None, "A2")]:
        result = call("session.tags.set", session_id="shared-id", tag=tag, assigned=True, profile=profile)
        assert "result" in result, result
        assert result["result"]["tags"] == (["B"] if profile else (["A"] if tag == "A" else ["A", "A2"]))
    registry.close_all()
    monkeypatch.setattr(server, "_db", None)
    for profile, tags in [(None, ["A", "A2"]), ("work", ["B"]), (None, ["A", "A2"])]:
        assert call("session.tags.list", profile=profile)["result"] == {"tags": tags}
        rows = call("session.list", profile=profile)["result"]["sessions"]
        assert rows[0]["tags"] == tags
        titled = call("session.list", profile=profile, title="Example")["result"]["sessions"]
        assert titled[0]["tags"] == tags
    # A live runtime id selects its owning profile, even without explicit profile.
    monkeypatch.setitem(server._sessions, "runtime", {"session_key": "shared-id", "profile_home": str(other), "agent": None})
    assert call("session.tags.set", session_id="runtime", tag="Live", assigned=True)["result"] == {"tags": ["B", "Live"]}
    assert call("session.tags.list")["result"] == {"tags": ["A", "A2"]}
    assert call("session.tags.set", session_id="runtime", profile="default", tag="Wrong", assigned=True)["error"]["code"] == 4001
    assert call("session.tags.set", session_id="shared-id", tag="A", assigned=False)["result"] == {"tags": ["A2"]}
    assert call("session.tags.list")["result"] == {"tags": ["A", "A2"]}

    db = server._get_db()
    db.end_session("shared-id", "compression")
    db.create_session("middle", source="desktop", parent_session_id="shared-id")
    db.end_session("middle", "compression")
    db.create_session("tip", source="desktop", parent_session_id="middle")
    db.append_message("tip", "user", "Continued")
    assert call("session.tags.set", session_id="tip", tag="Compressed", assigned=True)["result"] == {"tags": ["A2", "Compressed"]}
    rows = call("session.list")["result"]["sessions"]
    assert rows[0]["id"] == "tip"
    assert rows[0]["tags"] == ["A2", "Compressed"]
    assert call("session.tags.set", session_id="middle", tag="Compressed", assigned=False)["result"] == {"tags": ["A2"]}


def test_project_preview_and_hydrated_rows_preserve_tags(homes, monkeypatch):
    call("session.tags.set", session_id="shared-id", tag="Project", assigned=True)
    def no_single_lookup(*args):
        pytest.fail("project rows must use batch-hydrated tags")
    monkeypatch.setattr(SessionDB, "get_session_tags", no_single_lookup)
    overview = call("projects.tree")["result"]
    project = next(p for p in overview["projects"] if p["sessionCount"])
    assert project["previewSessions"][0]["tags"] == ["Project"]
    hydrated = call("projects.project_sessions", project_id=project["id"])["result"]["project"]
    rows = [s for repo in hydrated["repos"] for lane in repo["groups"] for s in lane["sessions"]]
    assert rows[0]["tags"] == ["Project"]


def test_rpc_rejects_invalid_names_types_missing_sessions_and_profiles(homes):
    for tag in [" ", "x" * 65, "a\nb", "\ttag", "a\x7fb", None, 7]:
        response = call("session.tags.set", session_id="shared-id", tag=tag, assigned=True)
        assert response["error"]["code"] == 4000, response
    for assigned in ["false", 1, None]:
        assert call("session.tags.set", session_id="shared-id", tag="Valid", assigned=assigned)["error"]["code"] == 4000
    assert call("session.tags.set", session_id="missing", tag="Ghost", assigned=True)["error"]["code"] == 4001
    assert call("session.tags.list", profile="absent")["error"]["code"] == 4064
    assert call("session.tags.list")["result"] == {"tags": []}
    assert call("session.tags.set", session_id="shared-id", tag="  Valid  ", assigned=True)["result"] == {"tags": ["Valid"]}
