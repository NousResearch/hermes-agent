"""Creation provenance is stable for one live object, never inferred from a runtime ID.

Drive authenticated JSON-RPC with real profile resolution. Agent/model execution is
suppressed; origin stamping, contract validation and session creation remain real.
"""
from __future__ import annotations

import copy
from pathlib import Path

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class LoginPeer:
    def __init__(self, user="alice", provider="basic"):
        self.auth_identity = {"provider": provider, "user_id": user} if user else None

    def write(self, frame):
        return True


def prepare(monkeypatch, tmp_path):
    home = tmp_path / "launch"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.profiles import get_profile_dir
    secondary = get_profile_dir("secondary")
    secondary.mkdir(parents=True)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_idempotency_keys", {})
    return home, secondary


def create(peer, *, method="session.create", **params):
    token = bind_transport(peer)
    try:
        response = server.handle_request({"id": "create", "method": method, "params": params})
        assert "error" not in response, response
        return response["result"]
    finally:
        reset_transport(token)


def test_origin_survives_retry_but_not_runtime_replacement(monkeypatch, tmp_path):
    home, secondary = prepare(monkeypatch, tmp_path)
    peer = LoginPeer()
    first = create(peer, idempotency_key="first")
    binding = copy.deepcopy(first["creation_binding"])
    assert binding["session_id"] == first["session_id"]
    assert binding["stored_session_id"] == first["stored_session_id"]
    assert binding["authenticated_owner"] == "basic:alice"
    assert create(LoginPeer(), idempotency_key="first")["creation_binding"] == binding
    # The launch -> named profile -> launch sequence uses the real resolver.
    other = create(peer, profile="secondary")
    last = create(peer)
    assert server._sessions[other["session_id"]]["profile_home"] == str(secondary)
    assert server._sessions[last["session_id"]]["profile_home"] is None
    for response in (first, other, last):
        wire = response["creation_binding"]
        assert str(home) not in str(wire) and str(secondary) not in str(wire)
        assert wire["runtime_incarnation"] != wire["profile_store_scope"]
    assert len({r["creation_binding"]["runtime_incarnation"] for r in (first, other, last)}) == 3
    # Mutating a returned dict cannot rewrite the origin retained by the server.
    first["creation_binding"]["runtime_incarnation"] = "client-forged"
    assert create(peer, idempotency_key="first")["creation_binding"] == binding
    # Deliberately reuse the public runtime ID; replacement must mint a new origin.
    monkeypatch.setattr(server, "_new_runtime_ids", lambda params: (binding["session_id"], "tui"))
    replacement = create(peer)
    assert replacement["session_id"] == binding["session_id"]
    assert replacement["creation_binding"]["runtime_incarnation"] != binding["runtime_incarnation"]
    # Whole stored branches use the same creation path but omit parent transcript on the wire.
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        db.create_session("saved-parent", source="tui")
        db.append_message("saved-parent", "user", "question")
        db.append_message("saved-parent", "assistant", "answer")
        branch = create(peer, method="session.branch_stored", parent_session_id="saved-parent",
                        idempotency_key="branch")
        assert branch["messages_omitted"] is True and "messages" not in branch
        assert branch["creation_binding"]["stored_session_id"] == branch["stored_session_id"]
        assert create(peer, method="session.branch_stored", parent_session_id="saved-parent",
                      idempotency_key="branch")["creation_binding"] == branch["creation_binding"]
    finally:
        db.close()


def test_binding_is_creator_only_and_never_restamps_mutable_state(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    peer = LoginPeer()
    result = create(peer, idempotency_key="same")
    original = result["creation_binding"]
    record = server._sessions[result["session_id"]]
    # Provider namespaces matter; a fresh peer for the same principal is allowed.
    for foreign in (LoginPeer("bob"), LoginPeer(provider="oidc"), LoginPeer(None)):
        assert "creation_binding" not in create(foreign, idempotency_key="same")
    assert create(LoginPeer(), idempotency_key="same")["creation_binding"] == original
    assert "creation_binding" not in create(LoginPeer(None))
    assert "creation_binding" not in create(LoginPeer("x" * 257))
    assert "creation_binding" not in create(peer, idempotency_key="same", profile="secondary")
    # Origin never becomes proof of current segment/profile/owner after rotation or adoption.
    for field, changed in (("session_key", "compressed-descendant"),
                           ("profile_home", str(tmp_path / "another-store")),
                           ("auth_user_id", "basic:bob")):
        old = record[field]
        record[field] = changed
        retry = create(peer, idempotency_key="same")
        assert retry["creation_binding"] == original
        if field == "session_key":
            assert retry["stored_session_id"] != original["stored_session_id"]
        record[field] = old
    assert create(peer, idempotency_key="same")["creation_binding"] == original


def test_creation_retry_preserves_profile_default_route(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_session_default_route", lambda _session: ("profile-model", "profile-provider"))
    peer = LoginPeer()
    first = create(peer, idempotency_key="profile-route")
    retry = create(LoginPeer(), idempotency_key="profile-route")
    for response in (first, retry):
        assert response["info"]["model"] == "profile-model"
        assert response["info"]["provider"] == "profile-provider"
    assert retry["creation_binding"] == first["creation_binding"]
