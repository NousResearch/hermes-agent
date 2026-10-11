"""Conditional subscription is an authenticated identity cut, not recovery or input fencing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class Peer:
    def __init__(self, user="alice", provider="basic"):
        self.auth_identity = {"provider": provider, "user_id": user} if user else None
        self._closed = False
        self.frames = []
        self.delivered = Event()

    def write(self, frame):
        self.frames.append(frame)
        self.delivered.set()
        return not self._closed


def rpc(peer, method, **params):
    token = bind_transport(peer)
    try:
        return server.handle_request({"id": "test", "method": method, "params": params})
    finally:
        reset_transport(token)


def setup(monkeypatch, tmp_path):
    home = tmp_path / "launch"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.profiles import get_profile_dir
    get_profile_dir("secondary").mkdir(parents=True)
    for name, value in (("_sessions", {}), ("_idempotency_keys", {}),
                        ("_served_profile_homes", set()), ("_pending_ws_reaps", {})):
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    creator = Peer()
    result = rpc(creator, "session.create")["result"]
    binding = result["creation_binding"]
    record = server._sessions[binding["session_id"]]
    return creator, binding, record


def activate(peer, binding, **params):
    return rpc(peer, "session.activate_bound", session_id=binding["session_id"],
               expected_binding=binding, **params)


def test_authenticated_cut_preserves_execution_and_refusals(monkeypatch, tmp_path):
    creator, binding, record = setup(monkeypatch, tmp_path)
    peer = Peer()
    receipt = {"attached": True, "accepted_binding": binding}
    assert activate(peer, binding) == {"jsonrpc": "2.0", "id": "test", "result": receipt}
    assert activate(peer, binding)["result"] == receipt  # lost response/repeated attempt
    server._emit("message.complete", binding["session_id"], {"text": "existing turn"})
    assert creator.delivered.wait(5) and peer.delivered.wait(5)
    assert not record["running"] and record["inflight_turn"] is None
    transport, viewers = record["transport"], dict(record["viewers"])
    timer = SimpleNamespace(cancel=lambda: (_ for _ in ()).throw(AssertionError("refusal cancelled reap")))
    server._pending_ws_reaps[binding["session_id"]] = timer
    dead = Peer()
    dead._closed = True
    for refused in (Peer("bob"), Peer(provider="oidc"), Peer(None), dead):
        assert activate(refused, binding)["error"]["code"] == 4007
        assert not refused.frames and record["transport"] is transport and record["viewers"] == viewers
        assert server._pending_ws_reaps[binding["session_id"]] is timer
    assert activate(Peer(), binding, profile="secondary")["error"]["code"] == 4007
    for field in binding:
        for invalid in (None, 123, "", "x" * 257):
            malformed = binding | {field: invalid}
            assert activate(Peer(), malformed)["error"]["code"] == 4000
        missing = {k: v for k, v in binding.items() if k != field}
        assert rpc(Peer(), "session.activate_bound", session_id=binding["session_id"],
                   expected_binding=missing)["error"]["code"] == 4000
        assert activate(Peer(), binding | {field: "different"})["error"]["code"] == 4007
    for field, value in (("session_key", "descendant"), ("auth_user_id", "basic:bob"),
                         ("profile_home", str(tmp_path / "secondary")),
                         ("_client_gone_interrupt_requested", True), ("resume_hydrating", True),
                         ("running", True), ("agent", SimpleNamespace(session_id=binding["stored_session_id"])),
                         ("_compute_host_active", True),
                         ("inflight_turn", {"error": "interrupted", "recoverable": True})):
        old = record.get(field)
        record[field] = value
        assert activate(Peer(), binding)["error"]["code"] == 4007
        record[field] = old
    late_dead = Peer()
    original_attach = server._attach_session_transport
    def close_before_attach(session, peer):
        peer._closed = True
        return original_attach(session, peer)
    with monkeypatch.context() as patch:
        patch.setattr(server, "_attach_session_transport", close_before_attach)
        assert activate(late_dead, binding)["error"]["code"] == 4007
    other = rpc(Peer(), "session.create", profile="secondary")["result"]["creation_binding"]
    assert activate(Peer(), other, profile="secondary")["result"]["accepted_binding"] == other
    with server._sessions_lock:
        server._sessions[binding["session_id"]] = dict(record)  # even copied origin is not this runtime
    assert activate(Peer(), binding)["error"]["code"] == 4007
    assert record["transport"] is transport and record["viewers"] == viewers
    assert server._pending_ws_reaps[binding["session_id"]] is timer


def test_compression_and_replacement_share_the_subscription_boundary(monkeypatch, tmp_path):
    creator, binding, record = setup(monkeypatch, tmp_path)
    assert "result" in activate(Peer(), binding)
    sid = binding["session_id"]
    entered, release, writer_started = Event(), Event(), Event()
    original_attach = server._attach_session_transport
    def attach(session, peer):
        entered.set()
        assert release.wait(5)
        return original_attach(session, peer)
    monkeypatch.setattr(server, "_attach_session_transport", attach)
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *a, **k: True)
    def adopt():
        writer_started.set()
        with record["history_lock"]:
            server._compute_host_adopt_frame_meta(record, {"sid": sid, "session_key": "compressed-tip"})
    peer = Peer()
    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = pool.submit(activate, peer, binding)
        assert entered.wait(5)
        writer = pool.submit(adopt)
        assert writer_started.wait(5)
        assert record["session_key"] == binding["stored_session_id"]
        release.set()
        assert accepted.result(timeout=5)["result"]["accepted_binding"] == binding
        writer.result(timeout=5)
    assert record["session_key"] == "compressed-tip"
    assert activate(Peer(), binding)["error"]["code"] == 4007
    # Compression wins first: park inside its real key writer while it owns history_lock.
    record["session_key"] = binding["stored_session_id"]
    entered.clear()
    release.clear()
    record["agent"] = SimpleNamespace(session_id="compressed-tip")
    def compress():
        server._sync_session_key_after_compress(sid, record, restart_slash_worker=False)
    def transfer(*a, **k):
        entered.set()
        assert release.wait(5)
        return True
    monkeypatch.setattr(server, "_transfer_active_session_slot", transfer)
    refused = Peer()
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(compress)
        assert entered.wait(5)
        pending = pool.submit(activate, refused, binding)
        release.set()
        writer.result(timeout=5)
        assert pending.result(timeout=5)["error"]["code"] == 4007
    assert not server._session_transport_contains(record, refused)
    # Registry replacement wins while activation waits for history: no stale-object attach.
    record["session_key"] = binding["stored_session_id"]
    record["agent"] = None
    raw_lock = record["history_lock"]
    candidate_seen = Event()
    class ObservedLock:
        def __enter__(self):
            candidate_seen.set()
            raw_lock.acquire()
        def __exit__(self, *args):
            raw_lock.release()
    record["history_lock"] = ObservedLock()
    with ThreadPoolExecutor(max_workers=1) as pool:
        raw_lock.acquire()
        try:
            pending = pool.submit(activate, Peer(), binding)
            assert candidate_seen.wait(5)
            with server._sessions_lock:
                server._sessions[sid] = dict(record)
        finally:
            raw_lock.release()
        assert pending.result(timeout=5)["error"]["code"] == 4007
