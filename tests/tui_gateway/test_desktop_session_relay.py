"""Offline, exact-session peer delivery across a named profile boundary."""
import threading
from types import SimpleNamespace

import pytest

from hermes_cli.active_sessions import transfer_active_session, try_acquire_active_session
from hermes_state import SessionDB
from tools.bot_live_delivery import read_delivery_result
from tui_gateway import session_notifications
from tui_gateway.method_ctx import rebind
from tui_gateway.session_lifecycle import _session_turn_admission


@pytest.fixture
def lisa_owner(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    lisa = root / "profiles" / "lisa"
    lisa.mkdir(parents=True)
    (lisa / "config.yaml").write_text("{}", encoding="utf-8")
    zara = root / "profiles" / "zara"
    zara.mkdir()
    (zara / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(zara))
    lease, refusal = try_acquire_active_session(
        session_id="desktop-exact", surface="desktop", config={}, registry_home=lisa,
        metadata={"live_session_id": "live-lisa", "bot_live_delivery_consumer": True},
        track_liveness=True)
    assert refusal is None
    yield lisa, lease
    lease.release()


def _transfer_lisa_owner_across_compression(lisa, lease, *, tip="desktop-tip"):
    db = SessionDB(db_path=lisa / "state.db")
    try:
        db.create_session(session_id="desktop-exact", source="desktop")
        db.end_session("desktop-exact", "compression")
        db.create_session(session_id=tip, source="desktop", parent_session_id="desktop-exact")
    finally:
        db.close()
    assert transfer_active_session(
        lease, session_id=tip,
        metadata={"live_session_id": "live-lisa", "bot_live_delivery_consumer": True})


def test_named_profile_exact_session_queues_then_drains_with_receipt(lisa_owner):
    from tools.session_relay import deliver_to_desktop_session

    lisa, lease = lisa_owner
    admitted = deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "Please inspect this")
    assert admitted["status"] == "queued"
    assert admitted["session_id"] == "desktop-exact"
    assert "Please inspect this" in admitted["message"]
    pending = read_delivery_result(lisa, admitted["delivery_id"])
    assert pending["status"] == "queued"
    seen = []

    def submit(rid, sid, session, text, **kwargs):
        seen.append(("submit", sid, text, kwargs["turn_author"]))
        kwargs["terminal_callback"]({"status": "settled", "text": "Received"})
        session["running"] = False
        return True

    poll = rebind(session_notifications._poll_bot_live_delivery_once, {
        "_session_home": lambda session: lisa,
        "_session_turn_admission": _session_turn_admission,
        "_run_prompt_submit": submit,
        "_emit": lambda event, sid, payload=None: seen.append((event, sid, payload)),
    })
    # No ``source`` key: the lease surface is what identifies a desktop owner.
    session = {"profile_home": str(lisa), "history_lock": threading.RLock(),
               "agent": object(), "session_key": "desktop-exact", "running": True,
               "active_session_lease": lease}
    assert poll("live-lisa", session) is False
    assert read_delivery_result(lisa, admitted["delivery_id"])["status"] == "queued"
    session["running"] = False
    assert poll("live-lisa", session) is True
    assert seen == [
        ("message.user", "live-lisa", {"text": admitted["message"], "author": admitted["author"],
                                         "delivery_id": admitted["delivery_id"]}),
        ("submit", "live-lisa", admitted["message"], admitted["author"]),
    ]
    assert read_delivery_result(lisa, admitted["delivery_id"])["reply"] == "Received"
    assert read_delivery_result(lisa, admitted["delivery_id"])["status"] == "settled"


def test_wrong_session_and_stale_lease_are_refused_without_mailbox(lisa_owner):
    from tools.session_relay import deliver_to_desktop_session
    lisa, lease = lisa_owner
    with pytest.raises(ValueError, match="owner"):
        deliver_to_desktop_session("lisa", "other-session", "ZARA", "no")
    lease.release()
    with pytest.raises(ValueError, match="owner"):
        deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "no")
    assert not (lisa / "runtime" / "bot_live_delivery").exists()


def test_unknown_profile_is_refused_without_default_fallback(lisa_owner):
    from tools.session_relay import deliver_to_desktop_session
    with pytest.raises(ValueError, match="profile"):
        deliver_to_desktop_session("missing", "desktop-exact", "ZARA", "no")


@pytest.mark.parametrize("target", ["default", " DEFAULT "])
def test_canonical_default_target_is_refused(lisa_owner, target):
    from tools.session_relay import deliver_to_desktop_session
    with pytest.raises(ValueError, match="named target"):
        deliver_to_desktop_session(target, "desktop-exact", "ZARA", "no")


def test_sender_cannot_impersonate_another_profile_or_human(lisa_owner):
    from tools.session_relay import deliver_to_desktop_session
    lisa, _ = lisa_owner
    for claimed_sender in ("lisa", "default", "Van"):
        with pytest.raises(ValueError, match="sender"):
            deliver_to_desktop_session("lisa", "desktop-exact", claimed_sender, "forged")
    assert not (lisa / "runtime" / "bot_live_delivery").exists()


@pytest.mark.parametrize("sender", ["default", " DEFAULT "])
def test_canonical_default_sender_is_refused(lisa_owner, sender):
    from tools.session_relay import deliver_to_desktop_session
    lisa, _ = lisa_owner
    with pytest.raises(ValueError, match="sender"):
        deliver_to_desktop_session("lisa", "desktop-exact", sender, "forged")
    assert not (lisa / "runtime" / "bot_live_delivery").exists()


def test_sender_is_canonicalized_before_attribution(lisa_owner):
    from tools.session_relay import deliver_to_desktop_session
    admitted = deliver_to_desktop_session(" LISA ", "desktop-exact", " ZARA ", "hello")
    assert admitted["author"] == {"id": "bot:zara", "name": "zara", "is_bot": True}
    assert admitted["message"].startswith("Message from 🤖 zara:\n")


def test_owner_change_during_admission_returns_terminal_refusal(lisa_owner, monkeypatch):
    from tools import session_relay
    lisa, lease = lisa_owner
    owner = {"profile_home": str(lisa.resolve()), "session_id": "desktop-exact",
             "lease_id": lease.lease_id, "live_session_id": "live-lisa"}
    monkeypatch.setattr(session_relay, "find_exact_desktop_owner", lambda *_: owner)
    monkeypatch.setattr(session_relay, "find_continuing_desktop_owner", lambda *_: None)
    refused = session_relay.deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "race")
    assert refused["status"] == "cancelled"
    assert "owner changed" in refused["reason"]
    assert read_delivery_result(lisa, refused["delivery_id"])["status"] == "cancelled"


def test_admission_accepts_same_owner_transferred_to_compression_descendant(lisa_owner, monkeypatch):
    from tools import session_relay
    lisa, lease = lisa_owner
    real_deliver = session_relay.deliver_to_live_owner

    def deliver_then_compress(*args, **kwargs):
        admitted = real_deliver(*args, **kwargs)
        _transfer_lisa_owner_across_compression(lisa, lease)
        return admitted

    monkeypatch.setattr(session_relay, "deliver_to_live_owner", deliver_then_compress)
    admitted = session_relay.deliver_to_desktop_session(
        "lisa", "desktop-exact", "ZARA", "compression race")
    assert admitted["status"] == "queued"
    assert read_delivery_result(lisa, admitted["delivery_id"])["status"] == "queued"


def test_claimed_delivery_reconciles_to_ambiguous_after_owner_release(lisa_owner):
    from tools import bot_live_delivery as mailbox
    from tools.session_relay import deliver_to_desktop_session, read_desktop_delivery_result
    lisa, lease = lisa_owner
    admitted = deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "crash window")
    owner = mailbox.find_exact_desktop_owner(lisa, "desktop-exact")
    assert mailbox.claim_pending_delivery(lisa, owner)["status"] == "claimed"
    assert read_desktop_delivery_result("lisa", admitted["delivery_id"])["status"] == "claimed"
    lease.release()
    reconciled = read_desktop_delivery_result("lisa", admitted["delivery_id"])
    assert reconciled["status"] == "ambiguous"
    assert "owner lease ended" in reconciled["reason"]


def test_claimed_delivery_stays_claimed_across_same_owner_compression_transfer(lisa_owner):
    from tools import bot_live_delivery as mailbox
    from tools.session_relay import deliver_to_desktop_session, read_desktop_delivery_result
    lisa, lease = lisa_owner
    admitted = deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "already started")
    owner = mailbox.find_exact_desktop_owner(lisa, "desktop-exact")
    assert mailbox.claim_pending_delivery(lisa, owner)["status"] == "claimed"
    _transfer_lisa_owner_across_compression(lisa, lease)
    reconciled = read_desktop_delivery_result("lisa", admitted["delivery_id"])
    assert reconciled["status"] == "claimed"


def test_queued_delivery_reconciles_to_cancelled_after_owner_release(lisa_owner):
    from tools.session_relay import deliver_to_desktop_session, read_desktop_delivery_result
    _lisa, lease = lisa_owner
    admitted = deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "never started")
    lease.release()
    reconciled = read_desktop_delivery_result("lisa", admitted["delivery_id"])
    assert reconciled["status"] == "cancelled"
    assert "owner lease ended" in reconciled["reason"]


def test_queued_delivery_stays_queued_across_same_owner_compression_transfer(lisa_owner):
    from tools.session_relay import deliver_to_desktop_session, read_desktop_delivery_result
    lisa, lease = lisa_owner
    admitted = deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "not started yet")
    _transfer_lisa_owner_across_compression(lisa, lease)
    reconciled = read_desktop_delivery_result("lisa", admitted["delivery_id"])
    assert reconciled["status"] == "queued"


def test_multiple_tickets_are_fifo_and_mailboxes_are_profile_and_owner_isolated(lisa_owner, tmp_path):
    from tools import bot_live_delivery as mailbox
    from tools.session_relay import deliver_to_desktop_session
    lisa, _ = lisa_owner
    first = deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "first")
    second = deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "second")
    owner = mailbox.find_exact_desktop_owner(lisa, "desktop-exact")
    assert mailbox.claim_pending_delivery(lisa, dict(owner, lease_id="other-lease")) is None
    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    assert mailbox.claim_pending_delivery(
        other_home, {**owner, "profile_home": str(other_home.resolve())}) is None
    assert mailbox.claim_pending_delivery(lisa, owner)["delivery_id"] == first["delivery_id"]
    assert mailbox.claim_pending_delivery(lisa, owner)["delivery_id"] == second["delivery_id"]


# ── the sender is whatever home this process is bound to ──────────────────────────────────────


@pytest.fixture
def default_home_sender(tmp_path, monkeypatch):
    """The root home (``default`` profile) is ZARA; the target is the named ``lisa`` profile."""
    root = tmp_path / "hermes"
    root.mkdir()
    (root / "profile.yaml").write_text("display_name: ZARA\n", encoding="utf-8")
    lisa = root / "profiles" / "lisa"
    lisa.mkdir(parents=True)
    (lisa / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    lease, refusal = try_acquire_active_session(
        session_id="desktop-exact", surface="desktop", config={}, registry_home=lisa,
        metadata={"live_session_id": "live-lisa", "bot_live_delivery_consumer": True},
        track_liveness=True)
    assert refusal is None
    yield root, lisa, lease
    lease.release()


def test_default_home_sender_speaks_as_its_manifest_label(default_home_sender):
    """ZARA runs in the root home: the label comes from profile.yaml, never from the argument."""
    from tools.session_relay import bound_profile_identity, deliver_to_desktop_session

    root, lisa, _lease = default_home_sender
    assert bound_profile_identity() == ("default", "ZARA")
    admitted = deliver_to_desktop_session("lisa", "desktop-exact", "default", "Halo Lisa")
    assert admitted["status"] == "queued"
    assert admitted["author"] == {"id": "bot:zara", "name": "ZARA", "is_bot": True}
    assert admitted["message"] == "Message from 🤖 ZARA:\nHalo Lisa"
    assert (lisa / "runtime" / "bot_live_delivery").is_dir()


def test_default_home_sender_cannot_claim_another_identity_or_label(default_home_sender):
    from tools.session_relay import deliver_to_desktop_session

    _root, lisa, _lease = default_home_sender
    for claimed_sender in ("lisa", "van", "zara", " LISA ", "Van"):
        with pytest.raises(ValueError, match="sender"):
            deliver_to_desktop_session("lisa", "desktop-exact", claimed_sender, "forged")
    assert not (lisa / "runtime" / "bot_live_delivery").exists()


def test_named_home_sender_refuses_the_default_identity(lisa_owner):
    """Bound to a named profile home, ``default`` is somebody else's identity."""
    from tools.session_relay import bound_profile_identity, deliver_to_desktop_session

    lisa, _lease = lisa_owner
    assert bound_profile_identity() == ("zara", "zara")
    for claimed_sender in ("default", " DEFAULT "):
        with pytest.raises(ValueError, match="sender"):
            deliver_to_desktop_session("lisa", "desktop-exact", claimed_sender, "forged")
    assert not (lisa / "runtime" / "bot_live_delivery").exists()


def test_exact_session_owner_turn_exposes_peer_message_in_transcript_and_returns_receipt(
        lisa_owner, monkeypatch, tmp_path):
    """Real turn runner + real mailbox; only the model/network boundary is fake."""
    from tools.session_relay import deliver_to_desktop_session
    from tui_gateway import server as srv

    class InlineThread:
        def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
            self.target, self.args, self.kwargs = target, args, kwargs or {}
        def start(self):
            self.target(*self.args, **self.kwargs)
        def is_alive(self):
            return False
        def join(self, timeout=None):
            pass

    monkeypatch.setattr(srv.threading, "Thread", InlineThread)
    emitted = []
    monkeypatch.setattr(srv, "_emit", lambda *args: emitted.append(args))
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd",
                 "_tts_stream_begin", "_sync_session_key_after_compress"):
        monkeypatch.setattr(srv, name, lambda *a, **k: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda agent: {})
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda session: True)
    seen = []

    def run_conversation(user_message, *, turn_author=None, conversation_history=None, **kwargs):
        seen.append((user_message, turn_author))
        return {"messages": [*conversation_history, {"role": "user", "content": user_message},
                             {"role": "assistant", "content": "Acknowledged"}],
                "final_response": "Acknowledged"}

    agent = SimpleNamespace(session_id="desktop-exact", run_conversation=run_conversation,
                            clear_interrupt=lambda: None)
    lisa, lease = lisa_owner
    ticket = deliver_to_desktop_session("lisa", "desktop-exact", "ZARA", "Deliver this")
    session = {"profile_home": str(lisa), "history_lock": threading.RLock(),
               "history": [], "history_version": 0, "agent": agent, "session_key": "desktop-exact",
               "running": False, "active_session_lease": lease, "attached_images": [],
               "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
               "tool_progress_mode": "all", "inflight_turn": None, "transport": None}
    poll = rebind(session_notifications._poll_bot_live_delivery_once, {
        "_session_home": lambda session: lisa,
        "_session_turn_admission": _session_turn_admission,
        "_run_prompt_submit": srv._run_prompt_submit,
        "_emit": srv._emit,
        "_notif_release_turn": lambda session: session.update(running=False),
    })
    assert poll("live-lisa", session) is True
    assert seen == [(ticket["message"], {"id": "bot:zara", "name": "zara", "is_bot": True})]
    assert any(msg.get("role") == "user" and "Message from 🤖 zara:" in str(msg.get("content"))
               for msg in session["history"])
    assert any(event == "message.complete" for event, *_ in emitted)
    assert next(i for i, event in enumerate(emitted) if event[0] == "message.user") < next(
        i for i, event in enumerate(emitted) if event[0] == "message.start")
    assert read_delivery_result(lisa, ticket["delivery_id"])["status"] == "settled"
