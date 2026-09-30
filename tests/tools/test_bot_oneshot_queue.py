"""One-shot Bot Chat owners queue relayed DMs instead of refusing them.

Regression for #122370: a headless one-shot Bot Chat turn holds the session
lease without ``bot_live_delivery_consumer is True``, so ``find_canonical_live_owner``
refused relayed ``message_agent`` DMs (``SESSION_NOT_OWNED`` -> ``target_busy``)
instead of queueing them into ``runtime/bot_live_delivery/`` for pickup.
"""
from types import SimpleNamespace


def _bot_chat_home(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    return db


def _acquire(tmp_path, **metadata):
    from hermes_cli.active_sessions import try_acquire_active_session

    meta = dict(live_session_id="cli-live", **metadata)
    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="cli", config={}, registry_home=tmp_path, metadata=meta)
    assert refusal is None
    assert lease is not None
    return lease


def test_oneshot_owner_queues_claims_and_settles(tmp_path):
    """A lease holder advertising ``"oneshot"`` (no ``True`` flag) is admitted,
    and its DM round-trips queued -> claimed -> settled under its own lease."""
    from tools import bot_live_delivery as mailbox
    from tools import bot_mode_dm

    db = _bot_chat_home(tmp_path)
    lease = _acquire(tmp_path, bot_live_delivery_consumer="oneshot")
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner is not None
        assert owner["lease_id"] == lease.lease_id
        payload = tmp_path / "message.txt"
        payload.write_text("hello one-shot", encoding="utf-8")
        record = bot_mode_dm._admit_live_dm(tmp_path, str(payload), None)
        assert record is not None and record["status"] == "queued"
        assert record["owner"]["lease_id"] == lease.lease_id
        claim = mailbox.claim_pending_delivery(tmp_path, owner)
        assert claim is not None and claim["delivery_id"] == record["delivery_id"]
        receipt = mailbox.complete_delivery(
            tmp_path, claim["delivery_id"], status="settled", reply="served!")
        assert receipt["status"] == "settled" and receipt["reply"] == "served!"
    finally:
        lease.release()
        db.close()


def test_stale_queued_ticket_adopted_after_lease_turnover(tmp_path):
    """A queued (never executed) ticket pinned to a released one-shot lease is
    adopted by the next live canonical owner instead of stranding."""
    from tools import bot_live_delivery as mailbox

    db = _bot_chat_home(tmp_path)
    first = _acquire(tmp_path, bot_live_delivery_consumer="oneshot")
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner is not None
        queued = mailbox.deliver_to_live_owner(tmp_path, owner, "stranded dm")
    finally:
        first.release()  # one-shot exited (or crashed) without draining
    second = _acquire(tmp_path, bot_live_delivery_consumer="oneshot")
    try:
        current = mailbox.find_canonical_live_owner(tmp_path)
        assert current is not None
        assert current["lease_id"] == second.lease_id
        claim = mailbox.claim_pending_delivery(tmp_path, current)
        assert claim is not None
        assert claim["delivery_id"] == queued["delivery_id"]
        receipt = mailbox.complete_delivery(
            tmp_path, claim["delivery_id"], status="settled", reply="adopted!")
        assert receipt["status"] == "settled"
    finally:
        second.release()
        db.close()


def test_claimed_or_live_pinned_tickets_are_never_adopted(tmp_path):
    """Adoption is queued-only and stale-only: a claimed ticket of a dead owner
    stays an unknown outcome (never re-executed), and a live owner's pin cannot
    be stolen by a forged owner dict."""
    from tools import bot_live_delivery as mailbox

    db = _bot_chat_home(tmp_path)
    first = _acquire(tmp_path, bot_live_delivery_consumer="oneshot")
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner is not None
        queued = mailbox.deliver_to_live_owner(tmp_path, owner, "executed?")
        claim = mailbox.claim_pending_delivery(tmp_path, owner)
        assert claim is not None
        assert claim["delivery_id"] == queued["delivery_id"]
        live = mailbox.deliver_to_live_owner(tmp_path, owner, "still owned")
        forged = dict(owner, lease_id="0" * 32, live_session_id="intruder")
        assert mailbox.claim_pending_delivery(tmp_path, forged) is None
    finally:
        first.release()
    second = _acquire(tmp_path, bot_live_delivery_consumer="oneshot")
    try:
        current = mailbox.find_canonical_live_owner(tmp_path)
        # The claimed ticket stays claimed (unknown outcome); only the still-queued
        # ticket pinned to the now-dead lease is adopted.
        claim = mailbox.claim_pending_delivery(tmp_path, current)
        assert claim is not None
        assert claim["delivery_id"] == live["delivery_id"]
        receipt = mailbox.read_delivery_result(tmp_path, queued["delivery_id"])
        assert receipt is not None
        assert receipt["status"] == "claimed"
    finally:
        second.release()
        db.close()


class _FakeAgent:
    def __init__(self, reply="served!"):
        self.reply = reply
        self.calls = []

    def run_conversation(self, user_message, conversation_history=None, **kwargs):
        self.calls.append(dict(message=user_message, author=kwargs.get("turn_author")))
        history = list(conversation_history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": self.reply},
        ]
        return {"final_response": self.reply, "messages": history}


def test_oneshot_drain_serves_queue_before_lease_release(tmp_path, monkeypatch):
    from hermes_cli.cli_shutdown import _drain_bot_live_deliveries
    from tools import bot_live_delivery as mailbox

    db = _bot_chat_home(tmp_path)
    lease = _acquire(tmp_path, bot_live_delivery_consumer="oneshot")
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        author = {"id": "bot:coder", "name": "coder", "is_bot": True}
        queued = mailbox.deliver_to_live_owner(tmp_path, owner, "dm text", author=author)
        agent = _FakeAgent()
        cli = SimpleNamespace(agent=agent, _active_session_lease=lease, conversation_history=[])
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert _drain_bot_live_deliveries(cli) == 1
        assert [call["message"] for call in agent.calls] == ["dm text"]
        assert agent.calls[0]["author"] == author
        receipt = mailbox.read_delivery_result(tmp_path, queued["delivery_id"])
        assert receipt is not None
        assert receipt["status"] == "settled" and receipt["reply"] == "served!"
        assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    finally:
        lease.release()
        db.close()


def test_oneshot_drain_is_bounded_and_skips_without_mailbox(tmp_path, monkeypatch):
    from hermes_cli import cli_shutdown
    from hermes_cli.cli_shutdown import _drain_bot_live_deliveries
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stranger = SimpleNamespace(agent=_FakeAgent(), _active_session_lease=None,
                               conversation_history=[])
    assert _drain_bot_live_deliveries(stranger) == 0  # no lease: nothing to serve
    db = _bot_chat_home(tmp_path)
    lease = _acquire(tmp_path, bot_live_delivery_consumer="oneshot")
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        for index in range(cli_shutdown._ONESHOT_DRAIN_MAX_TURNS + 2):
            mailbox.deliver_to_live_owner(tmp_path, owner, f"dm {index}")
        agent = _FakeAgent()
        cli = SimpleNamespace(agent=agent, _active_session_lease=lease, conversation_history=[])
        assert _drain_bot_live_deliveries(cli) == cli_shutdown._ONESHOT_DRAIN_MAX_TURNS
        # Overflow stays queued under the live lease for the next owner to adopt.
        assert mailbox.claim_pending_delivery(tmp_path, owner) is not None
    finally:
        lease.release()
        db.close()
