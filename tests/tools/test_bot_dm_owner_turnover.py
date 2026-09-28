"""A local Bot Chat DM reaches a terminal receipt when its live owner closes or its recipient is busy.

Before: an envelope queued for a live Bot Chat owner that closed stayed pinned to a dead lease
(sender saw "pending or unknown" forever), and a DM into a busy recipient failed ``target_busy``
after the short relay budget even though the runner is a detached background process.
"""
import contextlib
import json
import textwrap
from pathlib import Path

import pytest

from tools import bot_live_delivery as mailbox
from tools import bot_mode_dm, bot_mode_probe, bot_relay


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _owner(home, lease, live="live", session="chat"):
    return dict(profile_home=str(Path(home).resolve()), session_id=session, lease_id=lease, live_session_id=live)


def _bot_chat_lease(home, live):
    from hermes_cli.active_sessions import try_acquire_active_session

    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="desktop", config={}, registry_home=home,
        metadata=dict(live_session_id=live, bot_live_delivery_consumer=True))
    assert refusal is None and lease is not None
    return lease


# ── mailbox: orphaned envelopes are adopted or retired, never left pinned to a dead lease ─────────


def test_next_owner_of_the_same_bot_chat_adopts_an_envelope_its_closed_predecessor_left_queued(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    first = _bot_chat_lease(tmp_path, "live-1")
    try:
        queued = mailbox.deliver_to_live_owner(tmp_path, mailbox.find_canonical_live_owner(tmp_path), "hello")
    finally:
        first.release()
    second = _bot_chat_lease(tmp_path, "live-2")
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner["lease_id"] == second.lease_id
        assert mailbox.claim_pending_delivery(tmp_path, owner) is None  # pinned to the dead lease

        state: dict = {}
        mailbox.adopt_orphaned_deliveries(tmp_path, owner, state=state)
        claim = mailbox.claim_pending_delivery(tmp_path, owner)
        assert claim is not None and claim["delivery_id"] == queued["delivery_id"]
        assert claim["message"] == "hello"
        assert claim["repinned_from"][0]["lease_id"] == first.lease_id
        # Rate-limited per owner state: a second pass inside the window does not rescan.
        mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path, first.lease_id, "live-1"), "later")
        mailbox.adopt_orphaned_deliveries(tmp_path, owner, state=state)
        assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    finally:
        second.release()
        db.close()


def test_envelope_with_no_live_owner_is_retired_owner_gone_and_never_claimed(tmp_path):
    dead = _owner(tmp_path, "dead-lease")
    queued = mailbox.deliver_to_live_owner(tmp_path, dead, "hello")
    pins = mailbox.queued_pins(tmp_path)
    assert pins == {queued["delivery_id"]: "dead-lease"}

    moved = mailbox.reconcile_orphaned_deliveries(
        tmp_path, None, live_leases=mailbox.live_lease_ids(tmp_path), expected_pins=pins, adopt_max_age_seconds=60)
    assert moved == {"adopted": [], "retired": [queued["delivery_id"]]}
    receipt = mailbox.read_delivery_result(tmp_path, queued["delivery_id"])
    assert receipt["status"] == "cancelled" and receipt["reason"] == mailbox.OWNER_GONE_REASON
    assert mailbox.claim_pending_delivery(tmp_path, dead) is None


def test_reconcile_moves_only_a_still_queued_envelope_whose_observed_pin_is_dead(tmp_path):
    new_owner = _owner(tmp_path, "new-lease", "live-2")
    fresh = mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path, "dead-lease"), "fresh", delivery_id="a" * 32)
    live = mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path, "live-lease"), "live", delivery_id="b" * 32)
    raced = mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path, "dead-lease"), "raced", delivery_id="c" * 32)
    claimed_owner = _owner(tmp_path, "dead-lease", "live-x", session="other")
    claimed = mailbox.deliver_to_live_owner(tmp_path, claimed_owner, "claimed", delivery_id="e" * 32)
    assert mailbox.claim_pending_delivery(tmp_path, claimed_owner)["delivery_id"] == claimed["delivery_id"]

    pins = mailbox.queued_pins(tmp_path)
    assert claimed["delivery_id"] not in pins
    pins[raced["delivery_id"]] = "someone-else"  # re-pinned after the caller's liveness snapshot
    moved = mailbox.reconcile_orphaned_deliveries(
        tmp_path, new_owner, live_leases={"live-lease"}, expected_pins=pins, adopt_max_age_seconds=60,
        now_ns=fresh["created_at"] + 30 * 10**9)

    assert moved == {"adopted": [fresh["delivery_id"]], "retired": []}
    assert mailbox.read_delivery_result(tmp_path, live["delivery_id"])["owner"]["lease_id"] == "live-lease"
    assert mailbox.read_delivery_result(tmp_path, raced["delivery_id"])["status"] == "queued"
    assert mailbox.read_delivery_result(tmp_path, claimed["delivery_id"])["status"] == "claimed"
    # Older than the adoption budget: retired, not handed to a turn the sender stopped waiting for.
    expired = mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path, "dead-lease"), "x", delivery_id="f" * 32)
    moved = mailbox.reconcile_orphaned_deliveries(
        tmp_path, new_owner, live_leases=set(), expected_pins=mailbox.queued_pins(tmp_path, only_id="f" * 32),
        adopt_max_age_seconds=60, now_ns=expired["created_at"] + 61 * 10**9, only_id="f" * 32)
    assert moved == {"adopted": [], "retired": ["f" * 32]}


# ── runner: owner_gone falls back to the CLI transport; a busy recipient is queued behind ─────────


def _managed_home(tmp_path) -> Path:
    home = tmp_path / ".hermes"
    target = home / "profiles" / "researcher"
    target.mkdir(parents=True)
    (target / "profile.yaml").write_text(textwrap.dedent("""\
        description: teammate for tests
        ui_meta:
          hermes-bots:
            shape: cloud
        """), encoding="utf-8")
    return home


@pytest.fixture
def runner(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(bot_mode_dm, "_LOCAL_LIVE_WAIT_SECONDS", 5)
    monkeypatch.setattr(bot_mode_dm, "_OWNER_CHECK_SECONDS", 0.05)
    monkeypatch.setattr(bot_mode_dm, "_BUSY_SLICE_SECONDS", 0.05)
    monkeypatch.setattr(bot_relay, "delivery_env", lambda author, home=None: {})
    turns: list[str] = []

    def local_turn(argv, dm_file, *, env=None):
        turns.append(Path(dm_file).read_text(encoding="utf-8"))
        return 0

    monkeypatch.setattr(bot_mode_dm, "_run_local_turn", local_turn)
    dm_file = tmp_path / "dm-1.txt"
    dm_file.write_text("hello", encoding="utf-8")
    return home / "profiles" / "researcher", dm_file, turns


def test_owner_closing_before_it_ran_the_dm_hands_it_to_the_cli_transport(runner, monkeypatch, capsys):
    target, dm_file, turns = runner
    owners = [_owner(target, "gone-lease")]  # admitted once; the owner is gone afterwards
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda h: owners.pop() if owners else None)
    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", lambda *a, **k: contextlib.nullcontext())

    assert bot_mode_dm._run_delivery(["hermes", "-p", "researcher"], str(dm_file), stdin_file=False) == 0
    assert turns == ["hello"]  # delivered exactly once, over the CLI path
    first = mailbox.read_delivery_result(target, bot_mode_dm._dm_delivery_id(str(dm_file)))
    assert first["status"] == "cancelled" and first["reason"] == mailbox.OWNER_GONE_REASON
    for leftover in ("", ".live.json", ".gen"):
        assert not Path(str(dm_file) + leftover).exists()
    assert capsys.readouterr().out == ""  # the live path printed no receipt; the CLI turn owns the reply


def test_busy_recipient_is_queued_behind_then_delivered_instead_of_target_busy(runner, monkeypatch):
    _target, dm_file, turns = runner
    probes = []
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda h: probes.append(h))
    monkeypatch.setattr(bot_relay, "dm_queue_wait_seconds", lambda home=None: 5.0)
    attempts = []

    @contextlib.contextmanager
    def busy_twice(argv, *, stdin_file, timeout_seconds=None):
        attempts.append(timeout_seconds)
        if len(attempts) < 3:
            raise bot_relay.TurnBusyError(argv[2], timeout_seconds or 0)
        yield

    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", busy_twice)
    assert bot_mode_dm._run_delivery(["hermes", "-p", "researcher"], str(dm_file), stdin_file=False) == 0
    assert turns == ["hello"]
    assert attempts == [0.05, 0.05, 0.05]
    assert len(probes) >= 3  # a live owner is re-probed between lock slices
    assert not dm_file.exists()


def test_busy_recipient_past_the_queue_budget_fails_target_busy_with_the_real_wait(runner, monkeypatch):
    _target, dm_file, turns = runner
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda h: None)
    monkeypatch.setattr(bot_relay, "dm_queue_wait_seconds", lambda home=None: 0.2)

    @contextlib.contextmanager
    def always_busy(argv, *, stdin_file, timeout_seconds=None):
        raise bot_relay.TurnBusyError(argv[2], 0)
        yield  # pragma: no cover

    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", always_busy)
    with pytest.raises(bot_relay.TurnBusyError) as exc:
        bot_mode_dm._run_delivery(["hermes", "-p", "researcher"], str(dm_file), stdin_file=False)
    assert exc.value.reason == "target_busy" and exc.value.waited_seconds > 0
    assert turns == [] and not dm_file.exists()


def test_owner_that_opens_while_queued_behind_a_busy_turn_takes_the_dm(runner, monkeypatch, capsys):
    target, dm_file, turns = runner
    monkeypatch.setattr(bot_mode_dm, "_LOCAL_LIVE_WAIT_SECONDS", 0)
    owner = _owner(target, "late-lease")
    probes = []
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda h: owner if probes.append(h) or len(probes) > 1 else None)
    monkeypatch.setattr(mailbox, "live_lease_ids", lambda h: {"late-lease"})
    monkeypatch.setattr(bot_relay, "dm_queue_wait_seconds", lambda home=None: 5.0)

    @contextlib.contextmanager
    def busy(argv, *, stdin_file, timeout_seconds=None):
        raise bot_relay.TurnBusyError(argv[2], 0)
        yield  # pragma: no cover

    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", busy)
    assert bot_mode_dm._run_delivery(["hermes", "-p", "researcher"], str(dm_file), stdin_file=False) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "queued"
    assert mailbox.read_delivery_result(target, receipt["delivery_id"])["owner"]["lease_id"] == "late-lease"
    assert turns == []
    assert dm_file.exists()  # the live path keeps its evidence for a retry of the same delivery id


def test_dm_queue_wait_seconds_reads_the_recipients_config_not_the_senders(tmp_path, monkeypatch):
    """The queue belongs to the recipient; the detached runner inherits the sender's HERMES_HOME."""
    sender, recipient, bare = tmp_path / "sender", tmp_path / "recipient", tmp_path / "bare"
    for home in (sender, recipient, bare):
        home.mkdir()
    (sender / "config.yaml").write_text("bot_mode:\n  dm_queue_wait_seconds: 5\n", encoding="utf-8")
    (recipient / "config.yaml").write_text("bot_mode:\n  dm_queue_wait_seconds: 42\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(sender))

    assert bot_relay.dm_queue_wait_seconds() == 5.0
    assert bot_relay.dm_queue_wait_seconds(recipient) == 42.0
    assert bot_mode_dm._live_wait_budget(str(recipient)) == 42.0
    assert bot_relay.dm_queue_wait_seconds(bare) == float(bot_relay.DM_QUEUE_WAIT_SECONDS_FALLBACK)
    assert bot_relay.dm_queue_wait_seconds() == 5.0  # the per-call override does not leak


def test_cli_queue_budget_comes_from_the_recipients_home(runner, monkeypatch):
    target, dm_file, turns = runner
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda h: None)
    asked = []
    monkeypatch.setattr(bot_relay, "dm_queue_wait_seconds", lambda home=None: asked.append(home) or 5.0)
    assert bot_mode_dm._run_delivery(["hermes", "-p", "researcher"], str(dm_file), stdin_file=False) == 0
    assert turns == ["hello"]
    assert asked and all(h is not None and Path(h).resolve() == Path(target).resolve() for h in asked)


def test_dm_runner_lock_serializes_two_runners_of_one_dm(tmp_path):
    import threading

    dm_file = str(tmp_path / "dm.txt")
    order: list[str] = []
    inside = threading.Event()

    def second():
        with bot_mode_dm._dm_runner_lock(dm_file):
            order.append("second")

    with bot_mode_dm._dm_runner_lock(dm_file):
        t = threading.Thread(target=second)
        t.start()
        inside.wait(0.3)  # give the second runner time to reach the lock
        order.append("first-done")
    t.join(5)
    assert order == ["first-done", "second"]


def test_dm_queue_wait_seconds_default_is_registered():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["bot_mode"]["dm_queue_wait_seconds"] == bot_relay.DM_QUEUE_WAIT_SECONDS_FALLBACK
