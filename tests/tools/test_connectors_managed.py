"""Managed connectors on the connection operation.

Contracts:
- connect/reconnect mint ONE operation, block the turn via the callback, return per-target
  outcomes; links live on the target and never in the model result on a desktop session
- off-desktop (no callback): result carries connect_url and returns at once (PR3 delivers it)
- reconnect is a repair: active → connected with no gateway mint; force → always reinitiate
- the watcher polls the gateway once per tick, transitions targets, settles on all-resolved
- ``wait`` is gone from the schema
- ``statusReason`` from the mint is kept as detail; the generic list copy never overwrites it
"""

import json
import threading
import time
from unittest.mock import patch

import pytest

import tools.connectors.tool  # registers the tool
from tools.connectors import contract as c
from tools.connectors import live, managed
from tools.connectors import operation as op_module
from tools.connectors.gateway.errors import RateLimited
from tools.connectors.tool import MANAGE_CONNECTIONS_SCHEMA, manage_connections


@pytest.fixture(autouse=True)
def _clean_live():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


class GatewayFake:
    """Scripted gateway.

    ``flips`` maps connector -> the account read on which that account first answers ``active``;
    ``rows`` maps connector -> the status each successive read answers (the last value repeats, an
    entry may be ``None`` for a 404 or an exception to raise). ``list_connectors`` is the reconnect
    repair check only; the watcher reads accounts."""

    def __init__(self, connected=(), flips=None, rows=None, mint_status="initiated", status_reason=None,
                 mint_connection_id=True):
        self.connected = set(connected)
        self.flips = dict(flips or {})
        self.rows = dict(rows or {})
        self.mint_status = mint_status
        self.status_reason = status_reason
        self.mint_connection_id = mint_connection_id
        self.lists = 0
        self.mints = []
        self.reads = []  # (connection_id, timeout) in call order
        self.slug_of = {}  # connection id -> connector slug

    def list_connectors(self, *, timeout=None):
        self.lists += 1
        return [{"connector": s, "enabled": True, "connected": s in self.connected} for s in ("gmail", "notion")]

    def connections(self, connectors, *, reinitiate=False, return_to=None, op=None):
        self.mints.append({"connectors": tuple(connectors), "reinitiate": reinitiate, "return_to": return_to, "op": op})
        results = []
        for slug in connectors:
            row = {"connector": slug, "status": self.mint_status, "reinitiated": reinitiate}
            if self.mint_status == "initiated":
                row["connect_url"] = f"https://connect.example/{slug}/{len(self.mints)}"
            if self.mint_connection_id and self.mint_status in ("initiated", "active"):
                connection_id = f"ca_{slug}_{len(self.mints)}"
                self.slug_of[connection_id] = slug
                row["connection_id"] = connection_id
            if self.status_reason:
                row["status_reason"] = self.status_reason
            results.append(row)
        return {"results": results, "summary": {"total": len(connectors)}}

    def account_status(self, connection_id, *, timeout=None):
        self.reads.append((connection_id, timeout))
        slug = self.slug_of.get(connection_id, "")
        nth = sum(1 for cid, _ in self.reads if cid == connection_id)
        status = self._status_of(slug, nth)
        if isinstance(status, Exception):
            raise status
        if status is None:
            return None
        return {"connectionId": connection_id, "connector": slug, "status": status,
                "statusReason": self.status_reason or "", "label": f"{slug}_a", "active": status == "active",
                "createdAt": "2026-09-14T10:00:00.000Z", "updatedAt": "2026-09-14T10:00:00.000Z"}

    def _status_of(self, slug, nth):
        script = self.rows.get(slug)
        if script:
            return script[min(nth, len(script)) - 1]
        flip = self.flips.get(slug)
        if flip is not None and nth >= flip:
            return "active"
        return "active" if slug in self.connected and not self.flips else "pending"


def _desktop_callback(answer=None):
    """A callback that emits the card and returns immediately (fire-and-forget, PR2 shape)."""
    seen = []

    def cb(payload):
        seen.append(payload)
        return answer

    cb.seen = seen
    return cb


def _run(args, gw, *, callback=None, tick=0.0, platform="desktop"):
    # Two seams read the surface: managed decides whether a card exists, the client decides whether a
    # return target rides the mint.
    with patch("tools.connectors.managed.WATCH_TICK_SECONDS", tick), \
         patch("tools.connectors.managed.session_platform", return_value=platform), \
         patch("tools.connectors.gateway.client.session_platform", return_value=platform):
        return json.loads(manage_connections(
            args, client_factory=lambda: gw, connection_callback=callback, session_id="s1",
        ))


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_reason_is_gone_from_the_schema():
    assert "reason" not in MANAGE_CONNECTIONS_SCHEMA["parameters"]["properties"]


def test_wait_is_gone_and_force_exists():
    props = MANAGE_CONNECTIONS_SCHEMA["parameters"]["properties"]
    assert "wait" not in props["action"]["enum"]
    assert "timeout_seconds" not in props
    assert props["force"]["type"] == "boolean"
    out = json.loads(manage_connections({"action": "wait", "connectors": ["gmail"]}))
    assert "action must be one of" in out["error"]


# ---------------------------------------------------------------------------
# desktop: one op, blocks, no URL in the result
# ---------------------------------------------------------------------------


def test_card_targets_carry_the_account_id_the_mint_named():
    gw = GatewayFake(flips={"gmail": 2})
    cb = _desktop_callback()
    out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=cb)
    (target,) = cb.seen[0]["targets"]
    assert target["connection_id"] == "ca_gmail_1"
    assert out["targets"][0]["connection_id"] == "ca_gmail_1"


def test_every_watch_read_is_bounded_by_the_remaining_deadline():
    """A stalled gateway page cannot hold the operation past its deadline (P1-8 residual)."""
    gw = GatewayFake()
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 2.0):
        _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.2)
    watch_reads = [timeout for _, timeout in gw.reads]
    assert watch_reads and all(t is not None and 0 < t <= 2.0 for t in watch_reads)
    assert watch_reads == sorted(watch_reads, reverse=True)  # each read shrinks with the deadline, down to the floor


def test_a_pending_row_moves_nothing_and_a_revoked_row_is_failed():
    gw = GatewayFake(rows={"gmail": ["pending", "pending", "revoked"]})
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.3):
        out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    # revoked is a failure, and a failed target is unresolved: the deadline stamps it.
    assert out["settled_by"] == "deadline"
    assert out["targets"][0]["state"] == "not_connected"
    assert len(gw.reads) >= 3


def test_desktop_connect_mints_once_emits_the_card_and_returns_outcomes_without_urls():
    gw = GatewayFake(flips={"gmail": 2, "notion": 3})
    cb = _desktop_callback()
    out = _run({"action": "connect", "connectors": ["gmail", "notion"]}, gw, callback=cb)

    assert [m["connectors"] for m in gw.mints] == [("gmail", "notion")]  # one mint for every target, up front
    (payload,) = cb.seen
    assert payload["op_id"] == out["op_id"]
    assert [t["name"] for t in payload["targets"]] == ["gmail", "notion"]
    assert all(t["kind"] == "connector" and t["action"] == "connect" for t in payload["targets"])
    assert out["status"] == "settled" and out["settled_by"] == "all_resolved"
    assert {t["state"] for t in out["targets"]} == {"connected"}
    assert "connect_url" not in json.dumps(out)
    assert live.current("s1") is None  # closed on settle


def test_desktop_connect_url_stays_on_the_live_operation_for_the_panel():
    gw = GatewayFake(flips={"gmail": 2})
    captured = {}

    def cb(payload):
        captured["op"] = live.get("s1", payload["op_id"])
        return None

    _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=cb)
    snap = captured["op"].result()["targets"][0]
    assert snap["connect_url"].startswith("https://connect.example/gmail/")


def test_watcher_transitions_on_flip_and_settles_by_deadline_when_nothing_flips():
    gw = GatewayFake()
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    assert out["settled_by"] == "deadline"
    assert out["targets"][0]["state"] == "not_connected"
    assert len(gw.reads) >= 2  # it did poll


def test_the_watcher_reads_one_account_per_target_per_tick_and_never_the_list():
    """The per-account route replaced the list walk: the watch loop asks for the accounts the mint
    named and nothing else, once each per tick, and stops reading a target once it resolves."""
    gw = GatewayFake(flips={"gmail": 3, "notion": 2})
    _run({"action": "connect", "connectors": ["gmail", "notion"]}, gw, callback=_desktop_callback())
    reads = [connection_id for connection_id, _ in gw.reads]
    assert gw.lists == 0
    assert reads.count("ca_gmail_1") == 3
    assert reads.count("ca_notion_1") == 2  # connected on read 2; never read again
    assert set(reads) == {"ca_gmail_1", "ca_notion_1"}


def test_respond_from_the_card_skips_a_target_and_wakes_the_loop():
    gw = GatewayFake(flips={"gmail": 2})
    done = threading.Event()

    def cb(payload):
        def answer():
            operation = live.get("s1", payload["op_id"])
            operation.transition("notion", c.TargetState.skipped, c.Actor.user)
            done.set()
        threading.Timer(0.02, answer).start()
        return None

    out = _run({"action": "connect", "connectors": ["gmail", "notion"]}, gw, callback=cb, tick=0.01)
    assert done.is_set()
    by = {t["name"]: t for t in out["targets"]}
    assert by["gmail"]["state"] == "connected" and by["notion"]["state"] == "skipped"
    assert out["settled_by"] == "all_resolved"


def test_mint_failure_detail_survives_the_generic_list_copy():
    gw = GatewayFake(mint_status="failed", status_reason="vendor: bad scope")
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    target = out["targets"][0]
    assert target["state"] == "not_connected"  # failed is unresolved; deadline stamped it
    assert target["detail"] == "vendor: bad scope"


# ---------------------------------------------------------------------------
# reconnect = repair
# ---------------------------------------------------------------------------


def test_reconnect_on_an_active_target_makes_no_gateway_mint():
    gw = GatewayFake(connected={"gmail"})
    out = _run({"action": "reconnect", "connectors": ["gmail"]}, gw, callback=_desktop_callback())
    assert gw.mints == []
    assert out["targets"][0]["state"] == "connected"
    assert out["settled_by"] == "all_resolved"


def test_reconnect_force_always_reinitiates_even_when_active():
    gw = GatewayFake(connected={"gmail"}, flips={"gmail": 1})
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        _run({"action": "reconnect", "connectors": ["gmail"], "force": True}, gw, callback=_desktop_callback(), tick=0.01)
    assert [(m["connectors"], m["reinitiate"]) for m in gw.mints] == [(("gmail",), True)]


def test_reconnect_on_a_disconnected_target_reinitiates():
    gw = GatewayFake(flips={"gmail": 2})
    _run({"action": "reconnect", "connectors": ["gmail"]}, gw, callback=_desktop_callback())
    assert [(m["connectors"], m["reinitiate"]) for m in gw.mints] == [(("gmail",), True)]


# ---------------------------------------------------------------------------
# off-desktop: links in the result, returns at once
# ---------------------------------------------------------------------------


def test_off_desktop_connect_returns_links_and_does_not_block():
    gw = GatewayFake()
    out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=None, platform="cli")
    assert out["status"] == "initiated"
    assert out["targets"][0]["connect_url"].startswith("https://connect.example/gmail/")
    assert "op_id" in out
    assert gw.lists == 0 and gw.reads == []  # no watcher without a card
    assert live.current("s1") is None


def test_platform_not_callback_presence_decides_the_url():
    # The TUI-in-a-terminal has a gateway callback attached but no card; the URL must be in the result.
    gw = GatewayFake()
    cb = _desktop_callback()
    out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=cb, platform="tui")
    assert out["targets"][0]["connect_url"]
    assert cb.seen == []  # no card emitted off-desktop


# ---------------------------------------------------------------------------
# one open op per session
# ---------------------------------------------------------------------------


def test_second_connect_while_an_operation_is_open_is_refused():
    gw = GatewayFake()
    operation = live.open_new([("gmail", "connector", "connect")], "s1") if hasattr(live, "open_new") else None
    if operation is None:
        from tools.connectors import operation as op
        operation = op.ConnectionOperation([op.Target("gmail", "connector", "connect")], session_key="s1")
        live.open(operation)
    out = _run({"action": "connect", "connectors": ["notion"]}, gw, callback=_desktop_callback())
    assert "already open" in out["error"] and operation.op_id in out["error"]
    assert gw.mints == []


# ---------------------------------------------------------------------------
# settle races and terminal targets (verification findings P1-1, P1-7, P1-8)
# ---------------------------------------------------------------------------


def test_connected_read_on_a_failed_target_is_ignored_not_an_error():
    """A failed mint whose account later reads connected must not raise out of the watcher."""
    gw = GatewayFake(mint_status="failed", status_reason="denied", flips={"gmail": 1})
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    assert "error" not in out
    assert out["targets"][0]["state"] in {"failed", "not_connected"}
    assert out["targets"][0]["detail"] == "denied"


def test_continue_during_a_connected_read_keeps_the_settled_result():
    """Settling while an account read is in flight must not let that read's `active` raise into
    tool_error: the Continue landed first, so its frozen result stands and the read is dropped."""
    gw = GatewayFake()
    settled = threading.Event()
    original = gw.account_status

    def settle_mid_read(connection_id, **kwargs):
        row = original(connection_id, **kwargs)
        live_op = live.get("s1", op_id["v"])
        live_op.settle(c.SettleReason.continue_)
        settled.set()
        return dict(row, status="active", active=True)

    gw.account_status = settle_mid_read
    op_id = {}

    def cb(payload):
        op_id["v"] = payload["op_id"]
        return None

    out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=cb, tick=0.01)
    assert settled.is_set()
    assert "error" not in out
    assert out["settled_by"] == "continue"
    assert out["targets"][0]["state"] == "not_connected"


def test_settle_reason_is_not_written_into_the_row_detail():
    gw = GatewayFake()
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    assert out["settled_by"] == "deadline"
    assert out["targets"][0]["state"] == "not_connected"
    assert "detail" not in out["targets"][0]


def test_interrupt_wakes_the_loop_and_settles_before_the_next_tick():
    from tools.interrupt import set_interrupt

    gw = GatewayFake()
    worker = {}

    def cb(payload):
        worker["tid"] = threading.current_thread().ident
        def stop():
            set_interrupt(True, worker["tid"])
        threading.Timer(0.02, stop).start()
        return None

    import time
    started = time.monotonic()
    try:
        with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 10):
            out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=cb, tick=5.0)
    finally:
        set_interrupt(False, worker.get("tid"))
    assert out["settled_by"] == "interrupt"
    assert time.monotonic() - started < 2.0  # woke on the interrupt, not on the 5 s tick


# ---------------------------------------------------------------------------
# the watcher reads one account route per pending target (NS-880)
# ---------------------------------------------------------------------------


class _OneRead:
    """A gateway whose every account read answers the same row."""

    def __init__(self, status, reason=""):
        self.status = status
        self.reason = reason

    def account_status(self, connection_id, *, timeout=None):
        return {"connectionId": connection_id, "connector": "gmail", "status": self.status,
                "statusReason": self.reason, "label": "gmail_a", "active": self.status == "active",
                "createdAt": "2026-09-14T10:00:00.000Z", "updatedAt": "2026-09-14T10:00:00.000Z"}


def _one_initiated_target():
    operation = op_module.ConnectionOperation([op_module.Target("gmail", "connector", "connect")], session_key="s1")
    operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher, connection_id="ca_1")
    return operation


@pytest.mark.parametrize("status, state, actor", [
    ("active", "connected", "backend_watcher"),
    ("failed", "failed", "backend_watcher"),
    ("revoked", "failed", "backend_watcher"),
    ("inactive", "failed", "backend_watcher"),
    ("expired", "expired", "clock"),
    ("pending", "initiated", None),
])
def test_the_account_status_decides_the_state_and_who_caused_it(monkeypatch, status, state, actor):
    changes = []
    monkeypatch.setattr(op_module.ConnectionOperation, "on_change",
                        staticmethod(lambda operation, change, snapshot: changes.append(change)))
    operation = _one_initiated_target()
    managed._observe(_OneRead(status), operation)
    assert operation.target("gmail").state.value == state
    if actor is None:
        assert len(changes) == 1  # the mint's own transition; the read moved nothing
    else:
        assert changes[-1]["actor"] == actor


def test_the_account_status_reason_becomes_the_row_detail():
    operation = _one_initiated_target()
    managed._observe(_OneRead("failed", "the user revoked access"), operation)
    assert operation.target("gmail").detail == "the user revoked access"


def test_a_target_the_mint_named_no_account_for_is_never_read():
    """No connectionId means nothing to watch: a no-auth toolkit and a failed mint both answer that way."""
    gw = GatewayFake(mint_connection_id=False)
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.1):
        out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    assert gw.reads == []
    assert out["settled_by"] == "deadline"


def _two_initiated_targets():
    operation = op_module.ConnectionOperation(
        [op_module.Target("gmail", "connector", "connect"), op_module.Target("notion", "connector", "connect")],
        session_key="s1")
    for name, connection_id in (("gmail", "ca_1"), ("notion", "ca_2")):
        operation.transition(name, c.TargetState.initiated, c.Actor.backend_watcher, connection_id=connection_id)
    return operation


def test_a_rate_limit_parks_every_live_target_not_only_the_one_that_read():
    """The route's 429 budget is per principal, not per account: once one read is refused, the
    next target's read in the same tick spends the same budget and would be refused too."""
    operation = _two_initiated_targets()

    class Limited:
        def __init__(self):
            self.reads = []

        def account_status(self, connection_id, *, timeout=None):
            self.reads.append(connection_id)
            raise RateLimited("slow down", retry_after=2.0)

    client = Limited()
    parked_at = time.time()
    managed._observe(client, operation)
    managed._observe(client, operation)  # the next tick, well inside the retry window

    assert client.reads == ["ca_1"]
    assert all(t.next_read_at >= parked_at + 2.0 for t in operation.targets)


def test_a_skip_that_lands_during_the_read_drops_the_read_instead_of_raising():
    """The card's Skip on the RPC thread can resolve a target while the watcher's read of it is in
    flight; the read then has no live row to move, so it is dropped rather than raised into the
    tool result (which would leave the operation open with no card to answer it)."""
    operation = _one_initiated_target()

    class SkipMidRead(_OneRead):
        def account_status(self, connection_id, *, timeout=None):
            operation.transition("gmail", c.TargetState.skipped, c.Actor.user)
            return super().account_status(connection_id, timeout=timeout)

    managed._observe(SkipMidRead("active"), operation)
    assert operation.target("gmail").state == c.TargetState.skipped


def test_a_read_never_waits_longer_than_ten_seconds_whatever_the_deadline():
    """Continue must be able to return the tool within seconds: a hung gateway may not hold one
    read for the operation's whole deadline."""
    operation = _one_initiated_target()
    assert operation.remaining_seconds() > 200

    class Recording(_OneRead):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.timeouts = []

        def account_status(self, connection_id, *, timeout=None):
            self.timeouts.append(timeout)
            return super().account_status(connection_id, timeout=timeout)

    client = Recording("pending")
    managed._observe(client, operation)
    (timeout,) = client.timeouts
    assert 0 < timeout <= 10.0


def test_an_account_the_gateway_does_not_know_moves_nothing_until_the_deadline():
    gw = GatewayFake(rows={"gmail": [None]})
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.1):
        out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    assert len(gw.reads) > 1  # a 404 is "not yet", so it keeps asking
    assert out["settled_by"] == "deadline" and out["targets"][0]["state"] == "not_connected"


def test_the_managed_watcher_ticks_on_its_own_cadence_not_the_shared_default():
    """The single-account route carries its own rate budget, so the managed kind sets the tick it
    needs instead of the 5 s every other kind uses."""
    gw = GatewayFake()
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 30.0), \
         patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.2):
        _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    assert len(gw.reads) > 3


# ---------------------------------------------------------------------------
# return to the surface that asked (portal PR 2.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev_server, expected", [
    (None, "hermes-desktop"),
    ("http://127.0.0.1:5174", "hermes-desktop-dev"),
])
def test_the_desktop_mint_names_the_return_target_and_the_operation(monkeypatch, dev_server, expected):
    monkeypatch.delenv("HERMES_DESKTOP_DEV_SERVER", raising=False)
    if dev_server:
        monkeypatch.setenv("HERMES_DESKTOP_DEV_SERVER", dev_server)
    gw = GatewayFake(flips={"gmail": 1})
    out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=_desktop_callback(), tick=0.01)
    (mint,) = gw.mints
    assert mint["return_to"] == expected
    assert mint["op"] == out["op_id"]


def test_off_the_desktop_the_mint_names_no_return_target():
    """Only the desktop registers the hermes:// scheme; anywhere else the vendor keeps its own done page."""
    gw = GatewayFake()
    _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=None, platform="cli")
    (mint,) = gw.mints
    assert mint["return_to"] is None and mint["op"] is None
