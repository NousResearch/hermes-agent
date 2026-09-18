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
from unittest.mock import patch

import pytest

import tools.connectors.tool  # registers the tool
from tools.connectors import contract as c
from tools.connectors import live
from tools.connectors.tool import MANAGE_CONNECTIONS_SCHEMA, manage_connections


@pytest.fixture(autouse=True)
def _clean_live():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


class GatewayFake:
    """Scripted gateway. ``flips`` maps connector -> the list call number on which it reports connected."""

    def __init__(self, connected=(), flips=None, mint_status="initiated", status_reason=None):
        self.connected = set(connected)
        self.flips = dict(flips or {})
        self.mint_status = mint_status
        self.status_reason = status_reason
        self.lists = 0
        self.mints = []
        self.statuses = {}  # slug -> connectionStatus per list call (last value repeats)

    def list_connectors(self):
        self.lists += 1
        for slug, on in self.flips.items():
            if self.lists >= on:
                self.connected.add(slug)
        rows = []
        for s in ("gmail", "notion"):
            row = {"connector": s, "enabled": True, "connected": s in self.connected}
            script = self.statuses.get(s)
            if script:
                row["connectionStatus"] = script[min(self.lists, len(script)) - 1]
                row["connected"] = row["connectionStatus"] == "active"
            rows.append(row)
        return rows

    def connections(self, connectors, *, reinitiate=False):
        self.mints.append((tuple(connectors), reinitiate))
        results = []
        for slug in connectors:
            row = {"connector": slug, "status": self.mint_status, "reinitiated": reinitiate}
            if self.mint_status == "initiated":
                row["connect_url"] = f"https://connect.example/{slug}/{len(self.mints)}"
            if self.status_reason:
                row["status_reason"] = self.status_reason
            results.append(row)
        return {"results": results, "summary": {"total": len(connectors)}}


def _desktop_callback(answer=None):
    """A callback that emits the card and returns immediately (fire-and-forget, PR2 shape)."""
    seen = []

    def cb(payload):
        seen.append(payload)
        return answer

    cb.seen = seen
    return cb


def _run(args, gw, *, callback=None, tick=0.0, platform="desktop"):
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", tick), \
         patch("tools.connectors.managed.session_platform", return_value=platform):
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


def test_desktop_connect_mints_once_emits_the_card_and_returns_outcomes_without_urls():
    gw = GatewayFake(flips={"gmail": 2, "notion": 3})
    cb = _desktop_callback()
    out = _run({"action": "connect", "connectors": ["gmail", "notion"]}, gw, callback=cb)

    assert gw.mints == [(("gmail", "notion"), False)]  # one mint for every target, up front
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
    assert gw.lists >= 2  # it did poll


def test_watcher_polls_once_per_tick_for_the_whole_operation():
    gw = GatewayFake(flips={"gmail": 3, "notion": 3})
    _run({"action": "connect", "connectors": ["gmail", "notion"]}, gw, callback=_desktop_callback())
    assert gw.lists == 3  # shared scan, not one per target


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
    assert gw.mints == [(("gmail",), True)]


def test_force_does_not_settle_connected_from_the_old_account():
    """An account switch: the vendor keeps the old account active while the new link waits. `connected`
    on the list is the old account until the row has read as anything else once."""
    gw = GatewayFake(connected={"gmail"})
    gw.statuses = {"gmail": ["active", "active", "initializing", "active"]}
    out = _run({"action": "reconnect", "connectors": ["gmail"], "force": True}, gw, callback=_desktop_callback(), tick=0.01)
    assert out["settled_by"] == "all_resolved"
    assert out["targets"][0]["state"] == "connected"
    assert gw.lists == 4  # reads 1-2 were the old account; 3 was the new attempt; 4 saw it connected


def test_force_reads_a_failed_new_attempt_as_failed_not_as_still_waiting():
    gw = GatewayFake(connected={"gmail"})
    gw.statuses = {"gmail": ["active", "failed"]}
    seen = []

    def cb(payload):
        op_id["v"] = payload["op_id"]

    op_id = {}
    original = gw.list_connectors

    def spy():
        rows = original()
        op = live.get("s1", op_id["v"]) if op_id else None
        if op is not None:
            seen.append(op.target("gmail").state.value)
        return rows

    gw.list_connectors = spy
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.2):
        _run({"action": "reconnect", "connectors": ["gmail"], "force": True}, gw, callback=cb, tick=0.01)
    # Read 1 saw the old account (still initiated); the failed row on read 2 was applied, not swallowed.
    assert "failed" in seen


def test_reconnect_on_a_disconnected_target_reinitiates():
    gw = GatewayFake(flips={"gmail": 2})
    _run({"action": "reconnect", "connectors": ["gmail"]}, gw, callback=_desktop_callback())
    assert gw.mints == [(("gmail",), True)]


# ---------------------------------------------------------------------------
# off-desktop: links in the result, returns at once
# ---------------------------------------------------------------------------


def test_off_desktop_connect_returns_links_and_does_not_block():
    gw = GatewayFake()
    out = _run({"action": "connect", "connectors": ["gmail"]}, gw, callback=None, platform="cli")
    assert out["status"] == "initiated"
    assert out["targets"][0]["connect_url"].startswith("https://connect.example/gmail/")
    assert "op_id" in out
    assert gw.lists == 0  # no watcher without a card
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
    """Settling while a list read is in flight must not let that read's `connected` raise into tool_error."""
    gw = GatewayFake()
    settled = threading.Event()
    original = gw.list_connectors

    def slow_list():
        rows = original()
        if gw.lists == 2:
            live_op = live.get("s1", op_id["v"])
            live_op.settle(c.SettleReason.continue_)
            settled.set()
            gw.connected.add("gmail")
            rows = original()
        return rows

    gw.list_connectors = slow_list
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
