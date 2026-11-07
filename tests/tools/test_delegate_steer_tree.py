"""delegate_task steering across the whole helper tree.

A helper below the top level (depth > 0) joins its own sub-helpers synchronously, so a steer queued
on it used to wait unread until the whole sub-batch finished. Pinned here:

* relay: a steer on a helper that is waiting on live sub-helpers is also queued, labelled, into
  each of them at once; the helper keeps its own copy for when the join returns;
* yield: a steer on a helper running tools asks its tool workers to yield, so a long foreground
  command moves to the background registry instead of parking the steer;
* receipts: every steer records queued_at / delivered_at, shown in action='list' and logged;
* Codex app-server children: the steer goes to the active native turn (``turn/steer``);
* tree: list/steer/stop reach descendants for an ancestor in the owning chain only.
"""

import json
import logging
import threading
import time
import weakref

import pytest

from agent.interrupt_control import InterruptControlMixin
from tools import interrupt as interrupt_mod
from tools.delegate_tool import _handle_control_action, _register_subagent, _unregister_subagent, steer_subagent
from tools.delegate_tool_registry import _close_subagent_steering
from tests.waiting import wait_until


class _Agent(InterruptControlMixin):
    """The real steer/drain/yield surface without the rest of AIAgent."""

    _executing_tools = False
    _interrupt_requested = False
    _pending_steer = None
    _pending_redirect = None
    api_mode = "chat_completions"

    def __init__(self, parent=None, session_id=None):
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect_lock = threading.Lock()
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._execution_thread_id = None
        self.session_id = session_id
        self._delegate_parent_ref = weakref.ref(parent) if parent is not None else None


class _Root:
    def __init__(self, session_id="root-session"):
        self.session_id = session_id


def _register(sid, agent, *, parent_id=None, depth=0, owner_sid="root-session", **extra):
    _register_subagent({
        "subagent_id": sid, "parent_id": parent_id, "depth": depth, "goal": f"goal of {sid}",
        "model": "m", "started_at": time.time(), "status": "running", "tool_count": 0, "agent": agent,
        "owner_agent_session_id": owner_sid, **extra,
    })


@pytest.fixture
def tree():
    """root (top-level conversation) -> mid (depth-1 helper, joined) -> leaf (depth-2 helper)."""
    root = _Root()
    mid = _Agent(root, session_id="mid-session")
    leaf = _Agent(mid, session_id="leaf-session")
    _register("tree-mid", mid)
    _register("tree-leaf", leaf, parent_id="tree-mid", depth=1, owner_sid="mid-session")
    try:
        yield root, mid, leaf
    finally:
        _unregister_subagent("tree-leaf")
        _unregister_subagent("tree-mid")


def _entries(root):
    return {e["subagent_id"]: e for e in json.loads(_handle_control_action("list", None, None, root))["subagents"]}


# ---------------------------------------------------------------------------
# 1. Relay down
# ---------------------------------------------------------------------------


def test_steer_on_a_joined_helper_is_queued_on_its_sub_helper_before_the_call_returns(tree):
    root, mid, leaf = tree
    delivered, landed, stop = {}, threading.Event(), threading.Event()

    def leaf_loop():  # the sub-helper keeps reaching tool boundaries; the middle helper is blocked in its join
        while not stop.is_set():
            text = leaf._drain_pending_steer()
            if text:
                delivered["text"] = text
                landed.set()
                return
            stop.wait(0.05)

    t = threading.Thread(target=leaf_loop, daemon=True)
    t.start()
    try:
        out = json.loads(_handle_control_action("steer", "tree-mid", "use port 8080", root))
        assert out["status"] == "queued" and out["relayed_to"] == ["tree-leaf"]
        assert landed.wait(timeout=5), "the steer never reached the sub-helper while its parent was joined"
    finally:
        stop.set()
        t.join(timeout=5)
    assert "use port 8080" in delivered["text"]
    assert "tree-mid" in delivered["text"] and "relayed" in delivered["text"].lower()
    # The joined helper still gets its own, unlabelled copy when the join returns.
    assert mid._drain_pending_steer() == "use port 8080"


def test_relayed_to_reports_only_this_calls_recipients(tree):
    root, _mid, leaf = tree
    assert json.loads(_handle_control_action("steer", "tree-mid", "first", root))["relayed_to"] == ["tree-leaf"]
    _close_subagent_steering("tree-leaf", leaf)
    assert json.loads(_handle_control_action("steer", "tree-mid", "second", root))["relayed_to"] == []


def test_relay_skips_a_sub_helper_that_stopped_accepting(tree):
    _root, mid, leaf = tree
    _close_subagent_steering("tree-leaf", leaf)
    assert steer_subagent("tree-mid", "narrow the scope") is True
    assert leaf._pending_steer is None
    assert mid._pending_steer == "narrow the scope"


def test_flat_tree_steer_is_unchanged():
    """Default max_spawn_depth=1: a child has no sub-helpers, so nothing is relayed or labelled."""
    root = _Root()
    child, sibling = _Agent(root), _Agent(root)
    _register("flat-a", child)
    _register("flat-b", sibling)
    try:
        assert steer_subagent("flat-a", "check the logs") is True
        assert child._pending_steer == "check the logs"
        assert sibling._pending_steer is None
        assert set(_entries(root)) == {"flat-a", "flat-b"}
    finally:
        _unregister_subagent("flat-a")
        _unregister_subagent("flat-b")


# ---------------------------------------------------------------------------
# 2. Yield on steer
# ---------------------------------------------------------------------------


def test_steer_asks_a_busy_helpers_tool_workers_to_yield():
    agent = _Agent(_Root())
    agent._executing_tools = True
    tid = 987_654_321  # a fake worker tid, never a real thread here
    agent._tool_worker_threads.add(tid)
    _register("yield-1", agent)
    try:
        assert steer_subagent("yield-1", "wrap up") is True
        assert interrupt_mod.is_thread_yield_requested(tid)
    finally:
        interrupt_mod.consume_yield(tid)
        _unregister_subagent("yield-1")


def test_yield_is_armed_under_the_worker_lock(monkeypatch):
    """A pooled worker that exits between a snapshot and the arm must not keep the yield for its next command."""
    agent = _Agent(_Root())
    agent._executing_tools = True
    agent._tool_worker_threads.add(987_654_322)
    held = []
    monkeypatch.setattr("agent.interrupt_control._request_yield",
                        lambda tid: held.append(agent._tool_worker_threads_lock.locked()))
    _register("yield-locked", agent)
    try:
        assert steer_subagent("yield-locked", "wrap up") is True
        assert agent.redirect("user note") is True
    finally:
        _unregister_subagent("yield-locked")
    assert held == [True, True]


def test_joined_helper_does_not_yield_its_own_delegate_call(tree):
    """Its worker is inside delegate_task; the steer goes down to the sub-helpers instead."""
    _root, mid, leaf = tree
    mid._executing_tools = leaf._executing_tools = True
    mid._tool_worker_threads.add(111_111_111)
    leaf._tool_worker_threads.add(222_222_222)
    try:
        assert steer_subagent("tree-mid", "stop polling") is True
        assert not interrupt_mod.is_thread_yield_requested(111_111_111)
        assert interrupt_mod.is_thread_yield_requested(222_222_222)
    finally:
        interrupt_mod.consume_yield(111_111_111)
        interrupt_mod.consume_yield(222_222_222)


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass
def test_subagent_steer_yields_a_long_foreground_command(tmp_path, monkeypatch):
    """End to end on the local backend: the command is handed to the background registry, not killed."""
    from tools.process_registry import process_registry
    from tools.terminal_tool import terminal_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _Agent(_Root())
    agent._executing_tools = True
    _register("yield-e2e", agent)
    res = {}

    def worker():
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(threading.current_thread().ident)
        res["result"] = json.loads(terminal_tool(f"touch {started}; sleep 60", task_id="yield-e2e", timeout=90))

    started = tmp_path / "started"
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        wait_until(started.exists, timeout=15, what="the foreground command to start")
        assert steer_subagent("yield-e2e", "also check the session ids") is True
        t.join(timeout=15)
        assert not t.is_alive(), "the foreground command still parks the subagent steer"
        assert res["result"]["status"] == "yielded_to_background"
        assert agent._pending_steer == "also check the session ids"
    finally:
        _unregister_subagent("yield-e2e")
        if "result" in res and res["result"].get("session_id"):
            process_registry.kill_process(res["result"]["session_id"])


# ---------------------------------------------------------------------------
# 3. Delivery receipts
# ---------------------------------------------------------------------------


def test_list_shows_queued_then_delivered_receipts_and_logs_both(tree, caplog):
    root, _mid, leaf = tree
    caplog.set_level(logging.INFO, logger="tools.delegate_tool")
    assert steer_subagent("tree-leaf", "use the staging db") is True

    receipt = _entries(root)["tree-leaf"]["steers"][-1]
    assert receipt["status"] == "queued" and receipt["delivered_at"] is None
    assert isinstance(receipt["queued_at"], float)

    assert leaf._drain_pending_steer() == "use the staging db"
    receipt = _entries(root)["tree-leaf"]["steers"][-1]
    assert receipt["status"] == "delivered"
    assert receipt["delivered_at"] >= receipt["queued_at"]
    assert receipt["latency_s"] == round(receipt["delivered_at"] - receipt["queued_at"], 3)

    logs = [r.getMessage() for r in caplog.records]
    assert any("queued" in m and "tree-leaf" in m for m in logs)
    assert any("delivered" in m and "tree-leaf" in m for m in logs)


def test_a_drain_racing_the_steer_call_still_settles_its_receipt(tree):
    """The child may take the text before steer_subagent returns from agent.steer(): the receipt is already there."""
    root, _mid, leaf = tree
    original = leaf.steer
    leaf.steer = lambda text, **kw: original(text, **kw) and bool(leaf._drain_pending_steer())
    assert steer_subagent("tree-leaf", "racy note") is True
    assert _entries(root)["tree-leaf"]["steers"][-1]["status"] == "delivered"


def test_receipts_match_whole_submissions_not_substrings(tree):
    """A drain of other text that lands between publishing the receipt and queueing its text must not settle it."""
    root, _mid, leaf = tree
    leaf.steer("narrow the scope")  # queued by another path, no receipt
    original = leaf.steer
    leaf.steer = lambda text, **kw: (leaf._drain_pending_steer(), original(text, **kw))[1]
    assert steer_subagent("tree-leaf", "scope") is True
    assert _entries(root)["tree-leaf"]["steers"][-1]["status"] == "queued"
    assert leaf._pending_steer == "scope"


def test_steer_handed_off_by_the_turn_finalizer_is_receipted_as_missed(tree):
    """A child that ends its turn hands leftover steers to its result (``missed_steer``), bypassing the drain hook."""
    root, _mid, leaf = tree
    steer_subagent("tree-leaf", "after the last tool")
    assert leaf._drain_leftover_steer() == "after the last tool"
    _close_subagent_steering("tree-leaf", leaf)
    assert _entries(root)["tree-leaf"]["steers"][-1]["status"] == "missed"


def test_a_requeued_drain_reopens_the_receipt_until_it_really_lands(tree):
    """A drain with no tool result to attach to (the child's first model call) puts the text back."""
    from agent.interrupt_control import requeue_drained_steer

    root, _mid, leaf = tree
    steer_subagent("tree-leaf", "early note")
    requeue_drained_steer(leaf, leaf._drain_pending_steer())
    assert _entries(root)["tree-leaf"]["steers"][-1]["status"] == "queued"
    assert leaf._drain_leftover_steer() == "early note"  # the child then finished without another tool batch
    _close_subagent_steering("tree-leaf", leaf)
    assert _entries(root)["tree-leaf"]["steers"][-1]["status"] == "missed"


def test_relayed_receipt_names_the_helper_it_came_through(tree):
    root, _mid, _leaf = tree
    steer_subagent("tree-mid", "switch to plan B")
    assert _entries(root)["tree-leaf"]["steers"][-1]["relayed_from"] == "tree-mid"
    assert _entries(root)["tree-mid"]["steers"][-1]["relayed_from"] is None


def test_steer_left_at_completion_is_receipted_as_missed(tree):
    root, _mid, leaf = tree
    steer_subagent("tree-leaf", "too late")
    assert _close_subagent_steering("tree-leaf", leaf) == "too late"
    assert _entries(root)["tree-leaf"]["steers"][-1]["status"] == "missed"


# ---------------------------------------------------------------------------
# 4. Tree-aware list / steer / stop
# ---------------------------------------------------------------------------


def test_list_shows_descendants_with_depth(tree):
    root, _, _ = tree
    entries = _entries(root)
    assert entries["tree-leaf"]["parent_id"] == "tree-mid"
    assert (entries["tree-mid"]["depth"], entries["tree-leaf"]["depth"]) == (0, 1)


def test_ancestor_can_steer_and_stop_a_descendant(tree, monkeypatch):
    root, _, leaf = tree
    out = json.loads(_handle_control_action("steer", "tree-leaf", "direct note", root))
    assert out["status"] == "queued" and out["relayed_to"] == [] and leaf._pending_steer == "direct note"

    stopped = []
    monkeypatch.setattr("tools.delegate_tool_registry.request_hard_interrupt",
                        lambda agent, msg: stopped.append(agent) or True)
    assert json.loads(_handle_control_action("stop", "tree-leaf", None, root))["status"] == "interrupt_requested"
    assert stopped == [leaf]


def test_non_owner_cannot_see_steer_or_stop_a_descendant(tree):
    _, _, leaf = tree
    stranger = _Root(session_id="someone-else")
    assert "tree-leaf" not in _entries(stranger)
    for action in ("steer", "stop"):
        assert "error" in json.loads(_handle_control_action(action, "tree-leaf", "x", stranger))
    assert leaf._pending_steer is None


def test_descendant_stays_reachable_after_the_root_agent_is_rebuilt(tree):
    """A CLI /model swap rebuilds the root AIAgent: the middle helper is found by its durable owner id,
    and its sub-helper through it (the sub-helper's own owner id names the middle helper)."""
    _, _, leaf = tree
    rebuilt = _Root()  # same session id, new object: the weakref chain no longer reaches it
    assert {"tree-mid", "tree-leaf"} <= set(_entries(rebuilt))
    assert json.loads(_handle_control_action("steer", "tree-leaf", "after rebuild", rebuilt))["status"] == "queued"
    assert leaf._pending_steer == "after rebuild"


def test_gateway_authority_still_gates_a_descendant(tree):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

    class _T:
        pass

    owner_t, other_t = _T(), _T()
    session = {"transport": owner_t}
    with _active_subagents_lock:
        _active_subagents["tree-leaf"].update(
            owner_session_id="ui-1", owner_transport=owner_t, owner_session_record=session)
    _, _, leaf = tree
    assert steer_subagent("tree-leaf", "x", owner_session_id="ui-1", owner_transport=other_t,
                          owner_session_record=session) is False
    assert steer_subagent("tree-leaf", "x", owner_session_id="ui-2", owner_transport=owner_t,
                          owner_session_record=session) is False
    assert leaf._pending_steer is None
    assert steer_subagent("tree-leaf", "ok", owner_session_id="ui-1", owner_transport=owner_t,
                          owner_session_record=session) is True


# ---------------------------------------------------------------------------
# Codex app-server children (their own turn loop never drains the Hermes queue)
# ---------------------------------------------------------------------------


def _codex_child(turn_id="turn-1"):
    """A child on the codex_app_server runtime: the real native ``request_steer``, only its JSON-RPC stubbed."""
    from agent.transports.codex_app_server_session import CodexAppServerSession

    rpc = []
    session = CodexAppServerSession.__new__(CodexAppServerSession)
    session._active_turn_lock = threading.Lock()
    session._active_turn_id, session._thread_id = turn_id, "thread-1"
    session._client = type("_Rpc", (), {"request": lambda self, method, params, **kw: rpc.append((method, params))
                                        or {"turnId": turn_id}})()
    child = _Agent(_Root())
    child.api_mode, child._codex_session = "codex_app_server", session
    return child, rpc


def test_codex_child_gets_the_steer_in_its_active_turn():
    child, rpc = _codex_child()
    _register("codex-1", child)
    try:
        assert "error" in json.loads(_handle_control_action("steer", "codex-1", "x", _Root("someone-else")))
        assert rpc == []  # authority is checked before anything reaches the runtime
        assert json.loads(_handle_control_action("steer", "codex-1", "use the v2 api", _Root()))["status"] == "queued"
        assert [(m, p["input"][0]["text"], p["expectedTurnId"]) for m, p in rpc] == [
            ("turn/steer", "use the v2 api", "turn-1")]
        assert child._pending_steer is None  # not parked in the Hermes queue the native turn never reads
        receipt = _entries(_Root())["codex-1"]["steers"][-1]
        assert receipt["status"] == "delivered" and receipt["delivered_at"] >= receipt["queued_at"]
    finally:
        _unregister_subagent("codex-1")


def test_codex_child_with_no_active_turn_queues_the_steer_and_reports_a_miss():
    child, rpc = _codex_child(turn_id=None)
    _register("codex-2", child)
    try:
        assert steer_subagent("codex-2", "between turns") is True
        assert rpc == [] and child._pending_steer == "between turns"
        assert _close_subagent_steering("codex-2", child) == "between turns"
        assert _entries(_Root())["codex-2"]["steers"][-1]["status"] == "missed"
    finally:
        _unregister_subagent("codex-2")
