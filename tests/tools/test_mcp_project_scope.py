"""Stdio MCP servers run in the calling session's project, not the gateway's launch dir.

#151827 (port of #63759): a stdio connection is discovered once per process, and every session
shared its one child, spawned wherever the MCP loop resolved a cwd. A project-sensitive server
(Hindsight picks its memory bank from ``${PWD}``) answered every project with the launch dir's.

The probe is a REAL stdio MCP server that reports the cwd it was spawned in and its pid, so
nothing here can pass against a mocked transport.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from agent.runtime_cwd import clear_session_cwd, set_session_cwd

_PROBE = '''\
import json, os, sys, time
START = os.path.realpath(os.getcwd())
for line in sys.stdin:
    r = json.loads(line)
    method, ident = r.get("method"), r.get("id")
    if ident is None:
        continue
    if method == "initialize":
        result = {"protocolVersion": r["params"]["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "cwdprobe", "version": "1"}}
    elif method == "tools/list":
        obj = {"type": "object", "properties": {"seconds": {"type": "number"}}}
        result = {"tools": [{"name": "whereami", "description": "spawn cwd and pid", "inputSchema": obj},
                            {"name": "slow", "description": "sleep, then whereami", "inputSchema": obj}]}
    elif method == "tools/call":
        time.sleep(float((r["params"].get("arguments") or {}).get("seconds") or 0))
        result = {"content": [{"type": "text", "text": f"{START}|{os.getpid()}"}]}
    elif method == "events/list":
        result = {"events": [{"name": f"pid-{os.getpid()}"}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}), flush=True)
'''


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("mcp_project_scope")
    (root / "probe.py").write_text(_PROBE, encoding="utf-8")
    for name in ("a", "b", "c", "d"):
        (root / name).mkdir()
    yield Path(str(root))
    from tools.mcp_tool_lifecycle import _kill_orphaned_mcp_children, shutdown_mcp_servers
    shutdown_mcp_servers()
    _kill_orphaned_mcp_children(include_active=True)


def _register(root: Path, name: str) -> None:
    from tools import mcp_tool
    from tools.mcp_tool_discovery import register_mcp_servers
    register_mcp_servers({name: {"command": sys.executable, "args": [str(root / "probe.py")],
                                          "connect_timeout": 30, "timeout": 30}})
    assert name in mcp_tool._servers, mcp_tool._server_connect_errors.get(name)


@pytest.fixture
def probe(root):
    _register(root, "cwdprobe")
    yield type("Probe", (), {d: str(root / d) for d in "abcd"})
    clear_session_cwd()


def _call(name: str = "cwdprobe", tool: str = "whereami", cwd: str | None = None, **args) -> tuple[str, int]:
    from tools.mcp_tool_handlers import _make_tool_handler
    if cwd is not None:
        set_session_cwd(cwd)
    try:
        payload = _make_tool_handler(name, tool, 30)(args)
    finally:
        clear_session_cwd()
    text = json.loads(payload)
    assert "error" not in text, text
    spawn_cwd, pid = str(text["result"]).split("|")
    return spawn_cwd, int(pid)


def _clones(name: str = "cwdprobe") -> dict:
    from tools import mcp_tool_project_scope as scope
    return {k[1]: s for k, s in scope._scoped_servers.items() if k[0] == name}


def test_each_project_gets_its_own_process(probe):
    a1, b, a2 = _call(cwd=probe.a), _call(cwd=probe.b), _call(cwd=probe.a)

    assert (a1[0], b[0], a2[0]) == (probe.a, probe.b, probe.a)
    assert a1[1] != b[1], "two projects shared one stdio child"
    assert a1[1] == a2[1], "a project respawned its child on every call"


def test_concurrent_sessions_stay_isolated(probe):
    results, errors = {}, []
    barrier = threading.Barrier(2)

    def run(label, cwd):
        try:
            barrier.wait(timeout=30)
            results[label] = _call(cwd=cwd)
        except BaseException as exc:  # re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(k, getattr(probe, k))) for k in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    assert results["a"][0] == probe.a and results["b"][0] == probe.b
    assert results["a"][1] != results["b"][1]


def test_a_session_in_the_baseline_dir_never_clones(root, monkeypatch):
    """The gateway case: TERMINAL_CWD is not the process cwd. The baseline spawns there, and a
    session resolving to it must reuse it, or every session pays a duplicate child."""
    from tools import mcp_tool
    monkeypatch.setenv("TERMINAL_CWD", str(root / "a"))
    _register(root, "cwdprobe_baseline")

    spawn_cwd, pid = _call("cwdprobe_baseline")

    assert spawn_cwd == str(root / "a")
    assert not _clones("cwdprobe_baseline")
    assert mcp_tool._servers["cwdprobe_baseline"]._stdio_child_pids == {pid}


def test_clones_never_touch_the_tool_list(probe):
    from tools.registry import registry
    before = sorted(registry.get_all_tool_names())

    _call(cwd=probe.c)
    clone = _clones()[probe.c]

    assert clone._registered_tool_names == []
    assert sorted(registry.get_all_tool_names()) == before


def test_url_transport_is_never_cloned(probe):
    from tools import mcp_tool
    from tools.mcp_tool_discovery import _get_connected_server_for_call
    url_server = mcp_tool.MCPServerTask("fake-url")
    url_server._config = {"url": "https://example.invalid/mcp"}
    url_server.session = object()
    with mcp_tool._lock:
        mcp_tool._servers["fake-url"] = url_server
    try:
        for cwd in (probe.a, probe.b):
            set_session_cwd(cwd)
            assert _get_connected_server_for_call("fake-url") is url_server
    finally:
        clear_session_cwd()
        with mcp_tool._lock:
            mcp_tool._servers.pop("fake-url", None)
    assert not _clones("fake-url")


def test_idle_clones_past_the_cap_are_reaped_and_reconnect(probe, monkeypatch):
    from tests.waiting import wait_until
    from tools import mcp_tool_project_scope as scope
    monkeypatch.setattr(scope, "_MAX_SCOPED_TRANSPORTS_PER_SERVER", 1)
    _call(cwd=probe.c)
    old = _clones()[probe.c]

    assert _call(cwd=probe.d)[0] == probe.d

    assert list(_clones()) == [probe.d], "idle project children survived past the cap"
    assert not [k for k in scope._scoped_connect_locks if k[1] == probe.c]
    # The victim's shutdown is fire-and-forget on the MCP loop (review finding 6): it is no
    # longer complete when the evicting call returns — only scheduled.
    wait_until(lambda: old.session is None, timeout=15,
               what="the evicted clone's scheduled shutdown to finish")
    assert _call(cwd=probe.c)[0] == probe.c, "an evicted project did not reconnect"


def test_a_clone_in_use_is_never_evicted(probe, monkeypatch):
    from tools import mcp_tool_project_scope as scope
    monkeypatch.setattr(scope, "_MAX_SCOPED_TRANSPORTS_PER_SERVER", 1)
    _call(cwd=probe.a)
    busy = _clones()[probe.a]
    result = {}
    worker = threading.Thread(target=lambda: result.update(a=_call(tool="slow", cwd=probe.a, seconds=2)))
    worker.start()
    deadline = time.monotonic() + 10
    while not getattr(busy, "_scoped_leases", 0) and time.monotonic() < deadline:
        time.sleep(0.01)

    _call(cwd=probe.b)  # over the cap while project a's call is in flight
    worker.join(timeout=30)

    assert result["a"][0] == probe.a
    assert busy.session is not None, "the in-flight project's child was evicted"
    assert busy._scoped_leases == 0, "the handler's lease was never released"


def test_reload_tears_down_a_servers_clones(root, probe):
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    _register(root, "cwdprobe_reload")
    _call("cwdprobe_reload", cwd=probe.b)
    clone = _clones("cwdprobe_reload")[probe.b]

    shutdown_mcp_servers(scope=None, names={"cwdprobe_reload"})

    assert not _clones("cwdprobe_reload")
    assert clone._task.done() and clone.session is None
    assert _clones(), "pruning one server tore down another server's clones"


def test_a_clone_connecting_during_a_prune_is_discarded(root):
    """A clone whose connect is still in flight when its baseline is torn down (reload, config
    prune) must never be inserted afterwards: it would be a live child with no baseline, which
    no teardown will ever reap or re-register (#151827 port, review finding 1)."""
    from tests.waiting import wait_until
    from tools import mcp_tool, mcp_tool_project_scope as scope
    from tools.mcp_tool_lifecycle import _kill_orphaned_mcp_children, shutdown_mcp_servers

    name = "cwdprobe_prune_race"
    slow, go = root / "slow", root / "go"
    slow.mkdir(exist_ok=True)
    # The clone child answers `initialize` only when the trigger file appears, so the prune can
    # deterministically land inside the connect: a watcher fires the trigger the moment the
    # baseline disappears from the registry, while the MCP loop is still draining.
    (root / "slow_probe.py").write_text(_PROBE.replace(
        'if method == "initialize":',
        'if method == "initialize":\n'
        f'        if os.path.basename(START) == "slow":\n'
        f'            for _ in range(4000):\n'
        f'                if os.path.exists({str(go)!r}):\n'
        f'                    break\n'
        f'                time.sleep(0.01)'),
        encoding="utf-8")
    from tools.mcp_tool_discovery import register_mcp_servers
    # A second server keeps the MCP loop alive across the prune (the real /reload-mcp shape):
    # with nothing else registered, the post-prune drain would cancel the in-flight connect
    # itself and the race could never fire.
    register_mcp_servers({"cwdprobe_prune_keeper": {"command": sys.executable, "args": [str(root / "probe.py")],
                                                    "connect_timeout": 30, "timeout": 30}})
    register_mcp_servers({name: {"command": sys.executable, "args": [str(root / "slow_probe.py")],
                                 "connect_timeout": 30, "timeout": 30}})
    assert name in mcp_tool._servers
    result: dict = {}

    def race() -> None:
        from tools.mcp_tool_handlers import _make_tool_handler
        set_session_cwd(str(slow))
        try:
            result["r"] = _make_tool_handler(name, "whereami", 30)({})
        finally:
            clear_session_cwd()

    def trigger_when_pruned() -> None:
        wait_until(lambda: name not in mcp_tool._servers, timeout=20,
                   what="the prune to remove the baseline")
        go.write_text("go")

    worker = threading.Thread(target=race)
    trigger = threading.Thread(target=trigger_when_pruned)
    worker.start()
    try:
        # The connect holds its single-flight lock while spawning the child: tear the baseline
        # down inside that window, before the clone can be inserted.
        connect_lock = wait_until(
            lambda: scope._scoped_connect_locks.get((name, os.path.realpath(str(slow)))),
            what="the clone connect to take flight")
        assert connect_lock is not None
        wait_until(lambda: connect_lock.locked(), what="the connect lock to be held")
        trigger.start()
        shutdown_mcp_servers(scope=None, names={name})
        worker.join(timeout=60)
        trigger.join(timeout=30)

        assert name not in mcp_tool._servers, "test premise: the prune removed the baseline"
        assert not _clones(name), "a clone connected during the prune survived it"
        assert "not connected" in str(result.get("r")), result.get("r")
        assert not [k for k in scope._scoped_connect_locks if k[0] == name], \
            "the discarded generation left its connect lock behind"
    finally:
        clear_session_cwd()
        shutdown_mcp_servers()
        _kill_orphaned_mcp_children(include_active=True)


def test_the_insert_gate_refuses_a_torn_down_baseline(root, monkeypatch):
    """Gate-level twin of the prune race: when the baseline is already gone, the gate refuses
    BEFORE any spawn (a queued peer never pays a child spawn for a dead generation), inserts
    nothing, and leaves no connect lock behind for the dead generation."""
    import os as _os
    from tools import mcp_tool, mcp_tool_project_scope as scope

    name = "cwdprobe_gate"
    project = root / "gateproj"
    project.mkdir(exist_ok=True)
    _register(root, name)
    baseline = mcp_tool._servers[name]

    spawns = []

    def _spy_connect(*args, **kwargs):
        spawns.append(args)
        return mcp_tool.MCPServerTask(name)  # a real task shell, never started

    monkeypatch.setattr(scope, "_connect_clone", _spy_connect)
    with mcp_tool._lock:
        mcp_tool._servers.pop(name)  # the prune, before the call

    set_session_cwd(str(project))
    try:
        got = scope._server_for_call(name, name, baseline)
    finally:
        clear_session_cwd()

    assert got is None, "a clone was handed out for a torn-down baseline"
    assert not _clones(name), "the orphaned clone was inserted anyway"
    assert not spawns, "a dead baseline was refused only after a wasted spawn"
    assert not scope._scoped_connect_locks.get((name, _os.path.realpath(str(project)))), \
        "the refused generation left its connect lock behind"


def test_a_discarded_clone_whose_shutdown_failed_is_handed_to_the_reaper(root, monkeypatch):
    """When the MCP loop died with the prune, the discarded clone's shutdown can never run; its
    child PIDs must land in the orphan ledger so the next idle sweep reaps them instead of
    leaking until the final include_active sweep (independent review, suggestion 1)."""
    import os as _os
    from tools import mcp_tool, mcp_tool_project_scope as scope
    from tools.mcp_tool_lifecycle import _orphan_stdio_pids, _orphan_stdio_pid_servers

    name = "cwdprobe_reaper"
    project = root / "reaperproj"
    project.mkdir(exist_ok=True)
    _register(root, name)
    baseline = mcp_tool._servers[name]

    discarded = mcp_tool.MCPServerTask(name)
    discarded._stdio_child_pids = {4711}

    def _connect_and_prune(*args, **kwargs):
        with mcp_tool._lock:
            mcp_tool._servers.pop(name)  # the prune lands DURING the connect, after the pre-spawn gate
        return discarded

    monkeypatch.setattr(scope, "_connect_clone", _connect_and_prune)

    def _dead_loop(*args, **kwargs):
        raise RuntimeError("MCP event loop is not running")

    monkeypatch.setattr(scope._loop, "_run_on_mcp_loop", _dead_loop)
    # _release_spawned_children re-hands only PIDs that are (or whose group is) still alive;
    # 4711 is a stand-in, so fake the liveness probe it consults.
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: pid == 4711)
    try:
        set_session_cwd(str(project))
        try:
            assert scope._server_for_call(name, name, baseline) is None
        finally:
            clear_session_cwd()

        assert 4711 in _orphan_stdio_pids, "the discarded child leaked out of the reaper's reach"
        assert _orphan_stdio_pid_servers.get(4711) == name
    finally:
        with mcp_tool._lock:
            _orphan_stdio_pids.discard(4711)
            _orphan_stdio_pid_servers.pop(4711, None)


def test_event_utility_handlers_serve_the_baseline_not_the_callers_clone(probe):
    """The events utilities manage webhook routes keyed by server NAME, shared by every project:
    each clone resubscribing them would deliver the same event several times, and a clone child's
    terminal refusal would delete the route for every project. They always talk to the baseline
    (port review finding 2)."""
    from tools.mcp_tool_events import _make_list_event_types_handler
    _, clone_pid = _call(cwd=probe.a)
    _, baseline_pid = _call()
    assert clone_pid != baseline_pid
    set_session_cwd(probe.a)
    try:
        payload = _make_list_event_types_handler("cwdprobe", 30)({})
    finally:
        clear_session_cwd()
    listing = json.loads(payload)
    assert "error" not in listing, listing
    assert int(listing["events"][0]["name"].removeprefix("pid-")) == baseline_pid, \
        "the events handler talked to the caller's project clone"


def test_two_connect_generations_never_overwrite_each_others_clone(root, monkeypatch):
    """Two concurrent connects can hold different single-flight locks for one clone key when a
    teardown pops the lock entry under a still-outstanding seat. The insert must be first-wins
    under the registry lock and the duplicate child discarded off-lock: a plain overwrite would
    strand a live child no ledger will ever reap (#151827 port, review finding 4)."""
    import os as _os
    from tests.waiting import wait_until
    from tools import mcp_tool, mcp_tool_project_scope as scope
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    name = "cwdprobe_insert_race"
    project = root / "insertproj"
    project.mkdir(exist_ok=True)
    _register(root, name)
    baseline = mcp_tool._servers[name]
    cwd = _os.path.realpath(str(project))
    clone_key = (name, cwd)

    winner, loser = mcp_tool.MCPServerTask(name), mcp_tool.MCPServerTask(name)
    arrived, go = [], [threading.Event(), threading.Event()]

    def _gated_connect(server_name, baseline_, cwd_):
        with mcp_tool._lock:
            slot = len(arrived)
            arrived.append(slot)
        go[slot].wait(timeout=30)
        return (loser, winner)[slot]  # the first generation loses; the second wins the insert

    monkeypatch.setattr(scope, "_connect_clone", _gated_connect)
    results: dict = {}

    def _call_thread(tag):
        set_session_cwd(str(project))
        try:
            results[tag] = scope._server_for_call(name, name, baseline)
        finally:
            clear_session_cwd()

    first = threading.Thread(target=_call_thread, args=("first",))
    first.start()
    wait_until(lambda: len(arrived) == 1, timeout=10, what="the first connect to take flight")
    with mcp_tool._lock:  # the teardown race: the lock entry goes while the seat is outstanding
        scope._scoped_connect_locks.pop(clone_key, None)

    second = threading.Thread(target=_call_thread, args=("second",))
    second.start()
    wait_until(lambda: len(arrived) == 2, timeout=10, what="the second generation to connect")
    try:
        go[1].set()  # the second generation wins the insert
        wait_until(lambda: scope._scoped_servers.get(clone_key) is winner,
                   timeout=10, what="the winning clone to be installed")
        go[0].set()  # the first generation lands on an occupied key
        first.join(timeout=30)
        second.join(timeout=30)

        assert results["second"] is winner
        assert results["first"] is None, "the duplicate connect handed its clone out anyway"
        assert scope._scoped_servers.get(clone_key) is winner, "the winner lost its seat to the loser"
        assert loser._shutdown_event.is_set(), "the losing connect's child was never discarded"
        assert not winner._shutdown_event.is_set(), "the winning clone was discarded instead"
    finally:
        go[0].set()
        go[1].set()
        first.join(timeout=30)
        second.join(timeout=30)
        shutdown_mcp_servers(scope=None, names={name})


def test_a_peer_queued_on_the_same_lock_reuses_the_installed_clone(root, monkeypatch):
    """Two concurrent calls for one project queue on the SAME single-flight lock. When the first
    installs its clone and releases, the peer must lease and return that installed clone - never
    discard it (a regression here gives the peer a spurious 'not connected', shuts down a clone
    later callers are handed from the fast path, and can kill one a handler still holds a lease
    on) and never overwrite it (#151827 port, review finding 4, same-lock peer path)."""
    import os as _os
    from tests.waiting import wait_until
    from tools import mcp_tool, mcp_tool_project_scope as scope
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    name = "cwdprobe_same_lock_peer"
    project = root / "peerproj"
    project.mkdir(exist_ok=True)
    _register(root, name)
    baseline = mcp_tool._servers[name]
    cwd = _os.path.realpath(str(project))
    clone_key = (name, cwd)

    clone_shell = mcp_tool.MCPServerTask(name)
    connect_go = threading.Event()

    def _gated_connect(server_name, baseline_, cwd_):
        connect_go.wait(timeout=30)
        return clone_shell

    monkeypatch.setattr(scope, "_connect_clone", _gated_connect)
    results: dict = {}

    def _call_thread(tag):
        set_session_cwd(str(project))
        try:
            results[tag] = scope._server_for_call(name, name, baseline)
        finally:
            clear_session_cwd()

    first = threading.Thread(target=_call_thread, args=("first",))
    first.start()
    wait_until(lambda: scope._scoped_connect_waiters.get(clone_key) == 1,
               timeout=10, what="the first connect to take its seat")
    second = threading.Thread(target=_call_thread, args=("second",))
    second.start()
    wait_until(lambda: scope._scoped_connect_waiters.get(clone_key) == 2,
               timeout=10, what="the peer to queue on the same lock")
    try:
        connect_go.set()  # the first connect installs its clone; the peer follows on the same lock
        first.join(timeout=30)
        second.join(timeout=30)

        assert results["first"] is clone_shell and results["second"] is clone_shell, \
            f"the same-lock peer did not reuse the installed clone: {results}"
        assert scope._scoped_servers.get(clone_key) is clone_shell, "the installed clone was dropped"
        assert not clone_shell._shutdown_event.is_set(), "the installed clone was discarded"
        assert clone_shell._scoped_leases == 0, "the peer's lease was never released"
    finally:
        connect_go.set()
        first.join(timeout=30)
        second.join(timeout=30)
        shutdown_mcp_servers(scope=None, names={name})


def _clear_project_ledgers(keys) -> None:
    """Clear the per-project breaker and connect-cooldown ledgers these failing-clone tests arm,
    so neither test's leftovers reach the next one (inert today — unique server names, per-file
    subprocesses — but the ledgers stay honest)."""
    from tools import mcp_tool, mcp_tool_project_scope as scope
    for key in keys:
        mcp_tool._server_error_counts.pop(key, None)
        mcp_tool._server_breaker_opened_at.pop(key, None)
        mcp_tool._server_errors_all_application.pop(key, None)
        scope._scoped_connect_failures.pop(key, None)
        scope._scoped_connect_retry_after.pop(key, None)


def test_a_failing_projects_clones_never_trip_the_baseline_breaker(root, probe, monkeypatch):
    """Three failed clone connects in one project open THAT project's breaker only - the breaker
    bookkeeping is keyed on the clone (design PR63759-PORT-DESIGN.md section 4.0). On the un-keyed
    port every strike resolved the baseline connection, so one broken project paused the server in
    every project (review finding 5)."""
    import os as _os
    from tools import mcp_tool, mcp_tool_project_scope as scope

    name = "cwdprobe_breaker"
    _register(root, name)
    monkeypatch.setattr(scope, "_connect_clone", lambda *a, **k: None)  # every clone spawn fails

    def _failing_call(cwd: str) -> dict:
        from tools.mcp_tool_handlers import _make_tool_handler
        set_session_cwd(cwd)
        try:
            return json.loads(_make_tool_handler(name, "whereami", 30)({}))
        finally:
            clear_session_cwd()

    for _ in range(3):
        assert "error" in _failing_call(probe.a)
    baseline_key = name
    a_key = (name, _os.path.realpath(probe.a))
    b_key = (name, _os.path.realpath(probe.b))
    try:
        assert mcp_tool._server_error_counts.get(a_key) == 3, \
            f"the failing project's strikes did not land on its clone: {mcp_tool._server_error_counts}"
        assert mcp_tool._server_breaker_opened_at.get(a_key) is not None, "the project breaker never opened"
        assert not mcp_tool._server_error_counts.get(baseline_key), \
            "one project's failing clones opened the baseline's breaker for every project"

        # b's call still fails (every spawn is stubbed off); the contract is that its strike lands
        # on b's own clone and its breaker check stays closed, not that b is served.
        assert "error" in _failing_call(probe.b)
        from tools.mcp_tool_handlers import _check_circuit_breaker
        set_session_cwd(probe.b)
        try:
            assert _check_circuit_breaker(name) is None, "project b was paused by project a's breaker"
        finally:
            clear_session_cwd()
        assert not mcp_tool._server_error_counts.get(baseline_key)
    finally:
        _clear_project_ledgers((baseline_key, a_key, b_key))


def test_a_failed_clone_connect_arms_a_retry_cooldown(root, probe, monkeypatch):
    """A failed clone connect arms a geometric cooldown on the clone key: a broken project must
    not pay a full child spawn (up to connect_timeout+5) on every call while its waiters queue
    behind it (review finding 5)."""
    from tools import mcp_tool_project_scope as scope

    name = "cwdprobe_cooldown"
    _register(root, name)
    spawns = []
    monkeypatch.setattr(scope, "_connect_clone", lambda *a, **k: spawns.append(1))  # a failed clone connect

    def _failing_call(cwd: str) -> dict:
        from tools.mcp_tool_handlers import _make_tool_handler
        set_session_cwd(cwd)
        try:
            return json.loads(_make_tool_handler(name, "whereami", 30)({}))
        finally:
            clear_session_cwd()

    try:
        for _ in range(3):
            assert "error" in _failing_call(probe.a)
        assert len(spawns) == 1, \
            f"the connect cooldown did not gate the spawn retries: {len(spawns)} spawns for 3 calls"
    finally:
        import os as _os
        _clear_project_ledgers(((name, _os.path.realpath(probe.a)),))


def test_eviction_never_shuts_a_victim_down_on_the_callers_thread(root, probe, monkeypatch):
    """A call that tips a server over the clone cap evicts the LRU victim. The eviction must not
    run the victim's shutdown on the CALLER's thread: the first call from a 9th project would
    block up to 15s on another project's teardown, and an interrupt landing before the shutdown
    starts would leave the victim popped with its child alive (review finding 6). The shutdown is
    scheduled fire-and-forget on the MCP loop instead."""
    from tests.waiting import wait_until
    from tools import mcp_tool_project_scope as scope

    monkeypatch.setattr(scope, "_MAX_SCOPED_TRANSPORTS_PER_SERVER", 1)
    _call(cwd=probe.a)
    old = _clones()[probe.a]
    shutdown_go = threading.Event()
    real_run_on_loop = scope._loop._run_on_mcp_loop

    def _gated_run_on_loop(fn, *args, **kwargs):
        if getattr(fn, "__name__", "") != "shutdown":  # clone connects run for real
            return real_run_on_loop(fn, *args, **kwargs)
        return shutdown_go.wait(timeout=30)  # the victim's shutdown never starts while gated

    monkeypatch.setattr(scope._loop, "_run_on_mcp_loop", _gated_run_on_loop)
    result: dict = {}

    def _evicting_call():
        result["b"] = _call(cwd=probe.b)

    worker = threading.Thread(target=_evicting_call)
    worker.start()
    try:
        wait_until(lambda: result.get("b") is not None, timeout=5,
                   what="the evicting call to return without waiting on the victim's shutdown")
        assert list(_clones()) == [probe.b], "the eviction did not drop the victim from the ledger"
        wait_until(lambda: old.session is None, timeout=15,
                   what="the scheduled fire-and-forget shutdown to tear the victim down")
    finally:
        shutdown_go.set()  # release a base-tree worker parked inside the victim's shutdown
        worker.join(timeout=30)


def _reap_bare_keyed_clones() -> None:
    """Reap clones stranded under a bare baseline key (the rekey bug's signature), so a run of
    these tests never leaks its children into the next one."""
    from tools import mcp_tool, mcp_tool_loop
    from tools import mcp_tool_project_scope as pscope
    with mcp_tool._lock:
        strays = pscope.pop_clones({k[0] for k in pscope._scoped_servers if isinstance(k[0], str)})
    for clone in strays:
        try:
            mcp_tool_loop._run_on_mcp_loop(clone.shutdown, timeout=15)
        except Exception:
            pass


def test_a_runtime_multiplex_flip_rekeys_and_reuses_the_pre_flip_project_clones(root):
    """When the gateway flips to multiplexing at runtime, rekey_unscoped_connections moves the
    baseline under ``(scope, name)``; the project clones must follow their baseline, or the
    pre-flip clones (live children, up to the cap per server) turn invisible to lookup, eviction
    and scoped teardown, every project pays a fresh child, and only a wildcard shutdown reaps
    the old ones (#151827 port, review finding 3)."""
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool_scope import rekey_unscoped_connections
    from tools.registry import session_tool_scope

    name = "cwdprobe_rekey"
    _register(root, name)
    project = str(root / "a")
    spawn_cwd, pid = _call(name, cwd=project)
    from tools import mcp_tool_project_scope as pscope
    assert (name, os.path.realpath(project)) in pscope._scoped_servers, \
        "test premise: the pre-flip call created a project clone (else both legs use the baseline)"
    try:
        rekey_unscoped_connections("rekey-scope")
        with session_tool_scope("rekey-scope"):
            reuse_cwd, reuse_pid = _call(name, cwd=project)
    finally:
        shutdown_mcp_servers(scope="rekey-scope", names={name})
        _reap_bare_keyed_clones()
    assert (reuse_cwd, reuse_pid) == (spawn_cwd, pid), \
        "the multiplex flip stranded the project's child and spawned a second one"


def test_the_rekey_moves_every_project_scope_sidecar(root):
    """Gate-level twin: every project-scope sidecar follows the baseline's key. A connect still
    in flight under the old key keeps its own entries (its baseline check now fails, so it
    discards its clone and its release reaps them); a clone whose project a post-flip twin
    already serves is discarded instead of overwriting it."""
    from tools import mcp_tool, mcp_tool_project_scope as pscope
    from tools.mcp_tool_scope import rekey_unscoped_connections

    name = "cwdprobe_rekey_gate"
    with mcp_tool._lock:
        clone, twin = mcp_tool.MCPServerTask(name), mcp_tool.MCPServerTask(name)
        old_key, new_key = (name, "/p/a"), (("rekey-scope", name), "/p/a")
        pscope._scoped_servers[old_key] = clone
        pscope._scoped_servers[new_key] = twin  # a post-flip clone already serves this project
        busy_lock = pscope._scoped_connect_locks.setdefault((name, "/p/b"), threading.Lock())
        busy_lock.acquire()
        pscope._scoped_connect_waiters[(name, "/p/b")] = 1
        pscope._scoped_connect_locks.setdefault((name, "/p/c"), threading.Lock())
    try:
        rekey_unscoped_connections("rekey-scope")
        with mcp_tool._lock:
            assert pscope._scoped_servers.get(new_key) is clone, \
                "the live clone lost its seat to the idle twin"
            assert pscope._scoped_servers.get(old_key) is None \
                and not any(c is twin for c in pscope._scoped_servers.values()), \
                "a duplicate clone survived the rekey"
            assert (name, "/p/b") in pscope._scoped_connect_locks \
                and pscope._scoped_connect_waiters.get((name, "/p/b")) == 1, \
                "an in-flight connect lost its lock or waiter seat"
            assert (name, "/p/c") not in pscope._scoped_connect_locks, \
                "a settled dead lock survived the rekey"
    finally:
        busy_lock.release()
        from tools import mcp_tool_loop
        with mcp_tool._lock:  # the rekeyed clone was never a registered server's; reap it directly
            strays = pscope.pop_clones({("rekey-scope", name)})
        for stray in strays:
            try:
                mcp_tool_loop._run_on_mcp_loop(stray.shutdown, timeout=15)
            except Exception:
                pass
        _reap_bare_keyed_clones()


def test_an_in_flight_call_survives_a_rekey_and_stays_releasable(probe, monkeypatch):
    """Leases are counted on the clone object, so a multiplex flip mid-call neither evicts the
    busy clone nor strands its lease under the dead key: once the call returns, the clone is
    idle again and eviction reaps it like any other (review finding 3, lease half)."""
    from tools import mcp_tool_project_scope as pscope
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool_scope import rekey_unscoped_connections
    from tools.registry import session_tool_scope
    from tests.waiting import wait_until

    monkeypatch.setattr(pscope, "_MAX_SCOPED_TRANSPORTS_PER_SERVER", 1)
    _call(cwd=probe.a)
    busy = _clones()[probe.a]
    result: dict = {}
    worker = threading.Thread(target=lambda: result.update(a=_call(tool="slow", cwd=probe.a, seconds=2)))
    worker.start()
    wait_until(lambda: bool(getattr(pscope, "_scoped_active_calls", None))
               or bool(getattr(busy, "_scoped_leases", 0)),
               timeout=10, what="the in-flight call to take its lease")
    # The _scoped_active_calls leg is a red-on-base accommodation: pre-fix trees count leases in
    # that module dict, post-fix trees on the clone; the run must work under either accounting.
    try:
        rekey_unscoped_connections("rekey2-scope")  # the flip lands mid-call
        with session_tool_scope("rekey2-scope"):
            _call(cwd=probe.b)  # over the cap while the pre-flip call is still in flight
            worker.join(timeout=30)
            assert result["a"][0] == probe.a
            assert busy.session is not None, "the in-flight project's child was evicted mid-call"
            _call(cwd=probe.c)  # cap eviction: the returned clone must now count as idle
        assert not any(c is busy for c in pscope._scoped_servers.values()), \
            "the pre-flip clone was never evictable again (a stranded lease or a stranded key)"
    finally:
        worker.join(timeout=30)
        shutdown_mcp_servers(scope="rekey2-scope", names={"cwdprobe"})
        _reap_bare_keyed_clones()
