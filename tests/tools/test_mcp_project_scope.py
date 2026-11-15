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
    from tools import mcp_tool_project_scope as scope
    monkeypatch.setattr(scope, "_MAX_SCOPED_TRANSPORTS_PER_SERVER", 1)
    _call(cwd=probe.c)
    old = _clones()[probe.c]

    assert _call(cwd=probe.d)[0] == probe.d

    assert list(_clones()) == [probe.d], "idle project children survived past the cap"
    assert not [k for k in scope._scoped_connect_locks if k[1] == probe.c]
    assert old._task.done() and old.session is None
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
    while not scope._scoped_active_calls and time.monotonic() < deadline:
        time.sleep(0.01)

    _call(cwd=probe.b)  # over the cap while project a's call is in flight
    worker.join(timeout=30)

    assert result["a"][0] == probe.a
    assert busy.session is not None, "the in-flight project's child was evicted"
    assert scope._scoped_active_calls == {}


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
    finally:
        clear_session_cwd()
        shutdown_mcp_servers()
        _kill_orphaned_mcp_children(include_active=True)


def test_the_insert_gate_refuses_a_torn_down_baseline(root, monkeypatch):
    """Gate-level twin of the prune race: the clone insert must verify the baseline is still the
    live connection, and a clone connected for a torn-down baseline is shut down, not handed out."""
    from tools import mcp_tool, mcp_tool_project_scope as scope

    name = "cwdprobe_gate"
    project = root / "gateproj"
    project.mkdir(exist_ok=True)
    _register(root, name)
    baseline = mcp_tool._servers[name]

    class _OrphanClone:
        def __init__(self):
            self.shutdown_called = threading.Event()

        async def shutdown(self):
            self.shutdown_called.set()

    orphan = _OrphanClone()
    monkeypatch.setattr(scope, "_connect_clone", lambda *args, **kwargs: orphan)
    with mcp_tool._lock:
        mcp_tool._servers.pop(name)  # the prune, between the lookup and the insert

    set_session_cwd(str(project))
    try:
        got = scope._server_for_call(name, name, baseline)
    finally:
        clear_session_cwd()

    assert got is None, "a clone was handed out for a torn-down baseline"
    assert not _clones(name), "the orphaned clone was inserted anyway"
    assert orphan.shutdown_called.is_set(), "the discarded clone was never shut down"
