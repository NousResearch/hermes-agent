"""Declared MCP tool costs and the per-chat spend limit (``tools.mcp_tool_costs``).

Invariants: a costed call's prompt names the cost and the chat's running total; only dispatched
calls count; a call that would cross ``spend_limit`` asks even on a ``trust: full`` server and is
refused unattended without dispatching; chats and profiles never share a tally, a chat keeps its
total across a session-id rotation and each cron run has its own; parallel over-limit calls are asked
one at a time; malformed declarations fail with a named error (package and config.yaml) while unknown
keys only warn.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool, mcp_tool_costs
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_registration as registration
from tools.approval_context import reset_current_session_key, set_current_session_key

COSTED = {"trust": "full", "tool_costs": {"paid_render": 40, "batch": {"credits": [5, 20], "note": "per item"}},
          "spend_limit": 100}


def _run_on_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _run():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_run())
    finally:
        loop.close()


@pytest.fixture
def session():
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], isError=False, structuredContent=None)
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=result)
    with patch.dict(mcp_tool._servers, {"srv": SimpleNamespace(session=session, _rpc_lock=None)}), \
         patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=_run_on_loop), \
         patch.dict(mcp_tool._server_error_counts, {}, clear=True), \
         patch.dict(mcp_tool._server_trust_levels, {}, clear=True), \
         patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True), \
         patch.dict(mcp_tool._server_cost_policies, {}, clear=True), \
         patch.dict(mcp_tool_costs._spent, {}, clear=True), \
         patch.dict(mcp_tool_costs._ask_locks, {}, clear=True), \
         patch.object(mcp_tool_costs, "_unscoped_moved_to", None):
        yield session


def _register(config, name="srv", tools=("paid_render", "batch", "free_status")):
    registration._record_tool_trust_metadata(name, config, [SimpleNamespace(name=t, annotations=None) for t in tools])


def _call(tool, chat="chat-a", answer="accept"):
    asked = []
    token = set_current_session_key(chat)
    try:
        with patch("tools.approval_prompt.request_elicitation_consent",
                   side_effect=lambda msg, desc, **kw: asked.append(msg) or answer):
            raw = handlers._make_tool_handler("srv", tool, 30.0)({})
    finally:
        reset_current_session_key(token)
    return json.loads(raw), asked


def test_untrusted_prompt_names_cost_and_running_total_and_declines_add_nothing(session):
    _register(dict(COSTED, trust="untrusted"))
    _, asked = _call("paid_render", answer="decline")
    assert "Costs about 40 credits." in asked[0] and "used about" not in asked[0]
    result, asked = _call("paid_render")  # the declined call added nothing
    assert result == {"result": "ok"} and "used about" not in asked[0]
    _, asked = _call("batch")
    assert "Costs 5-20 credits (per item). This chat has used about 40 credits on 'srv' (limit 100)." in asked[0]
    # An undeclared tool on a costed server keeps main's exact prompt.
    _, asked = _call("free_status")
    assert asked == ["MCP tool 'free_status' on UNTRUSTED server 'srv' wants to run. This tool is write-capable "
                     "(no readOnlyHint=true annotation) and may modify external state."]
    assert session.call_tool.await_count == 3


def test_full_trust_asks_only_past_the_limit_and_unattended_refuses_without_dispatch(session):
    from gateway.session_context import clear_session_vars, set_session_vars
    _register(COSTED)
    for _ in range(2):  # 80 of 100: trust: full stays silent below the limit
        result, asked = _call("paid_render")
        assert result == {"result": "ok"} and asked == []
    _, asked = _call("free_status")
    assert asked == []
    result, asked = _call("paid_render", answer="decline")  # 80 + 40 > 100
    assert "would take this chat past its spend limit" in asked[0]
    assert "This chat has used about 80 credits on 'srv' (limit 100)." in asked[0]
    assert "spend limit is 100 credits" in result["error"] and "Do not retry" in result["error"]
    result, asked = _call("paid_render")  # approved past the limit: runs, and the next one asks again
    assert result == {"result": "ok"}
    _, asked = _call("batch")
    assert "used about 120 credits" in asked[0]
    dispatched = session.call_tool.await_count
    tokens = set_session_vars(cron_session="1")  # cron: nobody can answer, the real consent path declines
    token = set_current_session_key("chat-a")
    try:
        result = json.loads(handlers._make_tool_handler("srv", "paid_render", 30.0)({}))
    finally:
        reset_current_session_key(token)
        clear_session_vars(tokens)
    assert "used about 140 credits" in result["error"] and session.call_tool.await_count == dispatched


def test_chats_and_profiles_never_share_a_tally(session):
    _register(COSTED)
    for _ in range(2):
        _call("paid_render", chat="chat-a")
    assert _call("paid_render", chat="chat-a")[1]  # chat-a is at its limit
    assert _call("paid_render", chat="chat-b")[1] == []  # chat-b is not
    # Two profiles in one multiplexed process: the policy key carries the profile scope.
    policy = mcp_tool._server_cost_policies["srv"]
    mcp_tool._server_cost_policies[("profile-a", "srv")] = mcp_tool._server_cost_policies[("profile-b", "srv")] = policy
    a = mcp_tool_costs.pending_charge(("profile-a", "srv"), "srv", "paid_render", "chat-a")
    b = mcp_tool_costs.pending_charge(("profile-b", "srv"), "srv", "paid_render", "chat-a")
    assert a is not None and b is not None
    a.book()
    assert a.spent() == 40 and b.spent() == 0
    # A process that starts multiplexing moves bare-name policies under (scope, name); totals follow,
    # and so does a charge built before the move but booked after it.
    in_flight = mcp_tool_costs.pending_charge("srv", "srv", "paid_render", "chat-a")
    assert in_flight is not None
    mcp_tool_costs.rekey_unscoped("launch")
    in_flight.book()
    assert mcp_tool_costs._spent[(("launch", "srv"), "chat-a")] == 160  # 40 + 40 + the approved 40 + in flight


def test_declarations_validate_translate_and_survive_a_native_trust_override(tmp_path, caplog):
    from hermes_cli.agent_plugins import AgentPluginError, load_agent_plugin
    from tools.mcp_tool_config import _filter_suspicious_mcp_servers, _load_mcp_config, _portable_mcp_servers

    def package(server_decl):
        root = tmp_path / "pkg"
        root.mkdir(exist_ok=True)
        (root / "plugin.json").write_text(json.dumps({
            "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "pkg",
            "extensions": {"com.nousresearch.hermes": {"servers": {"srv": server_decl}}}}))
        (root / "mcp.json").write_text(json.dumps({
            "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
            "mcpServers": {"srv": {"type": "stdio", "command": "python3", "args": ["s.py"]}}}))
        return load_agent_plugin(root, tmp_path / "data")

    pkg = package({"trust": "untrusted", "costs": {"paid_render": [5, 20]}, "spend_limit": 50, "future": 1})
    assert pkg.mcp_servers["srv"]["tool_costs"] == {"paid_render": [5, 20]}
    assert pkg.mcp_servers["srv"]["spend_limit"] == 50
    assert any("future" in d.message for d in pkg.diagnostics)  # unknown keys still only warn
    with pytest.raises(AgentPluginError, match="cost of 'paid_render' must be a non-negative number"):
        package({"costs": {"paid_render": -1}})
    with caplog.at_level("ERROR"):
        kept = _filter_suspicious_mcp_servers({"bad": {"command": "x", "spend_limit": "lots"}, "ok": {"command": "x"}})
    assert list(kept) == ["ok"] and "spend_limit must be a non-negative number" in caplog.text

    # A same-name native entry widening trust keeps the package's costs; its own spend_limit (0 = off) wins.
    native = {"srv": {"command": "python3", "trust": "full", "spend_limit": 0}}
    manager = SimpleNamespace(get_portable_mcp_servers=lambda: dict(pkg.mcp_servers))
    with patch("hermes_cli.plugins.discover_plugins"), patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        _portable_mcp_servers(native)
        assert native["srv"]["trust"] == "full" and native["srv"]["spend_limit"] == 0
        assert native["srv"]["tool_costs"] == {"paid_render": [5, 20]}
        # A native entry rejected for a malformed budget is not revived by the package's looser one.
        with patch("hermes_cli.config.load_config", return_value={"mcp_servers": {"srv": {"command": "x", "spend_limit": -5}}}):
            assert "srv" not in _load_mcp_config()


def test_a_chat_keeps_its_total_across_a_rotation_and_each_cron_run_has_its_own(session, caplog):
    from cron.scheduler import _CronRunScope
    from gateway.session_context import clear_session_vars, set_session_vars
    config = dict(COSTED, tool_costs={**COSTED["tool_costs"], "mcp_srv_typo": 3})
    _register(config)
    with caplog.at_level("WARNING", logger="tools.mcp_tool"):  # live discovery sees the full tool list
        registration._warn_unknown_cost_keys("srv", "srv", ["paid_render", "batch", "free_status"])
    assert "mcp_srv_typo" in caplog.text  # a key the server does not offer is named, never silently dropped
    for _ in range(2):
        _call("paid_render", chat="parent")
    # Compression (CLI, TUI/Desktop) and /branch continue the chat under a new id through this one hook.
    from tools.approval_yolo import transfer_session_yolo
    transfer_session_yolo("parent", "child")
    assert "used about 80 credits" in _call("paid_render", chat="child", answer="decline")[1][0]
    # Cron: the approval session key is "" for every job; each run binds its own tally.
    tokens = set_session_vars(cron_session="1")
    try:
        for job in ("job-a", "job-b", "job-c"):
            scope = _CronRunScope({}, job, "run-1")
            scope.enter()
            try:
                token = set_current_session_key("")
                try:
                    for _ in range(2):
                        assert json.loads(handlers._make_tool_handler("srv", "paid_render", 30.0)({})) == {"result": "ok"}
                finally:
                    reset_current_session_key(token)
            finally:
                scope.exit()
    finally:
        clear_session_vars(tokens)


def test_parallel_over_limit_calls_are_asked_one_at_a_time(session):
    import threading
    _register(COSTED)
    for _ in range(2):
        _call("paid_render")  # 80 of 100: the next two parallel calls both cross the limit
    open_prompts, peak, prompts, gate = [0], [0], [], threading.Barrier(2, timeout=2)

    def consent(msg, desc, **kw):
        open_prompts[0] += 1
        peak[0] = max(peak[0], open_prompts[0])
        prompts.append(msg)
        try:
            gate.wait()  # unserialized prompts would meet here; serialized ones time out alone
        except threading.BrokenBarrierError:
            pass
        open_prompts[0] -= 1
        return "accept"

    def one():
        token = set_current_session_key("chat-a")
        try:
            handlers._make_tool_handler("srv", "paid_render", 30.0)({})
        finally:
            reset_current_session_key(token)

    with patch("tools.approval_prompt.request_elicitation_consent", side_effect=consent):
        threads = [threading.Thread(target=one) for _ in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(10)
    assert peak[0] == 1 and len(prompts) == 2
    assert {"used about 80 credits" in p for p in prompts} == {True, False}  # the second saw the first's booking
