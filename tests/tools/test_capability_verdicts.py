"""A vanished MCP tool must not be reported as a missing capability.

When an MCP server parks (reconnect budget exhausted) or fails to connect,
``_deregister_tools()`` pulls its tools out of the registry. The model's
schema, however, is byte-stable for the life of a conversation — that is a
prompt-cache invariant, not an oversight — so it goes on calling them.

It used to get back ``{"error": "Unknown tool: mcp__asana__create_task"}``,
read that as proof the capability does not exist, tell the user Hermes cannot
do that thing, and never try again. A transport outage lasting seconds became
a permanent-looking loss of capability — and for OAuth servers, which had
effectively no self-recovery after parking, permanently permanent.

The verdict must answer a different question than "is this name in the
registry": WHICH of reconnecting / parked / in backoff / genuinely unknown is
true.
"""
from __future__ import annotations

import json

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


@pytest.fixture
def mcp(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    mcp_tool._known_mcp_tool_owners.clear()
    yield mcp_tool
    mcp_tool._known_mcp_tool_owners.clear()
    mcp_tool._servers.pop("asana", None)
    mcp_tool._server_connect_retry_after.pop("asana", None)
    mcp_tool._server_connect_failures.pop("asana", None)


def _reg():
    from tools import mcp_tool_registration

    return mcp_tool_registration


class _Server:
    def __init__(self, parked=False, session=None, registered=()):
        self._was_parked = parked
        self.session = session
        self._registered_tool_names = list(registered)


# ---------------------------------------------------------------------------
# The registry hook
# ---------------------------------------------------------------------------


def test_a_genuinely_unknown_name_still_says_unknown():
    """The precise verdict must not blur the case it is not about."""
    from tools.registry import registry

    out = json.loads(registry.dispatch("no_such_tool_anywhere", {}))
    assert out["error"] == "Unknown tool: no_such_tool_anywhere"


def test_the_mcp_resolver_is_registered_at_import():
    import tools.mcp_tool as mcp_tool  # noqa: F401
    from tools.registry import registry

    assert mcp_tool._describe_unknown_mcp_tool in registry._unknown_tool_resolvers


def test_a_resolver_that_raises_falls_back_to_the_plain_verdict():
    """A broken resolver must never break dispatch."""
    from tools.registry import registry

    def _boom(name, scope=None):
        raise RuntimeError("resolver exploded")

    registry.register_unknown_tool_resolver(_boom)
    try:
        out = json.loads(registry.dispatch("still_unknown_xyz", {}))
        assert out["error"] == "Unknown tool: still_unknown_xyz"
    finally:
        registry._unknown_tool_resolvers.remove(_boom)


def test_a_resolver_returning_none_defers_to_the_next():
    from tools.registry import registry

    def _defer(name, scope=None):
        return None

    def _answer(name, scope=None):
        return "precise verdict for " + name

    registry.register_unknown_tool_resolver(_defer)
    registry.register_unknown_tool_resolver(_answer)
    try:
        out = json.loads(registry.dispatch("deferred_tool_xyz", {}))
        assert out["error"] == "precise verdict for deferred_tool_xyz"
    finally:
        registry._unknown_tool_resolvers.remove(_defer)
        registry._unknown_tool_resolvers.remove(_answer)


def test_resolvers_are_not_registered_twice():
    from tools.registry import registry

    def _r(name, scope=None):
        return None

    registry.register_unknown_tool_resolver(_r)
    registry.register_unknown_tool_resolver(_r)
    try:
        assert registry._unknown_tool_resolvers.count(_r) == 1
    finally:
        registry._unknown_tool_resolvers.remove(_r)


# ---------------------------------------------------------------------------
# Provenance survives deregistration
# ---------------------------------------------------------------------------


def test_provenance_outlives_deregistration(mcp):
    """The whole verdict rests on still knowing who owned the name."""
    _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana")
    _reg()._forget_mcp_tool_server("mcp__asana__create_task")

    # The LIVE map is cleared...
    assert "mcp__asana__create_task" not in mcp._mcp_tool_server_names
    # ...but the association we need is kept.
    assert mcp._known_mcp_tool_owners[(None, "mcp__asana__create_task")] == "asana"


def test_the_owner_map_is_bounded(mcp):
    """A server with churning dynamic discovery must not grow it forever."""
    for i in range(mcp._MAX_REMEMBERED_TOOL_OWNERS + 50):
        _reg()._track_mcp_tool_server(f"mcp__srv__tool_{i}", "srv")
    assert len(mcp._known_mcp_tool_owners) == mcp._MAX_REMEMBERED_TOOL_OWNERS


# ---------------------------------------------------------------------------
# The verdicts themselves
# ---------------------------------------------------------------------------


def _verdict(mcp, tool="mcp__asana__create_task"):
    return mcp._describe_unknown_mcp_tool(tool)


def test_an_unrelated_name_gets_no_mcp_verdict(mcp):
    assert _verdict(mcp, "read_file") is None


class TestParkedServer:
    @pytest.fixture(autouse=True)
    def _setup(self, mcp):
        _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana")
        _reg()._forget_mcp_tool_server("mcp__asana__create_task")
        mcp._servers["asana"] = _Server(parked=True)

    def test_it_names_the_server_and_the_state(self, mcp):
        v = _verdict(mcp)
        assert "asana" in v
        assert "parked" in v.lower()

    def test_it_gives_the_self_probe_interval(self, mcp):
        assert str(mcp._PARKED_RETRY_INTERVAL) in _verdict(mcp)

    def test_it_forbids_the_wrong_conclusion(self, mcp):
        """The regression, stated as plainly as the model needs it."""
        v = _verdict(mcp)
        assert "NOT a missing capability" in v
        assert "do NOT tell" in v

    def test_it_never_says_unknown_tool(self, mcp):
        assert "Unknown tool" not in _verdict(mcp)


class TestReconnectingServer:
    def test_a_live_server_reads_as_reconnecting(self, mcp):
        _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana")
        mcp._servers["asana"] = _Server(parked=False)
        v = _verdict(mcp)
        assert "reconnect" in v.lower()
        assert "NOT a missing capability" in v
        # Bounded: retry ONCE, then tell the user — not an open-ended loop.
        assert "retry this call ONCE" in v
        assert "stop retrying" in v


class TestConnectBackoff:
    def test_it_reports_the_remaining_cooldown(self, mcp):
        import time

        _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana")
        mcp._servers["asana"] = _Server(parked=False)
        mcp._server_connect_retry_after["asana"] = time.monotonic() + 42.0
        v = _verdict(mcp)
        assert "backoff" in v.lower()
        assert "s" in v and ("~42" in v or "~41" in v)
        assert "NOT a missing capability" in v


class TestServerGoneFromConfig:
    def test_it_points_at_the_config_instead_of_denying_the_capability(self, mcp):
        _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana")
        mcp._servers.pop("asana", None)
        v = _verdict(mcp)
        assert "hermes mcp list" in v
        assert "assuming the capability does not exist" in v
        assert "Unknown tool" not in v


# ---------------------------------------------------------------------------
# End-to-end through dispatch
# ---------------------------------------------------------------------------


def test_dispatch_returns_the_precise_verdict_for_a_parked_tool(mcp):
    """What the model actually receives when it calls a parked server's tool."""
    from tools.registry import registry

    _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana")
    _reg()._forget_mcp_tool_server("mcp__asana__create_task")
    mcp._servers["asana"] = _Server(parked=True)

    out = json.loads(registry.dispatch("mcp__asana__create_task", {}))
    assert "Unknown tool" not in out["error"]
    assert "asana" in out["error"]
    assert "NOT a missing capability" in out["error"]


def test_a_registered_tool_is_unaffected(mcp):
    """The resolver must only ever speak for names the registry lacks."""
    from tools.registry import registry, tool_result

    registry.register(
        name="mcp__asana__create_task",
        toolset="testing",
        schema={
            "name": "mcp__asana__create_task",
            "description": "test only",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kw: tool_result(ok=True),
    )
    _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana")
    mcp._servers["asana"] = _Server(parked=True)
    try:
        out = json.loads(registry.dispatch("mcp__asana__create_task", {}))
        assert out.get("ok") is True
    finally:
        registry._tools.pop("mcp__asana__create_task", None)


class TestConnectedServerNoLongerOffersTheTool:
    """The opposite failure: telling the model to wait for a live server.

    A tool removed, renamed, or filtered out of the config leaves its name in
    the conversation's byte-stable schema while the server itself is up and
    serving. Calling it "temporarily unavailable, retry" would send the model
    into a loop against a server that is already healthy and never coming
    back with that tool.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, mcp):
        _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana")
        _reg()._forget_mcp_tool_server("mcp__asana__create_task")
        mcp._servers["asana"] = _Server(
            parked=False,
            session=object(),
            registered=["mcp__asana__get_task", "mcp__asana__list_tasks"],
        )

    def test_it_says_the_tool_is_gone_not_that_the_server_is_down(self, mcp):
        v = _verdict(mcp)
        assert "no longer exists" in v
        assert "not an outage" in v

    def test_it_tells_the_model_to_stop_retrying(self, mcp):
        v = _verdict(mcp)
        assert "Do NOT retry it" in v
        assert "temporarily unavailable" not in v

    def test_it_reports_that_the_server_is_serving(self, mcp):
        assert "2 other tools" in _verdict(mcp)

    def test_a_connected_server_with_no_tools_still_reads_as_reconnecting(self, mcp):
        """Mid-reconnect the list is briefly empty — that IS an outage."""
        mcp._servers["asana"] = _Server(
            parked=False, session=object(), registered=[]
        )
        v = _verdict(mcp)
        assert "reconnect" in v.lower()
        assert "no longer exists" not in v


class TestUnderAProfileMultiplexer:
    """Connections are keyed ``(scope, name)`` when one process serves several profiles: the
    verdict must describe the CALLING profile's connection, not miss it and claim the server has
    no running connection."""

    def test_a_parked_scoped_server_is_found_through_the_callers_scope(self, mcp, monkeypatch):
        monkeypatch.setattr(mcp, "_mcp_registry_scope", lambda: "profile-b")
        _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana", "profile-b")
        _reg()._forget_mcp_tool_server("mcp__asana__create_task")
        mcp._servers[("profile-b", "asana")] = _Server(parked=True)
        try:
            v = _verdict(mcp)
            assert "parked" in v.lower()
            assert "hermes mcp list" not in v
        finally:
            mcp._servers.pop(("profile-b", "asana"), None)

    def test_another_profiles_connection_is_not_borrowed(self, mcp, monkeypatch):
        monkeypatch.setattr(mcp, "_mcp_registry_scope", lambda: "profile-b")
        _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana", "profile-b")
        mcp._servers[("profile-a", "asana")] = _Server(parked=True)
        try:
            assert "hermes mcp list" in _verdict(mcp)
        finally:
            mcp._servers.pop(("profile-a", "asana"), None)

    def test_the_dispatch_scope_wins_over_the_ambient_context(self, mcp, monkeypatch):
        """``registry.dispatch(scope=A)`` from a context bound to B must describe A's connection."""
        from tools.registry import registry

        monkeypatch.setattr(mcp, "_mcp_registry_scope", lambda: "profile-b")
        _reg()._track_mcp_tool_server("mcp__asana__create_task", "asana", "profile-a")
        mcp._servers[("profile-a", "asana")] = _Server(parked=True)
        try:
            out = json.loads(registry.dispatch("mcp__asana__create_task", {}, scope="profile-a"))
            assert "parked" in out["error"].lower()
        finally:
            mcp._servers.pop(("profile-a", "asana"), None)

    def test_another_profiles_tool_name_is_not_explained_to_this_one(self, mcp, monkeypatch):
        """Two profiles' servers can normalize to the same tool name: provenance is per scope."""
        monkeypatch.setattr(mcp, "_mcp_registry_scope", lambda: "profile-b")
        _reg()._track_mcp_tool_server("mcp__foo_bar__t", "foo-bar", "profile-a")
        assert _verdict(mcp, "mcp__foo_bar__t") is None

