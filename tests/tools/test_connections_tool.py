"""Behavior tests for manage_connections.

DI-callable idiom: a fake client injected through manage_connections'
seams; no module mocks, no network.
"""

import json
import time
from unittest.mock import patch

import pytest

import tools.connections_tool  # registers the tool
from tools.connections_tool import MANAGE_CONNECTIONS_SCHEMA, manage_connections


class FakeClient:
    def __init__(self):
        self.calls = []

    def list_connectors(self):
        self.calls.append(("list",))
        return [
            {"connector": "gmail", "enabled": True, "connected": False},
            {"connector": "linear", "enabled": True, "connected": True},
        ]

    def connections(self, connectors, *, reinitiate=False):
        self.calls.append(("connections", tuple(connectors), reinitiate))
        return {
            "results": [
                {
                    "connector": c,
                    "status": "initiated",
                    "connect_url": f"https://connect.example/{c}",
                    "instruction": f"finish authorizing {c} in the browser",
                    "reinitiated": reinitiate,
                }
                for c in connectors
            ],
            "summary": {"total": len(connectors), "initiated": len(connectors)},
        }


def test_status_lists_and_filters_connectors():
    client = FakeClient()
    out = json.loads(
        manage_connections(
            {"action": "status", "connectors": ["GMAIL"]},
            client_factory=lambda: client,
        )
    )
    assert out["connectors"] == [
        {"connector": "gmail", "enabled": True, "connected": False}
    ]


def test_connect_returns_link_and_instruction_once_per_session():
    client = FakeClient()
    seen = set()
    first = json.loads(
        manage_connections(
            {"action": "connect", "connectors": ["gmail"]},
            client_factory=lambda: client,
            seen_instructions=seen,
        )
    )
    entry = first["results"][0]
    assert entry["connect_url"] == "https://connect.example/gmail"
    assert "instruction" in entry

    second = json.loads(
        manage_connections(
            {"action": "connect", "connectors": ["gmail"]},
            client_factory=lambda: client,
            seen_instructions=seen,
        )
    )
    assert "instruction" not in second["results"][0]  # shown once per session
    assert ("connections", ("gmail",), False) in client.calls

    # A DIFFERENT session sharing the process still gets the guidance.
    other_session = json.loads(
        manage_connections(
            {"action": "connect", "connectors": ["gmail"]},
            client_factory=lambda: client,
            seen_instructions=seen,
            session_id="other-session",
        )
    )
    assert "instruction" in other_session["results"][0]


def test_reconnect_sets_reinitiate():
    client = FakeClient()
    manage_connections(
        {"action": "reconnect", "connectors": ["gmail"]},
        client_factory=lambda: client,
        seen_instructions=set(),
    )
    assert ("connections", ("gmail",), True) in client.calls


def test_connect_without_connectors_is_a_usage_error():
    out = json.loads(
        manage_connections({"action": "connect"}, client_factory=FakeClient)
    )
    assert "requires 'connectors'" in out["error"]


def test_disconnect_is_refused_before_any_gateway_call():
    # De-authentication is user-only: the tool rejects it up front and the
    # gateway never hears about it.
    client = FakeClient()
    out = json.loads(
        manage_connections(
            {"action": "disconnect", "connectors": ["gmail"]},
            client_factory=lambda: client,
        )
    )
    assert "error" in out
    assert client.calls == []


def test_gateway_failure_is_a_model_actionable_error():
    def exploding():
        raise RuntimeError("gateway on fire")

    out = json.loads(
        manage_connections({"action": "status"}, client_factory=exploding)
    )
    assert "connector gateway request failed" in out["error"]


def test_mcp_actions_are_not_this_tools_business():
    # Local MCP setup belongs to setup_mcp, which owns the desktop consent
    # callback. Folding those actions in here promised a flow this tool has no
    # way to reach, so they are rejected as unknown actions.
    out = json.loads(
        manage_connections({"action": "install", "server": "linear"})
    )
    assert "action must be one of" in out["error"]
    assert "install" not in MANAGE_CONNECTIONS_SCHEMA["parameters"]["properties"]["action"]["enum"]


# ---------------------------------------------------------------------------
# action "wait": the waiting happens inside the call, not in the model's head
# ---------------------------------------------------------------------------


class WaitClient(FakeClient):
    """Reports `connector` connected from the `flips_on`-th list call onward.

    `flips_on=None` never connects, which is the ordinary shape of a user who
    wandered off mid-authorization.
    """

    def __init__(self, connector="gmail", flips_on=None):
        super().__init__()
        self.connector = connector
        self.flips_on = flips_on
        self.polls = 0
        self.on_poll = None

    def list_connectors(self):
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        connected = self.flips_on is not None and self.polls >= self.flips_on
        return [{"connector": self.connector, "enabled": True, "connected": connected}]


@pytest.fixture
def no_sleep(monkeypatch):
    """Collect the wait slices instead of spending them, so tests run in ms."""
    slices = []
    monkeypatch.setattr(time, "sleep", lambda seconds: slices.append(seconds))
    return slices


def _aged(rendered):
    """Rewind every recorded stamp past the just-minted window.

    A real wait follows the connect across a turn boundary (a model round
    trip); these tests call the two back to back, so without the rewind every
    wait would hit the same-batch bounce instead of the path under test.
    """
    for slugs in rendered.values():
        for slug, stamp in list(slugs.items()):
            slugs[slug] = stamp - 60.0
    return rendered


def _wait(client, connectors=("gmail",), *, rendered=None, session_id=None, rewind=True, **extra):
    args = {"action": "wait", "connectors": list(connectors)}
    args.update(extra)
    if rendered is None:
        rendered = {str(session_id or ""): {c: time.monotonic() for c in connectors}}
    if rewind:
        _aged(rendered)
    return json.loads(
        manage_connections(
            args,
            client_factory=lambda: client,
            rendered_links=rendered,
            session_id=session_id,
        )
    )


def test_wait_returns_connected_when_the_gateway_flips_live(no_sleep):
    """The whole point: the link is shown, then the call absorbs the waiting.

    Goes through 'connect' first so the link-rendering bookkeeping wait relies
    on is exercised, not simulated.
    """
    client = WaitClient(flips_on=3)
    rendered = {}
    manage_connections(
        {"action": "connect", "connectors": ["gmail"]},
        client_factory=lambda: client,
        seen_instructions=set(),
        rendered_links=rendered,
    )

    out = _wait(client, rendered=rendered)
    assert out["status"] == "connected"
    assert out["pending"] == []
    assert out["connectors"] == [
        {"connector": "gmail", "enabled": True, "connected": True}
    ]
    assert client.polls == 3  # each poll is a live gateway read, none cached
    # Waits are taken in one-second slices so the interrupt flag stays answered.
    assert set(no_sleep) == {1.0}


def test_wait_timeout_lists_what_is_pending_and_denies_being_an_error(no_sleep):
    client = WaitClient(flips_on=None)
    out = _wait(client, timeout_seconds=20)

    assert out["status"] == "timeout"
    assert out["pending"] == ["gmail"]
    assert out["connectors"] == []
    assert "NOT an error" in out["note"]
    assert "ASK THE USER" in out["note"]
    # The three offers the model must put to the user.
    assert "keep waiting" in out["note"]
    assert "continue without" in out["note"]
    assert "fresh connect links" in out["note"]
    assert "timeout_note" not in out  # nothing was clamped
    assert client.polls == 5  # 20s of budget at a 5s cadence, the last gap partial


def test_wait_clamps_an_over_long_timeout_and_says_the_cap_was_applied(no_sleep):
    client = WaitClient(flips_on=None)
    out = _wait(client, timeout_seconds=600)

    assert out["status"] == "timeout"
    assert "180" in out["timeout_note"]
    assert "capped" in out["timeout_note"]
    assert client.polls == 37  # the cap, not the ask, bounded the loop


def test_wait_tolerates_transient_gateway_blips_but_not_a_dead_gateway(no_sleep):
    # One blip costs a poll, never the whole wait: the connection still
    # resolves when the gateway comes back. Three consecutive failures mean
    # the gateway is genuinely down — the wait ends as a NEVER-error timeout
    # that reports what the last good poll saw.
    flaky = WaitClient(flips_on=4)

    def blip_twice(n):
        if n in (2, 3):
            raise RuntimeError("gateway hiccup")

    flaky.on_poll = blip_twice
    out = _wait(flaky, timeout_seconds=180)
    assert out["status"] == "connected"
    assert flaky.polls == 4

    dead = WaitClient(flips_on=None)
    dead.on_poll = lambda n: (_ for _ in ()).throw(RuntimeError("gateway down"))
    out = _wait(dead, timeout_seconds=180)
    assert out["status"] == "timeout"
    assert "stopped answering" in out["note"]
    assert out["pending"] == ["gmail"]
    assert "NOT an error" in out["note"]
    assert dead.polls == 3  # gave up on the third consecutive failure


def test_wait_interrupted_mid_wait_reports_interrupted_not_an_error(no_sleep):
    from tools.interrupt import set_interrupt

    client = WaitClient(flips_on=None)
    client.on_poll = lambda n: set_interrupt(True)
    try:
        out = _wait(client, timeout_seconds=180)
    finally:
        set_interrupt(False)

    assert out["status"] == "interrupted"
    assert out["pending"] == ["gmail"]
    assert "NOT an error" in out["note"]
    # Stopped in the first slice of the first wait rather than polling on.
    assert client.polls == 1
    assert no_sleep == []


def test_wait_refuses_a_connector_whose_link_this_session_never_showed(no_sleep):
    """Structural anti-footgun: waiting for a link nobody rendered is a stall.

    Nothing is going to change, so the loop would burn its whole budget and
    then report a pending connector the user was never asked to authorize.
    """
    client = WaitClient(flips_on=1)
    out = _wait(client, ("gmail", "linear"), rendered={"": {"gmail": 1.0}})

    assert "wait refused" in out["error"]
    assert "linear" in out["error"]
    assert "connect" in out["error"]
    assert client.polls == 0  # refused before any gateway read


def test_wait_in_the_same_batch_as_connect_bounces_instead_of_blocking(no_sleep):
    """connect→wait in one assistant turn: the user has not seen the links.

    The bounce is a normal result, not an error — the model is told to send
    its message first and wait next turn. Zero polls, zero sleep.
    """
    client = WaitClient(flips_on=1)
    rendered = {}
    manage_connections(
        {"action": "connect", "connectors": ["gmail"]},
        client_factory=lambda: client,
        seen_instructions=set(),
        rendered_links=rendered,
    )

    out = _wait(client, rendered=rendered, rewind=False)
    assert out["status"] == "pending"
    assert out["pending"] == ["gmail"]
    assert "has not seen them" in out["note"]
    assert "next turn" in out["note"]
    assert client.polls == 0
    assert no_sleep == []


def test_wait_accepts_a_connector_that_was_already_connected(no_sleep):
    """connect on an already-live app mints no link; wait must still run.

    The refusal guard exists for connectors this session never addressed —
    an active one WAS addressed, and there is no link the user must see, so
    an immediate wait legitimately returns connected on the first poll.
    """

    class ActiveClient(WaitClient):
        def connections(self, connectors, *, reinitiate=False):
            self.calls.append(("connections", tuple(connectors), reinitiate))
            return {
                "results": [{"connector": c, "status": "active"} for c in connectors],
                "summary": {"total": len(connectors), "active": len(connectors)},
            }

    client = ActiveClient(flips_on=1)
    rendered = {}
    out = json.loads(
        manage_connections(
            {"action": "connect", "connectors": ["gmail"]},
            client_factory=lambda: client,
            seen_instructions=set(),
            rendered_links=rendered,
        )
    )
    assert out["results"][0]["status"] == "active"
    assert "connect_url" not in out["results"][0]
    assert "Already connected" in out["results"][0]["note"]

    # No rewind: even seconds after the connect, the wait runs (never_fresh).
    waited = _wait(client, rendered=rendered, rewind=False)
    assert waited["status"] == "connected"
    assert client.polls == 1


def test_wait_link_bookkeeping_is_per_session(no_sleep):
    """A link shown in session A does not license a wait in session B."""
    client = WaitClient(flips_on=1)
    rendered = {}
    manage_connections(
        {"action": "connect", "connectors": ["gmail"]},
        client_factory=lambda: client,
        seen_instructions=set(),
        rendered_links=rendered,
        session_id="session-a",
    )
    assert _wait(client, rendered=rendered, session_id="session-a")["status"] == (
        "connected"
    )
    other = _wait(client, rendered=rendered, session_id="session-b")
    assert "wait refused" in other["error"]


def test_wait_requires_connectors():
    out = json.loads(
        manage_connections({"action": "wait"}, client_factory=FakeClient)
    )
    assert "requires 'connectors'" in out["error"]


def test_wait_never_rides_a_parallel_batch():
    """A three-minute block must not hold a gathered batch's siblings hostage."""
    from agent.tool_dispatch_helpers import _NEVER_PARALLEL_TOOLS

    assert "manage_connections" in _NEVER_PARALLEL_TOOLS


# ---------------------------------------------------------------------------
# reachability: a registered tool nobody enables is a tool nobody can call
# ---------------------------------------------------------------------------


def _session_tool_names(enabled_toolsets, *, connectors, disabled_toolsets=None):
    """Tool names a session would actually receive, through the real assembly.

    Skips the tool_search step so the assertion is about NAME resolution and
    check_fn, not about how many MCP servers the developer running the suite
    happens to have configured.
    """
    from model_tools import _compute_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    with patch("tools.tool_gateway.config.connectors_available",
               return_value=connectors):
        invalidate_check_fn_cache()
        try:
            defs = _compute_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        finally:
            invalidate_check_fn_cache()
    return {d["function"]["name"] for d in defs}



def test_cli_session_gets_the_tool_outside_a_code_workspace(tmp_path, monkeypatch):
    """The path a plain `hermes` run takes: _get_platform_tools, no git cwd."""
    from hermes_cli.tools_config import _get_platform_tools

    monkeypatch.chdir(tmp_path)
    enabled = sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=True))

    assert "connections" in enabled
    assert "manage_connections" in _session_tool_names(enabled, connectors=True)


def test_cli_session_gets_the_tool_inside_a_code_workspace(monkeypatch):
    """Same resolver, run from this repo — the surface the live miss was on."""
    from pathlib import Path

    from hermes_cli.tools_config import _get_platform_tools

    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    enabled = sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=True))
    assert "manage_connections" in _session_tool_names(enabled, connectors=True)


def test_tui_and_desktop_sessions_get_the_tool(monkeypatch):
    """The path the TUI/desktop gateway takes to build its selection."""
    from tui_gateway.server import _load_enabled_toolsets

    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    for platform in ("tui", "desktop"):
        selection = _load_enabled_toolsets(platform)
        names = _session_tool_names(selection, connectors=True)
        assert "manage_connections" in names, platform


def test_focus_mode_coding_posture_gets_the_tool(monkeypatch):
    """An engineer pinned to the coding posture still sees their accounts."""
    from pathlib import Path

    from agent.coding_context import coding_selection

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo)
    selection = coding_selection(
        platform="cli", cwd=str(repo), config={"agent": {"coding_context": "focus"}}
    )
    assert selection == ["coding"]  # posture collapse still collapses
    assert "manage_connections" in _session_tool_names(selection, connectors=True)


def test_signed_out_session_sees_nothing(tmp_path, monkeypatch):
    """check_fn is the only entitlement gate, on every surface."""
    from hermes_cli.tools_config import _get_platform_tools
    from tui_gateway.server import _load_enabled_toolsets

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    selections = [
        sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=True)),
        _load_enabled_toolsets("tui"),
        ["coding"],
    ]
    for selection in selections:
        assert "manage_connections" not in _session_tool_names(
            selection, connectors=False
        ), selection


def test_operator_can_still_turn_it_off(tmp_path, monkeypatch):
    """`agent.disabled_toolsets: [connections]` wins; a bundle name does not.

    The name is added before the disabled subtraction, so the toolset behaves
    like any other. Naming a platform composite instead must NOT strip it —
    that branch preserves core tools on purpose (#33924).
    """
    from hermes_cli.tools_config import _get_platform_tools

    monkeypatch.chdir(tmp_path)
    enabled = sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=True))

    assert "manage_connections" not in _session_tool_names(
        enabled, connectors=True, disabled_toolsets=["connections"]
    )
    assert "manage_connections" in _session_tool_names(
        enabled, connectors=True, disabled_toolsets=["hermes-cli"]
    )


def test_tool_is_never_deferrable():
    from tools.tool_search import is_deferrable_tool_name

    # Core names short-circuit before the toolset check, so listing
    # "connections" in _DIRECT_SURFACE_TOOLSETS would be redundant.
    assert is_deferrable_tool_name("manage_connections") is False
