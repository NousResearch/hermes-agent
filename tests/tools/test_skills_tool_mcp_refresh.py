"""``refresh_catalogs`` decides what a listing may claim is current.

The listing surfaces re-read the servers before they answer, so this classification is the
whole contract: a server that answered is shown, a server that did not is reported instead
of being answered from its last snapshot, and a read that merely ran long is never turned
into a disconnect. Each claim is asserted through the manager stand-in's call log, so "no
RPC happened" and "nothing was marked disconnected" are proven by absence.
"""

import asyncio

import pytest

from agent.skills_contracts import SkillError, SkillErrorCode
from tools import skills_tool_mcp

TIMEOUT = 0.3


class _Manager:
    """Manager stand-in: only the calls a listing is allowed to make."""

    def __init__(self, *, sessions=(), results=None):
        self.sessions = set(sessions)
        self.results = dict(results or {})  # name -> "ok" | exception | Event that never arrives
        self.calls = []
        self.cancelled = []
        self.disconnects = []

    def has_session(self, name, scope):
        return name in self.sessions

    async def discover(self, name, scope, *, reason=""):
        self.calls.append(name)
        outcome = self.results.get(name, "ok")
        if isinstance(outcome, asyncio.Event):
            try:
                await outcome.wait()
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise
        elif isinstance(outcome, BaseException):
            raise outcome
        return object()

    def mark_disconnected(self, *args, **kwargs):
        self.disconnects.append((args, kwargs))


@pytest.fixture
def refresh_env(monkeypatch):
    """An opted-in server set, a recorded connect nudge, and a running MCP loop."""
    from tools.mcp_tool_loop import _ensure_mcp_loop

    state = {"servers": {"acme"}, "connected": []}
    monkeypatch.setattr(skills_tool_mcp, "known_server_labels", lambda: set(state["servers"]))
    monkeypatch.setattr(skills_tool_mcp, "_connect_servers",
                        lambda names: state["connected"].append(list(names)))
    _ensure_mcp_loop()
    return state


def test_an_answering_server_is_verified_and_a_failing_one_is_a_problem(refresh_env):
    refresh_env["servers"] = {"acme", "other"}
    manager = _Manager(sessions={"acme", "other"},
                       results={"acme": SkillError(SkillErrorCode.UNAVAILABLE, "session terminated")})

    verified, problems = skills_tool_mcp.refresh_catalogs(manager=manager, scope_id="", timeout=TIMEOUT)

    assert verified == ["other"], "one unreachable server does not cost the others their listing"
    assert problems == [{"server": "acme", "error": "session terminated"}]
    assert manager.disconnects == [], "an unreadable catalog is not proof the transport is gone"


def test_an_empty_catalog_is_an_answer_not_a_failure(refresh_env):
    manager = _Manager(sessions={"acme"})

    verified, problems = skills_tool_mcp.refresh_catalogs(manager=manager, scope_id="", timeout=TIMEOUT)

    assert verified == ["acme"] and problems == []


def test_a_server_without_the_channel_has_verified_nothing_to_show(refresh_env):
    """"I do not serve skills" is this server's current list, not an unreadable one."""
    manager = _Manager(sessions={"acme"},
                       results={"acme": SkillError(SkillErrorCode.UNSUPPORTED_EXTENSION, "no extension")})

    verified, problems = skills_tool_mcp.refresh_catalogs(manager=manager, scope_id="", timeout=TIMEOUT)

    assert verified == ["acme"] and problems == []


def test_a_read_that_outlives_the_listing_is_reported_and_left_running(refresh_env):
    """A slow server costs this listing its rows; the read it started still lands."""
    never = asyncio.Event()
    manager = _Manager(sessions={"acme"}, results={"acme": never})

    verified, problems = skills_tool_mcp.refresh_catalogs(manager=manager, scope_id="", timeout=TIMEOUT)

    assert verified == []
    assert problems == [{"server": "acme", "error": "the catalog did not answer in time"}]
    assert manager.cancelled == [], "the catalog it is reading is still the right one"
    never.set()


def test_a_server_with_no_live_connection_is_reported_without_a_read(refresh_env):
    manager = _Manager(sessions=set())

    verified, problems = skills_tool_mcp.refresh_catalogs(manager=manager, scope_id="", timeout=TIMEOUT)

    assert verified == []
    assert problems == [{"server": "acme", "error": "no live connection"}]
    assert manager.calls == [], "nothing is sent over a connection the host does not have"
    assert refresh_env["connected"] == [["acme"]], "the server was asked to come back first"


def test_a_server_already_connected_is_not_nudged(refresh_env):
    manager = _Manager(sessions={"acme"})

    skills_tool_mcp.refresh_catalogs(manager=manager, scope_id="", timeout=TIMEOUT)

    assert refresh_env["connected"] == []


def test_no_opted_in_server_means_no_connection_and_no_read(refresh_env):
    refresh_env["servers"] = set()
    manager = _Manager(sessions={"acme"})

    assert skills_tool_mcp.refresh_catalogs(manager=manager, scope_id="", timeout=TIMEOUT) == ([], [])
    assert refresh_env["connected"] == [] and manager.calls == []


def test_a_paginating_server_that_fails_mid_read_is_not_verified(refresh_env):
    """A half-listing is never an answer, however many pages it managed first."""
    manager = _Manager(sessions={"acme"},
                       results={"acme": SkillError(SkillErrorCode.INCOMPLETE_CATALOG, "repeated cursor")})

    verified, problems = skills_tool_mcp.refresh_catalogs(manager=manager, scope_id="", timeout=TIMEOUT)

    assert verified == [] and problems[0]["error"] == "repeated cursor"
