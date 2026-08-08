"""Regression tests for the s5 wave-1 split (w1a): projects + slash-mirror clusters.

Covers the two highest-agreement move clusters extracted from
``tui_gateway/server.py`` (god-file extraction wave 1):

- c2  -> ``tui_gateway/methods_projects.py``: ``_projects_payload``,
         ``_projects_method``, ``_require_project`` (plus ``_NoProject`` and
         the ``_E_PROJECTS``/``_E_NO_PROJECT``/``_E_PROJECT_ARG`` codes).
- c29 -> ``tui_gateway/slash_mirror.py``: the ``_format_live_*`` formatters,
         ``_live_slash_command_output``, ``_mirror_slash_side_effects``.

Like every other split module (see ``method_ctx.py``), the moved functions
are rebound onto ``server``'s globals at install time, so production callers
(methods_tools.py, compute_host.py) and these tests reach them through the
``tui_gateway.server`` namespace.
"""

import contextlib
import threading

import pytest

import tui_gateway.methods_projects as methods_projects
import tui_gateway.server as server
import tui_gateway.slash_mirror as slash_mirror


# ── c2: projects surface ─────────────────────────────────────────────


class _FakeProject:
    def __init__(self, pid: str):
        self.id = pid
        self.slug = pid

    def to_dict(self):
        return {"id": self.id, "slug": self.slug}


class _FakePdb:
    def __init__(self, project=None, error=None):
        self._project = project
        self._error = error

    def get_project(self, conn, pid):
        if self._error is not None:
            raise self._error
        return self._project


def test_projects_helpers_rebound_onto_server():
    # The three moved helpers must be the same objects server.py imports.
    assert server._projects_method is methods_projects._projects_method
    assert server._projects_payload is methods_projects._projects_payload
    assert server._require_project is methods_projects._require_project
    # _NoProject raised in the mixin must be the class the rebound handler
    # catches — a single shared object via the re-import.
    assert server._NoProject is methods_projects._NoProject
    assert server._E_PROJECTS == 5061
    assert server._E_NO_PROJECT == 5062
    assert server._E_PROJECT_ARG == 5063


def test_require_project_resolves_or_raises():
    proj = _FakeProject("p1")
    assert server._require_project(_FakePdb(project=proj), object(), {"id": "p1"}) is proj
    with pytest.raises(server._NoProject):
        server._require_project(_FakePdb(project=None), object(), {"id": "missing"})


def test_projects_method_error_mapping_via_installed_handlers(monkeypatch):
    """The moved _projects_method decorator maps failures to 5061/5062/5063."""
    import hermes_cli.projects_db as pdb_mod

    fake_conn = object()
    monkeypatch.setattr(
        pdb_mod, "connect_closing", lambda: contextlib.nullcontext(fake_conn)
    )

    # 5062: id resolves to nothing -> _NoProject.
    monkeypatch.setattr(pdb_mod, "get_project", lambda conn, pid: None)
    resp = server._methods["projects.get"](1, {"id": "missing"})
    assert resp["error"]["code"] == 5062
    assert resp["error"]["message"] == "no such project"

    # 5063: invalid argument.
    monkeypatch.setattr(
        pdb_mod, "get_project", lambda conn, pid: (_ for _ in ()).throw(ValueError("bad id"))
    )
    resp = server._methods["projects.get"](1, {"id": "bad"})
    assert resp["error"]["code"] == 5063

    # 5061: anything else.
    monkeypatch.setattr(
        pdb_mod, "get_project", lambda conn, pid: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    resp = server._methods["projects.get"](1, {"id": "boom"})
    assert resp["error"]["code"] == 5061


def test_projects_list_payload_via_installed_handler(monkeypatch):
    """_projects_payload (moved) feeds the staying projects.list handler."""
    import hermes_cli.projects_db as pdb_mod

    monkeypatch.setattr(
        pdb_mod, "connect_closing", lambda: contextlib.nullcontext(object())
    )
    monkeypatch.setattr(
        pdb_mod,
        "list_projects",
        lambda conn, include_archived=False: [_FakeProject("p1"), _FakeProject("p2")],
    )
    monkeypatch.setattr(pdb_mod, "get_active_id", lambda conn: "p1")

    resp = server._methods["projects.list"](1, {})
    assert resp["result"] == {
        "projects": [{"id": "p1", "slug": "p1"}, {"id": "p2", "slug": "p2"}],
        "active_id": "p1",
    }


# ── c29: slash-mirror live output ────────────────────────────────────


def test_format_live_model_output():
    assert server._format_live_model_output({}) == "Current model: (unknown)"


def test_format_live_usage_output_no_agent():
    out = server._format_live_usage_output({})
    assert out == "(._.) No active agent -- send a message first."


def test_format_live_usage_output_counts_and_model(monkeypatch):
    # Pure formatting path via the _metadata_mirror usage snapshot.
    session = {
        "_metadata_message_count": 5,
        "_compute_host_active": True,
        "_metadata_mirror": {
            "usage": {"model": "m1", "input": 10, "output": 20, "calls": 2}
        },
    }
    out = server._format_live_usage_output(session)
    assert "Session Token Usage" in out
    assert "Model: m1" in out
    assert "Messages:                     5" in out


def test_format_live_history_output_uses_in_memory_history(monkeypatch):
    # db branch is skipped without a session_key; _get_db stays untouched.
    monkeypatch.setattr(server, "_get_db", lambda: None)
    session = {
        "history": [{"role": "user", "content": "hi"}],
        "history_lock": threading.Lock(),
    }
    out = server._format_live_history_output(session)
    assert "Conversation History" in out
    assert "[You #1] hi" in out


def test_format_live_context_output_empty_and_with_usage(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    session = {"history": [], "history_lock": threading.Lock()}
    out = server._format_live_context_output(session)
    assert "Conversation is empty (no messages yet)." in out

    session = {
        "history": [],
        "history_lock": threading.Lock(),
        "_compute_host_active": True,
        "_metadata_mirror": {"usage": {"model": "m9", "total": 3}},
    }
    out = server._format_live_context_output(session)
    assert "Model: m9" in out


def test_live_slash_command_output_routing():
    # Unknown command: not a live command, not an isolated read -> None.
    assert server._live_slash_command_output("sid", None, "bogus", "") is None
    # usage with no session -> the no-agent message.
    assert server._live_slash_command_output("sid", None, "usage", "") == (
        "(._.) No active agent -- send a message first."
    )
    # model on an empty session -> formatter default.
    assert server._live_slash_command_output("sid", {}, "model", "") == (
        "Current model: (unknown)"
    )
    # clear is a direct live command.
    assert "terminal-only" in server._live_slash_command_output("sid", None, "clear", "")


def test_mirror_slash_side_effects_idle_and_busy(monkeypatch):
    # Keep the compute-host gate closed (default config has isolation off).
    monkeypatch.setattr(server, "_turn_isolation_enabled", lambda cfg=None: False)

    # Idle session: no-op mirror returns "".
    idle = {"agent": None, "running": False}
    assert server._mirror_slash_side_effects("sid", idle, "/model foo") == ""

    # Running session: mutating command is rejected by the busy guard.
    busy = {"agent": object(), "running": True}
    warning = server._mirror_slash_side_effects("sid", busy, "/model foo")
    assert "session busy" in warning
    assert "/interrupt" in warning


def test_slash_mirror_helpers_rebound_onto_server():
    # The moved formatters are rebound onto server's globals (rebound copies,
    # not the raw module-level function objects).
    assert server._format_live_usage_output is not slash_mirror._format_live_usage_output
    assert server._live_slash_command_output is not slash_mirror._live_slash_command_output
    assert server._mirror_slash_side_effects is not slash_mirror._mirror_slash_side_effects
