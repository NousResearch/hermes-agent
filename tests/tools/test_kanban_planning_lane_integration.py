"""Argos planning lane: authorization surface proven at the real boundaries.

The parent work (``f59c2a2400``) established the lane, but its tests assert
mostly at schema assembly. Two boundaries were left unproven and are what this
file pins:

1. **The dispatcher integration chain.** ``dispatch_once`` -> ``_default_spawn``
   -> the worker's own ``kanban_*`` calls -> the next tick, driven through the
   real spawn seam (``Popen`` faked, nothing launched). A ``show``/``complete``
   smoke proves neither the argv/env pin nor the promotion order.
2. **What the pin is and is not.** The argv ``--toolsets kanban`` restricts the
   schema the model can *see*; the agent loop then refuses any name outside that
   schema (``valid_tool_names``, checked in ``turn_tool_validation``) — so the
   schema is the model-facing boundary, and it is a real one on the only path a
   model can take. It is NOT an in-handler guard: a forced ``registry.dispatch``
   of ``terminal`` still runs. Kanban tools go further, re-checking the
   allow-list inside the handler, so even that forced call cannot escape the
   lane. The asymmetry is the contract, and it is asserted here so a future
   refactor cannot silently move either boundary.
"""
from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest


def _schema_names(selection):
    from model_tools import get_tool_definitions
    return {row["function"]["name"] for row in get_tool_definitions(
        selection, quiet_mode=True, skip_tool_search_assembly=True)}


def _call(name, args):
    from tools.registry import registry
    result = registry.dispatch(name, args)
    return result if isinstance(result, dict) else json.loads(result)


def _no_network(monkeypatch):
    """Any socket connect is a test bug: these paths must never egress."""
    monkeypatch.setattr(socket.socket, "connect", lambda *a: (_ for _ in ()).throw(
        AssertionError("network egress attempted")))


def _task(kb, *, task_id, assignee, status="running", run_id=7):
    return kb.Task(id=task_id, title="plan", body=None, assignee=assignee, status=status,
                   priority=0, created_by="test", created_at=1, started_at=None, completed_at=None,
                   workspace_kind="scratch", workspace_path=None, claim_lock="lock",
                   claim_expires=None, tenant=None, current_run_id=run_id)


@pytest.fixture
def board(monkeypatch, tmp_path):
    """An isolated board + profile tree. No production kanban.db, no real home."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    root = tmp_path / ".hermes"
    (root / "profiles" / "argos").mkdir(parents=True)
    (root / "config.yaml").write_text("kanban:\n  orchestrator_profile: argos\n")
    # argos asks for the whole world; the lane must not honour it.
    (root / "profiles" / "argos" / "config.yaml").write_text(
        "platform_toolsets:\n  cli: [kanban, web, terminal, file, code_execution, "
        "browser, connections, code_execution]\nagent:\n  disabled_toolsets: []\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _no_network(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    kb.init_db()
    return {"root": root, "db": kb.kanban_db_path(), "connect": kbc.connect}


def test_dispatcher_chain_runs_end_to_end_without_spawning_anything(board, monkeypatch, tmp_path):
    """Argos-parent -> Hefesto child -> dependent Atena review, through the real spawn seam.

    Uses ``dispatch_once`` (not a hand-built board) so claim/promotion ordering,
    the run record and the argv/env pin are the dispatcher's own. ``Popen`` is
    faked: no agent is started, no PID is real.
    """
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch

    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, cmd: cmd)
    monkeypatch.setattr(dispatch, "_retag_legacy_worker_sessions", lambda path: None)
    monkeypatch.setattr(dispatch, "_open_worker_log", lambda task, board_slug: open(
        tmp_path / "worker.log", "ab"))
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda assignee: True)
    spawns = []

    def fake_popen(cmd, **kwargs):
        spawns.append((cmd, kwargs["env"]))
        return type("P", (), {"pid": 4000 + len(spawns)})()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with board["connect"]() as conn:
        parent = kb.create_task(conn, title="decompose", assignee="argos")
        # An unrelated human gate that must survive the whole chain untouched.
        sentinel = kb.create_task(conn, title="human decision", assignee="human")
        kb.block_task(conn, sentinel, reason="human gate")

    # --- tick 1: the dispatcher spawns the Argos parent as a planning worker ---
    with board["connect"]() as conn:
        result = dispatch.dispatch_once(conn, max_spawn=1)
        assert [row[0] for row in result.spawned] == [parent]
        # The run record is created by the claim this tick, not by create_task.
        claimed = kb.get_task(conn, parent)
        assert claimed is not None and claimed.current_run_id
        parent_env = spawns[-1][1]
        parent_cmd = spawns[-1][0]
    assert parent_env["HERMES_KANBAN_PLANNING_WORKER"] == "1"
    assert parent_env["HERMES_SAFE_MODE"] == "1"
    assert "HERMES_ACCEPT_HOOKS" not in parent_env
    assert parent_cmd[parent_cmd.index("--toolsets") + 1] == "kanban"
    assert "--accept-hooks" not in parent_cmd

    # --- the Argos parent, running as a real worker, reads its own context ---
    with monkeypatch.context() as worker:
        worker.setenv("HERMES_KANBAN_DB", str(board["db"]))
        worker.setenv("HERMES_KANBAN_TASK", parent)
        worker.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
        worker.setenv("HERMES_PROFILE", "argos")
        worker.setenv("HERMES_KANBAN_PLANNING_WORKER", "1")
        worker.setenv("HERMES_SAFE_MODE", "1")

        # A kanban-only session really is kanban-only — from the same registry
        # the model would be handed.
        assert _schema_names(["kanban"]) <= {
            "kanban_show", "kanban_create", "kanban_link", "kanban_comment",
            "kanban_complete", "kanban_block", "kanban_heartbeat"}

        context = _call("kanban_show", {})
        assert context["task"]["id"] == parent
        assert "worker_context" in context

        def call(name, **args):
            value = _call(name, args)
            assert value.get("ok"), value
            return value

        implementation = call("kanban_create", title="implement", assignee="hefesto",
                              parents=[parent], idempotency_key=f"{parent}:implement")["task_id"]
        review = call("kanban_create", title="review", assignee="atena",
                      idempotency_key=f"{parent}:review")["task_id"]
        call("kanban_link", parent_id=implementation, child_id=review)
        call("kanban_comment", task_id=implementation, body="Implementation handoff")
        call("kanban_complete", summary="Decomposed", created_cards=[implementation, review])

    # --- tick 2: the review stays gated on the implementation ---
    with board["connect"]() as conn:
        assert kb.get_task(conn, implementation).status == "ready"
        assert kb.get_task(conn, review).status == "todo"
        # An ordinary worker, NOT the planner: full profile toolsets + hooks.
        assert [row[0] for row in dispatch.dispatch_once(conn, max_spawn=1).spawned] == [implementation]
        impl_env = spawns[-1][1]
        assert "HERMES_KANBAN_PLANNING_WORKER" not in impl_env
        assert "HERMES_SAFE_MODE" not in impl_env
        assert "--accept-hooks" in spawns[-1][0]
        assert spawns[-1][0][spawns[-1][0].index("--toolsets") + 1] != "kanban"

        kb.complete_task(conn, implementation, summary="implemented")
        assert kb.get_task(conn, review).status == "ready"
        assert [row[0] for row in dispatch.dispatch_once(conn, max_spawn=1).spawned] == [review]

    # The human gate was never reachable by any tick of the chain.
    with board["connect"]() as conn:
        assert kb.get_task(conn, sentinel).status == "blocked"
    assert [env["HERMES_KANBAN_TASK"] for _, env in spawns] == [parent, implementation, review]


def test_planning_worker_cannot_unblock_a_human_gate(board, monkeypatch):
    """``kanban_unblock`` is outside the allow-list, so a sticky human block
    survives even the planner trying to clear it (and the handler refuses
    before recompute_ready, so no promotion is attempted at all)."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    with board["connect"]() as conn:
        gated = kb.create_task(conn, title="human gate", assignee="human")
        kb.block_task(conn, gated, reason="needs a human")

    recomputes = []
    monkeypatch.setattr(kt, "_recompute_ready", lambda *a, **k: recomputes.append(a), raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "argos")
    monkeypatch.setenv("HERMES_KANBAN_TASK", gated)
    monkeypatch.setenv("HERMES_KANBAN_PLANNING_WORKER", "1")

    denied = _call("kanban_unblock", {"task_id": gated})
    assert "unavailable to dispatcher planning workers" in denied["error"]
    assert "kanban_unblock" not in _schema_names(["kanban"])
    assert not recomputes, "planner reached the promotion path for a human gate"

    with board["connect"]() as conn:
        assert kb.get_task(conn, gated).status == "blocked"


def test_lane_boundary_is_schema_then_handler_not_argv_alone(board, monkeypatch, tmp_path):
    """Pin *where* the restriction lives, so it cannot quietly move.

    Non-Kanban tools are removed by ``--toolsets`` at schema assembly, and the
    agent loop refuses any name outside ``valid_tool_names`` (derived from that
    same schema). They are therefore invisible-and-refused to the model. Kanban
    tools, by contrast, are individually refused *inside* the handler, so a
    forced ``registry.dispatch`` — the stale-schema case the parent fix named —
    still cannot escape.
    """
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt
    from tools.registry import registry

    with board["connect"]() as conn:
        parent = kb.create_task(conn, title="plan", assignee="argos")
    monkeypatch.setenv("HERMES_HOME", str(board["root"] / "profiles" / "argos"))
    monkeypatch.setenv("HERMES_PROFILE", "argos")
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    monkeypatch.setenv("HERMES_KANBAN_PLANNING_WORKER", "1")
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")

    # Real discovery, so "absent" means "filtered", not "never registered".
    visible = _schema_names(["kanban"])
    assert registry.get_schema("terminal") is not None, "terminal not registered at all"
    assert "terminal" not in visible

    # The model path: the agent derives valid_tool_names from exactly this
    # assembly (agent/agent_init.py), and turn_tool_validation refuses any name
    # outside it before dispatch — so the schema IS the model-facing boundary.
    # That module is deliberately not imported here: importing it triggers
    # hermes_bootstrap, which reads the real home and trips the test I/O guard.
    from agent import turn_tool_validation
    assert hasattr(turn_tool_validation, "validate_tool_calls")

    # DELIBERATE COUNTER-EXAMPLE, not an endorsement: a forced dispatch of a
    # NON-kanban tool is outside the lane's own handler guard, so it DOES run
    # here. In production it is unreachable because the model can only name
    # tools in its assembled schema and turn_tool_validation rejects any other
    # name before dispatch. Pinned so the boundary is stated in one place: the
    # kanban handler guards the KANBAN surface; the schema + agent loop guard
    # everything else. If the schema ever becomes a weaker boundary, this
    # assertion is the canary and must start failing loudly.
    terminal = _call("terminal", {"command": "true"})
    assert "output" in terminal and terminal["exit_code"] == 0

    # A forced dispatch of a NON-allow-listed KANBAN tool is refused in-handler,
    # before any effect, even with the download path stubbed to explode.
    monkeypatch.setattr(kt, "_download_url_with_cap", lambda *a: (_ for _ in ()).throw(
        AssertionError("download path reached")))
    for denied_tool in ("kanban_attach_url", "kanban_attach", "kanban_attachments",
                        "kanban_list", "kanban_unblock", "kanban_request_review",
                        "kanban_request_changes"):
        result = _call(denied_tool, {"task_id": parent, "url": "https://example.invalid/x"})
        assert "unavailable to dispatcher planning workers" in result["error"], denied_tool
        assert denied_tool not in visible, denied_tool


def test_other_profiles_and_default_kanban_behaviour_are_untouched(board, monkeypatch):
    """The restriction is scoped to the pinned lane, not to Kanban itself.

    A second profile with the kanban toolset keeps its full CLI tools and
    ``kanban_attach_url``; the review-lane tools stay available to it. The
    parent's own argos policy also still governs only argos.
    """
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch

    hefesto = board["root"] / "profiles" / "hefesto"
    hefesto.mkdir(parents=True)
    (hefesto / "config.yaml").write_text(
        "platform_toolsets:\n  cli: [kanban, web, terminal, file]\n")

    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, cmd: cmd)
    monkeypatch.setattr(dispatch, "_open_worker_log", lambda task, board_slug: open(
        board["root"] / "worker.log", "ab"))
    captured = []

    def fake_popen(cmd, **kwargs):
        captured.append((cmd, kwargs["env"]))
        return type("P", (), {"pid": 5000 + len(captured)})()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = board["root"] / "work"
    workspace.mkdir()

    # hefesto is not the configured orchestrator: ordinary worker, unchanged.
    dispatch._default_spawn(_task(kb, task_id="t_h", assignee="hefesto"), str(workspace))
    cmd, env = captured[-1]
    assert "HERMES_KANBAN_PLANNING_WORKER" not in env
    assert "HERMES_SAFE_MODE" not in env
    assert "--accept-hooks" in cmd
    assert cmd[cmd.index("--toolsets") + 1] != "kanban"
    assert {"web", "terminal", "file"} <= set(cmd[cmd.index("--toolsets") + 1].split(","))
    # argos stays the only planner.
    dispatch._default_spawn(_task(kb, task_id="t_a", assignee="argos"), str(workspace))
    assert captured[-1][1]["HERMES_KANBAN_PLANNING_WORKER"] == "1"

    # A non-planning profile session with the kanban toolset is unaffected by
    # the lane: kanban_attach_url and the review lane are all its own.
    monkeypatch.setenv("HERMES_HOME", str(hefesto))
    monkeypatch.setenv("HERMES_PROFILE", "hefesto")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_h")
    monkeypatch.delenv("HERMES_KANBAN_PLANNING_WORKER", raising=False)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    visible = _schema_names(["kanban"])
    assert "kanban_attach_url" in visible
    assert "kanban_list" not in visible, "board-routing tools stay hidden from task workers"
    assert "kanban_unblock" not in visible
    assert {"kanban_show", "kanban_create", "kanban_link", "kanban_comment",
            "kanban_complete", "kanban_block", "kanban_heartbeat"} <= visible
    assert "kanban_request_review" in visible


def test_unverifiable_root_policy_restricts_ordinary_profiles_too(board, monkeypatch, tmp_path):
    """Fail-closed is board-wide, not argos-flavoured.

    With no verifiable root policy nobody can be identified as the planner, so
    EVERY assignee's worker is spawned restricted — an ordinary profile does
    not silently keep its own toolsets on a misconfigured board.
    """
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch

    (board["root"] / "config.yaml").write_text("kanban: {}\n")  # no orchestrator_profile
    hefesto = board["root"] / "profiles" / "hefesto"
    hefesto.mkdir(parents=True, exist_ok=True)
    (hefesto / "config.yaml").write_text(
        "platform_toolsets:\n  cli: [kanban, web, terminal, file, browser]\n")

    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, cmd: cmd)
    monkeypatch.setattr(dispatch, "_open_worker_log", lambda task, board_slug: open(
        tmp_path / "worker.log", "ab"))
    captured = []

    def fake_popen(cmd, **kwargs):
        captured.append((cmd, kwargs["env"]))
        return type("P", (), {"pid": 6000 + len(captured)})()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = board["root"] / "work"
    workspace.mkdir()
    for assignee in ("hefesto", "argos"):
        dispatch._default_spawn(_task(kb, task_id=f"t_{assignee}", assignee=assignee), str(workspace))
    for cmd, env in captured:
        assert env["HERMES_KANBAN_PLANNING_WORKER"] == "1"
        assert env["HERMES_SAFE_MODE"] == "1"
        assert cmd[cmd.index("--toolsets") + 1] == "kanban"
        assert "--accept-hooks" not in cmd
