"""Tests for tools/plan_ready_tool.py — plan_ready → clarify → approve flow."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import plan_mode

    plan_mode._DB_CACHE.clear()
    yield home
    plan_mode._DB_CACHE.clear()


def test_approve_flow_marks_approved(hermes_home):
    from hermes_cli.plan_mode import PlanManager
    from tools.plan_ready_tool import plan_ready_tool

    PlanManager("s1").enter()
    seen = {}

    def cb(question, choices):
        seen["question"] = question
        seen["choices"] = choices
        return "Approve"

    out = json.loads(plan_ready_tool(session_id="s1", plan_path=".hermes/plans/p.md", callback=cb))
    assert out["status"] == "approved"
    assert seen["choices"] == ["Approve", "Keep planning"]
    assert not PlanManager("s1").is_active()
    assert PlanManager("s1").state.plan_path == ".hermes/plans/p.md"


def test_keep_planning_stays_active(hermes_home):
    from hermes_cli.plan_mode import PlanManager
    from tools.plan_ready_tool import plan_ready_tool

    PlanManager("s2").enter()
    out = json.loads(plan_ready_tool(session_id="s2", callback=lambda q, c: "Keep planning"))
    assert out["status"] == "planning"
    assert out["feedback"] == ""
    assert PlanManager("s2").is_active()


def test_free_text_returned_as_feedback(hermes_home):
    from hermes_cli.plan_mode import PlanManager
    from tools.plan_ready_tool import plan_ready_tool

    PlanManager("s3").enter()
    out = json.loads(plan_ready_tool(session_id="s3", callback=lambda q, c: "add a rollback step"))
    assert out["status"] == "planning"
    assert out["feedback"] == "add a rollback step"
    assert PlanManager("s3").is_active()


def test_error_when_not_in_plan_mode(hermes_home):
    from tools.plan_ready_tool import plan_ready_tool

    out = json.loads(plan_ready_tool(session_id="s4", callback=lambda q, c: "Approve"))
    assert "error" in out


def test_callback_exception_does_not_approve(hermes_home):
    from hermes_cli.plan_mode import PlanManager
    from tools.plan_ready_tool import plan_ready_tool

    PlanManager("s5").enter()

    def boom(q, c):
        raise RuntimeError("no ui")

    out = json.loads(plan_ready_tool(session_id="s5", callback=boom))
    assert "error" in out
    # Must NOT have approved on error.
    assert PlanManager("s5").is_active()
