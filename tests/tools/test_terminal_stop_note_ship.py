"""Terminal tool refuses merge/deploy/push-to-main when a stop note holds."""

from __future__ import annotations

import json

from tools.terminal_tool import _kanban_stop_note_ship_block, _pre_exec_block, _Rejected


def test_ship_block_skips_when_not_a_kanban_worker(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _kanban_stop_note_ship_block("gh pr merge 1") is None


def test_ship_block_skips_non_ship_commands(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    assert _kanban_stop_note_ship_block("git push origin feat/foo") is None
    assert _kanban_stop_note_ship_block("gh pr create --fill") is None


def test_ship_block_refuses_when_stop_note_holds(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    monkeypatch.setattr(
        "tools.terminal_tool._kanban_ship_abort_reason",
        lambda _tid: "Stop/close note on this kanban task — merge/deploy/push-to-main refused.",
    )
    raw = _kanban_stop_note_ship_block("gh pr merge 12")
    body = json.loads(raw)
    assert body["status"] == "blocked"
    assert body["exit_code"] == -1
    assert "Stop/close note" in body["error"]


def test_ship_block_allows_ship_after_stop_lifted(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    monkeypatch.setattr(
        "tools.terminal_tool._kanban_ship_abort_reason", lambda _tid: None,
    )
    assert _kanban_stop_note_ship_block("gh pr merge 12") is None


def test_pre_exec_raises_rejected_on_stop_note_ship(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    monkeypatch.setattr("tools.terminal_tool.gateway_lifecycle_block", lambda **_k: None)
    monkeypatch.setattr("tools.terminal_tool.self_repo_block", lambda **_k: None)
    monkeypatch.setattr(
        "tools.terminal_tool._kanban_ship_abort_reason",
        lambda _tid: "Stop/close note on this kanban task — merge/deploy/push-to-main refused.",
    )
    try:
        _pre_exec_block(
            "lab deploy backend", env=None, env_type="local", cwd="/tmp",
            workdir=None, session_key="s",
        )
    except _Rejected as rejected:
        body = json.loads(rejected.result_json)
        assert body["status"] == "blocked"
        return
    raise AssertionError("expected _Rejected")


def test_ship_block_refuses_when_card_not_running(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    monkeypatch.setattr(
        "tools.terminal_tool._kanban_ship_abort_reason",
        lambda _tid: "Kanban task is blocked — merge/deploy/push-to-main refused.",
    )
    raw = _kanban_stop_note_ship_block("gh pr merge 12")
    assert raw is not None
    body = json.loads(raw)
    assert "blocked" in body["error"]
