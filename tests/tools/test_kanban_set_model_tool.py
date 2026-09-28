"""kanban_set_model — in-process worker tool for the per-task model override.

Context (t_df250a0a): a dispatcher-spawned worker's OWN process never carries
HERMES_DELEGATED_CHILD_CONTEXT — only its *subprocess descendants* (terminal/
CLI calls) do, by the intentional write-fence added in commit b578261584 (see
tests/tools/test_kanban_descendant_scope.py). `hermes kanban set-model` run
via the terminal tool is therefore a real descendant process and is correctly
refused — that is not a bug and must not be "fixed" by weakening the guard.
The actual gap: a worker had no IN-PROCESS way to set/clear its own model
override, only the CLI (subprocess) path. This tool closes that gap.

RED (pre-fix): `kanban_set_model` does not exist in tools.kanban_tools_schemas
or the _TOOLS registration — importing KANBAN_SET_MODEL_SCHEMA / calling
_handle_set_model raises ImportError/AttributeError.
GREEN (post-fix): the tool exists, mutates model_override/provider_override
for the worker's own task while running fully in-process (no subprocess), is
scoped like kanban_heartbeat (cannot touch a foreign task_id), and a genuine
delegate_task child (HERMES_DELEGATED_CHILD_CONTEXT=1) is still refused.
"""
from __future__ import annotations

import json
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from tools import kanban_tools as kt


@pytest.fixture
def board(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    conn = connect(db)
    own = kb.create_task(conn, title="own", assignee="worker")
    foreign = kb.create_task(conn, title="foreign", assignee="worker")
    for tid in (own, foreign):
        kb.claim_task(conn, tid)
    task = kb.get_task(conn, own)
    for key, value in {
        "HERMES_KANBAN_DB": str(db), "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_TASK": own, "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
        "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock, "HOME": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    return conn, own, foreign


def test_kanban_set_model_registered_and_worker_scoped():
    names = {name for name, *_ in kt._TOOLS}
    assert "kanban_set_model" in names


def test_worker_can_set_and_clear_its_own_model_in_process(board):
    conn, own, foreign = board
    out = json.loads(kt._handle_set_model({"model": "glm-5", "provider": "openrouter"}))
    assert out["ok"], out
    task = kb.get_task(conn, own)
    assert task.model_override == "glm-5"
    assert task.provider_override == "openrouter"

    out = json.loads(kt._handle_set_model({"model": "none"}))
    assert out["ok"], out
    task = kb.get_task(conn, own)
    assert task.model_override is None
    assert task.provider_override is None


def test_worker_cannot_set_model_on_a_foreign_task(board):
    conn, own, foreign = board
    out = json.loads(kt._handle_set_model({"task_id": foreign, "model": "glm-5"}))
    assert "error" in out, out
    assert kb.get_task(conn, foreign).model_override is None


def test_provider_without_model_rejected(board):
    conn, own, foreign = board
    out = json.loads(kt._handle_set_model({"provider": "openrouter"}))
    assert "error" in out, out
    assert kb.get_task(conn, own).model_override is None


def test_delegate_child_context_still_refused_even_via_the_new_tool(board, monkeypatch):
    """The in-process tool must not become a bypass for the real invariant:
    a delegate_task child (marker set) is still not a Kanban run owner."""
    conn, own, foreign = board
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    out = json.loads(kt._handle_set_model({"model": "glm-5"}))
    assert "error" in out, out
    assert "delegate_task child" in out["error"]
    assert kb.get_task(conn, own).model_override is None


def test_real_subprocess_descendant_of_the_worker_is_still_denied_the_cli_path(board):
    """Documents the non-bug: `hermes kanban set-model` run through a real
    Hermes-managed spawn surface (terminal tool) is still correctly refused
    (write-fence intact, per tests/tools/test_kanban_descendant_scope.py);
    kanban_set_model is the sanctioned in-process replacement."""
    import shlex
    import subprocess
    import sys
    from pathlib import Path

    from tools.environments.local import LocalEnvironment

    conn, own, foreign = board
    root = Path(__file__).resolve().parents[2]
    script = f"cd {shlex.quote(str(root))} && {shlex.quote(sys.executable)} -m hermes_cli.main kanban set-model {own} glm-5"
    terminal = LocalEnvironment(cwd=str(root))
    try:
        result = terminal.execute(script)
    finally:
        terminal.cleanup()
    output = result.get("output", "")
    assert "delegate_task child" in output, output
    assert kb.get_task(conn, own).model_override is None
