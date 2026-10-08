"""P4 PARTE 2 wiring: off_board reaches complete_task from the CLI (--off-board) and kanban_complete tool."""
import json
from pathlib import Path
import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home)); monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for k in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear(); kb.init_db()
    return kb


def _card(kb):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="ob", assignee="builder", workspace_kind="scratch")
        assert kb.claim_task(c, tid) is not None
    return tid


def _status(kb, tid):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as c:
        return kb.get_task(c, tid).status


def test_tool_off_board_without_served_model_refused(env):
    from tools import kanban_tools  # noqa
    from tools.registry import registry
    tid = _card(env)
    r = json.loads(registry.dispatch("kanban_complete", {"task_id": tid, "summary": "s", "off_board": True}))
    assert r.get("error") and "served_model" in r["error"], r
    assert _status(env, tid) == "running"


def test_tool_off_board_with_served_model_completes(env):
    from tools import kanban_tools  # noqa
    from tools.registry import registry
    tid = _card(env)
    r = json.loads(registry.dispatch("kanban_complete", {"task_id": tid, "summary": "s", "off_board": True,
                                                          "metadata": {"served_model": "gpt-6.1-sol"}}))
    assert r.get("ok"), r
    assert _status(env, tid) == "done"


def test_tool_default_unchanged_no_served_model_needed(env):
    from tools import kanban_tools  # noqa
    from tools.registry import registry
    tid = _card(env)
    r = json.loads(registry.dispatch("kanban_complete", {"task_id": tid, "summary": "s"}))
    assert r.get("ok"), r


def test_cli_off_board_flag_refused_then_ok(env):
    from hermes_cli.kanban import run_slash
    tid = _card(env)
    out = run_slash(f"complete {tid} --result r --off-board")
    assert "served_model" in out or "off-board" in out.lower(), out
    assert _status(env, tid) == "running"
    out2 = run_slash(f"complete {tid} --result r --off-board --metadata " + json.dumps(json.dumps({"served_model": "gpt-6.1-sol"})))
    assert _status(env, tid) == "done", out2


def test_cli_default_unchanged(env):
    from hermes_cli.kanban import run_slash
    tid = _card(env)
    run_slash(f"complete {tid} --result r")
    assert _status(env, tid) == "done"
