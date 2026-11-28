"""``pre_kanban_board_access`` — a directive hook that VETOES access to a kanban
board before any operation runs (coordinator-only board gate).

The fire site is ``tools.kanban_tools._kanban_handler`` — the single chokepoint every
kanban tool funnels through — so a plugin gates one place and covers create/dispatch/
write (and read) alike. A plugin returns ``{"action": "block", "message"}`` to refuse
the operation; ``None`` / ``{"action": "allow"}`` (or any other return) passes.
"""
import json
from pathlib import Path

import pytest


@pytest.fixture
def kanban_env(tmp_path, monkeypatch):
    """Isolated kanban home with registered tool handlers."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_RUN_ID", "HERMES_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _dispatch_create(title="board access", assignee="atlas", board=None):
    from tools import kanban_tools  # register actual handlers
    from tools.registry import registry
    args = {"title": title, "assignee": assignee}
    if board is not None:
        args["board"] = board
    return json.loads(registry.dispatch("kanban_create", args))


def test_no_subscriber_short_circuits(kanban_env, monkeypatch):
    """has_hook=False -> invoke_hook is never called and the operation proceeds."""
    from tools import kanban_tools  # noqa: F401  (register handlers)
    from tools.registry import registry

    calls = []
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: False)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *a, **kw: calls.append((a, kw)) or [],
    )
    out = _dispatch_create()
    assert out.get("ok") is True, out
    assert calls == []


def test_block_directive_vetoes_operation(kanban_env, monkeypatch):
    """A plugin's {"action": "block", "message"} refuses the write before any row lands."""
    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *a, **kw: [{"action": "block", "message": "coordinator-only board"}],
    )
    out = _dispatch_create()
    assert out.get("ok") is not True, out
    assert "coordinator-only board" in out.get("error", "")
    with kbc.connect_closing() as conn:
        assert kanban_env.list_tasks(conn) == []


def test_allow_and_none_pass(kanban_env, monkeypatch):
    """None and {"action": "allow"} (and non-block dicts) let the operation proceed."""
    from tools import kanban_tools  # noqa: F401
    from tools.registry import registry

    for returned in (None, {"action": "allow"}, {"action": "allow", "message": "ok"}):
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: [returned],
        )
        out = _dispatch_create(title=f"allow-{returned!r}")
        assert out.get("ok") is True, (returned, out)


def test_non_dict_and_message_less_block_ignored(kanban_env, monkeypatch):
    """Garbage returns and a block without a message degrade to allow (fail-open)."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *a, **kw: ["garbage", 42, {"action": "block"}, {"action": "block", "message": ""}],
    )
    out = _dispatch_create()
    assert out.get("ok") is True, out


def test_hook_payload_carries_tool_board_profile_and_args(kanban_env, monkeypatch):
    """Kwargs name the tool, the target board, the runtime identity, and the args."""
    captured = {}
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *a, **kw: (captured.update(kw), [])[1],
    )
    _dispatch_create(title="payload probe", assignee="henry", board="hive")
    assert captured["tool_name"] == "kanban_create"
    assert captured["board"] == "hive"
    assert captured["profile_name"]  # runtime identity, never empty
    assert captured["args"]["title"] == "payload probe"
    assert captured["args"]["assignee"] == "henry"


def test_block_wins_over_allow(kanban_env, monkeypatch):
    """First valid block vetoes even if a later hook allowed (matches pre_tool_call)."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *a, **kw: [{"action": "allow"}, {"action": "block", "message": "later veto"}],
    )
    out = _dispatch_create()
    assert out.get("ok") is not True, out
    assert "later veto" in out.get("error", "")
