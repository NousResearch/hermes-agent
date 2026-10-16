"""A Telegram project session anchors terminal AND file tools, keeps `cd`, and releases cleanly (#138777)."""

import contextvars
import json

import pytest

from gateway.project_session import bind_workspace, release_workspace
from gateway.session_context import clear_session_vars, set_session_vars

ROUTE = "agent:main:telegram:dm:42"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "hello.txt").write_text("hi")
    (ws / "sub" / "in.txt").write_text("inner")
    return ws


def _in_worker(fn, *args, **kwargs):
    """Tool calls run in worker threads with a copied context (propagate_context_to_thread)."""
    return contextvars.copy_context().run(fn, *args, **kwargs)


def test_file_tools_resolve_relative_paths_against_the_project_workspace(workspace):
    from tools.file_tools import read_file_tool

    tokens = set_session_vars(platform="telegram", session_key=ROUTE, session_id="sid-1", cwd=str(workspace))
    try:
        # File tools receive the SESSION ID as task_id, terminal keys on the route.
        result = json.loads(_in_worker(read_file_tool, "hello.txt", task_id="sid-1"))
    finally:
        clear_session_vars(tokens)
        release_workspace(ROUTE, "sid-1")
    assert "error" not in result or not result["error"]
    assert "hi" in result["content"]


def test_cd_inside_the_session_survives_the_next_turn_binding(workspace):
    from tools.file_tools import read_file_tool
    from tools.terminal_tool import get_session_cwd, record_session_cwd

    def bind():
        return set_session_vars(platform="telegram", session_key=ROUTE, session_id="sid-1", cwd=str(workspace))

    tokens = bind()
    try:
        # What the terminal result handler records after the agent runs `cd sub`.
        record_session_cwd(ROUTE, str(workspace / "sub"))
    finally:
        clear_session_vars(tokens)
    # Next turn of the SAME session re-binds the project root but must not undo the agent's cd.
    tokens = bind()
    try:
        assert get_session_cwd(ROUTE) == str(workspace / "sub")
        result = json.loads(_in_worker(read_file_tool, "in.txt", task_id="sid-1"))
        assert "inner" in result["content"]
    finally:
        clear_session_vars(tokens)
        release_workspace(ROUTE, "sid-1")


def test_a_replacement_session_does_not_inherit_the_old_projects_cwd(workspace, tmp_path):
    from tools.terminal_tool import get_session_cwd, resolve_task_overrides

    bind_workspace(ROUTE, "sid-old", str(workspace))
    assert get_session_cwd(ROUTE) == str(workspace)
    # Ordinary /new: no project cwd for the new session id -> the stale pin is dropped.
    assert bind_workspace(ROUTE, "sid-new", "") == ""
    assert get_session_cwd(ROUTE) is None
    assert resolve_task_overrides(ROUTE) == {}


def test_a_pin_from_another_session_is_replaced_not_reused(workspace, tmp_path):
    from tools.terminal_tool import get_session_cwd

    other = tmp_path / "other"
    other.mkdir()
    bind_workspace(ROUTE, "sid-old", str(other))
    bind_workspace(ROUTE, "sid-new", str(workspace))
    assert get_session_cwd(ROUTE) == str(workspace)


def test_clear_keeps_unrelated_overrides_and_ignores_plain_cd_records(workspace):
    from tools.terminal_tool import (
        clear_task_session_cwd, get_session_cwd, record_session_cwd,
        register_task_env_overrides, resolve_task_overrides,
    )

    register_task_env_overrides(ROUTE, {"cwd": str(workspace), "cwd_source": "session", "docker_image": "img"})
    clear_task_session_cwd(ROUTE)
    assert get_session_cwd(ROUTE) is None
    assert resolve_task_overrides(ROUTE) == {"docker_image": "img"}
    record_session_cwd("other-route", str(workspace))  # an ordinary session's own cd record
    clear_task_session_cwd("other-route")
    assert get_session_cwd("other-route") == str(workspace)


def test_ordinary_telegram_session_keeps_its_cd_across_turns(workspace):
    from tools.terminal_tool import get_session_cwd, record_session_cwd

    record_session_cwd(ROUTE, str(workspace / "sub"))  # the agent's own cd in a plain session
    for _ in range(2):  # every turn re-binds session vars with no project cwd
        tokens = set_session_vars(platform="telegram", session_key=ROUTE, session_id="sid-plain", cwd="")
        clear_session_vars(tokens)
        assert get_session_cwd(ROUTE) == str(workspace / "sub")
