"""delegate_task children must not share the parent session\x27s cwd record.

Children run inside contextvars.copy_context() and therefore inherit the
parent\x27s session key. When the terminal cwd record was keyed by session key,
the parent and every child shared one record: a child\x27s cd moved the
parent and its siblings, and the per-child record seeded at spawn (the
isolated worktree path when delegation.worktree_isolation is on) was never
read, so children started in the parent\x27s or a sibling\x27s directory.
"""
import contextvars
import json

import pytest

from tools import terminal_tool as tt
from tools.approval_context import reset_current_session_key, set_current_session_key

PARENT = "parent-session"
CHILDREN = ("child-a", "child-b")


@pytest.fixture
def dirs(tmp_path):
    out = {name: tmp_path / name for name in ("parent", *CHILDREN)}
    for path in out.values():
        path.mkdir()
    return out


@pytest.fixture
def session(dirs):
    token = set_current_session_key(PARENT)
    tt.record_session_cwd(PARENT, str(dirs["parent"]))
    for child in CHILDREN:  # what the delegate spawn path does per child
        tt.record_session_cwd(child, str(dirs[child]))
        tt.register_container_alias(child, None)
    yield
    reset_current_session_key(token)
    for key in (PARENT, *CHILDREN):
        tt.clear_task_env_overrides(key)
        tt.clear_session_cwd(key)


def _pwd(task_id):
    return json.loads(tt.terminal_tool("pwd", task_id=task_id))["output"].strip()


def test_cwd_record_key_only_diverges_for_children(session):
    assert tt.cwd_record_key("child-a", PARENT) == "child-a"
    assert tt.cwd_record_key("not-a-child", PARENT) == PARENT
    assert tt.cwd_record_key(None, PARENT) == PARENT


def test_child_starts_in_its_seeded_directory(session, dirs):
    for child in CHILDREN:
        assert contextvars.copy_context().run(_pwd, child) == str(dirs[child])


def test_child_cd_does_not_move_parent_or_sibling(session, dirs, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    contextvars.copy_context().run(tt.terminal_tool, f"cd {elsewhere}", task_id="child-a")
    assert tt.get_session_cwd("child-a") == str(elsewhere)
    assert tt.get_session_cwd(PARENT) == str(dirs["parent"])
    assert contextvars.copy_context().run(_pwd, "child-b") == str(dirs["child-b"])
