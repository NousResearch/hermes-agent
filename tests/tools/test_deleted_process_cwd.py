"""A deleted process cwd (backend moved into a worktree that was later removed) must not
crash execute_code or the file tools: ``os.getcwd()`` raises FileNotFoundError there."""

import ast
import json
import os
from pathlib import Path

import pytest

from tests.tools._child_env_fixtures import child_env

pytestmark = pytest.mark.platforms("linux", "macos")  # Windows cannot delete a process cwd


@pytest.fixture
def deleted_cwd(tmp_path):
    """chdir into a directory, then delete it; restore the original cwd afterwards."""
    original = os.getcwd()
    gone = tmp_path / "removed-worktree"
    gone.mkdir()
    os.chdir(gone)
    gone.rmdir()
    with pytest.raises(FileNotFoundError):
        os.getcwd()
    try:
        yield gone
    finally:
        os.chdir(original)


def test_resolve_child_cwd_skips_deleted_process_cwd_without_terminal_cwd(deleted_cwd, monkeypatch):
    from tools import code_execution_env as ce
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    assert ce._resolve_child_cwd("project", "") == ""
    assert ce._resolve_child_cwd("project", "staging") == "staging"


def test_resolve_child_cwd_skips_deleted_process_cwd_with_dead_terminal_cwd(deleted_cwd, monkeypatch):
    from tools import code_execution_env as ce
    monkeypatch.setenv("TERMINAL_CWD", str(deleted_cwd))
    assert ce._resolve_child_cwd("project", "") == ""


@pytest.mark.usefixtures("child_env")
def test_execute_code_succeeds_with_deleted_process_cwd(deleted_cwd, monkeypatch):
    from tools.code_execution_tool import execute_code
    from unittest.mock import patch
    monkeypatch.setenv("TERMINAL_CWD", str(deleted_cwd))
    with patch("tools.code_execution_tool._load_config", return_value={"mode": "project"}):
        result = json.loads(execute_code(code="import os; print(os.path.isdir(os.getcwd()))",
                                         task_id="deleted-cwd-test", enabled_tools=[], reset=True))
    assert result["status"] == "success", result
    assert result["output"].strip() == "True"


def test_file_tools_base_dir_survives_deleted_process_cwd(deleted_cwd, monkeypatch):
    from tools import file_tools_paths as ftp
    monkeypatch.setattr(ftp, "_authoritative_workspace_root", lambda task_id="default": None)
    monkeypatch.setattr(ftp, "_terminal_env_type_for_task", lambda task_id="default": "local")
    monkeypatch.setattr(ftp, "_uses_container_paths", lambda task_id="default": False)
    base = ftp._resolve_base_dir("t", container_paths=False)
    assert base.is_absolute() and base.is_dir()
    resolved = ftp._resolve_path_for_task("notes.txt", "t")
    assert resolved.is_absolute() and resolved.parent.is_dir()


def test_process_cwd_helper_returns_empty_when_deleted(deleted_cwd):
    from agent.runtime_cwd import process_cwd
    assert process_cwd() == ""


def test_process_cwd_helper_returns_live_cwd(tmp_path):
    from agent.runtime_cwd import process_cwd
    original = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert os.path.samefile(process_cwd(), tmp_path)
    finally:
        os.chdir(original)


# --- detection guard -------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
# Sites where a bare os.getcwd() is the deliberate primitive behind the safe helpers.
_ALLOWED = {
    ("agent/runtime_cwd.py", "process_cwd"),
    # Documented upstream to propagate the OSError (prompt/context callers); not a tool path.
    ("agent/runtime_cwd.py", "resolve_agent_cwd"),
}
_CATCHES = {"OSError", "FileNotFoundError", "Exception", "BaseException", "EnvironmentError"}


def _handler_catches_oserror(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in _CATCHES for n in names)


def _bare_getcwd_sites(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def guarded(node):
        cur = node
        while cur in parents:
            parent = parents[cur]
            if isinstance(parent, ast.Try) and cur in parent.body and any(
                    _handler_catches_oserror(h) for h in parent.handlers):
                return True
            cur = parent
        return False

    def enclosing_function(node):
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
        return ""

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getcwd" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os" and not guarded(node)):
            yield node.lineno, enclosing_function(node)


def test_no_bare_os_getcwd_in_tools():
    """Every tools/ cwd consumer goes through agent.runtime_cwd.process_cwd() (or the terminal
    tool's _safe_getcwd built on it) or guards os.getcwd() with an OSError handler."""
    offenders = []
    for path in sorted([*(_REPO / "tools").rglob("*.py"), _REPO / "agent" / "runtime_cwd.py"]):
        rel = path.relative_to(_REPO).as_posix()
        for lineno, func in _bare_getcwd_sites(path):
            if (rel, func) not in _ALLOWED:
                offenders.append(f"{rel}:{lineno} ({func or '<module>'})")
    assert not offenders, ("bare os.getcwd() crashes when the process cwd was deleted; use "
                           "agent.runtime_cwd.process_cwd(): " + ", ".join(offenders))
