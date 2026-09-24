from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.environments.local import LocalEnvironment
from tools.terminal_tool import _ExecPlan, _run_foreground
from tools.terminal_workspace_mutation import (
    capture_git_workspace,
    detect_git_workspace_mutation,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Hermes Test")
    _git(root, "config", "user.email", "hermes@example.invalid")
    (root / "app.py").write_text("before\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-qm", "initial")
    return root


def test_detects_script_write_in_clean_repo(tmp_path):
    root = _repo(tmp_path)
    before = capture_git_workspace(root)

    (root / "app.py").write_text("after\n", encoding="utf-8")

    mutation = detect_git_workspace_mutation(before, root)
    assert mutation == {
        "workspace": str(root.resolve()),
        "operation": "unknown",
        "paths": [str((root / "app.py").resolve())],
    }


def test_detects_second_write_to_already_dirty_file(tmp_path):
    root = _repo(tmp_path)
    (root / "app.py").write_text("dirty once\n", encoding="utf-8")
    before = capture_git_workspace(root)

    (root / "app.py").write_text("dirty twice with a different size\n", encoding="utf-8")

    mutation = detect_git_workspace_mutation(before, root)
    assert mutation is not None
    assert mutation["paths"] == [str((root / "app.py").resolve())]


def test_detects_files_landed_in_a_commit(tmp_path):
    root = _repo(tmp_path)
    (root / "app.py").write_text("committed\n", encoding="utf-8")
    before = capture_git_workspace(root)

    _git(root, "add", "app.py")
    _git(root, "commit", "-qm", "change")

    mutation = detect_git_workspace_mutation(before, root)
    assert mutation is not None
    assert mutation["paths"] == [str((root / "app.py").resolve())]


def test_read_only_command_reports_no_mutation(tmp_path):
    root = _repo(tmp_path)
    before = capture_git_workspace(root)

    _git(root, "status", "--short")

    assert detect_git_workspace_mutation(before, root) is None


def test_foreground_terminal_result_reports_workspace_mutation(tmp_path):
    root = _repo(tmp_path)
    env = LocalEnvironment(cwd=str(root), timeout=30)
    plan = _ExecPlan(
        config={},
        env_type="local",
        effective_task_id="terminal-mutation",
        image="",
        cwd=str(root),
        host_cwd=str(root),
        effective_timeout=30,
    )
    try:
        payload = json.loads(_run_foreground(
            "printf 'after\\n' > app.py",
            env,
            plan,
            task_id="terminal-mutation",
            session_id="terminal-mutation",
            session_key="terminal-mutation",
            workdir=str(root),
            approval_note=None,
            clear_interrupt=False,
        ))
    finally:
        env.cleanup()

    assert payload["exit_code"] == 0
    assert payload["workspace_mutation"] == {
        "workspace": str(root.resolve()),
        "operation": "unknown",
        "paths": [str((root / "app.py").resolve())],
    }
