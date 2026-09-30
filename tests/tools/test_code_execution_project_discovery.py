"""Project-mode dependency discovery without ambient activation (#129101)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.tools._child_env_fixtures import child_env, project_python, run_code  # noqa: F401


@pytest.mark.platforms("linux", "macos", "windows")
def test_operator_project_venv_is_used_without_activation(child_env, project_python, monkeypatch):
    _, prefix = project_python
    project = child_env / "project"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet", str(project)], check=True)
    target = project / ".venv"
    shutil.move(str(prefix), target)
    monkeypatch.chdir(project)
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    assert not os.environ.get("VIRTUAL_ENV")
    assert not os.environ.get("CONDA_PREFIX")
    observed = run_code("import json, sys; print(json.dumps({'prefix': sys.prefix}))")
    assert Path(observed["prefix"]).resolve() == target.resolve()
    observed = run_code("import json, project_only_probe; print(json.dumps(project_only_probe.VALUE))")
    assert observed == "project-only → 雪"


@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("case", ["venv", "missing", "broken", "exception", "strict", "explicit", "untrusted", "nested", "kanban", "stale", "junction"])
def test_discovery_preserves_trust_and_fallbacks(child_env, project_python, monkeypatch, case):
    import sys
    from tools import code_execution_env as ce
    from agent.lsp.workspace import clear_cache

    python, prefix = project_python
    project = child_env / "project"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet", str(project)], check=True)
    monkeypatch.chdir(project)
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    cwd = project
    if case in ("untrusted", "nested"):
        cwd = (child_env if case == "untrusted" else project) / "clone"
        cwd.mkdir()
        subprocess.run(["git", "init", "--quiet", str(cwd)], check=True)
    if case == "kanban":
        monkeypatch.setenv("HERMES_KANBAN_TASK", "test-task")
    clear_cache()
    if case == "stale":
        from agent.lsp.workspace import find_git_worktree
        cwd = project / "new-clone"
        cwd.mkdir()
        assert find_git_worktree(str(cwd)) == str(project)
        subprocess.run(["git", "init", "--quiet", str(cwd)], check=True)
    if case == "junction":
        outside = child_env / "outside"
        outside.mkdir()
        cwd = project / "linked"
        if os.name == "nt":
            subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(cwd), str(outside)], check=True, capture_output=True)
        else:
            cwd.symlink_to(outside, target_is_directory=True)
    target = cwd / ("venv" if case == "venv" else ".venv")
    if case != "explicit":
        shutil.move(str(prefix), target)
    candidate = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if case == "missing":
        candidate.unlink()
    if case in ("broken", "exception"):
        outcome = subprocess.CompletedProcess([], 1, "")
        def failed_probe(*args, **kwargs):
            if case == "exception":
                raise OSError("fixture spawn failure")
            return outcome
        monkeypatch.setattr(ce.subprocess, "run", failed_probe)
    if case in ("strict", "untrusted", "nested", "kanban", "stale", "junction"):
        def forbidden_probe(*args, **kwargs):
            raise AssertionError("untrusted/strict interpreter must never be probed")
        monkeypatch.setattr(ce, "_probe_python", forbidden_probe)
    if case == "explicit":
        monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    ce._usable_python_cache.clear()
    try:
        actual = ce._resolve_child_python("strict" if case == "strict" else "project", str(cwd))
        expected = candidate if case == "venv" else python if case == "explicit" else Path(sys.executable)
        assert Path(actual).resolve() == expected.resolve()
    finally:
        clear_cache()
        ce._usable_python_cache.clear()
