"""PM runtime dependencies survive the kernel child boundary (#124049)."""

import json
import os
import sys
import venv
from pathlib import Path

import pytest

from pm.environments import install_state_dir, runtime_facts_path, site_packages, venv_python
from tools import code_execution_env, code_execution_tool
from tools.code_kernel import shutdown_all_kernels


@pytest.fixture
def runtime_dependency(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
    root = Path(code_execution_env.__file__).resolve().parents[1]
    generation = install_state_dir(root) / "environments" / "fixture" / "venv"
    generation.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text(
        f"version = {sys.version_info.major}.{sys.version_info.minor}.0\n", encoding="utf-8"
    )
    packages = site_packages(generation)
    packages.mkdir(parents=True)
    (packages / "pm_fixture_dependency.py").write_text("VALUE = 'runtime-dependency'\n", encoding="utf-8")
    facts = runtime_facts_path(root)
    facts.write_text(json.dumps({"packages": {"venv": {"environment": str(generation)}}}), encoding="utf-8")
    user_lib = tmp_path / "user-lib"
    user_lib.mkdir()
    (user_lib / "user_fixture_dependency.py").write_text("VALUE = 'user-dependency'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, (root, packages, user_lib))))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    shutdown_all_kernels()
    yield packages
    shutdown_all_kernels()


@pytest.mark.parametrize("mode", ["strict", "project"])
def test_runtime_dependencies_import_in_persistent_kernel(runtime_dependency, monkeypatch, mode):
    monkeypatch.setattr(code_execution_tool, "_load_config", lambda: {"mode": mode, "timeout": 15})
    first = json.loads(code_execution_tool.execute_code(
        "import pm_fixture_dependency as dep\nimport user_fixture_dependency as user\nprint(dep.VALUE, user.VALUE)",
        task_id="pm-runtime-import",
    ))
    assert first["status"] == "success", first
    assert "runtime-dependency user-dependency" in first["output"]
    second = json.loads(code_execution_tool.execute_code("print(dep.VALUE)", task_id="pm-runtime-import"))
    assert second["status"] == "success", second
    assert second["kernel"]["reused"] is True
    assert "runtime-dependency" in second["output"]


def test_project_interpreter_keeps_runtime_dependencies_out(runtime_dependency, tmp_path, monkeypatch):
    project = tmp_path / "project-env"
    venv.EnvBuilder(with_pip=False).create(project)
    monkeypatch.setenv("VIRTUAL_ENV", str(project))
    monkeypatch.setattr(code_execution_tool, "_load_config", lambda: {"mode": "project", "timeout": 15})
    assert code_execution_env._resolve_child_python("project") == str(venv_python(project))
    result = json.loads(code_execution_tool.execute_code(
        "import importlib.util\nimport user_fixture_dependency as user\n"
        "print(importlib.util.find_spec('pm_fixture_dependency') is None, user.VALUE)",
        task_id="pm-project-isolation",
    ))
    assert result["status"] == "success", result
    assert "True user-dependency" in result["output"]
