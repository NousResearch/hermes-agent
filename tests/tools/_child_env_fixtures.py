"""Isolated, real processes for the child-environment contract tests."""

import json
import os
import shlex
import subprocess
import sys
import venv
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def child_env(monkeypatch, tmp_path):
    # Never dump the operator's environment or source their shell startup files.
    names = (
        "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
        "TEMP", "TMP", "OS", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
    )
    seed = {k: v for k, v in os.environ.items() if k.upper() in names}
    seed.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), USER="env-test",
                HERMES_HOME=str(tmp_path / "hermes"),
                HERMES_RUNTIME_DIR=str(tmp_path / "runtime"),
                TERMINAL_ENV="local", TERMINAL_CWD=str(tmp_path), LANG="C.UTF-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from tools import env_passthrough
    from tools.environments import local
    monkeypatch.setattr(local, "_read_terminal_shell_init_config", lambda: ([], False))
    monkeypatch.setattr(env_passthrough, "_config_passthrough", {})
    token = env_passthrough._allowed_env_vars_var.set(set())
    with patch.dict(os.environ, seed, clear=True):
        try:
            yield tmp_path
        finally:
            from tools.code_kernel import shutdown_all_kernels
            shutdown_all_kernels()
            env_passthrough._allowed_env_vars_var.reset(token)


@pytest.fixture
def project_python(child_env):
    root = child_env / "project-venv"
    # A symlink to the parent binary triggers realpath's same-env shortcut;
    # copies=True semantics are essential to distinguish both interpreters.
    venv.EnvBuilder(with_pip=False, symlinks=False).create(root)
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run(
        [str(python), "-c", "import sys,sysconfig,json; print(json.dumps([sys.prefix,sysconfig.get_path('purelib')]))"],
        env=dict(os.environ), stdin=subprocess.DEVNULL, capture_output=True,
        text=True, check=True, timeout=30,
    )
    prefix, site = json.loads(result.stdout)
    assert Path(prefix).resolve() == root.resolve()
    assert Path(prefix).resolve() != Path(sys.prefix).resolve()
    assert python.resolve() != Path(sys.executable).resolve()
    Path(site, "project_only_probe.py").write_text("VALUE = 'project-only → 雪'\n", encoding="utf-8")
    return python, root


def run_code(code, mode="project", enabled_tools=("read_file",), reset=True):
    from tools.code_execution_tool import execute_code
    with patch("tools.code_execution_tool._load_config", return_value={"mode": mode}):
        result = json.loads(execute_code(code=code, task_id="child-env-test",
                                        enabled_tools=list(enabled_tools), reset=reset))
    assert result["status"] == "success", result
    return json.loads(result["output"])


def observe_terminal(env, names):
    code = "import json,os; print(json.dumps({k:os.environ.get(k) for k in " + repr(list(names)) + "}))"
    command = shlex.join([Path(sys.executable).as_posix(), "-c", code])
    result = env.execute(command)
    assert result["returncode"] == 0, result
    return json.loads(result["output"])


def observe_child(env, names):
    code = "import json,os; print(json.dumps({k:os.environ.get(k) for k in " + repr(list(names)) + "}))"
    result = subprocess.run([sys.executable, "-c", code], env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


class RunToCompletionEnv:
    """Real shell transport for the remote file-RPC path; no remote service."""

    def __init__(self, root):
        import shutil
        self.root = root
        self.bash = shutil.which("bash")
        assert self.bash, "bash is required for the POSIX file-RPC transport"
        self.env = dict(os.environ)
        self.env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), self.env["PATH"]])

    def get_temp_dir(self):
        return str(self.root)

    def execute(self, command, cwd=None, timeout=30):
        result = subprocess.run([self.bash, "-c", command], cwd=cwd or self.root,
                                env=self.env, stdin=subprocess.DEVNULL, timeout=timeout,
                                capture_output=True, text=True, encoding="utf-8")
        return {"returncode": result.returncode, "output": result.stdout + result.stderr}
