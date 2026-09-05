"""Exercise skill readiness through the registry with an externally owned FIFO."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("host", [
    pytest.param("linux", marks=pytest.mark.linux_only),
    pytest.param("macos", marks=pytest.mark.macos_only),
])
@pytest.mark.parametrize("credentials", ["none_required", "inherited", "scoped", "other_profile"])
def test_skill_dispatch_does_not_reread_fifo(tmp_path, host, credentials):
    skill = tmp_path / "skills" / "fifo-probe"
    skill.mkdir(parents=True)
    required = "" if credentials == "none_required" else "required_environment_variables: [OPENAI_API_KEY]\n"
    (skill / "SKILL.md").write_text(
        "---\nname: fifo-probe\ndescription: Synthetic skill.\n" + required + "---\nProbe body.\n",
        encoding="utf-8",
    )
    fifo = tmp_path / ".env"
    os.mkfifo(fifo, 0o600)
    before = fifo.stat()
    env = dict(os.environ, HERMES_HOME=str(tmp_path), HOME=str(tmp_path))
    env.pop("OPENAI_API_KEY", None)
    if credentials in {"inherited", "other_profile"}:
        env["OPENAI_API_KEY"] = "synthetic-inherited-key"
    code = """
import json, signal, sys
signal.alarm(5)
from agent import secret_scope
import tools.skills_tool
from tools.registry import registry
case = sys.argv[1]
if case in {'scoped', 'other_profile'}:
    secret_scope.set_multiplex_active(True)
    secret_scope.set_secret_scope({'OPENAI_API_KEY': 'synthetic-scoped-key'} if case == 'scoped' else {})
result = registry.dispatch('skill_view', {'name': 'fifo-probe'})
result = json.loads(result) if isinstance(result, str) else result
assert result.get('success'), result.get('error')
expected = ['OPENAI_API_KEY'] if case == 'other_profile' else []
assert result['missing_required_environment_variables'] == expected
assert 'Probe body.' in result['content']
"""
    result = subprocess.run(
        [sys.executable, "-c", code, credentials], env=env,
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, "skill discovery failed or hit the FIFO watchdog: " + result.stderr
    assert fifo.stat().st_ino == before.st_ino
    assert fifo.stat().st_mode == before.st_mode
