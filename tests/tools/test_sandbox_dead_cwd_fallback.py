"""A sandbox's working directory removed inside the sandbox must not brick the terminal (#62169).

Sandbox backends (docker, ssh, modal, daytona, vercel) cannot check their filesystem host-side, so
the shared wrapper script itself falls back. This drives a real bash through the shared
``BaseEnvironment.execute`` the way those backends do (no host-side ``_usable_cwd``); the
real-container version lives in ``test_docker_backend_real.py``.
"""

import shutil
import subprocess

import pytest

from tools.environments.base import BaseEnvironment, explicit_workdir_scope

pytestmark = pytest.mark.platforms("linux", "macos")


class _BashSandbox(BaseEnvironment):
    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        return subprocess.Popen(["bash", "-c", cmd_string], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, cwd="/")

    def cleanup(self):
        pass


def test_dead_sandbox_cwd_falls_back_to_nearest_ancestor_but_explicit_workdir_is_refused(tmp_path):
    doomed = tmp_path / "proj" / "sub"
    doomed.mkdir(parents=True)
    env = _BashSandbox(cwd=str(doomed), timeout=30)
    shutil.rmtree(tmp_path / "proj")

    result = env.execute("pwd", timeout=30)
    assert result["returncode"] == 0, result["output"]
    assert result["output"].strip() == str(tmp_path.resolve())
    assert result["cwd_fallback"] == {"requested": str(doomed), "used": str(tmp_path.resolve())}
    assert env.execute("pwd", timeout=30).get("cwd_fallback") is None  # healed: the marker moved the cwd

    with explicit_workdir_scope(True):
        refused = env.execute("echo ran", cwd=str(doomed), timeout=30)
    assert refused["returncode"] == 126 and "ran" not in refused["output"]
