"""A sandbox's working directory removed inside the sandbox must not brick the terminal (#62169).

Sandbox backends (docker, ssh, modal, daytona, vercel) cannot check their filesystem host-side, so
the shared wrapper script itself falls back. This drives a real bash through the shared
``BaseEnvironment.execute`` the way those backends do (no host-side ``_usable_cwd``); the
real-container version lives in ``test_docker_backend_real.py``.
"""

import shutil
import subprocess

import pytest

from tools.environments.base import BaseEnvironment

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

    result = env.execute("pwd", timeout=30, allow_cwd_fallback=True)
    assert result["returncode"] == 0, result["output"]
    assert result["output"].strip() == str(tmp_path.resolve())
    assert result["cwd_fallback"] == {"requested": str(doomed), "used": str(tmp_path.resolve())}
    healed = env.execute("pwd", timeout=30, allow_cwd_fallback=True)
    assert healed.get("cwd_fallback") is None  # the marker moved the cwd

    refused = env.execute("echo ran", cwd=str(doomed), timeout=30)  # explicit workdir: no opt-in
    assert refused["returncode"] == 126 and "ran" not in refused["output"]


class _RecordingCwdSandbox(_BashSandbox):
    """Records every value published to ``env.cwd``: a concurrent file op reads it at any instant."""

    @property
    def cwd(self):
        return self._cwd

    @cwd.setter
    def cwd(self, value):
        self.__dict__.setdefault("published", []).append(value)
        self._cwd = value


def test_housekeeping_command_from_root_never_publishes_root_as_the_session_cwd(tmp_path):
    (tmp_path / "sub").mkdir()
    env = _RecordingCwdSandbox(cwd=str(tmp_path), timeout=30)
    env.published.clear()

    env.execute("true", cwd="/", timeout=30)  # the background poller's shape
    assert env.published == [] and env.cwd == str(tmp_path)

    env.execute("cd sub", timeout=30)  # the session's own command still moves it
    assert env.cwd == str(tmp_path / "sub") and "/" not in env.published


@pytest.mark.parametrize("timed_out", [False, True], ids=["exited", "timed-out"])
def test_cwd_fallback_notice_survives_a_tiny_output_limit(tmp_path, monkeypatch, timed_out):
    import tools.tool_output_limits as limits
    monkeypatch.setattr(limits, "get_max_bytes", lambda: 200)
    doomed = tmp_path / "proj" / "sub"
    doomed.mkdir(parents=True)
    env = _BashSandbox(cwd=str(doomed), timeout=30)
    shutil.rmtree(tmp_path / "proj")

    result = env.execute("seq 1 5000" + ("; sleep 30" if timed_out else ""), timeout=2 if timed_out else 30,
                         bounded_capture=True, allow_cwd_fallback=True)
    assert result.get("hermes_timed_out", False) is timed_out, result
    assert result["cwd_fallback"] == {"requested": str(doomed), "used": str(tmp_path.resolve())}, result
    assert env._cwd_fallback_marker not in result["output"]
