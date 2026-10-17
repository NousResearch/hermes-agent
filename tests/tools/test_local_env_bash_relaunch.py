"""A cached bash that fails to launch is re-resolved on the next command
instead of failing every command until restart."""
import pytest

from pm import shell
from tools.environments import local
from tools.environments.local import LocalEnvironment


def test_launch_failure_re_resolves_bash_on_next_command(tmp_path, monkeypatch):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    resolutions = []
    real_resolve = shell._resolve_bash
    monkeypatch.setattr(shell, "_resolved", None)
    monkeypatch.setattr(shell, "_resolve_bash", lambda: resolutions.append(1) or real_resolve())

    def quarantined(*args, **kwargs):
        raise PermissionError(5, "Access is denied")

    with monkeypatch.context() as m:
        m.setattr(local.subprocess, "Popen", quarantined)
        with pytest.raises(PermissionError):
            env._run_bash("true")
    assert len(resolutions) == 1

    proc = env._run_bash("echo ok")
    assert proc.communicate(timeout=15)[0].strip() == "ok"
    assert len(resolutions) == 2
