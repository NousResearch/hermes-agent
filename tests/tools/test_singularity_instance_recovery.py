"""Tests for SingularityEnvironment out-of-band instance-removal recovery.

Mirrors DockerEnvironment's container-recreation recovery: an instance
reaped out-of-band (idle cleanup, OOM kill, host reboot) should be detected
and transparently recreated before retrying once, rather than failing every
subsequent command.
"""

import subprocess
from unittest.mock import MagicMock

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.singularity import SingularityEnvironment


@pytest.fixture(autouse=True)
def _mock_singularity(monkeypatch):
    monkeypatch.setattr("tools.environments.singularity.shutil.which",
                        lambda name: f"/usr/bin/{name}" if name == "apptainer" else None)
    monkeypatch.setattr("tools.environments.singularity.subprocess.run",
                        lambda *a, **k: subprocess.CompletedProcess([], 0))
    monkeypatch.setattr("tools.environments.singularity._popen_bash",
                        lambda *a, **k: MagicMock(stdout=iter([]),
                                                  stderr=iter([]),
                                                  stdin=MagicMock()))
    monkeypatch.setattr("tools.environments.base.time.sleep", lambda _: None)


def _make_env(**kwargs):
    kwargs.setdefault("image", "test-image")
    return SingularityEnvironment(**kwargs)


class TestIsInstanceGone:
    def test_detects_known_signatures(self):
        env = _make_env()
        assert env._is_instance_gone(255, "FATAL: no such instance hermes_abc123")
        assert env._is_instance_gone(255, "ERROR: instance does not exist")

    def test_ignores_non_255_returncode(self):
        env = _make_env()
        assert not env._is_instance_gone(1, "no such instance")

    def test_ignores_unrelated_255_exit(self):
        env = _make_env()
        assert not env._is_instance_gone(255, "some script inside the container exited 255")


class TestRecoveryFlow:
    def test_execute_recreates_instance_and_retries_once(self, monkeypatch):
        env = _make_env()
        calls = {"n": 0}

        def fake_execute(self, command, cwd="", **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"output": "FATAL: no such instance", "returncode": 255}
            return {"output": "ok", "returncode": 0}

        monkeypatch.setattr(BaseEnvironment, "execute", fake_execute)
        monkeypatch.setattr(env, "_recreate_instance", lambda: True)

        result = env.execute("echo hi")
        assert result == {"output": "ok", "returncode": 0}
        assert calls["n"] == 2

    def test_execute_does_not_retry_when_recreate_fails(self, monkeypatch):
        env = _make_env()
        calls = {"n": 0}

        def fake_execute(self, command, cwd="", **kwargs):
            calls["n"] += 1
            return {"output": "FATAL: no such instance", "returncode": 255}

        monkeypatch.setattr(BaseEnvironment, "execute", fake_execute)
        monkeypatch.setattr(env, "_recreate_instance", lambda: False)

        result = env.execute("echo hi")
        assert result == {"output": "FATAL: no such instance", "returncode": 255}
        assert calls["n"] == 1

    def test_recreate_instance_generates_new_id_and_reinits(self, monkeypatch):
        env = _make_env()
        old_id = env.instance_id

        started = {"n": 0}
        monkeypatch.setattr(env, "_start_instance",
                            lambda: started.__setitem__("n", started["n"] + 1))
        monkeypatch.setattr(env, "init_session", lambda: None)

        assert env._recreate_instance() is True
        assert env.instance_id != old_id
        assert started["n"] == 1

    def test_recreate_instance_returns_false_when_start_fails(self, monkeypatch):
        env = _make_env()

        def _raise():
            raise RuntimeError("start failed")

        monkeypatch.setattr(env, "_start_instance", _raise)
        assert env._recreate_instance() is False

    def test_recreate_instance_preserves_overlay_dir(self, monkeypatch, tmp_path):
        env = _make_env(persistent_filesystem=True, task_id="t1")
        overlay = env._overlay_dir
        assert overlay is not None

        monkeypatch.setattr(env, "_start_instance", lambda: None)
        monkeypatch.setattr(env, "init_session", lambda: None)

        assert env._recreate_instance() is True
        assert env._overlay_dir == overlay
