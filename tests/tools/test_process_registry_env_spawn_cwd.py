"""Background launches on environment-backed backends run in the requested cwd."""

import shutil
import time

import pytest

from tools.environments.local import LocalEnvironment
from tools.process_registry import ProcessRegistry


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_spawn_via_env_runs_command_in_requested_cwd(tmp_path):
    default_dir = tmp_path / "env-default"
    workdir = tmp_path / "requested"
    default_dir.mkdir()
    workdir.mkdir()
    env = LocalEnvironment(cwd=str(default_dir), timeout=30)
    registry = ProcessRegistry()
    try:
        session = registry.spawn_via_env(env, "pwd -P", cwd=str(workdir), task_id="bg-cwd")
        deadline = time.monotonic() + 30
        while not session.exited and time.monotonic() < deadline:
            time.sleep(0.2)

        assert session.exited and session.exit_code == 0, session.output_buffer
        assert session.output_buffer.strip() == str(workdir.resolve())
    finally:
        registry.kill_all()
        env.cleanup()
