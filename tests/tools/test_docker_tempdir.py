"""DockerEnvironment must place session snap/cwd artifacts on a writable path.

Docker's ``--tmpfs`` replicates the mode/ownership of the image's existing
mount point, so images whose /tmp is not world-writable get a root:root 0755
tmpfs at /tmp. A non-root container user (image ``USER`` or
``docker_run_as_host_user: true``) then can't write the snapshot wrapper or
cwd marker (EACCES) and session state tracking breaks. A ``TMPDIR`` configured
via ``terminal.docker_env`` must be honored for these artifacts.
"""

import subprocess

from tools.environments import docker as docker_env


def _mock_docker(monkeypatch):
    """Intercept docker CLI calls so no real docker daemon is needed.

    Pre-seeds the cgroup-limit probe cache so the throwaway probe container
    does not run (same pattern as test_docker_environment.py).
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    docker_env._cgroup_limits_ok = True

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)


class TestDockerTempDir:
    def test_honors_configured_tmpdir_for_session_artifacts(self, monkeypatch):
        _mock_docker(monkeypatch)

        env = docker_env.DockerEnvironment(
            image="python:3.11",
            task_id="test-tmpdir",
            env={"TMPDIR": "/workspace/tmp/"},
        )

        assert env.get_temp_dir() == "/workspace/tmp"
        assert env._snapshot_path == f"/workspace/tmp/hermes-snap-{env._session_id}.sh"
        assert env._cwd_file == f"/workspace/tmp/hermes-cwd-{env._session_id}.txt"

    def test_defaults_to_tmp_when_tmpdir_not_configured(self, monkeypatch):
        _mock_docker(monkeypatch)
        # The HOST TMPDIR must not leak into container artifact paths —
        # only the explicitly configured docker_env matters.
        monkeypatch.setenv("TMPDIR", "/host/private/tmp")

        env = docker_env.DockerEnvironment(
            image="python:3.11",
            task_id="test-tmpdir-default",
        )

        assert env.get_temp_dir() == "/tmp"
        assert env._snapshot_path == f"/tmp/hermes-snap-{env._session_id}.sh"
        assert env._cwd_file == f"/tmp/hermes-cwd-{env._session_id}.txt"
