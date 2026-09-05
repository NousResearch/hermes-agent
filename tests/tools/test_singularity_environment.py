"""Unit tests for tools.environments.singularity.

Extends the preflight tests in ``test_singularity_preflight.py`` with
coverage for the scratch/cache directory helpers, SIF build logic, and
the ``SingularityEnvironment`` class (constructor, instance start,
bash exec, cleanup).  All subprocess and filesystem interactions are
mocked so the suite runs on any host without apptainer/singularity
installed.

See: https://github.com/NousResearch/hermes-agent/issues/36552
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _load_snapshots / _save_snapshots
# ---------------------------------------------------------------------------

class TestSnapshotStore:
    """Thin wrappers over base._load_json_store / _save_json_store."""

    def test_load_snapshots_delegates_to_base(self, tmp_path, monkeypatch):
        import tools.environments.singularity as sing

        store = tmp_path / "snap.json"
        store.write_text('{"task1": "/overlay1"}')
        monkeypatch.setattr(sing, "_SNAPSHOT_STORE", store)
        result = sing._load_snapshots()
        assert result == {"task1": "/overlay1"}

    def test_load_snapshots_empty_when_missing(self, tmp_path, monkeypatch):
        import tools.environments.singularity as sing

        store = tmp_path / "nonexistent.json"
        monkeypatch.setattr(sing, "_SNAPSHOT_STORE", store)
        assert sing._load_snapshots() == {}

    def test_save_snapshots_writes_json(self, tmp_path, monkeypatch):
        import tools.environments.singularity as sing

        store = tmp_path / "snap.json"
        monkeypatch.setattr(sing, "_SNAPSHOT_STORE", store)
        sing._save_snapshots({"taskA": "/ovr-A"})
        assert store.exists()
        import json
        assert json.loads(store.read_text()) == {"taskA": "/ovr-A"}


# ---------------------------------------------------------------------------
# _get_scratch_dir
# ---------------------------------------------------------------------------

class TestGetScratchDir:
    """Scratch dir resolution: env override → /scratch → sandbox fallback."""

    def test_custom_scratch_dir_from_env(self, tmp_path, monkeypatch):
        import tools.environments.singularity as sing

        custom = tmp_path / "custom-scratch"
        monkeypatch.setenv("TERMINAL_SCRATCH_DIR", str(custom))
        result = sing._get_scratch_dir()
        assert result == custom
        assert custom.exists()

    def test_falls_back_to_scratch_when_writable(self, tmp_path, monkeypatch):
        """When /scratch exists and is writable, use it (Linux HPC clusters)."""
        import tools.environments.singularity as sing

        monkeypatch.delenv("TERMINAL_SCRATCH_DIR", raising=False)

        real_exists = Path.exists

        def patched_exists(self):
            if str(self) == "/scratch":
                return True
            return real_exists(self)

        # Redirect mkdir for /scratch paths to a temp dir so we don't
        # actually write to /scratch on the test host.
        redirect = tmp_path / "scratch-root"

        real_mkdir = Path.mkdir

        def patched_mkdir(self, *args, **kwargs):
            if str(self).startswith("/scratch"):
                return  # skip real mkdir for /scratch paths
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "exists", patched_exists)
        monkeypatch.setattr(Path, "mkdir", patched_mkdir)
        monkeypatch.setattr("os.access", lambda p, m: str(p) == "/scratch")
        monkeypatch.setenv("USER", "testuser")

        result = sing._get_scratch_dir()
        assert "hermes-agent" in str(result)
        assert "testuser" in str(result)

    def test_falls_back_to_sandbox_dir(self, tmp_path, monkeypatch):
        import tools.environments.singularity as sing

        monkeypatch.delenv("TERMINAL_SCRATCH_DIR", raising=False)

        # Make /scratch not exist
        real_exists = Path.exists

        def patched_exists(self):
            if str(self) == "/scratch":
                return False
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", patched_exists)

        sandbox_dir = tmp_path / "sandbox"
        monkeypatch.setattr(
            "tools.environments.base.get_sandbox_dir",
            lambda: sandbox_dir,
        )

        result = sing._get_scratch_dir()
        assert result == sandbox_dir / "singularity"
        assert result.exists()


# ---------------------------------------------------------------------------
# _get_apptainer_cache_dir
# ---------------------------------------------------------------------------

class TestGetApptainerCacheDir:
    """Cache dir resolution: APPTAINER_CACHEDIR env → scratch fallback."""

    def test_custom_cache_dir_from_env(self, tmp_path, monkeypatch):
        import tools.environments.singularity as sing

        custom = tmp_path / "custom-cache"
        monkeypatch.setenv("APPTAINER_CACHEDIR", str(custom))
        result = sing._get_apptainer_cache_dir()
        assert result == custom
        assert custom.exists()

    def test_falls_back_to_scratch_subdir(self, tmp_path, monkeypatch):
        import tools.environments.singularity as sing

        monkeypatch.delenv("APPTAINER_CACHEDIR", raising=False)
        scratch = tmp_path / "scratch"
        monkeypatch.setattr(sing, "_get_scratch_dir", lambda: scratch)

        result = sing._get_apptainer_cache_dir()
        assert result == scratch / ".apptainer"
        assert result.exists()


# ---------------------------------------------------------------------------
# _get_or_build_sif
# ---------------------------------------------------------------------------

class TestGetOrBuildSif:
    """SIF resolution: .sif passthrough → non-docker passthrough → build."""

    def test_existing_sif_path_returned_as_is(self, tmp_path):
        from tools.environments.singularity import _get_or_build_sif

        sif = tmp_path / "image.sif"
        sif.write_text("fake")
        assert _get_or_build_sif(str(sif)) == str(sif)

    def test_non_docker_non_sif_returned_as_is(self):
        from tools.environments.singularity import _get_or_build_sif

        assert _get_or_build_sif("ubuntu:22.04") == "ubuntu:22.04"

    def test_docker_url_with_cached_sif(self, tmp_path, monkeypatch):
        from tools.environments.singularity import _get_or_build_sif

        cache = tmp_path / "cache"
        monkeypatch.setattr(
            "tools.environments.singularity._get_apptainer_cache_dir",
            lambda: cache,
        )

        image = "docker://ubuntu:22.04"
        expected_name = "ubuntu-22.04.sif"
        sif_path = cache / expected_name
        sif_path.parent.mkdir(parents=True)
        sif_path.write_text("cached")

        result = _get_or_build_sif(image)
        assert result == str(sif_path)

    def test_docker_url_builds_sif_successfully(self, tmp_path, monkeypatch):
        from tools.environments.singularity import _get_or_build_sif

        cache = tmp_path / "cache"
        monkeypatch.setattr(
            "tools.environments.singularity._get_apptainer_cache_dir",
            lambda: cache,
        )

        def fake_run(cmd, **kwargs):
            # Simulate successful build: create the sif file
            sif_path = Path(cmd[2])
            sif_path.parent.mkdir(parents=True, exist_ok=True)
            sif_path.write_text("built")
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = _get_or_build_sif("docker://python:3.11", "apptainer")
        assert result.endswith("python-3.11.sif")
        assert (cache / "python-3.11.sif").exists()

    def test_docker_url_build_failure_falls_back(self, tmp_path, monkeypatch):
        from tools.environments.singularity import _get_or_build_sif

        cache = tmp_path / "cache"
        monkeypatch.setattr(
            "tools.environments.singularity._get_apptainer_cache_dir",
            lambda: cache,
        )

        fake_result = SimpleNamespace(returncode=1, stderr="build error", stdout="")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)

        result = _get_or_build_sif("docker://ubuntu:22.04", "apptainer")
        assert result == "docker://ubuntu:22.04"

    def test_docker_url_build_timeout_falls_back(self, tmp_path, monkeypatch):
        from tools.environments.singularity import _get_or_build_sif

        cache = tmp_path / "cache"
        monkeypatch.setattr(
            "tools.environments.singularity._get_apptainer_cache_dir",
            lambda: cache,
        )

        def fake_run(cmd, **kwargs):
            # Create a partial sif so the unlink path is exercised
            sif_path = Path(cmd[2])
            sif_path.parent.mkdir(parents=True, exist_ok=True)
            sif_path.write_text("partial")
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=600)

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = _get_or_build_sif("docker://ubuntu:22.04", "apptainer")
        assert result == "docker://ubuntu:22.04"
        assert not (cache / "ubuntu-22.04.sif").exists()

    def test_docker_url_build_exception_falls_back(self, tmp_path, monkeypatch):
        from tools.environments.singularity import _get_or_build_sif

        cache = tmp_path / "cache"
        monkeypatch.setattr(
            "tools.environments.singularity._get_apptainer_cache_dir",
            lambda: cache,
        )

        monkeypatch.setattr(
            subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))
        )

        result = _get_or_build_sif("docker://ubuntu:22.04", "apptainer")
        assert result == "docker://ubuntu:22.04"


# ---------------------------------------------------------------------------
# SingularityEnvironment
# ---------------------------------------------------------------------------

@pytest.fixture()
def sing_env_factory(monkeypatch, tmp_path):
    """Factory that creates a SingularityEnvironment with all externals mocked."""
    monkeypatch.setattr(
        "tools.environments.singularity._ensure_singularity_available",
        lambda: "apptainer",
    )
    monkeypatch.setattr(
        "tools.environments.singularity._get_or_build_sif",
        lambda image, exe="apptainer": image,
    )
    monkeypatch.setattr(
        "tools.environments.singularity._get_scratch_dir",
        lambda: tmp_path / "scratch",
    )
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        lambda: [],
    )
    # get_skills_directory_mount may accept kwargs; return empty list
    monkeypatch.setattr(
        "tools.credential_files.get_skills_directory_mount",
        lambda **kw: [],
    )
    monkeypatch.setattr(
        "tools.environments.base.is_interrupted",
        lambda: False,
    )
    # Prevent init_session from spawning a real bash process
    monkeypatch.setattr(
        "tools.environments.base.BaseEnvironment.init_session",
        lambda self: None,
    )

    def _factory(
        image="docker://ubuntu:22.04",
        cwd="~",
        timeout=60,
        cpu=0,
        memory=0,
        persistent_filesystem=False,
        task_id="default",
        start_returncode=0,
        run_callable=None,
    ):
        # Mock subprocess.run for instance start
        if run_callable is not None:
            monkeypatch.setattr(subprocess, "run", run_callable)
        else:
            start_result = SimpleNamespace(
                returncode=start_returncode,
                stderr="",
                stdout="",
            )
            monkeypatch.setattr(subprocess, "run", lambda *a, **kw: start_result)

        from tools.environments.singularity import SingularityEnvironment

        env = SingularityEnvironment(
            image=image,
            cwd=cwd,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            persistent_filesystem=persistent_filesystem,
            task_id=task_id,
        )
        return env

    return _factory


class TestSingularityEnvironmentConstructor:
    """Constructor wires executable, image, instance ID, and persistent overlay."""

    def test_basic_construction(self, sing_env_factory):
        env = sing_env_factory()
        assert env.executable == "apptainer"
        assert env.image == "docker://ubuntu:22.04"
        assert env.instance_id.startswith("hermes_")
        assert env._instance_started is True

    def test_persistent_filesystem_creates_overlay(self, sing_env_factory, tmp_path):
        env = sing_env_factory(persistent_filesystem=True, task_id="task-42")
        assert env._persistent is True
        assert env._overlay_dir is not None
        assert "overlay-task-42" in str(env._overlay_dir)
        assert env._overlay_dir.exists()

    def test_non_persistent_no_overlay(self, sing_env_factory):
        env = sing_env_factory(persistent_filesystem=False)
        assert env._persistent is False
        assert env._overlay_dir is None

    def test_instance_start_failure_raises(self, sing_env_factory):
        with pytest.raises(RuntimeError, match="Failed to start instance"):
            sing_env_factory(start_returncode=1)


class TestStartInstance:
    """_start_instance builds the correct apptainer command line."""

    def test_writable_tmpfs_when_not_persistent(self, sing_env_factory):
        captured_cmds = []

        def capture_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        env = sing_env_factory(persistent_filesystem=False, run_callable=capture_run)
        cmd = captured_cmds[0]
        assert "instance" in cmd
        assert "start" in cmd
        assert "--writable-tmpfs" in cmd
        assert "--containall" in cmd
        assert "--no-home" in cmd

    def test_overlay_when_persistent(self, sing_env_factory):
        captured_cmds = []

        def capture_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        env = sing_env_factory(persistent_filesystem=True, task_id="t1", run_callable=capture_run)
        cmd = captured_cmds[0]
        assert "--overlay" in cmd
        assert "--writable-tmpfs" not in cmd

    def test_memory_limit_added(self, sing_env_factory):
        captured_cmds = []

        def capture_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        env = sing_env_factory(memory=512, run_callable=capture_run)
        cmd = captured_cmds[0]
        assert "--memory" in cmd
        mem_idx = cmd.index("--memory")
        assert cmd[mem_idx + 1] == "512M"

    def test_cpu_limit_added(self, sing_env_factory):
        captured_cmds = []

        def capture_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        env = sing_env_factory(cpu=2.0, run_callable=capture_run)
        cmd = captured_cmds[0]
        assert "--cpus" in cmd
        cpu_idx = cmd.index("--cpus")
        assert cmd[cpu_idx + 1] == "2.0"

    def test_credential_mounts_added(self, sing_env_factory, monkeypatch):
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {"host_path": "/host/cred", "container_path": "/container/cred"},
            ],
        )
        captured_cmds = []

        def capture_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        env = sing_env_factory(run_callable=capture_run)
        cmd = captured_cmds[0]
        assert "--bind" in cmd
        bind_idx = cmd.index("--bind")
        assert "/host/cred:/container/cred:ro" in cmd[bind_idx + 1]

    def test_start_timeout_raises(self, sing_env_factory):
        def timeout_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=["apptainer"], timeout=120)

        with pytest.raises(RuntimeError, match="Instance start timed out"):
            sing_env_factory(run_callable=timeout_run)


class TestRunBash:
    """_run_bash spawns exec inside the running instance."""

    def test_non_login_command(self, sing_env_factory, monkeypatch):
        env = sing_env_factory()
        captured_cmd = []

        def fake_popen(cmd, stdin_data=None, **kwargs):
            captured_cmd.extend(cmd)
            proc = MagicMock()
            proc.stdout = iter(["output\n"])
            proc.returncode = 0
            proc.wait = lambda: 0
            proc.poll = lambda: 0
            return proc

        monkeypatch.setattr(
            "tools.environments.singularity._popen_bash", fake_popen
        )
        env._run_bash("echo hello")
        assert "exec" in captured_cmd
        assert "instance://" in captured_cmd[2]
        assert "bash" in captured_cmd
        assert "-c" in captured_cmd
        assert "echo hello" in captured_cmd

    def test_login_command(self, sing_env_factory, monkeypatch):
        env = sing_env_factory()
        captured_cmd = []

        def fake_popen(cmd, stdin_data=None, **kwargs):
            captured_cmd.extend(cmd)
            proc = MagicMock()
            proc.stdout = iter(["output\n"])
            proc.returncode = 0
            proc.wait = lambda: 0
            proc.poll = lambda: 0
            return proc

        monkeypatch.setattr(
            "tools.environments.singularity._popen_bash", fake_popen
        )
        env._run_bash("echo hello", login=True)
        assert "-l" in captured_cmd

    def test_raises_when_instance_not_started(self, sing_env_factory):
        env = sing_env_factory()
        env._instance_started = False
        with pytest.raises(RuntimeError, match="instance not started"):
            env._run_bash("echo hello")


class TestCleanup:
    """cleanup stops the instance and saves snapshots when persistent."""

    def test_cleanup_stops_instance(self, sing_env_factory, monkeypatch):
        env = sing_env_factory()
        stop_called = []

        def capture_run(cmd, **kwargs):
            stop_called.append(list(cmd))
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(subprocess, "run", capture_run)
        env.cleanup()
        assert len(stop_called) == 1
        assert "instance" in stop_called[0]
        assert "stop" in stop_called[0]
        assert env._instance_started is False

    def test_cleanup_skips_stop_when_not_started(self, sing_env_factory, monkeypatch):
        env = sing_env_factory()
        env._instance_started = False

        run_called = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: run_called.append(a))
        env.cleanup()
        assert len(run_called) == 0

    def test_cleanup_handles_stop_error(self, sing_env_factory, monkeypatch):
        env = sing_env_factory()
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("stop failed")),
        )
        # Should not raise
        env.cleanup()
        assert env._instance_started is False

    def test_cleanup_persistent_saves_snapshot(self, sing_env_factory, monkeypatch, tmp_path):
        env = sing_env_factory(persistent_filesystem=True, task_id="snap-task")

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stderr="", stdout=""),
        )
        saved = {}
        monkeypatch.setattr(
            "tools.environments.singularity._save_snapshots",
            lambda data: saved.update(data),
        )
        monkeypatch.setattr(
            "tools.environments.singularity._load_snapshots",
            lambda: {},
        )

        env.cleanup()
        assert "snap-task" in saved
        assert saved["snap-task"] == str(env._overlay_dir)

    def test_cleanup_non_persistent_no_snapshot(self, sing_env_factory, monkeypatch):
        env = sing_env_factory(persistent_filesystem=False)

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stderr="", stdout=""),
        )
        save_called = []
        monkeypatch.setattr(
            "tools.environments.singularity._save_snapshots",
            lambda data: save_called.append(data),
        )

        env.cleanup()
        assert len(save_called) == 0
