"""Tests for Docker runtime identity propagation in file tools."""

import threading
from unittest.mock import MagicMock, patch

import pytest

import tools.file_tools as file_tools


_RUNTIME_FINGERPRINT = "v1-file-tools-test"
_FROZEN_INIT_ENV = {"APP_MODE": "approved"}
_FROZEN_IMPLICIT_MOUNTS = ["/host/cache:/root/.cache:ro"]


def _make_env_config(**overrides):
    base = {
        "env_type": "docker",
        "docker_image": "test-image:latest",
        "singularity_image": "docker://test",
        "modal_image": "test",
        "daytona_image": "test",
        "cwd": "/workspace",
        "host_cwd": None,
        "timeout": 180,
        "container_cpu": 2,
        "container_memory": 4096,
        "container_disk": 20480,
        "container_persistent": False,
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
        "docker_env": {"APP_MODE": "approved"},
        "docker_run_as_host_user": True,
        "docker_extra_args": ["--read-only"],
        "docker_network": False,
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": False,
    }
    base.update(overrides)
    return base


class TestFileToolsContainerConfig:
    def _run(self, env_config, task_id, task_env_overrides=None):
        captured = {}
        mock_env = MagicMock()
        if env_config.get("env_type") == "docker":
            mock_env.runtime_fingerprint = _RUNTIME_FINGERPRINT

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", task_env_overrides or {}), \
             patch("tools.terminal_tool._session_cwd", {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch(
                 "tools.terminal_tool._resolve_docker_runtime_identity",
                 return_value=(
                     _FROZEN_INIT_ENV,
                     "sha256:file-tools-test",
                     _RUNTIME_FINGERPRINT,
                     _FROZEN_IMPLICIT_MOUNTS,
                 ),
             ), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_disk_usage_warning"), \
             patch("tools.file_tools._file_ops_cache", {}), \
             patch("tools.file_tools._file_ops_lock", threading.Lock()):
            file_tools._get_file_ops(task_id)

        return captured

    def test_docker_mount_cwd_to_workspace_passed(self):
        """docker_mount_cwd_to_workspace is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_mount_cwd_to_workspace=True), "t1").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is True

    def test_docker_forward_env_passed(self):
        """docker_forward_env is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_forward_env=["MY_SECRET"]), "t2").get("container_config", {})
        assert cc.get("docker_forward_env") == ["MY_SECRET"]

    def test_docker_mount_cwd_defaults_to_false(self):
        """docker_mount_cwd_to_workspace defaults to False when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_mount_cwd_to_workspace"]
        cc = self._run(cfg, "t3").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is False

    def test_docker_forward_env_defaults_to_empty_list(self):
        """docker_forward_env defaults to [] when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_forward_env"]
        cc = self._run(cfg, "t4").get("container_config", {})
        assert cc.get("docker_forward_env") == []

    def test_full_docker_runtime_config_and_frozen_identity_are_forwarded(self):
        captured = self._run(_make_env_config(), "t-runtime")

        assert captured["container_config"] == {
            "container_cpu": 2,
            "container_memory": 4096,
            "container_disk": 20480,
            "container_persistent": False,
            "modal_mode": "auto",
            "docker_volumes": [],
            "docker_mount_cwd_to_workspace": True,
            "docker_forward_env": ["MY_SECRET", "API_KEY"],
            "docker_env": {"APP_MODE": "approved"},
            "docker_run_as_host_user": True,
            "docker_extra_args": ["--read-only"],
            "docker_network": False,
            "docker_persist_across_processes": False,
            "docker_orphan_reaper": False,
        }
        assert captured["resolved_docker_init_env"] is _FROZEN_INIT_ENV
        assert (
            captured["resolved_docker_implicit_mounts"]
            is _FROZEN_IMPLICIT_MOUNTS
        )

    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"

    def test_mismatched_cached_docker_runtime_is_replaced_without_cleanup(self):
        config = _make_env_config()
        stale_env = MagicMock()
        stale_env.runtime_fingerprint = "v1-stale"
        stale_ops = MagicMock()
        stale_ops.cwd = "/workspace"
        fresh_env = MagicMock()
        fresh_env.runtime_fingerprint = _RUNTIME_FINGERPRINT
        active = {"default": stale_env}
        last_activity = {"default": 1.0}
        created = []

        def create_environment(**kwargs):
            created.append(kwargs)
            return fresh_env

        with patch("tools.terminal_tool._get_env_config", return_value=config), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._session_cwd", {}), \
             patch("tools.terminal_tool._active_environments", active), \
             patch("tools.terminal_tool._last_activity", last_activity), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch(
                 "tools.terminal_tool._resolve_docker_runtime_identity",
                 return_value=(
                     _FROZEN_INIT_ENV,
                     "sha256:file-tools-test",
                     _RUNTIME_FINGERPRINT,
                     _FROZEN_IMPLICIT_MOUNTS,
                 ),
             ), \
             patch(
                 "tools.terminal_tool._create_environment",
                 side_effect=create_environment,
             ), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.file_tools._file_ops_cache", {"default": stale_ops}), \
             patch("tools.file_tools._file_ops_lock", threading.Lock()):
            result = file_tools._get_file_ops("session-a")

        assert result.env is fresh_env
        assert active == {"default": fresh_env}
        assert len(created) == 1
        stale_env.cleanup.assert_not_called()

    def test_exact_active_runtime_rebuilds_stale_cached_wrapper(self):
        config = _make_env_config()
        stale_env = MagicMock()
        stale_env.runtime_fingerprint = "v1-stale"
        exact_env = MagicMock()
        exact_env.runtime_fingerprint = _RUNTIME_FINGERPRINT
        stale_ops = MagicMock()
        stale_ops.env = stale_env
        cache = {"default": stale_ops}
        active = {"default": exact_env}
        create_environment = MagicMock()

        with patch("tools.terminal_tool._get_env_config", return_value=config), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._session_cwd", {}), \
             patch("tools.terminal_tool._active_environments", active), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch(
                 "tools.terminal_tool._resolve_docker_runtime_identity",
                 return_value=(
                     _FROZEN_INIT_ENV,
                     "sha256:file-tools-test",
                     _RUNTIME_FINGERPRINT,
                     _FROZEN_IMPLICIT_MOUNTS,
                 ),
             ), \
             patch(
                 "tools.terminal_tool._create_environment",
                 create_environment,
             ), \
             patch("tools.file_tools._file_ops_cache", cache), \
             patch("tools.file_tools._file_ops_lock", threading.Lock()):
            result = file_tools._get_file_ops("session-a")

        assert result.env is exact_env
        assert cache["default"] is result
        create_environment.assert_not_called()
        stale_env.cleanup.assert_not_called()

    def test_missing_environment_preserves_cached_project_cwd_for_rebuild(self):
        config = _make_env_config(env_type="local", cwd="/config-cwd")
        stale_ops = MagicMock()
        stale_ops.env = MagicMock()
        stale_ops.cwd = "/project"
        captured = {}
        session_cwd = {}

        def create_environment(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("tools.terminal_tool._get_env_config", return_value=config), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._session_cwd", session_cwd), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch(
                 "tools.terminal_tool._create_environment",
                 side_effect=create_environment,
             ), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.file_tools._file_ops_cache", {"default": stale_ops}), \
             patch("tools.file_tools._file_ops_lock", threading.Lock()):
            file_tools._get_file_ops("default")

        assert captured["cwd"] == "/project"
        assert session_cwd["default"] == "/project"

    def test_backend_switch_detaches_docker_before_creating_local(self):
        config = _make_env_config(env_type="local", cwd="/project")

        class DockerEnvironment:
            pass

        stale_env = DockerEnvironment()
        stale_env.cleanup = MagicMock()
        stale_ops = MagicMock()
        stale_ops.env = stale_env
        fresh_env = MagicMock()
        active = {"default": stale_env}
        created = []

        def create_environment(**kwargs):
            created.append(kwargs)
            return fresh_env

        with patch("tools.terminal_tool._get_env_config", return_value=config), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._session_cwd", {}), \
             patch("tools.terminal_tool._active_environments", active), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch(
                 "tools.terminal_tool._create_environment",
                 side_effect=create_environment,
             ), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.file_tools._file_ops_cache", {"default": stale_ops}), \
             patch("tools.file_tools._file_ops_lock", threading.Lock()):
            result = file_tools._get_file_ops("default")

        assert result.env is fresh_env
        assert created[0]["env_type"] == "local"
        stale_env.cleanup.assert_not_called()

    def test_post_create_fingerprint_mismatch_cleans_only_new_environment(self):
        config = _make_env_config()
        wrong_env = MagicMock()
        wrong_env.runtime_fingerprint = "v1-wrong"
        active = {}
        cache = {}

        with patch("tools.terminal_tool._get_env_config", return_value=config), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._session_cwd", {}), \
             patch("tools.terminal_tool._active_environments", active), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch(
                 "tools.terminal_tool._resolve_docker_runtime_identity",
                 return_value=(
                     _FROZEN_INIT_ENV,
                     "sha256:file-tools-test",
                     _RUNTIME_FINGERPRINT,
                     _FROZEN_IMPLICIT_MOUNTS,
                 ),
             ), \
             patch(
                 "tools.terminal_tool._create_environment",
                 return_value=wrong_env,
             ), \
             patch("tools.file_tools._file_ops_cache", cache), \
             patch("tools.file_tools._file_ops_lock", threading.Lock()):
            with pytest.raises(RuntimeError, match="runtime identity changed"):
                file_tools._get_file_ops("default")

        wrong_env.cleanup.assert_called_once_with(force_remove=True)
        assert active == {}
        assert cache == {}

    @pytest.mark.parametrize("backend", ["singularity", "modal", "daytona"])
    def test_cwd_only_session_reuses_shared_non_docker_runtime(
        self,
        backend,
    ):
        """Logical session cwd must not become shared sandbox identity."""
        config = _make_env_config(
            env_type=backend,
            cwd="/root",
            modal_mode="direct",
        )
        image = config[f"{backend}_image"]

        class FakeContainer:
            cwd = "/root"

        env_class = type(f"{backend.title()}Environment", (FakeContainer,), {})
        env = env_class()
        cached = MagicMock()
        cached.env = env
        active = {"default": env}
        cache = {"default": cached}

        with patch(
            "tools.terminal_tool._get_modal_backend_state",
            return_value={"selected_backend": "direct"},
        ), patch(
            "tools.terminal_tool._get_env_config", return_value=config
        ), patch(
            "tools.terminal_tool._task_env_overrides",
            {"desktop-session": {"cwd": "/workspace/declared"}},
        ), patch(
            "tools.terminal_tool._session_cwd",
            {"desktop-session": "/workspace/after-cd"},
        ), patch(
            "tools.terminal_tool._active_environments", active
        ), patch(
            "tools.terminal_tool._last_activity", {}
        ), patch(
            "tools.terminal_tool._creation_locks", {}
        ), patch(
            "tools.terminal_tool._creation_locks_lock", threading.Lock()
        ), patch(
            "tools.terminal_tool._create_environment"
        ) as create_environment, patch(
            "tools.file_tools._file_ops_cache", cache
        ), patch(
            "tools.file_tools._file_ops_lock", threading.Lock()
        ):
            env._hermes_runtime_identity = (
                file_tools._requested_runtime_identity_for_task(
                    "desktop-session"
                )
            )
            result = file_tools._get_file_ops("desktop-session")

        assert result is cached
        assert active == {"default": env}
        create_environment.assert_not_called()
