"""Tests for docker container_config key propagation in file_tools."""

import threading
from unittest.mock import patch, MagicMock
import pytest
import tools.code_execution_tool as code_execution_tool
import tools.file_tools as file_tools
import tools.terminal_tool as terminal_tool


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
        "docker_env": {"A": "B"},
        "docker_extra_args": ["--shm-size=1g"],
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": False,
        "tenki_api_endpoint": "https://api.tenki.test",
        "tenki_workspace_id": "ws-123",
        "tenki_name_prefix": "agent",
        "tenki_allow_inbound": True,
        "tenki_allow_outbound": False,
        "tenki_max_duration": 7200,
        "tenki_idle_timeout": 600,
        "tenki_pause_retention": 3600,
        "tenki_sync_hermes_home": True,
        "tenki_forward_env": ["GITHUB_TOKEN"],
    }
    base.update(overrides)
    return base


class TestFileToolsContainerConfig:
    def _run(self, env_config, task_id, task_env_overrides=None):
        captured = {}
        mock_env = MagicMock()

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", task_env_overrides or {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_disk_usage_warning"), \
             patch("tools.file_tools._file_ops_cache", {}), \
             patch("tools.file_tools._file_ops_lock", __import__("threading").Lock()):
            file_tools._get_file_ops(task_id)

        return captured

    def test_docker_mount_cwd_to_workspace_passed(self):
        """docker_mount_cwd_to_workspace is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_mount_cwd_to_workspace=True), "t1").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is True


    def test_shared_container_config_fields_are_forwarded(self):
        """File tools use the same container-config builder as terminal execution."""
        cc = self._run(_make_env_config(), "t5").get("container_config", {})

        assert cc.get("docker_env") == {"A": "B"}
        assert cc.get("docker_extra_args") == ["--shm-size=1g"]
        assert cc.get("docker_persist_across_processes") is False
        assert cc.get("docker_orphan_reaper") is False
        assert cc.get("tenki_name_prefix") == "agent"
        assert cc.get("tenki_api_endpoint") == "https://api.tenki.test"
        assert cc.get("tenki_workspace_id") == "ws-123"
        assert cc.get("tenki_allow_inbound") is True
        assert cc.get("tenki_allow_outbound") is False
        assert cc.get("tenki_max_duration") == 7200
        assert cc.get("tenki_idle_timeout") == 600
        assert cc.get("tenki_pause_retention") == 3600
        assert cc.get("tenki_sync_hermes_home") is True
        assert cc.get("tenki_forward_env") == ["GITHUB_TOKEN"]

    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"


class TestExecuteCodeContainerConfig:
    def test_execute_code_uses_shared_container_config_for_tenki(self):
        captured = {}
        mock_env = MagicMock()
        env_config = _make_env_config(
            env_type="tenki",
            tenki_image="tenki-image",
            cwd="/home/tenki",
        )

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._retiring_environments", {}), \
             patch("tools.terminal_tool._env_lock", threading.Lock()), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            env, env_type = code_execution_tool._get_or_create_env("exec-tenki")

        assert env is mock_env
        assert env_type == "tenki"
        assert captured["env_type"] == "tenki"
        assert captured["image"] == "tenki-image"
        assert captured["cwd"] == "/home/tenki"
        assert captured["task_id"].startswith("tenki:")
        assert captured["task_id"].endswith(":default")
        cc = captured["container_config"]
        assert cc["container_persistent"] is False
        assert cc["tenki_api_endpoint"] == "https://api.tenki.test"
        assert cc["tenki_workspace_id"] == "ws-123"
        assert cc["tenki_name_prefix"] == "agent"
        assert cc["tenki_allow_inbound"] is True
        assert cc["tenki_allow_outbound"] is False
        assert cc["tenki_max_duration"] == 7200
        assert cc["tenki_idle_timeout"] == 600
        assert cc["tenki_pause_retention"] == 3600
        assert cc["tenki_sync_hermes_home"] is True
        assert cc["tenki_forward_env"] == ["GITHUB_TOKEN"]

    def test_execute_code_sanitizes_host_cwd_when_it_creates_tenki_first(self):
        captured = {}
        mock_env = MagicMock()
        env_config = _make_env_config(
            env_type="tenki",
            tenki_image="tenki-image",
            cwd="/home/tenki",
        )
        overrides = {
            "desktop-session": {
                "cwd": "/Users/alice/workspace",
            },
        }

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", overrides), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._retiring_environments", {}), \
             patch("tools.terminal_tool._env_lock", threading.Lock()), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            code_execution_tool._get_or_create_env("desktop-session")

        assert captured["cwd"] == "/home/tenki"


def test_tenki_live_cwd_registration_is_sanitized_for_file_operations():
    env_config = _make_env_config(
        env_type="tenki",
        tenki_image="tenki-image",
        cwd="/home/tenki",
    )
    raw_task_id = "desktop-live-cwd"
    effective_task_id = terminal_tool._resolve_environment_cache_key(
        raw_task_id,
        "tenki",
    )
    live_env = MagicMock()
    live_env.cwd = "/home/tenki"
    live_env.execute.return_value = {
        "output": "ok",
        "returncode": 0,
    }
    active = {effective_task_id: live_env}
    session_cwds = {}

    with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
         patch("tools.terminal_tool._task_env_overrides", {}), \
         patch("tools.terminal_tool._session_cwd", session_cwds), \
         patch("tools.terminal_tool._active_environments", active), \
         patch("tools.terminal_tool._last_activity", {}), \
         patch("tools.terminal_tool._retiring_environments", {}), \
         patch("tools.terminal_tool._env_lock", threading.Lock()), \
         patch("tools.terminal_tool._creation_locks", {}), \
         patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
         patch("tools.file_tools._file_ops_cache", {}), \
         patch("tools.file_tools._file_ops_lock", threading.Lock()):
        terminal_tool.register_task_env_overrides(
            raw_task_id,
            {"cwd": "/Users/alice/project"},
        )
        file_ops = file_tools._get_file_ops(raw_task_id)
        file_ops._exec("pwd")

    # Keep the host workspace in session state for Desktop/ACP, but never send
    # that host-only path to the Tenki guest.
    assert session_cwds[raw_task_id] == "/Users/alice/project"
    assert live_env.cwd == "/home/tenki"
    live_env.execute.assert_called_once_with("pwd", cwd="/home/tenki")


def test_environment_creation_slot_survives_retirement_until_last_waiter():
    task_id = "tenki:profile:creation-slot"
    old_env = MagicMock()
    completed = threading.Event()
    active = {task_id: old_env}
    retirements = {task_id: completed}
    creation_slots = {}
    env_lock = threading.Lock()
    creation_slots_lock = threading.Lock()
    first_selected = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def first_creator():
        with terminal_tool._environment_creation_lock(task_id):
            _key, env = terminal_tool._select_active_environment(task_id)
            assert env is None
            first_selected.set()
            assert release_first.wait(timeout=2)

    def second_creator():
        with terminal_tool._environment_creation_lock(task_id):
            second_acquired.set()

    with patch("tools.terminal_tool._active_environments", active), \
         patch("tools.terminal_tool._last_activity", {task_id: 0}), \
         patch("tools.terminal_tool._retiring_environments", retirements), \
         patch("tools.terminal_tool._env_lock", env_lock), \
         patch("tools.terminal_tool._creation_locks", creation_slots), \
         patch("tools.terminal_tool._creation_locks_lock", creation_slots_lock), \
         patch("tools.file_tools._file_ops_cache", {}), \
         patch("tools.file_tools._file_ops_lock", threading.Lock()):
        first = threading.Thread(target=first_creator)
        first.start()

        # The first creator owns the slot but waits behind the retirement
        # tombstone. Finishing retirement must release the tombstone without
        # deleting that slot generation.
        for _ in range(100):
            with creation_slots_lock:
                if task_id in creation_slots:
                    break
            threading.Event().wait(0.01)
        terminal_tool._finish_environment_retirement(
            task_id,
            old_env,
            completed,
        )
        assert first_selected.wait(timeout=2)

        with creation_slots_lock:
            original_slot = creation_slots[task_id]
        second = threading.Thread(target=second_creator)
        second.start()
        threading.Event().wait(0.05)

        with creation_slots_lock:
            assert creation_slots[task_id] is original_slot
            assert original_slot.users == 2
        assert second_acquired.is_set() is False

        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_acquired.is_set() is True
    assert creation_slots == {}


def test_environment_registration_discards_losing_candidate():
    task_id = "tenki:profile:registration-loser"
    existing = MagicMock()
    candidate = MagicMock()
    candidate.shares_remote_resource_with.return_value = False

    with patch(
        "tools.terminal_tool._active_environments",
        {task_id: existing},
    ), patch("tools.terminal_tool._env_lock", threading.Lock()):
        selected = terminal_tool._register_active_environment(
            task_id,
            candidate,
        )

    assert selected is existing
    candidate.discard.assert_called_once_with()


@pytest.mark.parametrize("entrypoint", ["terminal", "file", "code"])
def test_environment_entrypoints_wait_before_selecting_retiring_tenki(
    entrypoint,
):
    env_config = _make_env_config(
        env_type="tenki",
        tenki_image="tenki-image",
        cwd="/home/tenki",
    )
    raw_task_id = "retirement-barrier"
    effective_task_id = terminal_tool._resolve_environment_cache_key(
        raw_task_id,
        "tenki",
    )
    old_env = MagicMock()
    old_env.cwd = "/home/tenki"
    replacement = MagicMock()
    replacement.cwd = "/home/tenki"
    replacement.execute.return_value = {
        "output": "ok",
        "returncode": 0,
    }
    active = {effective_task_id: old_env}
    last_activity = {effective_task_id: 0}
    retirements = {}
    env_lock = threading.Lock()
    wait_entered = threading.Event()
    create_called = threading.Event()

    class TrackingEvent(threading.Event):
        def wait(self, timeout=None):
            wait_entered.set()
            return super().wait(timeout)

    retirement = TrackingEvent()
    retirements[effective_task_id] = retirement

    def fake_create_env(**_kwargs):
        create_called.set()
        return replacement

    result = []

    def invoke():
        if entrypoint == "terminal":
            result.append(
                terminal_tool.terminal_tool(
                    "echo ok",
                    task_id=raw_task_id,
                    force=True,
                )
            )
        elif entrypoint == "file":
            result.append(file_tools._get_file_ops(raw_task_id))
        else:
            result.append(
                code_execution_tool._get_or_create_env(raw_task_id)
            )

    with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
         patch("tools.terminal_tool._active_environments", active), \
         patch("tools.terminal_tool._last_activity", last_activity), \
         patch("tools.terminal_tool._retiring_environments", retirements), \
         patch("tools.terminal_tool._env_lock", env_lock), \
         patch("tools.terminal_tool._creation_locks", {}), \
         patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
         patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.file_tools._file_ops_cache", {}), \
         patch("tools.file_tools._file_ops_lock", threading.Lock()):
        worker = threading.Thread(target=invoke)
        worker.start()
        assert wait_entered.wait(timeout=2)
        assert create_called.is_set() is False
        assert old_env.execute.called is False

        with env_lock:
            active.pop(effective_task_id)
            retirements.pop(effective_task_id)
        retirement.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert create_called.is_set() is True
    assert old_env.execute.called is False
    if entrypoint == "file":
        assert result[0].env is replacement
    elif entrypoint == "code":
        assert result[0] == (replacement, "tenki")
