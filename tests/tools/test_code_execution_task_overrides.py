"""Regression test: execute_code must honor raw-task-id-keyed cwd overrides.

``resolve_task_overrides`` (added in #46552) reads a registered task/session
override under its *raw* id first, then falls back to the collapsed
container id -- because a CWD-only override collapses
``_resolve_container_task_id`` to ``"default"`` and would otherwise be
silently dropped. ``file_tools._get_file_ops`` and ``terminal_tool`` were
migrated to this helper; ``code_execution_tool._get_or_create_env`` still
read ``_task_env_overrides.get(effective_task_id, {})`` directly, which only
sees the collapsed id and misses a CWD-only override registered under the
originating session's raw task id.
"""

import threading
from unittest.mock import MagicMock, patch

import tools.code_execution_tool as code_execution_tool


def _make_env_config(**overrides):
    base = {
        "env_type": "local",
        "docker_image": "test-image:latest",
        "singularity_image": "docker://test",
        "modal_image": "test",
        "daytona_image": "test",
        "cwd": "/config-cwd",
        "host_cwd": None,
        "timeout": 180,
        "local_persistent": False,
    }
    base.update(overrides)
    return base


class TestCodeExecutionRespectsRawTaskCwdOverride:
    def _run(self, env_config, task_id, task_env_overrides=None):
        captured = {}
        mock_env = MagicMock()

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", task_env_overrides or {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._env_lock", threading.Lock()), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            code_execution_tool._get_or_create_env(task_id)

        return captured

    def test_cwd_only_raw_task_override_reaches_execute_code_env(self):
        """A CWD-only override registered under a raw session id (which
        collapses to "default") must still set the sandbox cwd."""
        captured = self._run(
            _make_env_config(),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"

    def test_no_override_falls_back_to_config_cwd(self):
        """Without any registered override, the configured default cwd is used."""
        captured = self._run(_make_env_config(), "default", task_env_overrides={})

        assert captured["cwd"] == "/config-cwd"

    def test_isolation_keyed_override_resolves_under_its_own_task_id(self):
        """An override with an isolation key (e.g. docker_image) keeps the
        task isolated and its cwd override must still apply."""
        captured = self._run(
            _make_env_config(env_type="docker"),
            "rollout-123",
            task_env_overrides={
                "rollout-123": {"docker_image": "custom:latest", "cwd": "/rollout/workspace"},
            },
        )

        assert captured["task_id"] == "rollout-123"
        assert captured["cwd"] == "/rollout/workspace"
        assert captured["image"] == "custom:latest"
