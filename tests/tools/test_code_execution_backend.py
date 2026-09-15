"""Tests for execute_code backend resolution.

The key contract: ``code_execution.backend`` overrides ``terminal.backend``
for the ``execute_code`` tool only. When unset, execute_code inherits the
terminal backend.
"""

from unittest.mock import patch

from tools import code_execution_tool as ce


class TestBackendResolution:
    def test_defaults_to_terminal_backend_when_override_unset(self):
        """Historical behavior: execute_code inherits terminal.backend."""
        with patch.object(ce, "_load_config", return_value={}), patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "docker"},
        ):
            assert ce._resolve_code_execution_env_type() == "docker"

    def test_override_wins_over_terminal_backend(self):
        """Setting code_execution.backend isolates execute_code from terminal."""
        with patch.object(ce, "_load_config", return_value={"backend": "nsjail"}), patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "local"},
        ):
            assert ce._resolve_code_execution_env_type() == "nsjail"

    def test_empty_override_is_ignored(self):
        """An empty backend must not shadow the terminal value."""
        with patch.object(ce, "_load_config", return_value={"backend": ""}), patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "singularity"},
        ):
            assert ce._resolve_code_execution_env_type() == "singularity"

    def test_whitespace_override_is_ignored(self):
        """Whitespace-only values don't count."""
        with patch.object(ce, "_load_config", return_value={"backend": "   "}), patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "local"},
        ):
            assert ce._resolve_code_execution_env_type() == "local"

    def test_falls_back_to_local_when_everything_missing(self):
        """Safe default when neither terminal nor code_execution is set."""
        with patch.object(ce, "_load_config", return_value={}), patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": None},
        ):
            assert ce._resolve_code_execution_env_type() == "local"


class TestCacheNamespacing:
    def test_separate_cache_key_when_backends_differ(self):
        """A distinct backend gets a namespaced task id and separate env."""
        active = {}
        fake_config = {
            "env_type": "local",
            "docker_image": "", "singularity_image": "", "modal_image": "",
            "daytona_image": "", "container_cpu": 1, "container_memory": 256,
            "container_disk": 64, "container_persistent": True,
            "docker_volumes": [], "nsjail_config": "", "nsjail_allow_net": False,
            "nsjail_forward_env": [], "cwd": "/tmp", "timeout": 30,
            "host_cwd": None,
        }
        captured_task_ids = []

        def _fake_create_environment(env_type, image, cwd, timeout, **kwargs):
            captured_task_ids.append((env_type, kwargs.get("task_id")))

            class _StubEnv:
                def cleanup(self):
                    pass

            return _StubEnv()

        import threading as _threading
        with patch.object(ce, "_load_config", return_value={"backend": "nsjail"}), \
             patch("tools.terminal_tool._active_environments", active), \
             patch("tools.terminal_tool._env_lock", _threading.RLock()), \
             patch("tools.terminal_tool_backends._create_environment", _fake_create_environment), \
             patch("tools.terminal_tool._get_env_config", return_value=fake_config), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._start_cleanup_thread", lambda: None), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", _threading.RLock()), \
             patch("tools.terminal_tool.resolve_task_overrides", return_value={}):
            env, env_type = ce._get_or_create_env("session-42")

        assert env_type == "nsjail"
        assert len(captured_task_ids) == 1
        captured_env_type, captured_id = captured_task_ids[0]
        assert captured_env_type == "nsjail"
        assert captured_id.startswith("_code_exec/nsjail/")
        assert "session-42" in captured_id
