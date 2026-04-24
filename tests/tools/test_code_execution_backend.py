"""Tests for execute_code backend resolution.

The key contract: ``code_execution.backend`` (synced to the
``CODE_EXECUTION_ENV`` environment variable) overrides ``terminal.backend``
for the ``execute_code`` tool only. When unset, execute_code inherits the
terminal backend.
"""

import os
from unittest.mock import patch

import pytest

from tools import code_execution_tool as ce


@pytest.fixture(autouse=True)
def _clear_env():
    """Ensure no test inherits a stray CODE_EXECUTION_ENV / TERMINAL_ENV."""
    saved = {k: os.environ.get(k) for k in ("CODE_EXECUTION_ENV", "TERMINAL_ENV")}
    for k in ("CODE_EXECUTION_ENV", "TERMINAL_ENV"):
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestBackendResolution:
    def test_defaults_to_terminal_backend_when_override_unset(self):
        """Historical behavior: execute_code inherits TERMINAL_ENV."""
        with patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "docker"},
        ):
            assert ce._resolve_code_execution_env_type() == "docker"

    def test_override_wins_over_terminal_backend(self):
        """Setting CODE_EXECUTION_ENV isolates execute_code from terminal."""
        os.environ["CODE_EXECUTION_ENV"] = "nsjail"
        with patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "local"},
        ):
            assert ce._resolve_code_execution_env_type() == "nsjail"

    def test_empty_override_is_ignored(self):
        """An empty CODE_EXECUTION_ENV must not shadow the terminal value."""
        os.environ["CODE_EXECUTION_ENV"] = ""
        with patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "singularity"},
        ):
            assert ce._resolve_code_execution_env_type() == "singularity"

    def test_whitespace_override_is_ignored(self):
        """Whitespace-only values (stray spaces in .env) don't count."""
        os.environ["CODE_EXECUTION_ENV"] = "   "
        with patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "local"},
        ):
            assert ce._resolve_code_execution_env_type() == "local"

    def test_falls_back_to_local_when_everything_missing(self):
        """Safe default when neither terminal nor code_execution is set."""
        with patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": None},
        ):
            assert ce._resolve_code_execution_env_type() == "local"


class TestCacheNamespacing:
    def test_separate_cache_key_when_backends_differ(self):
        """When execute_code uses a different backend than terminal, the
        cache must use a namespaced task_id so the two tools never reuse
        each other's live environment."""
        os.environ["CODE_EXECUTION_ENV"] = "nsjail"

        # Patch the terminal_tool module-level state so _get_or_create_env
        # hits our fake caches and environment factory.
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
        with patch("tools.terminal_tool._active_environments", active), \
             patch("tools.terminal_tool._env_lock", _threading.RLock()), \
             patch("tools.terminal_tool._create_environment", _fake_create_environment), \
             patch("tools.terminal_tool._get_env_config", return_value=fake_config), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._start_cleanup_thread", lambda: None), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", _threading.RLock()), \
             patch("tools.terminal_tool._task_env_overrides", {}):
            env, env_type = ce._get_or_create_env("session-42")

        assert env_type == "nsjail"
        assert len(captured_task_ids) == 1
        captured_env_type, captured_id = captured_task_ids[0]
        assert captured_env_type == "nsjail"
        # The cache key must be namespaced so it can't collide with a
        # terminal-owned entry for the same raw task id.
        assert captured_id.startswith("_code_exec/nsjail/")
        assert "session-42" in captured_id
