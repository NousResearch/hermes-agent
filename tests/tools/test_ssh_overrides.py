"""Tests for per-task SSH environment overrides."""

from tools.terminal_tool import (
    register_task_env_overrides,
    clear_task_env_overrides,
    _task_env_overrides,
)


class TestSSHOverridesInConfig:
    """Verify SSH config assembly respects per-task overrides."""

    def setup_method(self):
        self._saved = dict(_task_env_overrides)
        _task_env_overrides.clear()

    def teardown_method(self):
        _task_env_overrides.clear()
        _task_env_overrides.update(self._saved)

    def _build_ssh_config(self, task_id: str, global_config: dict) -> dict:
        """Replicate the SSH config assembly logic from terminal_tool.py."""
        overrides = _task_env_overrides.get(task_id, {})
        return {
            "host": overrides.get("ssh_host") or global_config.get("ssh_host", ""),
            "user": overrides.get("ssh_user") or global_config.get("ssh_user", ""),
            "port": overrides.get("ssh_port") or global_config.get("ssh_port", 22),
            "key": overrides.get("ssh_key") or global_config.get("ssh_key", ""),
            "persistent": overrides.get("ssh_persistent", global_config.get("ssh_persistent", False)),
        }

    def test_no_overrides_uses_global(self):
        """Without per-task overrides, global config is used."""
        global_config = {
            "ssh_host": "global.example.com",
            "ssh_user": "root",
            "ssh_port": 22,
            "ssh_key": "/root/.ssh/id_rsa",
            "ssh_persistent": True,
        }
        result = self._build_ssh_config("task-1", global_config)
        assert result["host"] == "global.example.com"
        assert result["user"] == "root"
        assert result["port"] == 22
        assert result["key"] == "/root/.ssh/id_rsa"
        assert result["persistent"] is True

    def test_override_port_and_key(self):
        """Per-task overrides for port and key take precedence."""
        global_config = {
            "ssh_host": "dojo.pwncollege.com",
            "ssh_user": "hacker",
            "ssh_port": 22,
            "ssh_key": "/default/key",
        }
        register_task_env_overrides("task-42", {
            "ssh_port": 2264,
            "ssh_key": "/tmp/keys/episode_42",
        })
        result = self._build_ssh_config("task-42", global_config)
        assert result["port"] == 2264
        assert result["key"] == "/tmp/keys/episode_42"
        # Non-overridden fields fall through to global
        assert result["host"] == "dojo.pwncollege.com"
        assert result["user"] == "hacker"

    def test_different_tasks_get_different_ports(self):
        """128 parallel rollouts each get their own SSH port."""
        global_config = {
            "ssh_host": "dojo.pwncollege.com",
            "ssh_user": "hacker",
            "ssh_port": 22,
            "ssh_key": "",
        }
        for i in range(128):
            tid = f"task-{i}"
            register_task_env_overrides(tid, {"ssh_port": 2222 + i})

        for i in range(128):
            tid = f"task-{i}"
            result = self._build_ssh_config(tid, global_config)
            assert result["port"] == 2222 + i

    def test_clear_overrides_reverts_to_global(self):
        """After clearing, config falls back to global."""
        global_config = {"ssh_port": 22}
        register_task_env_overrides("task-99", {"ssh_port": 9999})
        assert self._build_ssh_config("task-99", global_config)["port"] == 9999

        clear_task_env_overrides("task-99")
        assert self._build_ssh_config("task-99", global_config)["port"] == 22

    def test_persistent_false_not_clobbered_by_or(self):
        """ssh_persistent=False override must not be skipped due to falsy `or`."""
        global_config = {"ssh_persistent": True}
        register_task_env_overrides("task-x", {"ssh_persistent": False})
        result = self._build_ssh_config("task-x", global_config)
        assert result["persistent"] is False
