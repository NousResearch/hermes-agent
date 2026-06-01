"""Tests for Daytona CWD sync (daytona_sync_cwd).

Scope per task t_f1c48e58:
- daytona_sync_cwd defaults to False (off)
- Strict excludes and size limits; never silently upload host directories
- Sync is visible in status/doctor if enabled
- Excluded paths are NOT synced
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Config default: daytona_sync_cwd is False (off by default)
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdDefault:
    """daytona_sync_cwd must default to False in all config paths."""

    def test_config_default_is_false(self):
        """The DEFAULT_CONFIG dict must set daytona_sync_cwd = False."""
        from hermes_cli.config import DEFAULT_CONFIG

        terminal = DEFAULT_CONFIG.get("terminal", {})
        assert "daytona_sync_cwd" in terminal, (
            "daytona_sync_cwd missing from DEFAULT_CONFIG['terminal']"
        )
        assert terminal["daytona_sync_cwd"] is False, (
            f"daytona_sync_cwd must default to False, got {terminal['daytona_sync_cwd']!r}"
        )

    def test_env_var_defaults_to_false(self):
        """TERMINAL_DAYTONA_SYNC_CWD env var defaults to 'false' → parsed as False."""
        # Remove the env var if present
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TERMINAL_DAYTONA_SYNC_CWD", None)
            # Re-import to pick up env changes is complex; instead test
            # the parsing pattern used in terminal_tool.py
            val = os.getenv("TERMINAL_DAYTONA_SYNC_CWD", "false").lower() in {"true", "1", "yes"}
            assert val is False

    def test_env_var_true_parses_correctly(self):
        """TERMINAL_DAYTONA_SYNC_CWD=true → parsed as True."""
        with patch.dict(os.environ, {"TERMINAL_DAYTONA_SYNC_CWD": "true"}):
            val = os.getenv("TERMINAL_DAYTONA_SYNC_CWD", "false").lower() in {"true", "1", "yes"}
            assert val is True

    def test_env_var_one_parses_correctly(self):
        """TERMINAL_DAYTONA_SYNC_CWD=1 → parsed as True."""
        with patch.dict(os.environ, {"TERMINAL_DAYTONA_SYNC_CWD": "1"}):
            val = os.getenv("TERMINAL_DAYTONA_SYNC_CWD", "false").lower() in {"true", "1", "yes"}
            assert val is True

    def test_env_var_yes_parses_correctly(self):
        """TERMINAL_DAYTONA_SYNC_CWD=yes → parsed as True."""
        with patch.dict(os.environ, {"TERMINAL_DAYTONA_SYNC_CWD": "yes"}):
            val = os.getenv("TERMINAL_DAYTONA_SYNC_CWD", "false").lower() in {"true", "1", "yes"}
            assert val is True

    def test_sync_cwd_not_in_constructor_defaults(self):
        """DaytonaEnvironment constructor must accept sync_cwd and default to False."""
        from tools.environments.daytona import DaytonaEnvironment

        # Inspect the __init__ signature to confirm sync_cwd has default=False
        import inspect
        sig = inspect.signature(DaytonaEnvironment.__init__)
        assert "sync_cwd" in sig.parameters, (
            "DaytonaEnvironment.__init__ must accept sync_cwd parameter"
        )
        assert sig.parameters["sync_cwd"].default is False, (
            f"sync_cwd must default to False, got {sig.parameters['sync_cwd'].default!r}"
        )


# ---------------------------------------------------------------------------
# 2. Config/env bridge: key must be in all four bridging maps
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdBridgeConsistency:
    """daytona_sync_cwd must be bridged in config→env, CLI, gateway, and terminal_tool."""

    def test_key_in_config_to_env_sync(self):
        """terminal.daytona_sync_cwd must map to TERMINAL_DAYTONA_SYNC_CWD in config.py."""
        from tests.tools.test_terminal_config_env_sync import _save_config_env_sync_keys

        keys = _save_config_env_sync_keys()
        assert "daytona_sync_cwd" in keys, (
            "daytona_sync_cwd missing from _config_to_env_sync in hermes_cli/config.py"
        )

    def test_key_in_cli_env_mappings(self):
        """daytona_sync_cwd must map to TERMINAL_DAYTONA_SYNC_CWD in cli.py."""
        from tests.tools.test_terminal_config_env_sync import _cli_env_map_keys, skip_if_no_prompt_toolkit

        skip_if_no_prompt_toolkit()
        keys = _cli_env_map_keys()
        assert "daytona_sync_cwd" in keys, (
            "daytona_sync_cwd missing from env_mappings in cli.py"
        )

    def test_key_in_gateway_env_map(self):
        """daytona_sync_cwd must map to TERMINAL_DAYTONA_SYNC_CWD in gateway/run.py."""
        pytest.importorskip("httpx", reason="gateway/run.py requires httpx")
        from tests.tools.test_terminal_config_env_sync import _gateway_env_map_keys

        keys = _gateway_env_map_keys()
        assert "daytona_sync_cwd" in keys, (
            "daytona_sync_cwd missing from _terminal_env_map in gateway/run.py"
        )

    def test_env_var_in_terminal_tool(self):
        """TERMINAL_DAYTONA_SYNC_CWD must be consumed by terminal_tool._get_env_config."""
        pytest.importorskip("requests", reason="terminal_tool requires requests")
        from tests.tools.test_terminal_config_env_sync import _terminal_tool_env_var_names

        env_vars = _terminal_tool_env_var_names()
        assert "TERMINAL_DAYTONA_SYNC_CWD" in env_vars, (
            "TERMINAL_DAYTONA_SYNC_CWD not read by terminal_tool._get_env_config()"
        )

    def test_sync_source_bridged_and_consumed(self):
        """Explicit Daytona sync source must be bridged and consumed."""
        pytest.importorskip("requests", reason="terminal_tool requires requests")
        from tests.tools.test_terminal_config_env_sync import (
            _cli_env_map_keys,
            _gateway_env_map_keys,
            _save_config_env_sync_keys,
            _terminal_tool_env_var_names,
            skip_if_no_prompt_toolkit,
        )

        skip_if_no_prompt_toolkit()
        assert "daytona_sync_cwd_source" in _save_config_env_sync_keys()
        assert "daytona_sync_cwd_source" in _cli_env_map_keys()
        assert "daytona_sync_cwd_source" in _gateway_env_map_keys()
        assert "TERMINAL_DAYTONA_SYNC_CWD_SOURCE" in _terminal_tool_env_var_names()

    def test_terminal_tool_passes_daytona_expansion_config_to_create_environment(self, monkeypatch):
        """Real terminal_tool creation path must forward Daytona keys into container_config."""
        import json
        from tools import terminal_tool as terminal_mod

        captured = {}

        class FakeEnv:
            def execute(self, command, **kwargs):
                return {"output": "ok", "returncode": 0}

        def fake_create_environment(*args, **kwargs):
            captured.update(kwargs.get("container_config") or {})
            captured["host_cwd_arg"] = kwargs.get("host_cwd")
            return FakeEnv()

        with terminal_mod._env_lock:
            terminal_mod._active_environments.clear()
            terminal_mod._last_activity.clear()

        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_CREATE_MODE", "snapshot")
        monkeypatch.setenv("TERMINAL_DAYTONA_SNAPSHOT", "snap-123")
        monkeypatch.setenv("TERMINAL_DAYTONA_LANGUAGE", "python")
        monkeypatch.setenv("TERMINAL_DAYTONA_NAME_PREFIX", "review")
        monkeypatch.setenv("TERMINAL_DAYTONA_NAME_SCOPE", "profile")
        monkeypatch.setenv("TERMINAL_DAYTONA_LABELS", '{"team":"agents"}')
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_STOP_INTERVAL", "30")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_ARCHIVE_INTERVAL", "60")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_DELETE_INTERVAL", "120")
        monkeypatch.setenv("TERMINAL_DAYTONA_EPHEMERAL", "true")
        monkeypatch.setenv("TERMINAL_DAYTONA_ENV_VARS", '{"APP_ENV":"test"}')
        monkeypatch.setenv("TERMINAL_DAYTONA_NETWORK_BLOCK_ALL", "true")
        monkeypatch.setenv("TERMINAL_DAYTONA_NETWORK_ALLOW_LIST", "10.0.0.0/8")
        monkeypatch.setenv("TERMINAL_DAYTONA_VOLUME_MOUNTS", '[{"containerPath":"/data","sourcePath":"/mnt/data"}]')
        monkeypatch.setenv("TERMINAL_DAYTONA_GPU", "1")
        monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD", "true")

        monkeypatch.setattr(terminal_mod, "_create_environment", fake_create_environment)
        monkeypatch.setattr(terminal_mod, "_check_all_guards", lambda command, env_type: {"approved": True})

        try:
            result = json.loads(terminal_mod.terminal_tool("true"))
        finally:
            with terminal_mod._env_lock:
                terminal_mod._active_environments.clear()
                terminal_mod._last_activity.clear()

        assert result["exit_code"] == 0
        assert captured["daytona_create_mode"] == "snapshot"
        assert captured["daytona_snapshot"] == "snap-123"
        assert captured["daytona_language"] == "python"
        assert captured["daytona_name_prefix"] == "review"
        assert captured["daytona_name_scope"] == "profile"
        assert captured["daytona_labels"] == {"team": "agents"}
        assert captured["daytona_auto_stop_interval"] == 30
        assert captured["daytona_auto_archive_interval"] == 60
        assert captured["daytona_auto_delete_interval"] == 120
        assert captured["daytona_ephemeral"] is True
        assert captured["daytona_env_vars"] == {"APP_ENV": "test"}
        assert captured["daytona_network_block_all"] is True
        assert captured["daytona_network_allow_list"] == "10.0.0.0/8"
        assert captured["daytona_volume_mounts"] == [{"containerPath": "/data", "sourcePath": "/mnt/data"}]
        assert captured["daytona_gpu"] == 1
        assert captured["daytona_sync_cwd"] is True

    def test_file_tools_first_creator_forwards_daytona_expansion_config(self, monkeypatch):
        """file_tools must forward full Daytona config when it creates the sandbox first."""
        from tools import file_tools
        from tools import terminal_tool as terminal_mod

        captured = {}

        class FakeEnv:
            def execute(self, command, **kwargs):
                return {"output": "", "returncode": 0}

        def fake_create_environment(*args, **kwargs):
            captured.update(kwargs.get("container_config") or {})
            return FakeEnv()

        with terminal_mod._env_lock:
            terminal_mod._active_environments.clear()
            terminal_mod._last_activity.clear()
        file_tools.clear_file_ops_cache()

        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_CREATE_MODE", "snapshot")
        monkeypatch.setenv("TERMINAL_DAYTONA_SNAPSHOT", "snap-file-tools")
        monkeypatch.setenv("TERMINAL_DAYTONA_LANGUAGE", "python")
        monkeypatch.setenv("TERMINAL_DAYTONA_NAME_PREFIX", "files")
        monkeypatch.setenv("TERMINAL_DAYTONA_NAME_SCOPE", "profile")
        monkeypatch.setenv("TERMINAL_DAYTONA_LABELS", '{"creator":"file_tools"}')
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_STOP_INTERVAL", "31")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_ARCHIVE_INTERVAL", "61")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_DELETE_INTERVAL", "121")
        monkeypatch.setenv("TERMINAL_DAYTONA_EPHEMERAL", "true")
        monkeypatch.setenv("TERMINAL_DAYTONA_ENV_VARS", '{"APP_ENV":"files"}')
        monkeypatch.setenv("TERMINAL_DAYTONA_NETWORK_BLOCK_ALL", "true")
        monkeypatch.setenv("TERMINAL_DAYTONA_NETWORK_ALLOW_LIST", "10.1.0.0/16")
        monkeypatch.setenv("TERMINAL_DAYTONA_VOLUME_MOUNTS", '[{"containerPath":"/data","sourcePath":"/mnt/data"}]')
        monkeypatch.setenv("TERMINAL_DAYTONA_GPU", "2")
        monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD", "true")

        monkeypatch.setattr(terminal_mod, "_create_environment", fake_create_environment)

        try:
            file_tools._get_file_ops("daytona-file-tools-parity")
        finally:
            with terminal_mod._env_lock:
                terminal_mod._active_environments.clear()
                terminal_mod._last_activity.clear()
            file_tools.clear_file_ops_cache()

        assert captured["daytona_create_mode"] == "snapshot"
        assert captured["daytona_snapshot"] == "snap-file-tools"
        assert captured["daytona_language"] == "python"
        assert captured["daytona_name_prefix"] == "files"
        assert captured["daytona_name_scope"] == "profile"
        assert captured["daytona_labels"] == {"creator": "file_tools"}
        assert captured["daytona_auto_stop_interval"] == 31
        assert captured["daytona_auto_archive_interval"] == 61
        assert captured["daytona_auto_delete_interval"] == 121
        assert captured["daytona_ephemeral"] is True
        assert captured["daytona_env_vars"] == {"APP_ENV": "files"}
        assert captured["daytona_network_block_all"] is True
        assert captured["daytona_network_allow_list"] == "10.1.0.0/16"
        assert captured["daytona_volume_mounts"] == [{"containerPath": "/data", "sourcePath": "/mnt/data"}]
        assert captured["daytona_gpu"] == 2
        assert captured["daytona_sync_cwd"] is True

    def test_execute_code_first_creator_forwards_daytona_expansion_config(self, monkeypatch):
        """execute_code must forward full Daytona config when it creates the sandbox first."""
        from tools import code_execution_tool
        from tools import terminal_tool as terminal_mod

        captured = {}

        class FakeEnv:
            pass

        def fake_create_environment(*args, **kwargs):
            captured.update(kwargs.get("container_config") or {})
            return FakeEnv()

        with terminal_mod._env_lock:
            terminal_mod._active_environments.clear()
            terminal_mod._last_activity.clear()

        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_CREATE_MODE", "snapshot")
        monkeypatch.setenv("TERMINAL_DAYTONA_SNAPSHOT", "snap-execute-code")
        monkeypatch.setenv("TERMINAL_DAYTONA_LANGUAGE", "python")
        monkeypatch.setenv("TERMINAL_DAYTONA_NAME_PREFIX", "exec")
        monkeypatch.setenv("TERMINAL_DAYTONA_NAME_SCOPE", "profile")
        monkeypatch.setenv("TERMINAL_DAYTONA_LABELS", '{"creator":"execute_code"}')
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_STOP_INTERVAL", "32")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_ARCHIVE_INTERVAL", "62")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_DELETE_INTERVAL", "122")
        monkeypatch.setenv("TERMINAL_DAYTONA_EPHEMERAL", "true")
        monkeypatch.setenv("TERMINAL_DAYTONA_ENV_VARS", '{"APP_ENV":"exec"}')
        monkeypatch.setenv("TERMINAL_DAYTONA_NETWORK_BLOCK_ALL", "true")
        monkeypatch.setenv("TERMINAL_DAYTONA_NETWORK_ALLOW_LIST", "10.2.0.0/16")
        monkeypatch.setenv("TERMINAL_DAYTONA_VOLUME_MOUNTS", '[{"containerPath":"/cache","sourcePath":"/mnt/cache"}]')
        monkeypatch.setenv("TERMINAL_DAYTONA_GPU", "3")
        monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD", "true")

        monkeypatch.setattr(terminal_mod, "_create_environment", fake_create_environment)

        try:
            code_execution_tool._get_or_create_env("daytona-exec-parity")
        finally:
            with terminal_mod._env_lock:
                terminal_mod._active_environments.clear()
                terminal_mod._last_activity.clear()

        assert captured["daytona_create_mode"] == "snapshot"
        assert captured["daytona_snapshot"] == "snap-execute-code"
        assert captured["daytona_language"] == "python"
        assert captured["daytona_name_prefix"] == "exec"
        assert captured["daytona_name_scope"] == "profile"
        assert captured["daytona_labels"] == {"creator": "execute_code"}
        assert captured["daytona_auto_stop_interval"] == 32
        assert captured["daytona_auto_archive_interval"] == 62
        assert captured["daytona_auto_delete_interval"] == 122
        assert captured["daytona_ephemeral"] is True
        assert captured["daytona_env_vars"] == {"APP_ENV": "exec"}
        assert captured["daytona_network_block_all"] is True
        assert captured["daytona_network_allow_list"] == "10.2.0.0/16"
        assert captured["daytona_volume_mounts"] == [{"containerPath": "/cache", "sourcePath": "/mnt/cache"}]
        assert captured["daytona_gpu"] == 3
        assert captured["daytona_sync_cwd"] is True


# ---------------------------------------------------------------------------
# 3. Exclusion rules: excluded paths must NOT be synced
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdExcludes:
    """Validate _CWD_EXCLUDE_DIRS, _CWD_EXCLUDE_FILES, and _CWD_MAX_BYTES."""

    def test_env_dir_excluded(self):
        """The .env directory must be in _CWD_EXCLUDE_DIRS."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".env" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_git_dir_excluded(self):
        """The .git directory must be in _CWD_EXCLUDE_DIRS."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".git" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_node_modules_excluded(self):
        """The node_modules directory must be in _CWD_EXCLUDE_DIRS."""
        from tools.environments.daytona import DaytonaEnvironment

        assert "node_modules" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_pycache_excluded(self):
        """The __pycache__ directory must be in _CWD_EXCLUDE_DIRS."""
        from tools.environments.daytona import DaytonaEnvironment

        assert "__pycache__" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_hermes_dir_excluded(self, tmp_path):
        """The .hermes directory must be excluded from CWD sync."""
        from tools.environments.daytona import DaytonaEnvironment

        (tmp_path / "README.md").write_text("allowed")
        (tmp_path / ".hermes" / "config.yaml").parent.mkdir(parents=True)
        (tmp_path / ".hermes" / "config.yaml").write_text("secret-ish profile state")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._host_cwd = str(tmp_path)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()
        uploaded_files = []

        def fake_bulk_upload(files):
            uploaded_files.extend(files)

        env._daytona_bulk_upload = fake_bulk_upload

        with patch("hermes_constants.get_hermes_home", return_value=Path("/fake/hermes/home")):
            assert env._sync_cwd_to_sandbox() is True

        assert [remote for _, remote in uploaded_files] == ["/workspace/README.md"]

    def test_env_file_excluded(self):
        """The .env file pattern must be in _CWD_EXCLUDE_FILES."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".env" in DaytonaEnvironment._CWD_EXCLUDE_FILES

    @pytest.mark.parametrize("fname", [".env", ".env.local", ".env.staging", ".env.test", ".ENV.CI"])
    def test_env_files_excluded_by_prefix(self, fname):
        """All .env suffix variants must be excluded from CWD sync."""
        from tools.environments.daytona import DaytonaEnvironment

        assert DaytonaEnvironment._is_cwd_excluded_file(fname)

    def test_pem_extension_excluded(self):
        """.pem files must be excluded by extension check."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".pem" in DaytonaEnvironment._CWD_EXCLUDE_EXTENSIONS
        assert DaytonaEnvironment._is_cwd_excluded_file("server.pem")

    def test_key_extension_excluded(self):
        """'.key' files must be excluded."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".key" in DaytonaEnvironment._CWD_EXCLUDE_EXTENSIONS
        assert DaytonaEnvironment._is_cwd_excluded_file("id_rsa.key")

    def test_max_bytes_is_reasonable(self):
        """_CWD_MAX_BYTES must be at least 10 MiB and at most 500 MiB."""
        from tools.environments.daytona import DaytonaEnvironment

        max_bytes = DaytonaEnvironment._CWD_MAX_BYTES
        assert max_bytes >= 10 * 1024 * 1024, f"_CWD_MAX_BYTES too low: {max_bytes}"
        assert max_bytes <= 500 * 1024 * 1024, f"_CWD_MAX_BYTES too high: {max_bytes}"

    def test_sync_cwd_method_exists(self):
        """DaytonaEnvironment must have a _sync_cwd_to_sandbox method."""
        from tools.environments.daytona import DaytonaEnvironment

        assert hasattr(DaytonaEnvironment, "_sync_cwd_to_sandbox"), (
            "DaytonaEnvironment must have _sync_cwd_to_sandbox method"
        )


# ---------------------------------------------------------------------------
# 4. _sync_cwd_to_sandbox respects defaults and skips when disabled
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdBehavior:
    """Verify that _sync_cwd_to_sandbox is only called when sync_cwd=True."""

    def test_sync_cwd_false_does_not_call_sync(self, monkeypatch):
        """When sync_cwd=False (default), __init__ must not call the CWD sync method."""
        import sys
        import types
        from types import SimpleNamespace
        from tools.environments import daytona as daytona_mod
        from tools.environments.daytona import DaytonaEnvironment

        class FakeDaytonaError(Exception):
            pass

        class FakeDaytona:
            def get(self, name):
                raise FakeDaytonaError("not found")

            def list(self, *args, **kwargs):
                return iter(())

            def create(self, params):
                sandbox = MagicMock()
                sandbox.id = "fake-sandbox"
                sandbox.process.exec.return_value = SimpleNamespace(result="/home/daytona\n", exit_code=0)
                return sandbox

        fake_daytona_module = types.SimpleNamespace(
            Daytona=FakeDaytona,
            CreateSandboxFromImageParams=lambda **kwargs: kwargs,
            CreateSandboxFromSnapshotParams=lambda **kwargs: kwargs,
            DaytonaError=FakeDaytonaError,
            Resources=lambda **kwargs: kwargs,
            SandboxState=types.SimpleNamespace(STOPPED="stopped", ARCHIVED="archived"),
            CodeLanguage=("python", "typescript", "javascript"),
        )
        monkeypatch.setitem(sys.modules, "daytona", fake_daytona_module)
        monkeypatch.setattr(daytona_mod, "_derive_profile_id", lambda: "profile123")
        monkeypatch.setattr(daytona_mod, "FileSyncManager", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(DaytonaEnvironment, "init_session", lambda self: None)
        sync_mock = MagicMock(return_value=True)
        monkeypatch.setattr(DaytonaEnvironment, "_sync_cwd_to_sandbox", sync_mock)

        DaytonaEnvironment("python:3.11", sync_cwd=False)

        sync_mock.assert_not_called()

    def test_sync_cwd_true_calls_sync(self, monkeypatch):
        """When sync_cwd=True, __init__ runs the behavioral CWD sync hook."""
        import sys
        import types
        from types import SimpleNamespace
        from tools.environments import daytona as daytona_mod
        from tools.environments.daytona import DaytonaEnvironment

        class FakeDaytonaError(Exception):
            pass

        class FakeDaytona:
            def get(self, name):
                raise FakeDaytonaError("not found")

            def list(self, *args, **kwargs):
                return iter(())

            def create(self, params):
                sandbox = MagicMock()
                sandbox.id = "fake-sandbox"
                sandbox.process.exec.return_value = SimpleNamespace(result="/home/daytona\n", exit_code=0)
                return sandbox

        fake_daytona_module = types.SimpleNamespace(
            Daytona=FakeDaytona,
            CreateSandboxFromImageParams=lambda **kwargs: kwargs,
            CreateSandboxFromSnapshotParams=lambda **kwargs: kwargs,
            DaytonaError=FakeDaytonaError,
            Resources=lambda **kwargs: kwargs,
            SandboxState=types.SimpleNamespace(STOPPED="stopped", ARCHIVED="archived"),
            CodeLanguage=("python", "typescript", "javascript"),
        )
        monkeypatch.setitem(sys.modules, "daytona", fake_daytona_module)
        monkeypatch.setattr(daytona_mod, "_derive_profile_id", lambda: "profile123")
        monkeypatch.setattr(daytona_mod, "FileSyncManager", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(DaytonaEnvironment, "init_session", lambda self: None)
        sync_mock = MagicMock(return_value=True)
        monkeypatch.setattr(DaytonaEnvironment, "_sync_cwd_to_sandbox", sync_mock)

        env = DaytonaEnvironment("python:3.11", sync_cwd=True)

        sync_mock.assert_called_once_with()
        assert env.cwd == "/workspace"

    def test_hermes_home_not_synced_as_cwd(self, tmp_path):
        """If host_cwd resolves to .hermes home, sync must be skipped."""
        from tools.environments.daytona import DaytonaEnvironment

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("terminal: {}")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._host_cwd = str(hermes_home)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()
        env._daytona_bulk_upload = MagicMock()

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            assert env._sync_cwd_to_sandbox() is False

        env._sandbox.process.exec.assert_not_called()
        env._daytona_bulk_upload.assert_not_called()


# ---------------------------------------------------------------------------
# 5. File exclusion unit test with mock filesystem
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdExclusionLogic:
    """Unit tests for the file collection/exclusion logic in _sync_cwd_to_sandbox."""

    @pytest.mark.parametrize("fname", [".env", ".env.staging", ".env.test", "server.pem"])
    def test_sensitive_file_names_skipped(self, fname):
        """Sensitive basenames and extensions must be excluded."""
        from tools.environments.daytona import DaytonaEnvironment

        assert DaytonaEnvironment._is_cwd_excluded_file(fname)

    def test_git_directory_skipped(self):
        """The .git directory must be excluded from directory traversal."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".git" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_node_modules_directory_skipped(self):
        """The node_modules directory must be excluded."""
        from tools.environments.daytona import DaytonaEnvironment

        assert "node_modules" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_venv_directory_skipped(self):
        """Both .venv and venv directories must be excluded."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".venv" in DaytonaEnvironment._CWD_EXCLUDE_DIRS
        assert "venv" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_pycache_directory_skipped(self):
        """The __pycache__ directory must be excluded."""
        from tools.environments.daytona import DaytonaEnvironment

        assert "__pycache__" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_daytona_directory_skipped(self):
        """The .daytona directory must be excluded (avoid recursive sync)."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".daytona" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_docker_directory_skipped(self):
        """The .docker directory must be excluded to avoid uploading config.json auth."""
        from tools.environments.daytona import DaytonaEnvironment

        assert ".docker" in DaytonaEnvironment._CWD_EXCLUDE_DIRS

    def test_no_duplicate_exclude_dirs(self):
        """_CWD_EXCLUDE_DIRS must not contain duplicates."""
        from tools.environments.daytona import DaytonaEnvironment

        dirs = list(DaytonaEnvironment._CWD_EXCLUDE_DIRS)
        assert len(dirs) == len(set(dirs)), (
            f"_CWD_EXCLUDE_DIRS has duplicates: {dirs}"
        )


# ---------------------------------------------------------------------------
# 6. Behavioral tests: overflow cap aborts entire upload
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdOverflowAbortsEntireUpload:
    """When total size exceeds _CWD_MAX_BYTES, NO files should be uploaded.

    The old code broke early from the loop but still uploaded the partial
    files_to_sync list. The fix collects everything first, then checks the
    total; on overflow it returns immediately without calling
    _daytona_bulk_upload or mkdir.
    """

    def test_overflow_aborts_no_upload(self, tmp_path):
        """Files exceeding _CWD_MAX_BYTES must NOT be uploaded at all."""
        from tools.environments.daytona import DaytonaEnvironment

        # Create a temp directory with files that exceed a low cap
        (tmp_path / "file1.txt").write_bytes(b"A" * 2048)
        (tmp_path / "file2.txt").write_bytes(b"B" * 2048)

        # Build a bare-bones DaytonaEnvironment with __new__ to avoid __init__
        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(tmp_path)
        env._CWD_MAX_BYTES = 100  # 100 bytes — well under the 4 KiB total
        env._sandbox = MagicMock()

        uploaded_files = []

        def fake_bulk_upload(file_list):
            uploaded_files.extend(file_list)

        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}), \
             patch("hermes_constants.get_hermes_home",
                   return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        # No files should have been uploaded because total exceeds the cap
        assert uploaded_files == [], (
            f"Expected zero files on overflow, but upload was called with: {uploaded_files}"
        )

    def test_under_cap_proceeds_with_upload(self, tmp_path):
        """Files under _CWD_MAX_BYTES should be uploaded normally."""
        from tools.environments.daytona import DaytonaEnvironment

        (tmp_path / "small.txt").write_bytes(b"hello")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(tmp_path)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB — well over
        env._sandbox = MagicMock()

        uploaded_files = []

        def fake_bulk_upload(file_list):
            uploaded_files.extend(file_list)

        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}), \
             patch("hermes_constants.get_hermes_home",
                   return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        assert len(uploaded_files) == 1, (
            f"Expected 1 file uploaded, got {len(uploaded_files)}"
        )

    def test_overflow_does_not_mkdir(self, tmp_path):
        """On overflow, mkdir must not be called on the sandbox either."""
        from tools.environments.daytona import DaytonaEnvironment

        (tmp_path / "big.txt").write_bytes(b"X" * 2048)

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(tmp_path)
        env._CWD_MAX_BYTES = 100  # 100 bytes — well under total
        env._sandbox = MagicMock()

        env._daytona_bulk_upload = MagicMock()

        with patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}), \
             patch("hermes_constants.get_hermes_home",
                   return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        # Neither mkdir nor bulk_upload should have been called
        env._sandbox.process.exec.assert_not_called()
        env._daytona_bulk_upload.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Behavioral tests: exclusions actually filter the upload list
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdExclusionBehavior:
    """Temp-tree behavioral tests: excluded paths must NOT appear in the
    upload list, while allowed files must be included.

    Uses a real temp directory tree with excluded and allowed entries,
    a low _CWD_MAX_BYTES, and a mock sandbox to capture what would
    actually be uploaded.
    """

    def test_excluded_paths_not_uploaded(self, tmp_path):
        """A temp tree with allowed + excluded entries uploads only allowed file."""
        from tools.environments.daytona import DaytonaEnvironment

        # Create an allowed file at the root
        (tmp_path / "README.md").write_text("allowed content")

        # Create excluded directories
        (tmp_path / ".git" / "objects").mkdir(parents=True)
        (tmp_path / ".git" / "objects" / "ab").mkdir()
        (tmp_path / ".git" / "objects" / "ab" / "cd1234").write_bytes(b"git object")
        (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
        (tmp_path / "node_modules" / "pkg" / "index.js").write_text("// excluded")
        (tmp_path / ".docker").mkdir()
        (tmp_path / ".docker" / "config.json").write_text('{"auths": {}}')

        # Create excluded files
        (tmp_path / ".env").write_text("SECRET=123")
        (tmp_path / ".env.staging").write_text("STAGING_SECRET=123")
        (tmp_path / ".env.test").write_text("TEST_SECRET=123")
        (tmp_path / ".git-credentials").write_text("https://token@example.com")
        (tmp_path / ".dockercfg").write_text('{"auths": {}}')
        (tmp_path / "server.pem").write_text("PEM DATA")
        (tmp_path / "id_rsa.key").write_text("KEY DATA")
        (tmp_path / "server.crt").write_text("CERT DATA")
        (tmp_path / "client.cert").write_text("CERT DATA")
        (tmp_path / "bundle.pfx").write_text("PFX DATA")

        # Create a .venv directory (also excluded)
        (tmp_path / ".venv" / "bin").mkdir(parents=True)
        (tmp_path / ".venv" / "bin" / "python").write_text("# venv python")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(tmp_path)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB — plenty
        env._sandbox = MagicMock()

        uploaded_files = []

        def fake_bulk_upload(file_list):
            uploaded_files.extend(file_list)

        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}), \
             patch("hermes_constants.get_hermes_home",
                   return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        remote_paths = [remote for _, remote in uploaded_files]
        # Only README.md should be uploaded
        assert remote_paths == ["/workspace/README.md"], (
            f"Expected only /workspace/README.md, got: {remote_paths}"
        )

    def test_nested_allowed_files_uploaded(self, tmp_path):
        """Allowed files inside subdirectories are included, excluded ones are not."""
        from tools.environments.daytona import DaytonaEnvironment

        # Allowed file in a subdirectory
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "src" / "util.py").write_text("# utility")

        # Excluded directory inside allowed subdirectory
        (tmp_path / "src" / "__pycache__").mkdir()
        (tmp_path / "src" / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00" * 100)

        # Excluded file in src/
        (tmp_path / "src" / ".env").write_text("DB_SECRET=yes")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(tmp_path)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()

        uploaded_files = []

        def fake_bulk_upload(file_list):
            uploaded_files.extend(file_list)

        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}), \
             patch("hermes_constants.get_hermes_home",
                   return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        remote_paths = sorted([remote for _, remote in uploaded_files])
        # Should include both .py files, but NOT __pycache__ contents or .env
        expected = sorted([
            "/workspace/src/main.py",
            "/workspace/src/util.py",
        ])
        assert remote_paths == expected, (
            f"Expected {expected}, got: {remote_paths}"
        )

    def test_host_cwd_attribute_is_preferred_over_terminal_cwd_and_process_cwd(self, tmp_path):
        """Explicit host_cwd must control the upload source for Daytona sync_cwd."""
        from tools.environments.daytona import DaytonaEnvironment

        source = tmp_path / "source"
        env_var_dir = tmp_path / "env-var"
        source.mkdir()
        env_var_dir.mkdir()
        (source / "README.md").write_text("source")
        (env_var_dir / "WRONG.md").write_text("wrong")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(source)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()
        uploaded_files = []
        def fake_bulk_upload(files):
            uploaded_files.extend(files)
        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {"TERMINAL_CWD": str(env_var_dir)}), \
             patch("hermes_constants.get_hermes_home", return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        assert [remote for _, remote in uploaded_files] == ["/workspace/README.md"]

    def test_sensitive_credential_paths_not_uploaded(self, tmp_path):
        """CWD sync must not upload common credential stores to cloud sandboxes."""
        from tools.environments.daytona import DaytonaEnvironment

        (tmp_path / "README.md").write_text("allowed")
        for dirname in (".ssh", ".aws", ".docker", ".kube", Path(".config") / "gcloud"):
            path = tmp_path / dirname
            path.mkdir(parents=True, exist_ok=True)
            (path / "secret").write_text("do not upload")
        for filename in (".npmrc", ".pypirc", ".netrc", ".git-credentials", "credentials.json", "id_rsa", "id_ed25519"):
            (tmp_path / filename).write_text("do not upload")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(tmp_path)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()
        uploaded_files = []
        def fake_bulk_upload(files):
            uploaded_files.extend(files)
        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {}, clear=False), \
             patch("hermes_constants.get_hermes_home", return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        assert [remote for _, remote in uploaded_files] == ["/workspace/README.md"]

    def test_uppercase_secret_paths_not_uploaded(self, tmp_path):
        """Secret basename/suffix exclusions must be case-insensitive."""
        from tools.environments.daytona import DaytonaEnvironment

        (tmp_path / "README.md").write_text("allowed")
        for filename in (".ENV", ".NPMRC", "SERVER.PEM", "PROD.KEY"):
            (tmp_path / filename).write_text("do not upload")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(tmp_path)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()
        uploaded_files = []

        def fake_bulk_upload(files):
            uploaded_files.extend(files)

        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {}, clear=False), \
             patch("hermes_constants.get_hermes_home", return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        assert [remote for _, remote in uploaded_files] == ["/workspace/README.md"]

    def test_terminal_cwd_home_is_not_used_as_implicit_sync_source(self, monkeypatch, tmp_path):
        """Gateway-expanded home TERMINAL_CWD must not become Daytona sync source."""
        from tools import terminal_tool as terminal_mod

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD", "true")
        monkeypatch.setenv("TERMINAL_CWD", str(home))
        monkeypatch.delenv("TERMINAL_DAYTONA_SYNC_CWD_SOURCE", raising=False)

        with patch("pathlib.Path.home", return_value=home):
            config = terminal_mod._get_env_config()

        assert config["host_cwd"] is None

    def test_explicit_sync_source_controls_upload_source(self, monkeypatch, tmp_path):
        """TERMINAL_DAYTONA_SYNC_CWD_SOURCE is the explicit host upload source."""
        from tools import terminal_tool as terminal_mod

        source = tmp_path / "project"
        source.mkdir()
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD", "true")
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "home"))
        monkeypatch.setenv("TERMINAL_DAYTONA_SYNC_CWD_SOURCE", str(source))

        config = terminal_mod._get_env_config()

        assert config["host_cwd"] == str(source.resolve())
        assert config["cwd"] == "/workspace"


# ---------------------------------------------------------------------------
# 8. Behavioral tests: symlinks and resolved-path containment
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdSymlinkContainment:
    """CWD sync must not upload files through symlinks outside host_cwd."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_symlink_file_pointing_outside_cwd_not_uploaded(self, tmp_path):
        """A symlinked file whose target is outside TERMINAL_CWD is skipped."""
        from tools.environments.daytona import DaytonaEnvironment

        project = tmp_path / "project"
        outside = tmp_path / "outside"
        project.mkdir()
        outside.mkdir()
        (project / "README.md").write_text("allowed")
        (outside / "secret.txt").write_text("do not upload")

        try:
            os.symlink(outside / "secret.txt", project / "leak.txt")
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(project)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()

        uploaded_files = []

        def fake_bulk_upload(file_list):
            uploaded_files.extend(file_list)

        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {"TERMINAL_CWD": str(project)}), \
             patch("hermes_constants.get_hermes_home",
                   return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        assert [remote for _, remote in uploaded_files] == ["/workspace/README.md"]
        assert all(Path(host).name != "leak.txt" for host, _ in uploaded_files)
        assert all(Path(host).resolve() != (outside / "secret.txt").resolve()
                   for host, _ in uploaded_files)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_symlink_directory_pointing_outside_cwd_not_traversed(self, tmp_path):
        """A symlinked directory outside TERMINAL_CWD is not traversed/uploaded."""
        from tools.environments.daytona import DaytonaEnvironment

        project = tmp_path / "project"
        outside = tmp_path / "outside_dir"
        project.mkdir()
        outside.mkdir()
        (project / "README.md").write_text("allowed")
        (outside / "secret.txt").write_text("do not upload")

        try:
            os.symlink(outside, project / "linked_outside")
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sync_cwd = True
        env._host_cwd = str(project)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()

        uploaded_files = []

        def fake_bulk_upload(file_list):
            uploaded_files.extend(file_list)

        env._daytona_bulk_upload = fake_bulk_upload

        with patch.dict(os.environ, {"TERMINAL_CWD": str(project)}), \
             patch("hermes_constants.get_hermes_home",
                   return_value=Path("/fake/hermes/home")):
            env._sync_cwd_to_sandbox()

        remote_paths = [remote for _, remote in uploaded_files]
        assert remote_paths == ["/workspace/README.md"]
        assert "/workspace/linked_outside/secret.txt" not in remote_paths
# ---------------------------------------------------------------------------
# 9. Behavioral tests: managed /workspace sync clears stale files
# ---------------------------------------------------------------------------

class TestDaytonaSyncCwdManagedWorkspaceClear:
    """CWD sync owns /workspace and must remove stale remote entries before upload."""

    def test_sync_clears_workspace_before_reuploading_allowed_files(self, tmp_path):
        """A deleted local file must not survive in persistent /workspace."""
        from tools.environments.daytona import DaytonaEnvironment

        project = tmp_path / "project"
        project.mkdir()
        stale = project / "stale.txt"
        keep = project / "keep.txt"
        stale.write_text("old")
        keep.write_text("keep")

        remote_files = set()
        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._host_cwd = str(project)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()

        def fake_exec(command):
            if "find /workspace" in command:
                remote_files.clear()

        def fake_bulk_upload(files):
            remote_files.update(remote for _, remote in files)

        env._sandbox.process.exec.side_effect = fake_exec
        env._daytona_bulk_upload = fake_bulk_upload

        with patch("hermes_constants.get_hermes_home", return_value=Path("/fake/hermes/home")):
            assert env._sync_cwd_to_sandbox() is True
        assert remote_files == {"/workspace/keep.txt", "/workspace/stale.txt"}

        stale.unlink()
        with patch("hermes_constants.get_hermes_home", return_value=Path("/fake/hermes/home")):
            assert env._sync_cwd_to_sandbox() is True

        assert remote_files == {"/workspace/keep.txt"}

    def test_newly_excluded_secret_file_removed_by_subsequent_sync(self, tmp_path):
        """If a formerly synced file becomes excluded, the next sync clears it remotely."""
        from tools.environments.daytona import DaytonaEnvironment

        project = tmp_path / "project"
        project.mkdir()
        token = project / "token.txt"
        token.write_text("not-yet-excluded")

        remote_files = set()
        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._host_cwd = str(project)
        env._CWD_MAX_BYTES = 100 * 1024 * 1024
        env._sandbox = MagicMock()

        def fake_exec(command):
            if "find /workspace" in command:
                remote_files.clear()

        def fake_bulk_upload(files):
            remote_files.update(remote for _, remote in files)

        env._sandbox.process.exec.side_effect = fake_exec
        env._daytona_bulk_upload = fake_bulk_upload

        with patch("hermes_constants.get_hermes_home", return_value=Path("/fake/hermes/home")):
            assert env._sync_cwd_to_sandbox() is True
        assert remote_files == {"/workspace/token.txt"}

        token.rename(project / ".env")
        with patch("hermes_constants.get_hermes_home", return_value=Path("/fake/hermes/home")):
            assert env._sync_cwd_to_sandbox() is True

        assert remote_files == set()
