"""Tests for long-lived foreground process detection in terminal_tool.

Ensures that package manager commands like ``npm update vite`` are not
incorrectly flagged as long-lived server processes (false positive fix
for issue #42620).
"""
import json
from unittest.mock import patch, MagicMock


def _make_env_config(**overrides):
    """Return a minimal _get_env_config()-shaped dict with optional overrides."""
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _run_terminal(command: str) -> dict:
    """Run terminal_tool with mocked env and return the parsed result."""
    from tools.terminal_tool import terminal_tool

    with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
         patch("tools.terminal_tool._start_cleanup_thread"):

        mock_env = MagicMock()
        mock_env.execute.return_value = {"output": "done", "returncode": 0}

        with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
             patch("tools.terminal_tool._last_activity", {"default": 0}), \
             patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
            return json.loads(terminal_tool(command=command))


class TestViteFalsePositive:
    """npm/pnpm/yarn/bun package operations with 'vite' as a package name
    should NOT be blocked as long-lived processes."""

    def test_npm_update_vite_not_blocked(self):
        """npm update vite should run in foreground."""
        result = _run_terminal("npm update vite")
        assert result.get("error") is None

    def test_npm_install_vite_not_blocked(self):
        """npm install vite should run in foreground."""
        result = _run_terminal("npm install vite")
        assert result.get("error") is None

    def test_npm_install_vite_save_dev_not_blocked(self):
        """npm install vite --save-dev should run in foreground."""
        result = _run_terminal("npm install vite --save-dev")
        assert result.get("error") is None

    def test_npm_remove_vite_not_blocked(self):
        """npm remove vite should run in foreground."""
        result = _run_terminal("npm remove vite")
        assert result.get("error") is None

    def test_pnpm_add_vite_not_blocked(self):
        """pnpm add vite should run in foreground."""
        result = _run_terminal("pnpm add vite")
        assert result.get("error") is None

    def test_yarn_add_vite_not_blocked(self):
        """yarn add vite should run in foreground."""
        result = _run_terminal("yarn add vite")
        assert result.get("error") is None

    def test_bun_add_vite_not_blocked(self):
        """bun add vite should run in foreground."""
        result = _run_terminal("bun add vite")
        assert result.get("error") is None

    def test_npm_uninstall_vite_not_blocked(self):
        """npm uninstall vite should run in foreground."""
        result = _run_terminal("npm uninstall vite")
        assert result.get("error") is None

    def test_npm_update_vite_with_other_packages_not_blocked(self):
        """npm update vite lodash axios should run in foreground."""
        result = _run_terminal("npm update vite lodash axios")
        assert result.get("error") is None


class TestViteStillBlocked:
    """vite as a standalone dev server command should still be blocked."""

    def test_vite_standalone_blocked(self):
        """vite (standalone) should be blocked."""
        result = _run_terminal("vite")
        assert result.get("exit_code") == -1
        assert "long-lived" in result["error"].lower()

    def test_vite_dev_blocked(self):
        """vite dev should be blocked."""
        result = _run_terminal("vite dev")
        assert result.get("exit_code") == -1
        assert "long-lived" in result["error"].lower()

    def test_vite_build_blocked(self):
        """vite build should be blocked."""
        result = _run_terminal("vite build")
        assert result.get("exit_code") == -1
        assert "long-lived" in result["error"].lower()

    def test_vite_preview_blocked(self):
        """vite preview --port 3000 should be blocked."""
        result = _run_terminal("vite preview --port 3000")
        assert result.get("exit_code") == -1
        assert "long-lived" in result["error"].lower()


class TestOtherPmPackageArgsNotBlocked:
    """Other package manager commands with server-tool package names should
    also not be falsely blocked."""

    def test_npm_install_nodemon_not_blocked(self):
        """npm install nodemon should run in foreground (nodemon is a package name)."""
        result = _run_terminal("npm install nodemon")
        assert result.get("error") is None

    def test_npm_update_nodemon_not_blocked(self):
        """npm update nodemon should run in foreground."""
        result = _run_terminal("npm update nodemon")
        assert result.get("error") is None


class TestNonPmViteCommandsStillBlocked:
    """Commands that use vite as a server (not package manager arg) should
    still be blocked."""

    def test_npx_vite_blocked(self):
        """npx vite should be blocked."""
        result = _run_terminal("npx vite")
        assert result.get("exit_code") == -1
        assert "long-lived" in result["error"].lower()

    def test_npm_run_dev_still_blocked(self):
        """npm run dev (uses the npm run pattern, not vite pattern) should be blocked."""
        result = _run_terminal("npm run dev")
        assert result.get("exit_code") == -1
        assert "long-lived" in result["error"].lower()
