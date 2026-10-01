"""Long-lived foreground detection must not fire on package-manager package names.

``npm update vite`` names a package to update; it does not start the vite dev
server. Covers the false-positive fix for #42620 and keeps standalone server
invocations blocked.
"""
import json
from unittest.mock import patch, MagicMock

from tools.terminal_tool import _foreground_background_guidance


class TestPmPackageArgumentsAllowed:
    """Package-manager commands whose package name matches a server keyword
    must NOT be flagged as long-lived processes."""

    def _assert_allowed(self, command: str) -> None:
        assert _foreground_background_guidance(command) is None, command

    def test_npm_update_vite(self):
        self._assert_allowed("npm update vite")

    def test_npm_install_vite(self):
        self._assert_allowed("npm install vite")

    def test_npm_install_vite_with_flag(self):
        self._assert_allowed("npm install vite --save-dev")

    def test_npm_remove_vite(self):
        self._assert_allowed("npm remove vite")

    def test_npm_uninstall_vite(self):
        self._assert_allowed("npm uninstall vite")

    def test_pnpm_add_vite(self):
        self._assert_allowed("pnpm add vite")

    def test_yarn_add_vite(self):
        self._assert_allowed("yarn add vite")

    def test_bun_add_vite(self):
        self._assert_allowed("bun add vite")

    def test_npm_update_vite_with_other_packages(self):
        self._assert_allowed("npm update vite lodash axios")

    def test_npm_install_nodemon(self):
        self._assert_allowed("npm install nodemon")

    def test_npm_update_nodemon(self):
        self._assert_allowed("npm update nodemon")


class TestStandaloneServersStillBlocked:
    """Keyword invocations that are NOT package arguments stay blocked."""

    def _assert_blocked(self, command: str) -> None:
        msg = _foreground_background_guidance(command)
        assert msg is not None, command
        assert "long-lived" in msg.lower()

    def test_vite_standalone(self):
        self._assert_blocked("vite")

    def test_vite_dev(self):
        self._assert_blocked("vite dev")

    def test_vite_build(self):
        self._assert_blocked("vite build")

    def test_vite_preview(self):
        self._assert_blocked("vite preview --port 3000")

    def test_npx_vite(self):
        self._assert_blocked("npx vite")

    def test_npm_run_dev(self):
        self._assert_blocked("npm run dev")

    def test_uvicorn(self):
        self._assert_blocked("uvicorn main:app")


class TestCompoundCommandBoundary:
    """The package-argument exemption must not escape its own command segment."""

    def test_vite_after_separator_still_blocked(self):
        # `vite` here is a second command, not an npm package argument.
        msg = _foreground_background_guidance("npm install && vite")
        assert msg is not None
        assert "long-lived" in msg.lower()

    def test_package_arg_before_separator_still_allowed(self):
        assert _foreground_background_guidance("npm update vite && npm test") is None


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


class TestIntegratedTerminalPath:
    """The exemption reaches the full terminal_tool guard path."""

    def test_npm_update_vite_runs(self):
        result = _run_terminal("npm update vite")
        assert result.get("error") is None
        assert result.get("output") == "done"

    def test_vite_dev_still_blocked_integrated(self):
        result = _run_terminal("vite dev")
        assert result.get("error") is not None
        assert "long-lived" in result["error"].lower()
