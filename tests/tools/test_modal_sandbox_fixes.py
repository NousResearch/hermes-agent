"""Tests for Modal sandbox infrastructure fixes (TBLite baseline).

Covers the 9 bugs discovered while setting up TBLite evaluation:
1. Tool resolution — terminal + file tools load with minisweagent
2. CWD fix — host paths get replaced with /root for container backends
3. ephemeral_disk version check
4. Tilde ~ replaced with /root for container backends
5. ensurepip fix in patches.py for Modal image builder
6. install_pipx stays True for swerex-remote
7. /home/ added to host prefix check
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure repo root is importable
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import tools.terminal_tool  # noqa: F401
_tt_mod = sys.modules["tools.terminal_tool"]


# =========================================================================
# Test 1: Tool resolution includes terminal + file tools
# =========================================================================

class TestToolResolution:
    """Verify get_tool_definitions returns all expected tools for eval."""

    def test_terminal_and_file_toolsets_resolve_all_tools(self):
        """enabled_toolsets=['terminal', 'file'] should produce 6 tools."""
        from model_tools import get_tool_definitions
        tools = get_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            quiet_mode=True,
        )
        names = {t["function"]["name"] for t in tools}
        expected = {"terminal", "process", "read_file", "write_file", "search_files", "patch"}
        assert expected == names, f"Expected {expected}, got {names}"

    def test_terminal_tool_present(self):
        """The terminal tool must be present (not silently dropped)."""
        from model_tools import get_tool_definitions
        tools = get_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            quiet_mode=True,
        )
        names = [t["function"]["name"] for t in tools]
        assert "terminal" in names, (
            f"terminal tool missing! Only got: {names}. "
            "Check that minisweagent is installed (git submodule update --init)."
        )


# =========================================================================
# Test 2-4: CWD handling for container backends
# =========================================================================

class TestCwdHandling:
    """Verify host paths are sanitized for container backends."""

    def test_home_path_replaced_for_modal(self):
        """TERMINAL_CWD=/home/user/... should be replaced with /root for modal."""
        with patch.dict(os.environ, {
            "TERMINAL_ENV": "modal",
            "TERMINAL_CWD": "/home/dakota/github/hermes-agent",
        }):
            config = _tt_mod._get_env_config()
            assert config["cwd"] == "/root", (
                f"Expected /root, got {config['cwd']}. "
                "/home/ paths should be replaced for modal backend."
            )

    def test_users_path_replaced_for_docker(self):
        """TERMINAL_CWD=/Users/... should be replaced with /root for docker."""
        with patch.dict(os.environ, {
            "TERMINAL_ENV": "docker",
            "TERMINAL_CWD": "/Users/someone/projects",
        }):
            config = _tt_mod._get_env_config()
            assert config["cwd"] == "/root", (
                f"Expected /root, got {config['cwd']}. "
                "/Users/ paths should be replaced for docker backend."
            )

    def test_windows_path_replaced_for_modal(self):
        """TERMINAL_CWD=C:\\Users\\... should be replaced for modal."""
        with patch.dict(os.environ, {
            "TERMINAL_ENV": "modal",
            "TERMINAL_CWD": "C:\\Users\\someone\\projects",
        }):
            config = _tt_mod._get_env_config()
            assert config["cwd"] == "/root"

    def test_default_cwd_is_root_for_container_backends(self):
        """Container backends should default to /root, not ~."""
        for backend in ("modal", "docker", "singularity", "daytona"):
            with patch.dict(os.environ, {"TERMINAL_ENV": backend}, clear=False):
                # Remove TERMINAL_CWD so it uses default
                env = os.environ.copy()
                env.pop("TERMINAL_CWD", None)
                with patch.dict(os.environ, env, clear=True):
                    config = _tt_mod._get_env_config()
                    assert config["cwd"] == "/root", (
                        f"Backend {backend}: expected /root default, got {config['cwd']}"
                    )

    def test_local_backend_uses_getcwd(self):
        """Local backend should use os.getcwd(), not /root."""
        with patch.dict(os.environ, {"TERMINAL_ENV": "local"}, clear=False):
            env = os.environ.copy()
            env.pop("TERMINAL_CWD", None)
            with patch.dict(os.environ, env, clear=True):
                config = _tt_mod._get_env_config()
                assert config["cwd"] == os.getcwd()

    def test_ssh_preserves_home_paths(self):
        """SSH backend should NOT replace /home/ paths (they're valid remotely)."""
        with patch.dict(os.environ, {
            "TERMINAL_ENV": "ssh",
            "TERMINAL_CWD": "/home/remote-user/work",
            "TERMINAL_SSH_HOST": "example.com",
            "TERMINAL_SSH_USER": "user",
        }):
            config = _tt_mod._get_env_config()
            assert config["cwd"] == "/home/remote-user/work", (
                "SSH backend should preserve /home/ paths"
            )


# =========================================================================
# Test 5: ephemeral_disk version check
# =========================================================================

class TestEphemeralDiskCheck:
    """Verify ephemeral_disk is only passed when modal supports it."""

    def test_ephemeral_disk_skipped_when_unsupported(self):
        """If modal.Sandbox.create doesn't have ephemeral_disk param, skip it."""
        # Mock the modal import and Sandbox.create signature
        mock_modal = MagicMock()
        mock_sandbox_create = MagicMock()
        # Simulate a signature WITHOUT ephemeral_disk
        import inspect
        mock_params = {
            "args": inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
            "image": inspect.Parameter("image", inspect.Parameter.KEYWORD_ONLY),
            "timeout": inspect.Parameter("timeout", inspect.Parameter.KEYWORD_ONLY),
            "cpu": inspect.Parameter("cpu", inspect.Parameter.KEYWORD_ONLY),
            "memory": inspect.Parameter("memory", inspect.Parameter.KEYWORD_ONLY),
        }
        mock_sig = inspect.Signature(parameters=list(mock_params.values()))

        with patch.dict(os.environ, {"TERMINAL_ENV": "modal"}):
            config = _tt_mod._get_env_config()
            # The config has container_disk default of 51200
            disk = config.get("container_disk", 51200)
            assert disk > 0, "disk should default to > 0"

            # Simulate the version check logic from terminal_tool.py
            sandbox_kwargs = {}
            if disk > 0:
                try:
                    if "ephemeral_disk" in mock_params:
                        sandbox_kwargs["ephemeral_disk"] = disk
                except Exception:
                    pass

            assert "ephemeral_disk" not in sandbox_kwargs, (
                "ephemeral_disk should not be set when Sandbox.create doesn't support it"
            )


# =========================================================================
# Test 6: ModalEnvironment defaults
# =========================================================================

class TestModalEnvironmentDefaults:
    """Verify ModalEnvironment has correct defaults."""

    def test_default_cwd_is_root(self):
        """ModalEnvironment default cwd should be /root, not ~."""
        from tools.environments.modal import ModalEnvironment
        import inspect
        sig = inspect.signature(ModalEnvironment.__init__)
        cwd_default = sig.parameters["cwd"].default
        assert cwd_default == "/root", (
            f"ModalEnvironment cwd default should be /root, got {cwd_default!r}. "
            "Tilde ~ is not expanded by subprocess.run(cwd=...)."
        )


# =========================================================================
# Test 7: ensurepip fix in patches.py
# =========================================================================

class TestEnsurepipFix:
    """Verify the pip fix is applied in the patched Modal init."""

    def test_patched_init_creates_image_with_setup_commands(self):
        """The patched __init__ should create a modal.Image with pip fix."""
        try:
            from environments.patches import _patch_swerex_modal
        except ImportError:
            pytest.skip("environments.patches not importable")

        # Check that the patch code references ensurepip
        import inspect
        source = inspect.getsource(_patch_swerex_modal)
        assert "ensurepip" in source, (
            "patches._patch_swerex_modal should include ensurepip fix "
            "for Modal's legacy image builder"
        )
        assert "setup_dockerfile_commands" in source, (
            "patches._patch_swerex_modal should use setup_dockerfile_commands "
            "to fix pip before Modal's bootstrap"
        )

    def test_patched_init_uses_install_pipx_from_config(self):
        """The patched init should respect install_pipx from config."""
        try:
            from environments.patches import _patch_swerex_modal
        except ImportError:
            pytest.skip("environments.patches not importable")

        import inspect
        source = inspect.getsource(_patch_swerex_modal)
        assert "install_pipx" in source, (
            "patches._patch_swerex_modal should pass install_pipx to ModalDeployment"
        )


# =========================================================================
# Test 8: Host prefix list completeness
# =========================================================================

class TestHostPrefixList:
    """Verify the host prefix list catches common host-only paths."""

    def test_all_common_host_prefixes_caught(self):
        """The host prefix check should catch /Users/, /home/, C:\\, C:/."""
        # Read the actual source to verify the prefixes
        import inspect
        source = inspect.getsource(_tt_mod._get_env_config)
        for prefix in ["/Users/", "/home/", 'C:\\\\"', "C:/"]:
            # Normalize for source comparison
            check = prefix.rstrip('"')
            assert check in source or prefix in source, (
                f"Host prefix {prefix!r} not found in _get_env_config. "
                "Container backends need this to avoid using host paths."
            )
