"""Tests for Modal sandbox infrastructure fixes (TBLite baseline).

Covers the bugs discovered while setting up TBLite evaluation:
1. Tool resolution — terminal + file tools load correctly
2. CWD fix — host paths get replaced with /root for container backends
3. ephemeral_disk version check
4. ensurepip fix in Modal image builder
5. No swe-rex dependency — uses native Modal SDK
6. /home/ added to host prefix check
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure repo root is importable
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    import tools.terminal_tool  # noqa: F401
    _tt_mod = sys.modules["tools.terminal_tool"]
except ImportError:
    pytest.skip("hermes-agent tools not importable (missing deps)", allow_module_level=True)


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
        assert "terminal" in names, f"terminal tool missing! Only got: {names}."


# =========================================================================
# Test 2-4: CWD handling for container backends
# =========================================================================

class TestCwdHandling:
    """Verify host paths are sanitized for container backends."""

    def test_home_path_replaced_for_modal(self, monkeypatch):
        """TERMINAL_CWD=/home/user/... should be replaced with /root for modal."""
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.setenv("TERMINAL_CWD", "/home/dakota/github/hermes-agent")
        config = _tt_mod._get_env_config()
        assert config["cwd"] == "/root", (
            f"Expected /root, got {config['cwd']}. "
            "/home/ paths should be replaced for modal backend."
        )

    def test_users_path_replaced_for_docker_by_default(self, monkeypatch):
        """Docker should keep host paths out of the sandbox unless explicitly enabled."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CWD", "/Users/someone/projects")
        config = _tt_mod._get_env_config()
        assert config["cwd"] == "/root", (
            f"Expected /root, got {config['cwd']}. "
            "Host paths should be discarded for docker backend by default."
        )
        assert config["host_cwd"] is None
        assert config["docker_mount_cwd_to_workspace"] is False

    def test_users_path_maps_to_workspace_for_docker_when_enabled(self, monkeypatch):
        """Docker should map the host cwd into /workspace only when explicitly enabled."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CWD", "/Users/someone/projects")
        monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
        config = _tt_mod._get_env_config()
        assert config["cwd"] == "/workspace"
        assert config["host_cwd"] == "/Users/someone/projects"
        assert config["docker_mount_cwd_to_workspace"] is True

    def test_windows_path_replaced_for_modal(self, monkeypatch):
        """TERMINAL_CWD=C:\\Users\\... should be replaced for modal."""
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.setenv("TERMINAL_CWD", "C:\\Users\\someone\\projects")
        config = _tt_mod._get_env_config()
        assert config["cwd"] == "/root"

    @pytest.mark.parametrize("backend", ["modal", "docker", "singularity", "daytona"])
    def test_default_cwd_is_root_for_container_backends(self, backend, monkeypatch):
        """Container backends should default to /root, not ~."""
        monkeypatch.setenv("TERMINAL_ENV", backend)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.delenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", raising=False)
        config = _tt_mod._get_env_config()
        assert config["cwd"] == "/root", (
            f"Backend {backend}: expected /root default, got {config['cwd']}"
        )

    def test_docker_default_cwd_maps_current_directory_when_enabled(self, monkeypatch):
        """Docker should use /workspace when cwd mounting is explicitly enabled."""
        monkeypatch.setattr("tools.terminal_tool.os.getcwd", lambda: "/home/user/project")
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        config = _tt_mod._get_env_config()
        assert config["cwd"] == "/workspace"
        assert config["host_cwd"] == "/home/user/project"

    def test_local_backend_uses_getcwd(self, monkeypatch):
        """Local backend should use os.getcwd(), not /root."""
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        config = _tt_mod._get_env_config()
        assert config["cwd"] == os.getcwd()

    def test_create_environment_passes_docker_host_cwd_and_flag(self, monkeypatch):
        """Docker host cwd and mount flag should reach DockerEnvironment."""
        captured = {}
        sentinel = object()

        def _fake_docker_environment(**kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(_tt_mod, "_DockerEnvironment", _fake_docker_environment)

        env = _tt_mod._create_environment(
            env_type="docker",
            image="python:3.11",
            cwd="/workspace",
            timeout=60,
            container_config={"docker_mount_cwd_to_workspace": True},
            host_cwd="/home/user/project",
        )

        assert env is sentinel
        assert captured["cwd"] == "/workspace"
        assert captured["host_cwd"] == "/home/user/project"
        assert captured["auto_mount_cwd"] is True

    def test_ssh_preserves_home_paths(self, monkeypatch):
        """SSH backend should NOT replace /home/ paths (they're valid remotely)."""
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_CWD", "/home/remote-user/work")
        monkeypatch.setenv("TERMINAL_SSH_HOST", "example.com")
        monkeypatch.setenv("TERMINAL_SSH_USER", "user")
        config = _tt_mod._get_env_config()
        assert config["cwd"] == "/home/remote-user/work", (
            "SSH backend should preserve /home/ paths"
        )


# =========================================================================
# Test 5: ephemeral_disk version check
# =========================================================================

class TestEphemeralDiskCheck:
    """Verify ephemeral_disk is only passed when modal supports it."""

    def test_ephemeral_disk_skipped_when_unsupported(self, monkeypatch):
        """If modal.Sandbox.create doesn't have ephemeral_disk param, skip it."""
        import inspect
        mock_params = {
            "args": inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
            "image": inspect.Parameter("image", inspect.Parameter.KEYWORD_ONLY),
            "timeout": inspect.Parameter("timeout", inspect.Parameter.KEYWORD_ONLY),
            "cpu": inspect.Parameter("cpu", inspect.Parameter.KEYWORD_ONLY),
            "memory": inspect.Parameter("memory", inspect.Parameter.KEYWORD_ONLY),
        }

        monkeypatch.setenv("TERMINAL_ENV", "modal")
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
# Test 7: ensurepip fix in ModalEnvironment
# =========================================================================

class TestEnsurepipFix:
    """Verify the pip fix is applied in the ModalEnvironment init."""

    def test_modal_environment_creates_image_with_setup_commands(self):
        """_resolve_modal_image should create a modal.Image with pip fix."""
        try:
            from tools.environments.modal import _resolve_modal_image
        except ImportError:
            pytest.skip("tools.environments.modal not importable")

        import inspect
        source = inspect.getsource(_resolve_modal_image)
        assert "ensurepip" in source, (
            "_resolve_modal_image should include ensurepip fix "
            "for Modal's legacy image builder"
        )
        assert "setup_dockerfile_commands" in source, (
            "_resolve_modal_image should use setup_dockerfile_commands "
            "to fix pip before Modal's bootstrap"
        )

    def test_modal_environment_uses_native_sdk(self):
        """ModalEnvironment should use Modal SDK directly, not swe-rex."""
        try:
            from tools.environments.modal import ModalEnvironment
        except ImportError:
            pytest.skip("tools.environments.modal not importable")

        import inspect
        source = inspect.getsource(ModalEnvironment)
        assert "swerex" not in source.lower(), (
            "ModalEnvironment should not depend on swe-rex; "
            "use Modal SDK directly via Sandbox.create() + exec()"
        )
        assert "Sandbox.create.aio" in source, (
            "ModalEnvironment should use async Modal Sandbox.create.aio()"
        )
        assert "exec.aio" in source, (
            "ModalEnvironment should use Sandbox.exec.aio() for command execution"
        )


# =========================================================================
# Test 8: Host prefix list completeness
# =========================================================================

class TestHostPrefixList:
    """Verify the host prefix list catches common host-only paths.

    The prefixes used to live as an inline literal inside ``_get_env_config``;
    they now live in the module-level ``_HOST_CWD_PREFIXES`` constant shared by
    both the ``_get_env_config`` sanitizer and the override-resolution guard
    (``_is_unusable_container_cwd``). Assert the *behavior* (each common host
    prefix is flagged as unusable inside a container) rather than grepping a
    function's source — the latter is a change-detector that breaks on any
    refactor that moves the constant.
    """

    def test_all_common_host_prefixes_present_in_constant(self):
        """The shared prefix constant must list the common host-only roots."""
        for prefix in ("/Users/", "/home/", "C:\\", "C:/"):
            assert prefix in _tt_mod._HOST_CWD_PREFIXES, (
                f"Host prefix {prefix!r} missing from _HOST_CWD_PREFIXES. "
                "Container backends need this to avoid using host paths."
            )

    def test_all_common_host_paths_flagged_unusable(self):
        """A host path under each prefix must be rejected as a container cwd."""
        for host_path in ("/Users/me/proj", "/home/me/proj",
                           "C:\\Users\\me", "C:/Users/me"):
            assert _tt_mod._is_unusable_container_cwd(host_path) is True, (
                f"Host path {host_path!r} should be rejected as a container "
                "cwd but was accepted."
            )


# =========================================================================
# Test 7: Host-bound Docker sandboxes must not bypass dangerous-command
# approval. Isolated Docker keeps the container fast-path; once a host path
# is bind-mounted into the container, a command like `rm -rf /workspace` can
# reach real host files, so it goes through the normal approval flow.
# (PR #6436, @Kolektori)
# =========================================================================

class TestDockerHostBindApproval:
    """Docker host bind mounts disable the container approval fast-path."""

    def test_docker_host_access_detection(self):
        """_docker_has_host_access flags bind-mounted host paths only."""
        # Isolated docker (no host binds) -> not host access.
        assert _tt_mod._docker_has_host_access(
            {"env_type": "docker", "docker_volumes": [],
             "host_cwd": None, "docker_mount_cwd_to_workspace": False}) is False
        # Host-path bind mount -> host access.
        assert _tt_mod._docker_has_host_access(
            {"env_type": "docker", "docker_volumes": ["/tmp:/hosttmp"]}) is True
        # Named volume (not a host path) -> not host access.
        assert _tt_mod._docker_has_host_access(
            {"env_type": "docker", "docker_volumes": ["myvol:/data"]}) is False
        # cwd auto-mount flag -> host access.
        assert _tt_mod._docker_has_host_access(
            {"env_type": "docker", "host_cwd": "/home/u/p",
             "docker_mount_cwd_to_workspace": True}) is True
        # Windows host path -> host access.
        assert _tt_mod._docker_has_host_access(
            {"env_type": "docker", "docker_volumes": ["C:\\Users:/data"]}) is True
        # Other container backends never report host access.
        assert _tt_mod._docker_has_host_access(
            {"env_type": "modal", "docker_volumes": ["/tmp:/x"]}) is False

    def test_should_skip_container_guards(self):
        """Docker skips only when isolated; other sandboxes always skip."""
        import tools.approval as A
        assert A._should_skip_container_guards("docker", has_host_access=False) is True
        assert A._should_skip_container_guards("docker", has_host_access=True) is False
        assert A._should_skip_container_guards("modal", has_host_access=True) is True
        assert A._should_skip_container_guards("singularity") is True
        assert A._should_skip_container_guards("daytona") is True
        assert A._should_skip_container_guards("local") is False

    def test_isolated_docker_keeps_fast_path(self, monkeypatch):
        """Isolated Docker still bypasses dangerous-command approval."""
        import tools.approval as A
        self._isolate_approval_state(monkeypatch)
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setattr(
            "tools.tirith_security.check_command_security",
            lambda _c: {"action": "allow", "findings": [], "summary": ""})
        res = A.check_all_command_guards("rm -rf /workspace", "docker",
                                         has_host_access=False)
        assert res["approved"] is True

    @staticmethod
    def _isolate_approval_state(monkeypatch):
        """Clear approval state that leaks in from the real user config.

        ``tools.approval`` loads ``command_allowlist`` into module-level
        ``_permanent_approved`` at import time. This file imports
        ``tools.terminal_tool`` at module level (collection time — BEFORE the
        hermetic HERMES_HOME fixture runs), so on a dev machine whose real
        config permanently allowlists e.g. "delete in root path" the guard
        under test silently approves and the assertions flip. CI never has
        such an allowlist, making this a local-only flake.
        """
        import tools.approval as A
        monkeypatch.setattr(A, "_permanent_approved", set())
        monkeypatch.setattr(A, "_session_approved", {})

    def test_host_bound_docker_requires_approval(self, monkeypatch):
        """Host-bound Docker dangerous command escalates instead of bypassing."""
        import tools.approval as A
        self._isolate_approval_state(monkeypatch)
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setattr(
            "tools.tirith_security.check_command_security",
            lambda _c: {"action": "allow", "findings": [], "summary": ""})
        res = A.check_all_command_guards("rm -rf /workspace", "docker",
                                         has_host_access=True)
        # Must NOT take the silent container fast-path.
        assert res.get("approved") is not True
        assert res.get("status") == "pending_approval"

    def test_execute_code_isolated_docker_keeps_fast_path(self, monkeypatch):
        """Isolated Docker execute_code still bypasses the guard."""
        import tools.approval as A
        self._isolate_approval_state(monkeypatch)
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        res = A.check_execute_code_guard("import os", "docker",
                                         has_host_access=False)
        assert res["approved"] is True

    def test_execute_code_host_bound_docker_requires_approval(self, monkeypatch):
        """Host-bound Docker execute_code does not get the container fast-path."""
        import tools.approval as A
        self._isolate_approval_state(monkeypatch)
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        res = A.check_execute_code_guard(
            "import os; os.system('rm -rf /workspace')", "docker",
            has_host_access=True)
        assert res.get("approved") is not True
        assert res.get("status") == "pending_approval"

    def test_execute_code_vercel_sandbox_always_skips(self, monkeypatch):
        """vercel_sandbox has no host-bind concept and stays always-skipped."""
        import tools.approval as A
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        res = A.check_execute_code_guard("import os", "vercel_sandbox",
                                         has_host_access=True)
        assert res["approved"] is True
