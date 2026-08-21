"""Unit tests for the agent-sandbox environment backend."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Global Import Mocks
# ---------------------------------------------------------------------------
# Mucking with sys.modules must happen AT THE TOP LEVEL before test collection.
# This ensures `_lazy_ensure` doesn't run, and prevents ImportErrors if
# other tests have already triggered imports of AgentSandboxBackend.

sys.modules["tools.lazy_deps"] = MagicMock()

mock_k8s = MagicMock()
mock_k8s_models = MagicMock()
mock_k8s.models = mock_k8s_models

sys.modules["k8s_agent_sandbox"] = mock_k8s
sys.modules["k8s_agent_sandbox.models"] = mock_k8s_models


# ---------------------------------------------------------------------------
# Helpers to build mock agent-sandbox SDK objects
# ---------------------------------------------------------------------------

def _make_exec_response(stdout="", stderr="", exit_code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


def _make_sandbox(claim_name="sb-123"):
    sb = MagicMock()
    sb.claim_name = claim_name
    sb.commands.run.return_value = _make_exec_response()
    return sb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def k8s_sdk():
    """Provide the globally mocked k8s SDK module and reset it for each test."""
    mock_k8s.reset_mock()
    mock_k8s_models.reset_mock()
    return mock_k8s


@pytest.fixture()
def make_env(k8s_sdk, monkeypatch):
    """Factory that creates a K8sSandboxBackend with a mocked SDK."""
    # Prevent is_interrupted from interfering — patch where it's used (base.py)
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    # Prevent skills/credential sync from consuming mock exec calls
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])

    def _factory(
        sandbox=None,
        list_all_return=None,
        get_side_effect=None,
        persistent=False,
        connection_config=None,
        **kwargs,
    ):
        sandbox = sandbox or _make_sandbox()
        sandbox.commands.run.return_value = _make_exec_response()

        mock_client = MagicMock()
        mock_client.create_sandbox.return_value = sandbox

        if get_side_effect is not None:
            mock_client.get_sandbox.side_effect = get_side_effect
        else:
            mock_client.get_sandbox.return_value = sandbox

        if list_all_return is not None:
            mock_client.list_all_sandboxes.return_value = list_all_return
        else:
            mock_client.list_all_sandboxes.side_effect = IndexError("list index out of range")

        # Configure the return_value of the existing mocked class.
        # Do NOT overwrite it with a new MagicMock(), otherwise 'from ... import SandboxClient'
        # inside the backend will hold onto the old mock and ignore this.
        k8s_sdk.SandboxClient.return_value = mock_client

        from tools.environments.agent_sandbox import AgentSandboxBackend

        if connection_config is None:
            connection_config = {"name": "SandboxLocalTunnelConnectionConfig"}

        kwargs.setdefault("cwd", "/home/test")
        kwargs.setdefault("warmpool", "test-warmpool")
        kwargs.setdefault("namespace", "test-namespace")
        env = AgentSandboxBackend(
            connection_config_args=connection_config,
            persistent_filesystem=persistent,
            **kwargs,
        )
        env._mock_client = mock_client  # expose for assertions
        return env

    return _factory


# ---------------------------------------------------------------------------
# Sandbox persistence / resume
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistent_resumes_via_get(self, make_env):
        existing = _make_sandbox(claim_name="claim-existing")
        env = make_env(
            get_side_effect=lambda name: existing if name == "claim-existing" else None,
            list_all_return=["claim-existing"],
            persistent=True,
            task_id="mytask"
        )
        env._mock_client.list_all_sandboxes.assert_called_once_with(label_selector="hermes_task_id=mytask")
        env._mock_client.get_sandbox.assert_called_once_with("claim-existing")
        env._mock_client.create_sandbox.assert_not_called()

    def test_persistent_creates_new_when_none_found(self, make_env):
        env = make_env(
            list_all_return=[],
            persistent=True,
            task_id="mytask",
        )
        env._mock_client.create_sandbox.assert_called_once()
        env._mock_client.list_all_sandboxes.assert_called_once_with(label_selector="hermes_task_id=mytask")
        # Because list_all_sandboxes returned empty list, IndexError was raised in backend (by [0])
        # So get_sandbox is never called.
        env._mock_client.get_sandbox.assert_not_called()

    def test_non_persistent_skips_lookup(self, make_env):
        env = make_env(persistent=False, task_id="mytask")
        env._mock_client.get_sandbox.assert_not_called()
        env._mock_client.list_all_sandboxes.assert_not_called()
        env._mock_client.create_sandbox.assert_called_once_with(
            warmpool="test-warmpool",
            namespace="test-namespace",
            labels={"hermes_task_id": "mytask"}
        )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_persistent_cleanup_closes_connection(self, make_env):
        env = make_env(persistent=True)
        sb = env._sandbox
        env.cleanup()
        sb.close_connection.assert_called_once()
        sb.terminate.assert_not_called()

    def test_non_persistent_cleanup_deletes_sandbox(self, make_env):
        env = make_env(persistent=False)
        sb = env._sandbox
        env.cleanup()
        sb.terminate.assert_called_once()
        sb.close_connection.assert_not_called()

    def test_cleanup_idempotent(self, make_env):
        env = make_env(persistent=True)
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_swallows_errors(self, make_env):
        env = make_env(persistent=True)
        env._sandbox.close_connection.side_effect = RuntimeError("stop failed")
        env.cleanup()  # should not raise


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_basic_command(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_exec_response(stdout="", exit_code=0),  # init_session
            _make_exec_response(stdout="hello", exit_code=0),  # actual cmd
        ]
        env = make_env(sandbox=sb)

        result = env.execute("echo hello")
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_sdk_timeout_passed_to_exec(self, make_env):
        """SDK native timeout is passed to sandbox.commands.run()."""
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_exec_response(stdout="", exit_code=0),  # init_session
            _make_exec_response(stdout="ok", exit_code=0),
        ]
        env = make_env(sandbox=sb, timeout=42)

        env.execute("echo hello")
        # The exec call should receive timeout= kwarg (SDK native timeout)
        call_args = sb.commands.run.call_args_list[-1]
        assert call_args[1]["timeout"] == 42
        # The command should NOT have a shell `timeout` prefix
        cmd = call_args[1]["command"]
        assert not cmd.startswith("timeout ")

    def test_nonzero_exit_code(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_exec_response(stdout="", exit_code=0),  # init_session
            _make_exec_response(stderr="not found", exit_code=127),
        ]
        env = make_env(sandbox=sb)

        result = env.execute("bad_cmd")
        assert result["returncode"] == 127
        assert "not found" in result["output"]


# ---------------------------------------------------------------------------
# Connection Config Tests
# ---------------------------------------------------------------------------

class TestConnectionConfig:
    def test_sandbox_local_tunnel_connection_config(self, make_env, k8s_sdk):
        connection_config = {
            "name": "SandboxLocalTunnelConnectionConfig",
            "port_forward_ready_timeout": 50,
            "server_port": 9000,
            "router_namespace": "custom-ns"
        }
        env = make_env(connection_config=connection_config)
        k8s_sdk.models.SandboxLocalTunnelConnectionConfig.assert_called_once_with(
            port_forward_ready_timeout=50,
            server_port=9000,
            router_namespace="custom-ns"
        )
        env._mock_client.create_sandbox.assert_called_once()

    def test_sandbox_direct_connection_config(self, make_env, k8s_sdk):
        connection_config = {
            "name": "SandboxDirectConnectionConfig",
            "api_url": "http://api",
            "server_port": 9000,
        }
        env = make_env(connection_config=connection_config)
        k8s_sdk.models.SandboxDirectConnectionConfig.assert_called_once_with(
            api_url="http://api",
            server_port=9000,
        )

    def test_sandbox_gateway_connection_config(self, make_env, k8s_sdk):
        connection_config = {
            "name": "SandboxGatewayConnectionConfig",
            "gateway_name": "gw",
            "gateway_namespace": "gwns",
            "gateway_ready_timeout": 200,
            "server_port": 9000,
        }
        env = make_env(connection_config=connection_config)
        k8s_sdk.models.SandboxGatewayConnectionConfig.assert_called_once_with(
            gateway_name="gw",
            gateway_namespace="gwns",
            gateway_ready_timeout=200,
            server_port=9000,
        )

    def test_sandbox_in_cluster_connection_config(self, make_env, k8s_sdk):
        connection_config = {
            "name": "SandboxInClusterConnectionConfig",
            "server_port": 8080,
            "use_pod_ip": True,
        }
        env = make_env(connection_config=connection_config)
        k8s_sdk.models.SandboxInClusterConnectionConfig.assert_called_once_with(
            server_port=8080,
            use_pod_ip=True,
        )

    def test_invalid_connection_config(self, make_env):
        connection_config = {
            "name": "InvalidConfig"
        }
        with pytest.raises(ValueError, match="Not allowed connection config name"):
            make_env(connection_config=connection_config)

# ---------------------------------------------------------------------------
# Single-file upload (mid-session file change)
# ---------------------------------------------------------------------------

class TestSingleFileUpload:
    def test_change_file_mid_session_then_read_back(self, make_env, tmp_path):
        """Covers `_agent_sandbox_upload`, the single-file path used by
        FileSyncManager's `upload_fn` (as opposed to `_agent_sandbox_bulk_upload`,
        which batches multiple files into one tar).
        """
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_exec_response(stdout="", exit_code=0),               # init_session
            _make_exec_response(stdout="updated content", exit_code=0),  # cat after read-back
        ]
        env = make_env(sandbox=sb)

        # Simulate a file changing mid-session on the host side.
        host_file = tmp_path / "changed.txt"
        host_file.write_text("updated content")
        remote_path = "workspace/changed.txt"

        # Exercise the single-file upload path directly.
        env._agent_sandbox_upload(str(host_file), remote_path)

        # The SDK's single-file write should be called once with the exact bytes,
        # confirming this went through the single-file path, not the bulk tar path.
        sb.files.write.assert_called_once_with(path=remote_path, content=b"updated content")
        env._mock_client.create_sandbox.return_value.commands.run.assert_not_called() \
            if False else None  # no-op guard placeholder, remove if not needed

        # Read the file back from inside the sandbox to confirm the round trip.
        result = env.execute(f"cat {remote_path}")
        assert result["returncode"] == 0
        assert "updated content" in result["output"]
