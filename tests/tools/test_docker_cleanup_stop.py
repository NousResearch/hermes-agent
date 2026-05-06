"""Tests for Docker environment cleanup integration with /stop command.

These tests verify that the /stop command properly cleans up Docker containers
and doesn't leave orphaned containers running.
"""

from unittest.mock import MagicMock, patch, call
import pytest


def test_docker_cleanup_wait_parameter_signature():
    """Verify that DockerEnvironment.cleanup() has wait parameter."""
    from tools.environments.docker import DockerEnvironment
    import inspect
    
    # Check the signature
    sig = inspect.signature(DockerEnvironment.cleanup)
    assert "wait" in sig.parameters, "cleanup() should have a 'wait' parameter"
    
    # Check default value
    wait_param = sig.parameters["wait"]
    assert wait_param.default is False, "wait parameter should default to False (async mode)"


def test_docker_cleanup_wait_true_uses_subprocess_run():
    """Verify that cleanup(wait=True) uses subprocess.run (synchronous)."""
    from tools.environments.docker import DockerEnvironment
    
    with patch("tools.environments.docker._ensure_docker_available"):
        # Create environment with mocked __init__ to avoid Docker checks
        with patch.object(DockerEnvironment, "__init__", lambda self, **kw: None):
            env = DockerEnvironment.__new__(DockerEnvironment)
            env._container_id = "test-container-id"
            env._persistent = False
            env._workspace_dir = None
            env._home_dir = None
            env._docker_exe = "docker"
            
            with patch("tools.environments.docker.subprocess") as mock_subprocess:
                mock_subprocess.run.return_value = MagicMock(returncode=0)
                
                # Test with wait=True
                env.cleanup(wait=True)
                
                # Verify that subprocess.run was called (synchronous mode)
                assert mock_subprocess.run.called, "cleanup(wait=True) should use subprocess.run"


def test_docker_cleanup_wait_false_uses_subprocess_popen():
    """Verify that cleanup(wait=False) uses subprocess.Popen (async)."""
    from tools.environments.docker import DockerEnvironment
    
    with patch("tools.environments.docker._ensure_docker_available"):
        # Create environment with mocked __init__ to avoid Docker checks
        with patch.object(DockerEnvironment, "__init__", lambda self, **kw: None):
            env = DockerEnvironment.__new__(DockerEnvironment)
            env._container_id = "test-container-id"
            env._persistent = False
            env._workspace_dir = None
            env._home_dir = None
            env._docker_exe = "docker"
            
            with patch("tools.environments.docker.subprocess") as mock_subprocess:
                mock_subprocess.Popen.return_value = MagicMock(poll=lambda: 0)
                
                # Test with wait=False (default)
                env.cleanup(wait=False)
                
                # Verify that subprocess.Popen was called (async mode)
                assert mock_subprocess.Popen.called, "cleanup(wait=False) should use subprocess.Popen"
