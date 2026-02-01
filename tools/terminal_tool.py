#!/usr/bin/env python3
"""
Terminal Tool Module (mini-swe-agent backend)

A terminal tool that executes commands using mini-swe-agent's execution environments.
Supports local execution, Docker containers, and Modal cloud sandboxes.

Environment Selection (via TERMINAL_ENV environment variable):
- "local": Execute directly on the host machine (default, fastest)
- "docker": Execute in Docker containers (isolated, requires Docker)
- "modal": Execute in Modal cloud sandboxes (scalable, requires Modal account)

Features:
- Multiple execution backends (local, docker, modal)
- Background task support
- VM/container lifecycle management
- Automatic cleanup after inactivity

Usage:
    from terminal_tool import terminal_tool

    # Execute a simple command
    result = terminal_tool("ls -la")

    # Execute in background
    result = terminal_tool("python server.py", background=True)
"""

import json
import os
import sys
import time
import threading
import atexit
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

# Add mini-swe-agent to path if not installed
mini_swe_path = Path(__file__).parent.parent / "mini-swe-agent" / "src"
if mini_swe_path.exists():
    sys.path.insert(0, str(mini_swe_path))


# =============================================================================
# Custom Singularity Environment with more space
# =============================================================================

def _get_scratch_dir() -> Path:
    """Get the best directory for Singularity sandboxes - prefers /scratch if available."""
    # Check for configurable scratch directory first (highest priority)
    custom_scratch = os.getenv("TERMINAL_SCRATCH_DIR")
    if custom_scratch:
        scratch_path = Path(custom_scratch)
        scratch_path.mkdir(parents=True, exist_ok=True)
        return scratch_path
    
    # Check for /scratch (common on HPC clusters, especially GPU nodes)
    scratch = Path("/scratch")
    if scratch.exists() and os.access(scratch, os.W_OK):
        # Create user-specific subdirectory
        user_scratch = scratch / os.getenv("USER", "hermes") / "hermes-agent"
        user_scratch.mkdir(parents=True, exist_ok=True)
        if not os.getenv("HERMES_QUIET"):
            print(f"[Terminal] Using /scratch for sandboxes: {user_scratch}")
        return user_scratch
    
    # Fall back to /tmp
    if not os.getenv("HERMES_QUIET"):
        print("[Terminal] Warning: /scratch not available, using /tmp (limited space)")
    return Path(tempfile.gettempdir())


def _get_apptainer_cache_dir() -> Path:
    """Get the Apptainer cache directory for SIF images."""
    # Check for APPTAINER_CACHEDIR env var
    cache_dir = os.getenv("APPTAINER_CACHEDIR")
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    # Use scratch dir parent for cache (one level up from sandboxes)
    scratch = _get_scratch_dir()
    cache_path = scratch.parent / ".apptainer"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


# Lock for SIF building to prevent race conditions
_sif_build_lock = threading.Lock()


def _get_or_build_sif(image: str, executable: str = "apptainer") -> str:
    """
    Get or build a SIF image from a docker:// URL.
    
    If the image is already a .sif file, returns it as-is.
    If the image is a docker:// URL, checks for cached SIF and builds if needed.
    
    Args:
        image: Image path (docker://... URL or .sif path)
        executable: apptainer or singularity
        
    Returns:
        Path to SIF file, or original image if not a docker:// URL
    """
    # If already a .sif file, use it directly
    if image.endswith('.sif') and Path(image).exists():
        return image
    
    # If not a docker:// URL, return as-is (could be a local sandbox or other format)
    if not image.startswith('docker://'):
        return image
    
    # Generate SIF filename from docker image name
    # docker://nikolaik/python-nodejs:python3.11-nodejs20 -> python-nodejs-python3.11-nodejs20.sif
    image_name = image.replace('docker://', '').replace('/', '-').replace(':', '-')
    cache_dir = _get_apptainer_cache_dir()
    sif_path = cache_dir / f"{image_name}.sif"
    
    # Check if SIF already exists
    if sif_path.exists():
        return str(sif_path)
    
    # Build SIF with lock to prevent multiple workers building simultaneously
    with _sif_build_lock:
        # Double-check after acquiring lock (another thread may have built it)
        if sif_path.exists():
            return str(sif_path)
        
        print(f"[Terminal] Building SIF image (one-time setup)...")
        print(f"[Terminal]   Source: {image}")
        print(f"[Terminal]   Target: {sif_path}")
        
        # Ensure tmp directory exists for build
        tmp_dir = cache_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set APPTAINER_TMPDIR for the build
        env = os.environ.copy()
        env["APPTAINER_TMPDIR"] = str(tmp_dir)
        env["APPTAINER_CACHEDIR"] = str(cache_dir)
        
        try:
            result = subprocess.run(
                [executable, "build", str(sif_path), image],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for pulling and building
                env=env
            )
            if result.returncode != 0:
                print(f"[Terminal] ⚠️ SIF build failed, falling back to docker:// URL")
                print(f"[Terminal]   Error: {result.stderr[:500]}")
                return image
            
            print(f"[Terminal] ✅ SIF image built successfully")
            return str(sif_path)
            
        except subprocess.TimeoutExpired:
            print(f"[Terminal] ⚠️ SIF build timed out, falling back to docker:// URL")
            # Clean up partial file
            if sif_path.exists():
                sif_path.unlink()
            return image
        except Exception as e:
            print(f"[Terminal] ⚠️ SIF build error: {e}, falling back to docker:// URL")
            return image


# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = float(os.getenv("TERMINAL_DISK_WARNING_GB", "500"))


def _check_disk_usage_warning():
    """Check if total disk usage exceeds warning threshold."""
    scratch_dir = _get_scratch_dir()
    
    try:
        # Get total size of hermes directories
        total_bytes = 0
        import glob
        for path in glob.glob(str(scratch_dir / "hermes-*")):
            for f in Path(path).rglob('*'):
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                    except:
                        pass
        
        total_gb = total_bytes / (1024 ** 3)
        
        if total_gb > DISK_USAGE_WARNING_THRESHOLD_GB:
            print(f"⚠️  [Terminal] WARNING: Disk usage ({total_gb:.1f}GB) exceeds threshold ({DISK_USAGE_WARNING_THRESHOLD_GB}GB)")
            print(f"    Consider running cleanup_all_environments() or reducing parallel workers")
            return True
        
        return False
    except Exception as e:
        return False


def _transform_sudo_command(command: str) -> str:
    """
    Transform sudo commands to use -S flag if SUDO_PASSWORD is available.
    
    This is a shared helper used by all execution environments to provide
    consistent sudo handling across local, SSH, and container environments.
    
    If SUDO_PASSWORD is set, transforms:
      'sudo apt install curl' -> password piped via sudo -S
      
    If SUDO_PASSWORD is not set, command runs as-is (will fail gracefully
    with "sudo: a password is required" error due to stdin=DEVNULL).
    """
    sudo_password = os.getenv("SUDO_PASSWORD", "")
    
    if not sudo_password:
        return command  # No password, let it fail gracefully
    
    # Check if command contains sudo (simple detection)
    # Handle: "sudo cmd", "sudo -flag cmd", "cmd && sudo cmd2", etc.
    import re
    
    def replace_sudo(match):
        # Replace 'sudo' with password-piped version
        # The -S flag makes sudo read password from stdin
        # The -p '' suppresses the password prompt
        return f"echo '{sudo_password}' | sudo -S -p ''"
    
    # Match 'sudo' at word boundaries (not 'visudo' or 'sudoers')
    # This handles: sudo, sudo -flag, etc.
    return re.sub(r'\bsudo\b', replace_sudo, command)


class _LocalEnvironment:
    """
    Local execution environment with sudo support and non-blocking stdin.
    
    Features:
    - Uses stdin=DEVNULL to prevent hanging on interactive prompts (sudo, etc.)
    - Optional SUDO_PASSWORD support: if set, transforms `sudo` commands to use `sudo -S`
    - Graceful failure: sudo commands fail fast with clear error if no password configured
    
    Environment variables:
    - SUDO_PASSWORD: If set, enables sudo commands by piping password via `sudo -S`
    """
    
    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        self.cwd = cwd or os.getcwd()
        self.timeout = timeout
        self.env = env or {}
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command locally with sudo support."""
        work_dir = cwd or self.cwd or os.getcwd()
        effective_timeout = timeout or self.timeout
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        try:
            result = subprocess.run(
                exec_command,
                shell=True,
                text=True,
                cwd=work_dir,
                env=os.environ | self.env,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent hanging on interactive prompts
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {effective_timeout}s", "returncode": 124}
        except Exception as e:
            return {"output": f"Execution error: {str(e)}", "returncode": 1}
    
    def cleanup(self):
        """No cleanup needed for local environment."""
        pass
    
    def stop(self):
        """Alias for cleanup."""
        pass


class _SingularityEnvironment:
    """
    Custom Singularity/Apptainer environment with better space management.
    
    - Automatically builds/caches SIF images from docker:// URLs
    - Builds sandbox in /scratch (if available) or configurable location
    - Binds a large working directory into the container
    - Keeps container isolated from host filesystem
    """
    
    def __init__(self, image: str, cwd: str = "/workspace", timeout: int = 60):
        self.cwd = cwd
        self.timeout = timeout
        
        # Use apptainer if available, otherwise singularity
        self.executable = "apptainer" if shutil.which("apptainer") else "singularity"
        
        # Get or build SIF from docker:// URL (fast if already cached)
        self.image = _get_or_build_sif(image, self.executable)
        
        # Get scratch directory for sandbox
        self.scratch_dir = _get_scratch_dir()
        
        # Create unique sandbox directory
        self.sandbox_id = f"hermes-{uuid.uuid4().hex[:12]}"
        self.sandbox_dir = self.scratch_dir / self.sandbox_id
        
        # Create a working directory that will be bound into the container
        self.work_dir = self.scratch_dir / f"{self.sandbox_id}-work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the sandbox
        self._build_sandbox()
    
    def _build_sandbox(self):
        """Build a writable sandbox from the container image (SIF or other)."""
        try:
            result = subprocess.run(
                [self.executable, "build", "--sandbox", str(self.sandbox_dir), self.image],
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout for building
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to build sandbox: {result.stderr}")
            
            # Create /workspace directory inside the sandbox for bind mounting
            workspace_in_sandbox = self.sandbox_dir / "workspace"
            workspace_in_sandbox.mkdir(parents=True, exist_ok=True)
            
        except subprocess.TimeoutExpired:
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            raise RuntimeError("Sandbox build timed out")
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command in the Singularity container."""
        cmd = [self.executable, "exec"]
        
        # Isolation flags - contain but allow network
        cmd.extend(["--contain", "--cleanenv"])
        
        # Bind the working directory into the container at /workspace
        # This gives the container access to a large writable space
        cmd.extend(["--bind", f"{self.work_dir}:/workspace"])
        
        # Also bind it to /tmp inside container for pip cache etc.
        cmd.extend(["--bind", f"{self.work_dir}:/tmp"])
        
        # Set working directory
        work_dir = cwd or self.cwd
        cmd.extend(["--pwd", work_dir])
        
        # Use writable sandbox
        cmd.extend(["--writable", str(self.sandbox_dir)])
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        # Execute the command
        cmd.extend(["bash", "-c", exec_command])
        
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent hanging on interactive prompts
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {timeout or self.timeout}s", "returncode": 124}
    
    def cleanup(self):
        """Clean up sandbox and working directory."""
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)
    
    def stop(self):
        """Alias for cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


class _SSHEnvironment:
    """
    SSH-based remote execution environment.
    
    Runs commands on a remote machine over SSH, keeping the agent code
    completely isolated from the execution environment. Uses SSH ControlMaster
    for connection persistence (faster subsequent commands).
    
    Security benefits:
    - Agent cannot modify its own code
    - Remote machine acts as a sandbox
    - Clear separation between agent and execution environment
    """
    
    def __init__(self, host: str, user: str, cwd: str = "/tmp", timeout: int = 60,
                 port: int = 22, key_path: str = ""):
        self.host = host
        self.user = user
        self.cwd = cwd
        self.timeout = timeout
        self.port = port
        self.key_path = key_path
        
        # Create control socket directory for connection persistence
        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.control_socket = self.control_dir / f"{user}@{host}:{port}.sock"
        
        # Test connection and establish ControlMaster
        self._establish_connection()
    
    def _build_ssh_command(self, extra_args: list = None) -> list:
        """Build base SSH command with connection options."""
        cmd = ["ssh"]
        
        # Connection multiplexing for performance
        cmd.extend(["-o", f"ControlPath={self.control_socket}"])
        cmd.extend(["-o", "ControlMaster=auto"])
        cmd.extend(["-o", "ControlPersist=300"])  # Keep connection alive for 5 min
        
        # Standard options
        cmd.extend(["-o", "BatchMode=yes"])  # No password prompts
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])  # Accept new hosts
        cmd.extend(["-o", "ConnectTimeout=10"])
        
        # Port
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        
        # Private key
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        
        # Extra args (like -t for TTY)
        if extra_args:
            cmd.extend(extra_args)
        
        # Target
        cmd.append(f"{self.user}@{self.host}")
        
        return cmd
    
    def _establish_connection(self):
        """Test SSH connection and establish ControlMaster."""
        cmd = self._build_ssh_command()
        cmd.append("echo 'SSH connection established'")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"SSH connection failed: {error_msg}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SSH connection to {self.user}@{self.host} timed out")
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command on the remote host via SSH."""
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        # Wrap command to run in the correct directory
        # Use bash -c to handle complex commands properly
        wrapped_command = f'cd {work_dir} && {exec_command}'
        
        cmd = self._build_ssh_command()
        cmd.extend(["bash", "-c", wrapped_command])
        
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent hanging on interactive prompts
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {effective_timeout}s", "returncode": 124}
        except Exception as e:
            return {"output": f"SSH execution error: {str(e)}", "returncode": 1}
    
    def cleanup(self):
        """Close the SSH ControlMaster connection."""
        if self.control_socket.exists():
            try:
                # Send exit command to ControlMaster
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}", "-O", "exit", 
                       f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5)
            except:
                pass
            
            # Remove socket file
            try:
                self.control_socket.unlink()
            except:
                pass
    
    def stop(self):
        """Alias for cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


class _DockerEnvironment:
    """
    Docker execution environment wrapper with sudo support and non-blocking stdin.
    
    Wraps mini-swe-agent's DockerEnvironment but adds:
    - stdin=DEVNULL to prevent hanging on interactive prompts
    - SUDO_PASSWORD support via _transform_sudo_command
    """
    
    def __init__(self, image: str, cwd: str = "/", timeout: int = 60):
        from minisweagent.environments.docker import DockerEnvironment
        self._inner = DockerEnvironment(image=image, cwd=cwd, timeout=timeout)
        self.cwd = cwd
        self.timeout = timeout
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command in the Docker container with sudo support."""
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout
        
        # Get container_id from inner environment
        assert self._inner.container_id, "Container not started"
        
        cmd = [self._inner.config.executable, "exec", "-w", work_dir]
        for key in self._inner.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self._inner.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._inner.container_id, "bash", "-lc", exec_command])
        
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent hanging on interactive prompts
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {effective_timeout}s", "returncode": 124}
    
    def cleanup(self):
        """Cleanup the Docker container."""
        self._inner.cleanup()
    
    def stop(self):
        """Alias for cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


class _ModalEnvironment:
    """
    Modal cloud execution environment wrapper with sudo support.
    
    Wraps mini-swe-agent's SwerexModalEnvironment but adds:
    - SUDO_PASSWORD support via _transform_sudo_command
    
    Note: stdin handling is not needed for Modal since it uses remote async execution.
    """
    
    def __init__(self, image: str, cwd: str = "/", timeout: int = 60):
        from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
        self._inner = SwerexModalEnvironment(image=image, cwd=cwd, timeout=timeout)
        self.cwd = cwd
        self.timeout = timeout
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command in Modal with sudo support."""
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        # Delegate to inner environment with transformed command
        return self._inner.execute(exec_command, cwd=cwd, timeout=timeout)
    
    def cleanup(self):
        """Cleanup the Modal deployment."""
        if hasattr(self._inner, 'stop'):
            self._inner.stop()
    
    def stop(self):
        """Stop the Modal deployment."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute commands on a secure Linux environment.

**Environment:**
- Isolated execution environment (local, Docker, or Modal cloud based on configuration)
- Filesystem persists between tool calls within the same task
- Internet access available

**Command Execution:**
- Simple commands: Just provide the 'command' parameter
- Background processes: Set 'background': True for servers/long-running tasks
- Command timeout: Optional 'timeout' parameter in seconds

**Examples:**
- Run command: `{"command": "ls -la"}`
- Background task: `{"command": "source venv/bin/activate && python server.py", "background": True}`
- With timeout: `{"command": "long_task.sh", "timeout": 300}`

**Best Practices:**
- Run servers/long processes in background
- Monitor disk usage for large tasks
- Install whatever tools you need with apt-get or pip
- Do not be afraid to run pip with --break-system-packages

**Things to avoid:**
- Do NOT use interactive tools such as tmux, vim, nano, python repl - you will get stuck.
- Even git sometimes becomes interactive if the output is large. If you're not sure, pipe to cat.
"""

# Global state for environment lifecycle management
_active_environments: Dict[str, Any] = {}
_task_workdirs: Dict[str, str] = {}  # Maps task_id to working directory
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_cleanup_thread = None
_cleanup_running = False

# Configuration from environment variables
def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    return {
        "env_type": os.getenv("TERMINAL_ENV", "local"),  # local, docker, singularity, modal, or ssh
        "docker_image": os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11"),
        "singularity_image": os.getenv("TERMINAL_SINGULARITY_IMAGE", "docker://python:3.11"),
        "modal_image": os.getenv("TERMINAL_MODAL_IMAGE", "python:3.11"),
        "cwd": os.getenv("TERMINAL_CWD", "/tmp"),
        "timeout": int(os.getenv("TERMINAL_TIMEOUT", "60")),
        "lifetime_seconds": int(os.getenv("TERMINAL_LIFETIME_SECONDS", "300")),
        # SSH-specific config
        "ssh_host": os.getenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": os.getenv("TERMINAL_SSH_USER", ""),
        "ssh_port": int(os.getenv("TERMINAL_SSH_PORT", "22")),
        "ssh_key": os.getenv("TERMINAL_SSH_KEY", ""),  # Path to private key (optional, uses ssh-agent if empty)
    }


def _create_environment(env_type: str, image: str, cwd: str, timeout: int, ssh_config: dict = None):
    """
    Create an execution environment from mini-swe-agent.
    
    Args:
        env_type: One of "local", "docker", "singularity", "modal", "ssh"
        image: Docker/Singularity/Modal image name (ignored for local/ssh)
        cwd: Working directory
        timeout: Default command timeout
        ssh_config: SSH connection config (for env_type="ssh")
        
    Returns:
        Environment instance with execute() method
    """
    if env_type == "local":
        # Use our custom LocalEnvironment with sudo support and non-blocking stdin
        return _LocalEnvironment(cwd=cwd, timeout=timeout)
    
    elif env_type == "docker":
        # Use custom Docker wrapper with sudo support and non-blocking stdin
        return _DockerEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    elif env_type == "singularity":
        # Use custom Singularity environment with better space management
        return _SingularityEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    elif env_type == "modal":
        # Use custom Modal wrapper with sudo support
        return _ModalEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    elif env_type == "ssh":
        if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
            raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
        return _SSHEnvironment(
            host=ssh_config["host"],
            user=ssh_config["user"],
            port=ssh_config.get("port", 22),
            key_path=ssh_config.get("key", ""),
            cwd=cwd,
            timeout=timeout
        )
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Use 'local', 'docker', 'singularity', 'modal', or 'ssh'")


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    global _active_environments, _last_activity

    current_time = time.time()
    tasks_to_cleanup = []

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > lifetime_seconds:
                tasks_to_cleanup.append(task_id)

        for task_id in tasks_to_cleanup:
            try:
                if task_id in _active_environments:
                    env = _active_environments[task_id]
                    # Try various cleanup methods
                    if hasattr(env, 'cleanup'):
                        env.cleanup()
                    elif hasattr(env, 'stop'):
                        env.stop()
                    elif hasattr(env, 'terminate'):
                        env.terminate()

                    del _active_environments[task_id]
                    if not os.getenv("HERMES_QUIET"):
                        print(f"[Terminal Cleanup] Cleaned up inactive environment for task: {task_id}")

                if task_id in _last_activity:
                    del _last_activity[task_id]
                if task_id in _task_workdirs:
                    del _task_workdirs[task_id]

            except Exception as e:
                error_str = str(e)
                if not os.getenv("HERMES_QUIET"):
                    if "404" in error_str or "not found" in error_str.lower():
                        print(f"[Terminal Cleanup] Environment for task {task_id} already cleaned up")
                    else:
                        print(f"[Terminal Cleanup] Error cleaning up environment for task {task_id}: {e}")
                
                # Always remove from tracking dicts
                if task_id in _active_environments:
                    del _active_environments[task_id]
                if task_id in _last_activity:
                    del _last_activity[task_id]
                if task_id in _task_workdirs:
                    del _task_workdirs[task_id]


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    global _cleanup_running

    while _cleanup_running:
        try:
            config = _get_env_config()
            _cleanup_inactive_envs(config["lifetime_seconds"])
        except Exception as e:
            if not os.getenv("HERMES_QUIET"):
                print(f"[Terminal Cleanup] Error in cleanup thread: {e}")

        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _env_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()


def _stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        _cleanup_thread.join(timeout=5)


def get_active_environments_info() -> Dict[str, Any]:
    """Get information about currently active environments."""
    info = {
        "count": len(_active_environments),
        "task_ids": list(_active_environments.keys()),
        "workdirs": dict(_task_workdirs),
    }
    
    # Calculate total disk usage
    total_size = 0
    for task_id in _active_environments.keys():
        # Check sandbox and workdir sizes
        scratch_dir = _get_scratch_dir()
        for pattern in [f"hermes-*{task_id[:8]}*"]:
            import glob
            for path in glob.glob(str(scratch_dir / "hermes-*")):
                try:
                    size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
                    total_size += size
                except:
                    pass
    
    info["total_disk_usage_mb"] = round(total_size / (1024 * 1024), 2)
    return info


def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    global _active_environments, _last_activity, _task_workdirs
    
    task_ids = list(_active_environments.keys())
    cleaned = 0
    
    for task_id in task_ids:
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            print(f"[Terminal Cleanup] Error cleaning {task_id}: {e}")
    
    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"[Terminal Cleanup] Removed orphaned: {path}")
        except:
            pass
    
    print(f"[Terminal Cleanup] Cleaned {cleaned} environments")
    return cleaned


def cleanup_vm(task_id: str):
    """Manually clean up a specific environment by task_id."""
    global _active_environments, _last_activity, _task_workdirs

    with _env_lock:
        try:
            if task_id in _active_environments:
                env = _active_environments[task_id]
                if hasattr(env, 'cleanup'):
                    env.cleanup()
                elif hasattr(env, 'stop'):
                    env.stop()
                elif hasattr(env, 'terminate'):
                    env.terminate()

                del _active_environments[task_id]
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Terminal Cleanup] Manually cleaned up environment for task: {task_id}")

            if task_id in _task_workdirs:
                del _task_workdirs[task_id]

            if task_id in _last_activity:
                del _last_activity[task_id]

        except Exception as e:
            if not os.getenv("HERMES_QUIET"):
                error_str = str(e)
                if "404" in error_str or "not found" in error_str.lower():
                    print(f"[Terminal Cleanup] Environment for task {task_id} already cleaned up")
                else:
                    print(f"[Terminal Cleanup] Error cleaning up environment for task {task_id}: {e}")


atexit.register(_stop_cleanup_thread)


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None
) -> str:
    """
    Execute a command using mini-swe-agent's execution environments.

    Args:
        command: The command to execute
        background: Whether to run in background (default: False)
        timeout: Command timeout in seconds (default: from config)
        task_id: Unique identifier for environment isolation (optional)

    Returns:
        str: JSON string with output, exit_code, and error fields

    Examples:
        # Execute a simple command
        >>> result = terminal_tool(command="ls -la /tmp")

        # Run a background task
        >>> result = terminal_tool(command="python server.py", background=True)

        # With custom timeout
        >>> result = terminal_tool(command="long_task.sh", timeout=300)
    """
    global _active_environments, _last_activity

    try:
        # Get configuration
        config = _get_env_config()
        env_type = config["env_type"]
        
        # Select image based on env type
        if env_type == "docker":
            image = config["docker_image"]
        elif env_type == "singularity":
            image = config["singularity_image"]
        elif env_type == "modal":
            image = config["modal_image"]
        else:
            image = ""
        
        cwd = config["cwd"]
        default_timeout = config["timeout"]
        effective_timeout = timeout or default_timeout

        # Use task_id for environment isolation
        effective_task_id = task_id or "default"

        # For local environment in batch mode, create a unique subdirectory per task
        # This prevents parallel tasks from overwriting each other's files
        # In CLI mode (HERMES_QUIET), use the cwd directly without subdirectories
        if env_type == "local" and not os.getenv("HERMES_QUIET"):
            import uuid
            with _env_lock:
                if effective_task_id not in _task_workdirs:
                    task_workdir = Path(cwd) / f"hermes-{effective_task_id}-{uuid.uuid4().hex[:8]}"
                    task_workdir.mkdir(parents=True, exist_ok=True)
                    _task_workdirs[effective_task_id] = str(task_workdir)
                cwd = _task_workdirs[effective_task_id]

        # Start cleanup thread
        _start_cleanup_thread()

        # Get or create environment
        with _env_lock:
            if effective_task_id not in _active_environments:
                # Check disk usage before creating new environment
                _check_disk_usage_warning()
                
                try:
                    # Build SSH config if using SSH environment
                    ssh_config = None
                    if env_type == "ssh":
                        ssh_config = {
                            "host": config.get("ssh_host", ""),
                            "user": config.get("ssh_user", ""),
                            "port": config.get("ssh_port", 22),
                            "key": config.get("ssh_key", ""),
                        }
                    
                    _active_environments[effective_task_id] = _create_environment(
                        env_type=env_type,
                        image=image,
                        cwd=cwd,
                        timeout=effective_timeout,
                        ssh_config=ssh_config
                    )
                except ImportError as e:
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": f"Terminal tool disabled: mini-swe-agent not available ({e})",
                        "status": "disabled"
                    }, ensure_ascii=False)

            # Update last activity time
            _last_activity[effective_task_id] = time.time()
            env = _active_environments[effective_task_id]

        # Prepare command for execution
        if background:
            # Run in background with nohup and redirect output
            exec_command = f"nohup {command} > /tmp/bg_output.log 2>&1 &"
            try:
                result = env.execute(exec_command, timeout=10)
                return json.dumps({
                    "output": "Background task started successfully",
                    "exit_code": 0,
                    "error": None
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": f"Failed to start background task: {str(e)}"
                }, ensure_ascii=False)
        else:
            # Run foreground command with retry logic
            max_retries = 3
            retry_count = 0
            result = None
            
            while retry_count <= max_retries:
                try:
                    result = env.execute(command, timeout=effective_timeout)
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        return json.dumps({
                            "output": "",
                            "exit_code": 124,
                            "error": f"Command timed out after {effective_timeout} seconds"
                        }, ensure_ascii=False)
                    
                    # Retry on transient errors
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        print(f"⚠️  Terminal: execution error, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": f"Command execution failed: {str(e)}"
                    }, ensure_ascii=False)
                
                # Got a result
                break
            
            # Extract output
            output = result.get("output", "")
            returncode = result.get("returncode", 0)
            
            # Truncate output if too long
            MAX_OUTPUT_CHARS = 50000
            if len(output) > MAX_OUTPUT_CHARS:
                truncated_notice = f"\n\n... [OUTPUT TRUNCATED - showing last {MAX_OUTPUT_CHARS} chars of {len(output)} total] ..."
                output = truncated_notice + output[-MAX_OUTPUT_CHARS:]

            return json.dumps({
                "output": output.strip() if output else "",
                "exit_code": returncode,
                "error": None
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Failed to execute command: {str(e)}",
            "status": "error"
        }, ensure_ascii=False)


def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met."""
    config = _get_env_config()
    env_type = config["env_type"]
    
    try:
        if env_type == "local":
            from minisweagent.environments.local import LocalEnvironment
            return True
        elif env_type == "docker":
            from minisweagent.environments.docker import DockerEnvironment
            # Check if docker is available
            import subprocess
            result = subprocess.run(["docker", "version"], capture_output=True, timeout=5)
            return result.returncode == 0
        elif env_type == "singularity":
            from minisweagent.environments.singularity import SingularityEnvironment
            # Check if singularity/apptainer is available
            import subprocess
            import shutil
            executable = shutil.which("apptainer") or shutil.which("singularity")
            if executable:
                result = subprocess.run([executable, "--version"], capture_output=True, timeout=5)
                return result.returncode == 0
            return False
        elif env_type == "modal":
            from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
            # Check for modal token
            return os.getenv("MODAL_TOKEN_ID") is not None or Path.home().joinpath(".modal.toml").exists()
        else:
            return False
    except Exception as e:
        print(f"Terminal requirements check failed: {e}")
        return False


if __name__ == "__main__":
    """Simple test when run directly."""
    print("Terminal Tool Module (mini-swe-agent backend)")
    print("=" * 50)
    
    config = _get_env_config()
    print(f"\nCurrent Configuration:")
    print(f"  Environment type: {config['env_type']}")
    print(f"  Docker image: {config['docker_image']}")
    print(f"  Modal image: {config['modal_image']}")
    print(f"  Working directory: {config['cwd']}")
    print(f"  Default timeout: {config['timeout']}s")
    print(f"  Lifetime: {config['lifetime_seconds']}s")

    if not check_terminal_requirements():
        print("\n❌ Requirements not met. Please check the messages above.")
        exit(1)

    print("\n✅ All requirements met!")
    print("\nAvailable Tool:")
    print("  - terminal_tool: Execute commands using mini-swe-agent environments")

    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = terminal_tool(command='ls -la')")
    print("  ")
    print("  # Run a background task")
    print("  result = terminal_tool(command='python server.py', background=True)")

    print("\nEnvironment Variables:")
    print(f"  TERMINAL_ENV: {os.getenv('TERMINAL_ENV', 'local')} (local/docker/modal)")
    print(f"  TERMINAL_DOCKER_IMAGE: {os.getenv('TERMINAL_DOCKER_IMAGE', 'python:3.11-slim')}")
    print(f"  TERMINAL_MODAL_IMAGE: {os.getenv('TERMINAL_MODAL_IMAGE', 'python:3.11-slim')}")
    print(f"  TERMINAL_CWD: {os.getenv('TERMINAL_CWD', '/tmp')}")
    print(f"  TERMINAL_TIMEOUT: {os.getenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {os.getenv('TERMINAL_LIFETIME_SECONDS', '300')}")
