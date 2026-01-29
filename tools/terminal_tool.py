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
        print(f"[Terminal] Using /scratch for sandboxes: {user_scratch}")
        return user_scratch
    
    # Fall back to /tmp
    print("[Terminal] Warning: /scratch not available, using /tmp (limited space)")
    return Path(tempfile.gettempdir())


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


class _SingularityEnvironment:
    """
    Custom Singularity/Apptainer environment with better space management.
    
    - Builds sandbox in /scratch (if available) or configurable location
    - Binds a large working directory into the container
    - Keeps container isolated from host filesystem
    """
    
    def __init__(self, image: str, cwd: str = "/workspace", timeout: int = 60):
        self.image = image
        self.cwd = cwd
        self.timeout = timeout
        
        # Use apptainer if available, otherwise singularity
        self.executable = "apptainer" if shutil.which("apptainer") else "singularity"
        
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
        """Build a writable sandbox from the container image."""
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
        
        # Execute the command
        cmd.extend(["bash", "-c", command])
        
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
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
        "env_type": os.getenv("TERMINAL_ENV", "local"),  # local, docker, singularity, or modal
        "docker_image": os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11"),
        "singularity_image": os.getenv("TERMINAL_SINGULARITY_IMAGE", "docker://python:3.11"),
        "modal_image": os.getenv("TERMINAL_MODAL_IMAGE", "python:3.11"),
        "cwd": os.getenv("TERMINAL_CWD", "/tmp"),
        "timeout": int(os.getenv("TERMINAL_TIMEOUT", "60")),
        "lifetime_seconds": int(os.getenv("TERMINAL_LIFETIME_SECONDS", "300")),
    }


def _create_environment(env_type: str, image: str, cwd: str, timeout: int):
    """
    Create an execution environment from mini-swe-agent.
    
    Args:
        env_type: One of "local", "docker", "singularity", "modal"
        image: Docker/Singularity/Modal image name (ignored for local)
        cwd: Working directory
        timeout: Default command timeout
        
    Returns:
        Environment instance with execute() method
    """
    if env_type == "local":
        from minisweagent.environments.local import LocalEnvironment
        return LocalEnvironment(cwd=cwd, timeout=timeout)
    
    elif env_type == "docker":
        from minisweagent.environments.docker import DockerEnvironment
        return DockerEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    elif env_type == "singularity":
        # Use custom Singularity environment with better space management
        return _SingularityEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    elif env_type == "modal":
        from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
        return SwerexModalEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Use 'local', 'docker', 'singularity', or 'modal'")


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
                    print(f"[Terminal Cleanup] Cleaned up inactive environment for task: {task_id}")

                if task_id in _last_activity:
                    del _last_activity[task_id]
                if task_id in _task_workdirs:
                    del _task_workdirs[task_id]

            except Exception as e:
                error_str = str(e)
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
                print(f"[Terminal Cleanup] Manually cleaned up environment for task: {task_id}")

            if task_id in _task_workdirs:
                del _task_workdirs[task_id]

            if task_id in _last_activity:
                del _last_activity[task_id]

        except Exception as e:
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

        # For local environment, create a unique subdirectory per task
        # This prevents parallel tasks from overwriting each other's files
        if env_type == "local":
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
                    _active_environments[effective_task_id] = _create_environment(
                        env_type=env_type,
                        image=image,
                        cwd=cwd,
                        timeout=effective_timeout
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
