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
import logging
import os
import signal
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global interrupt event: set by the agent when a user interrupt arrives.
# The terminal tool polls this during command execution so it can kill
# long-running subprocesses immediately instead of blocking until timeout.
# ---------------------------------------------------------------------------
from tools.interrupt import set_interrupt as set_interrupt_event, is_interrupted, _interrupt_event


# Add mini-swe-agent to path if not installed
mini_swe_path = Path(__file__).parent.parent / "mini-swe-agent" / "src"
if mini_swe_path.exists():
    sys.path.insert(0, str(mini_swe_path))


# =============================================================================
# Custom Singularity Environment with more space
# =============================================================================

# Singularity helpers (scratch dir, SIF cache) now live in tools/environments/singularity.py
from tools.environments.singularity import _get_scratch_dir


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
                    except OSError:
                        pass
        
        total_gb = total_bytes / (1024 ** 3)
        
        if total_gb > DISK_USAGE_WARNING_THRESHOLD_GB:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
            return True
        
        return False
    except Exception as e:
        return False


# Session-cached sudo password (persists until CLI exits)
_cached_sudo_password: str = ""

# Optional UI callbacks for interactive prompts. When set, these are called
# instead of the default /dev/tty or input() readers. The CLI registers these
# so prompts route through prompt_toolkit's event loop.
#   _sudo_password_callback() -> str  (return password or "" to skip)
#   _approval_callback(command, description) -> str  ("once"/"session"/"always"/"deny")
_sudo_password_callback = None
_approval_callback = None


def set_sudo_password_callback(cb):
    """Register a callback for sudo password prompts (used by CLI)."""
    global _sudo_password_callback
    _sudo_password_callback = cb


def set_approval_callback(cb):
    """Register a callback for dangerous command approval prompts (used by CLI)."""
    global _approval_callback
    _approval_callback = cb

# =============================================================================
# Dangerous Command Approval System
# =============================================================================

# Dangerous command detection + approval now consolidated in tools/approval.py
from tools.approval import (
    detect_dangerous_command as _detect_dangerous_command,
    check_dangerous_command as _check_dangerous_command_impl,
    load_permanent_allowlist as _load_permanent_allowlist,
    DANGEROUS_PATTERNS,
)


def _check_dangerous_command(command: str, env_type: str) -> dict:
    """Delegate to the consolidated approval module, passing the CLI callback."""
    return _check_dangerous_command_impl(command, env_type,
                                         approval_callback=_approval_callback)


def _handle_sudo_failure(output: str, env_type: str) -> str:
    """
    Check for sudo failure and add helpful message for messaging contexts.
    
    Returns enhanced output if sudo failed in messaging context, else original.
    """
    is_gateway = os.getenv("HERMES_GATEWAY_SESSION")
    
    if not is_gateway:
        return output
    
    # Check for sudo failure indicators
    sudo_failures = [
        "sudo: a password is required",
        "sudo: no tty present",
        "sudo: a terminal is required",
    ]
    
    for failure in sudo_failures:
        if failure in output:
            return output + "\n\n💡 Tip: To enable sudo over messaging, add SUDO_PASSWORD to ~/.hermes/.env on the agent machine."
    
    return output


def _prompt_for_sudo_password(timeout_seconds: int = 45) -> str:
    """
    Prompt user for sudo password with timeout.
    
    Returns the password if entered, or empty string if:
    - User presses Enter without input (skip)
    - Timeout expires (45s default)
    - Any error occurs
    
    Only works in interactive mode (HERMES_INTERACTIVE=1).
    If a _sudo_password_callback is registered (by the CLI), delegates to it
    so the prompt integrates with prompt_toolkit's UI.  Otherwise reads
    directly from /dev/tty with echo disabled.
    """
    import sys
    import time as time_module
    
    # Use the registered callback when available (prompt_toolkit-compatible)
    if _sudo_password_callback is not None:
        try:
            return _sudo_password_callback() or ""
        except Exception:
            return ""

    result = {"password": None, "done": False}
    
    def read_password_thread():
        """Read password from /dev/tty with echo disabled."""
        tty_fd = None
        old_attrs = None
        try:
            import termios
            tty_fd = os.open("/dev/tty", os.O_RDONLY)
            old_attrs = termios.tcgetattr(tty_fd)
            new_attrs = termios.tcgetattr(tty_fd)
            new_attrs[3] = new_attrs[3] & ~termios.ECHO
            termios.tcsetattr(tty_fd, termios.TCSAFLUSH, new_attrs)
            chars = []
            while True:
                b = os.read(tty_fd, 1)
                if not b or b in (b"\n", b"\r"):
                    break
                chars.append(b)
            result["password"] = b"".join(chars).decode("utf-8", errors="replace")
        except (EOFError, KeyboardInterrupt, OSError):
            result["password"] = ""
        except Exception:
            result["password"] = ""
        finally:
            if tty_fd is not None and old_attrs is not None:
                try:
                    import termios as _termios
                    _termios.tcsetattr(tty_fd, _termios.TCSAFLUSH, old_attrs)
                except Exception:
                    pass
            if tty_fd is not None:
                try:
                    os.close(tty_fd)
                except Exception:
                    pass
            result["done"] = True
    
    try:
        os.environ["HERMES_SPINNER_PAUSE"] = "1"
        time_module.sleep(0.2)
        
        print()
        print("┌" + "─" * 58 + "┐")
        print("│  🔐 SUDO PASSWORD REQUIRED" + " " * 30 + "│")
        print("├" + "─" * 58 + "┤")
        print("│  Enter password below (input is hidden), or:            │")
        print("│    • Press Enter to skip (command fails gracefully)     │")
        print(f"│    • Wait {timeout_seconds}s to auto-skip" + " " * 27 + "│")
        print("└" + "─" * 58 + "┘")
        print()
        print("  Password (hidden): ", end="", flush=True)
        
        password_thread = threading.Thread(target=read_password_thread, daemon=True)
        password_thread.start()
        password_thread.join(timeout=timeout_seconds)
        
        if result["done"]:
            password = result["password"] or ""
            print()  # newline after hidden input
            if password:
                print("  ✓ Password received (cached for this session)")
            else:
                print("  ⏭ Skipped - continuing without sudo")
            print()
            sys.stdout.flush()
            return password
        else:
            print("\n  ⏱ Timeout - continuing without sudo")
            print("    (Press Enter to dismiss)")
            print()
            sys.stdout.flush()
            return ""
            
    except (EOFError, KeyboardInterrupt):
        print()
        print("  ⏭ Cancelled - continuing without sudo")
        print()
        sys.stdout.flush()
        return ""
    except Exception as e:
        print(f"\n  [sudo prompt error: {e}] - continuing without sudo\n")
        sys.stdout.flush()
        return ""
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]


def _transform_sudo_command(command: str) -> str:
    """
    Transform sudo commands to use -S flag if SUDO_PASSWORD is available.
    
    This is a shared helper used by all execution environments to provide
    consistent sudo handling across local, SSH, and container environments.
    
    If SUDO_PASSWORD is set (via env, config, or interactive prompt):
      'sudo apt install curl' -> password piped via sudo -S
      
    If SUDO_PASSWORD is not set and in interactive mode (HERMES_INTERACTIVE=1):
      Prompts user for password with 45s timeout, caches for session.
      
    If SUDO_PASSWORD is not set and NOT interactive:
      Command runs as-is (fails gracefully with "sudo: a password is required").
    """
    global _cached_sudo_password
    import re
    
    # Check if command even contains sudo
    if not re.search(r'\bsudo\b', command):
        return command  # No sudo in command, return as-is
    
    # Try to get password from: env var -> session cache -> interactive prompt
    sudo_password = os.getenv("SUDO_PASSWORD", "") or _cached_sudo_password
    
    if not sudo_password:
        # No password configured - check if we're in interactive mode
        if os.getenv("HERMES_INTERACTIVE"):
            # Prompt user for password
            sudo_password = _prompt_for_sudo_password(timeout_seconds=45)
            if sudo_password:
                _cached_sudo_password = sudo_password  # Cache for session
    
    if not sudo_password:
        return command  # No password, let it fail gracefully
    
    def replace_sudo(match):
        # Replace 'sudo' with password-piped version
        # The -S flag makes sudo read password from stdin
        # The -p '' suppresses the password prompt
        # Use shlex.quote() to prevent shell injection via password content
        import shlex
        return f"echo {shlex.quote(sudo_password)} | sudo -S -p ''"
    
    # Match 'sudo' at word boundaries (not 'visudo' or 'sudoers')
    # This handles: sudo, sudo -flag, etc.
    return re.sub(r'\bsudo\b', replace_sudo, command)


# Environment classes now live in tools/environments/
from tools.environments.local import LocalEnvironment as _LocalEnvironment
from tools.environments.singularity import SingularityEnvironment as _SingularityEnvironment
from tools.environments.ssh import SSHEnvironment as _SSHEnvironment
from tools.environments.docker import DockerEnvironment as _DockerEnvironment
from tools.environments.modal import ModalEnvironment as _ModalEnvironment


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on a Linux environment. Filesystem persists between calls.

Do NOT use cat/head/tail to read files — use read_file instead.
Do NOT use grep/rg/find to search — use search_files instead.
Do NOT use ls to list directories — use search_files(target='files') instead.
Do NOT use sed/awk to edit files — use patch instead.
Do NOT use echo/cat heredoc to create files — use write_file instead.
Reserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell.

Background processes: Set background=true to get a session_id, then use the 'process' tool to poll/wait/kill/write.
Working directory: Use 'workdir' for per-command cwd.
PTY mode: Set pty=true for interactive CLI tools (Codex, Claude Code, Python REPL).

Do NOT use vim/nano/interactive tools without pty=true — they hang without a pseudo-terminal. Pipe git output to cat if it might page.
"""

# Global state for environment lifecycle management
_active_environments: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_creation_locks: Dict[str, threading.Lock] = {}  # Per-task locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Per-task environment overrides registry.
# Allows environments (e.g., TerminalBench2Env) to specify a custom Docker/Modal
# image for a specific task_id BEFORE the agent loop starts. When the terminal or
# file tools create a new sandbox for that task_id, they check this registry first
# and fall back to the TERMINAL_MODAL_IMAGE (etc.) env var if no override is set.
#
# This is never exposed to the model -- only infrastructure code calls it.
# Thread-safe because each task_id is unique per rollout.
_task_env_overrides: Dict[str, Dict[str, Any]] = {}


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """
    Register environment overrides for a specific task/rollout.

    Called by Atropos environments before the agent loop to configure
    per-task sandbox settings (e.g., a custom Dockerfile for the Modal image).

    Supported override keys:
        - modal_image: str -- Path to Dockerfile or Docker Hub image name
        - docker_image: str -- Docker image name
        - cwd: str -- Working directory inside the sandbox

    Args:
        task_id: The rollout's unique task identifier
        overrides: Dict of config keys to override
    """
    _task_env_overrides[task_id] = overrides


def clear_task_env_overrides(task_id: str):
    """
    Clear environment overrides for a task after rollout completes.

    Called during cleanup to avoid stale entries accumulating.
    """
    _task_env_overrides.pop(task_id, None)

# Configuration from environment variables
def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    # Default image with Python and Node.js for maximum compatibility
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    env_type = os.getenv("TERMINAL_ENV", "local")
    
    # Default cwd: local uses the host's current directory, everything
    # else starts in the user's home (~ resolves to whatever account
    # is running inside the container/remote).
    if env_type == "local":
        default_cwd = os.getcwd()
    else:
        default_cwd = "~"
    
    # Read TERMINAL_CWD but sanity-check it for container backends.
    # If the CWD looks like a host-local path that can't exist inside a
    # container/sandbox, fall back to the backend's own default. This
    # catches the case where cli.py (or .env) leaked the host's CWD.
    # SSH is excluded since /home/ paths are valid on remote machines.
    cwd = os.getenv("TERMINAL_CWD", default_cwd)
    if env_type in ("modal", "docker", "singularity") and cwd:
        host_prefixes = ("/Users/", "C:\\", "C:/")
        if any(cwd.startswith(p) for p in host_prefixes) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host path won't exist in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    return {
        "env_type": env_type,
        "docker_image": os.getenv("TERMINAL_DOCKER_IMAGE", default_image),
        "singularity_image": os.getenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": os.getenv("TERMINAL_MODAL_IMAGE", default_image),
        "cwd": cwd,
        "timeout": int(os.getenv("TERMINAL_TIMEOUT", "60")),
        "lifetime_seconds": int(os.getenv("TERMINAL_LIFETIME_SECONDS", "300")),
        # SSH-specific config
        "ssh_host": os.getenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": os.getenv("TERMINAL_SSH_USER", ""),
        "ssh_port": int(os.getenv("TERMINAL_SSH_PORT", "22")),
        "ssh_key": os.getenv("TERMINAL_SSH_KEY", ""),
        # Container resource config (applies to docker, singularity, modal -- ignored for local/ssh)
        "container_cpu": float(os.getenv("TERMINAL_CONTAINER_CPU", "1")),
        "container_memory": int(os.getenv("TERMINAL_CONTAINER_MEMORY", "5120")),     # MB (default 5GB)
        "container_disk": int(os.getenv("TERMINAL_CONTAINER_DISK", "51200")),        # MB (default 50GB)
        "container_persistent": os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in ("true", "1", "yes"),
    }


def _create_environment(env_type: str, image: str, cwd: str, timeout: int,
                        ssh_config: dict = None, container_config: dict = None,
                        task_id: str = "default"):
    """
    Create an execution environment from mini-swe-agent.
    
    Args:
        env_type: One of "local", "docker", "singularity", "modal", "ssh"
        image: Docker/Singularity/Modal image name (ignored for local/ssh)
        cwd: Working directory
        timeout: Default command timeout
        ssh_config: SSH connection config (for env_type="ssh")
        container_config: Resource config for container backends (cpu, memory, disk, persistent)
        task_id: Task identifier for environment reuse and snapshot keying
        
    Returns:
        Environment instance with execute() method
    """
    cc = container_config or {}
    cpu = cc.get("container_cpu", 1)
    memory = cc.get("container_memory", 5120)
    disk = cc.get("container_disk", 51200)
    persistent = cc.get("container_persistent", True)

    if env_type == "local":
        return _LocalEnvironment(cwd=cwd, timeout=timeout)
    
    elif env_type == "docker":
        return _DockerEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "singularity":
        return _SingularityEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "modal":
        sandbox_kwargs = {}
        if cpu > 0:
            sandbox_kwargs["cpu"] = cpu
        if memory > 0:
            sandbox_kwargs["memory"] = memory
        if disk > 0:
            sandbox_kwargs["ephemeral_disk"] = disk
        
        return _ModalEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "ssh":
        if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
            raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
        return _SSHEnvironment(
            host=ssh_config["host"],
            user=ssh_config["user"],
            port=ssh_config.get("port", 22),
            key_path=ssh_config.get("key", ""),
            cwd=cwd,
            timeout=timeout,
        )
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Use 'local', 'docker', 'singularity', 'modal', or 'ssh'")


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    global _active_environments, _last_activity

    current_time = time.time()

    # Check the process registry -- skip cleanup for sandboxes with active
    # background processes (their _last_activity gets refreshed to keep them alive).
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            if process_registry.has_active_processes(task_id):
                _last_activity[task_id] = current_time  # Keep sandbox alive
    except ImportError:
        pass

    # Phase 1: collect stale entries and remove them from tracking dicts while
    # holding the lock.  Do NOT call env.cleanup() inside the lock -- Modal and
    # Docker teardown can block for 10-15s, which would stall every concurrent
    # terminal/file tool call waiting on _env_lock.
    envs_to_stop = []  # list of (task_id, env) pairs

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > lifetime_seconds:
                env = _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
                if env is not None:
                    envs_to_stop.append((task_id, env))

        # Also purge per-task creation locks for cleaned-up tasks
        with _creation_locks_lock:
            for task_id, _ in envs_to_stop:
                _creation_locks.pop(task_id, None)

    # Phase 2: stop the actual sandboxes OUTSIDE the lock so other tool calls
    # are not blocked while Modal/Docker sandboxes shut down.
    for task_id, env in envs_to_stop:
        # Invalidate stale file_ops cache entry (Bug fix: prevents
        # ShellFileOperations from referencing a dead sandbox)
        try:
            from tools.file_tools import clear_file_ops_cache
            clear_file_ops_cache(task_id)
        except ImportError:
            pass

        try:
            if hasattr(env, 'cleanup'):
                env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Cleaned up inactive environment for task: %s", task_id)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", task_id)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    global _cleanup_running

    while _cleanup_running:
        try:
            config = _get_env_config()
            _cleanup_inactive_envs(config["lifetime_seconds"])
        except Exception as e:
            logger.warning("Error in cleanup thread: %s", e, exc_info=True)

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
        try:
            _cleanup_thread.join(timeout=5)
        except (SystemExit, KeyboardInterrupt):
            pass


def get_active_environments_info() -> Dict[str, Any]:
    """Get information about currently active environments."""
    info = {
        "count": len(_active_environments),
        "task_ids": list(_active_environments.keys()),
        "workdirs": {},
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
                except OSError:
                    pass
    
    info["total_disk_usage_mb"] = round(total_size / (1024 * 1024), 2)
    return info


def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    global _active_environments, _last_activity
    
    task_ids = list(_active_environments.keys())
    cleaned = 0
    
    for task_id in task_ids:
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)
    
    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)
        except OSError:
            pass
    
    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(task_id: str):
    """Manually clean up a specific environment by task_id."""
    global _active_environments, _last_activity

    # Remove from tracking dicts while holding the lock, but defer the
    # actual (potentially slow) env.cleanup() call to outside the lock
    # so other tool calls aren't blocked.
    env = None
    with _env_lock:
        env = _active_environments.pop(task_id, None)
        _last_activity.pop(task_id, None)

    # Clean up per-task creation lock
    with _creation_locks_lock:
        _creation_locks.pop(task_id, None)

    # Invalidate stale file_ops cache entry
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(task_id)
    except ImportError:
        pass

    if env is None:
        return

    try:
        if hasattr(env, 'cleanup'):
            env.cleanup()
        elif hasattr(env, 'stop'):
            env.stop()
        elif hasattr(env, 'terminate'):
            env.terminate()

        logger.info("Manually cleaned up environment for task: %s", task_id)

    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _atexit_cleanup():
    """Stop cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        count = len(_active_environments)
        logger.info("Shutting down %d remaining sandbox(es)...", count)
        cleanup_all_environments()

atexit.register(_atexit_cleanup)


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    check_interval: Optional[int] = None,
    pty: bool = False,
) -> str:
    """
    Execute a command using mini-swe-agent's execution environments.

    Args:
        command: The command to execute
        background: Whether to run in background (default: False)
        timeout: Command timeout in seconds (default: from config)
        task_id: Unique identifier for environment isolation (optional)
        force: If True, skip dangerous command check (use after user confirms)
        workdir: Working directory for this command (optional, uses session cwd if not set)
        check_interval: Seconds between auto-checks for background processes (gateway only, min 30)
        pty: If True, use pseudo-terminal for interactive CLI tools (local backend only)

    Returns:
        str: JSON string with output, exit_code, and error fields

    Examples:
        # Execute a simple command
        >>> result = terminal_tool(command="ls -la /tmp")

        # Run a background task
        >>> result = terminal_tool(command="python server.py", background=True)

        # With custom timeout
        >>> result = terminal_tool(command="long_task.sh", timeout=300)
        
        # Force run after user confirmation
        # Note: force parameter is internal only, not exposed to model API
    """
    global _active_environments, _last_activity

    try:
        # Get configuration
        config = _get_env_config()
        env_type = config["env_type"]

        # Use task_id for environment isolation
        effective_task_id = task_id or "default"

        # Check per-task overrides (set by environments like TerminalBench2Env)
        # before falling back to global env var config
        overrides = _task_env_overrides.get(effective_task_id, {})
        
        # Select image based on env type, with per-task override support
        if env_type == "docker":
            image = overrides.get("docker_image") or config["docker_image"]
        elif env_type == "singularity":
            image = overrides.get("singularity_image") or config["singularity_image"]
        elif env_type == "modal":
            image = overrides.get("modal_image") or config["modal_image"]
        else:
            image = ""
        
        cwd = overrides.get("cwd") or config["cwd"]
        default_timeout = config["timeout"]
        effective_timeout = timeout or default_timeout

        # Start cleanup thread
        _start_cleanup_thread()

        # Get or create environment.
        # Use a per-task creation lock so concurrent tool calls for the same
        # task_id wait for the first one to finish creating the sandbox,
        # instead of each creating their own (wasting Modal resources).
        with _env_lock:
            if effective_task_id in _active_environments:
                _last_activity[effective_task_id] = time.time()
                env = _active_environments[effective_task_id]
                needs_creation = False
            else:
                needs_creation = True

        if needs_creation:
            # Per-task lock: only one thread creates the sandbox, others wait
            with _creation_locks_lock:
                if effective_task_id not in _creation_locks:
                    _creation_locks[effective_task_id] = threading.Lock()
                task_lock = _creation_locks[effective_task_id]

            with task_lock:
                # Double-check after acquiring the per-task lock
                with _env_lock:
                    if effective_task_id in _active_environments:
                        _last_activity[effective_task_id] = time.time()
                        env = _active_environments[effective_task_id]
                        needs_creation = False

                if needs_creation:
                    if env_type == "singularity":
                        _check_disk_usage_warning()
                    logger.info("Creating new %s environment for task %s...", env_type, effective_task_id[:8])
                    try:
                        ssh_config = None
                        if env_type == "ssh":
                            ssh_config = {
                                "host": config.get("ssh_host", ""),
                                "user": config.get("ssh_user", ""),
                                "port": config.get("ssh_port", 22),
                                "key": config.get("ssh_key", ""),
                            }

                        container_config = None
                        if env_type in ("docker", "singularity", "modal"):
                            container_config = {
                                "container_cpu": config.get("container_cpu", 1),
                                "container_memory": config.get("container_memory", 5120),
                                "container_disk": config.get("container_disk", 51200),
                                "container_persistent": config.get("container_persistent", True),
                            }

                        new_env = _create_environment(
                            env_type=env_type,
                            image=image,
                            cwd=cwd,
                            timeout=effective_timeout,
                            ssh_config=ssh_config,
                            container_config=container_config,
                            task_id=effective_task_id,
                        )
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": f"Terminal tool disabled: mini-swe-agent not available ({e})",
                            "status": "disabled"
                        }, ensure_ascii=False)

                    with _env_lock:
                        _active_environments[effective_task_id] = new_env
                        _last_activity[effective_task_id] = time.time()
                        env = new_env
                    logger.info("%s environment ready for task %s", env_type, effective_task_id[:8])

        # Check for dangerous commands (only for local/ssh in interactive modes)
        # Skip check if force=True (user has confirmed they want to run it)
        if not force:
            approval = _check_dangerous_command(command, env_type)
            if not approval["approved"]:
                # Check if this is an approval_required (gateway ask mode)
                if approval.get("status") == "approval_required":
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": approval.get("message", "Waiting for user approval"),
                        "status": "approval_required",
                        "command": approval.get("command", command),
                        "description": approval.get("description", "dangerous command"),
                        "pattern_key": approval.get("pattern_key", ""),
                    }, ensure_ascii=False)
                # Command was blocked - include the pattern category so the caller knows why
                desc = approval.get("description", "potentially dangerous operation")
                fallback_msg = (
                    f"Command denied: matches '{desc}' pattern. "
                    "Use the approval prompt to allow it, or rephrase the command."
                )
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": approval.get("message", fallback_msg),
                    "status": "blocked"
                }, ensure_ascii=False)

        # Prepare command for execution
        if background:
            # Spawn a tracked background process via the process registry.
            # For local backends: uses subprocess.Popen with output buffering.
            # For non-local backends: runs inside the sandbox via env.execute().
            from tools.process_registry import process_registry

            session_key = os.getenv("HERMES_SESSION_KEY", "")
            effective_cwd = workdir or cwd
            try:
                if env_type == "local":
                    proc_session = process_registry.spawn_local(
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                        env_vars=env.env if hasattr(env, 'env') else None,
                        use_pty=pty,
                    )
                else:
                    proc_session = process_registry.spawn_via_env(
                        env=env,
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                    )

                result_data = {
                    "output": "Background process started",
                    "session_id": proc_session.id,
                    "pid": proc_session.pid,
                    "exit_code": 0,
                    "error": None,
                }

                # Transparent timeout clamping note
                max_timeout = effective_timeout
                if timeout and timeout > max_timeout:
                    result_data["timeout_note"] = (
                        f"Requested timeout {timeout}s was clamped to "
                        f"configured limit of {max_timeout}s"
                    )

                # Register check_interval watcher (gateway picks this up after agent run)
                if check_interval and background:
                    effective_interval = max(30, check_interval)
                    if check_interval < 30:
                        result_data["check_interval_note"] = (
                            f"Requested {check_interval}s raised to minimum 30s"
                        )
                    process_registry.pending_watchers.append({
                        "session_id": proc_session.id,
                        "check_interval": effective_interval,
                        "session_key": session_key,
                        "platform": os.getenv("HERMES_SESSION_PLATFORM", ""),
                        "chat_id": os.getenv("HERMES_SESSION_CHAT_ID", ""),
                    })

                return json.dumps(result_data, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": f"Failed to start background process: {str(e)}"
                }, ensure_ascii=False)
        else:
            # Run foreground command with retry logic
            max_retries = 3
            retry_count = 0
            result = None
            
            while retry_count <= max_retries:
                try:
                    execute_kwargs = {"timeout": effective_timeout}
                    if workdir:
                        execute_kwargs["cwd"] = workdir
                    result = env.execute(command, **execute_kwargs)
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
                        logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                       wait_time, retry_count, max_retries, command[:200], type(e).__name__, e, effective_task_id, env_type)
                        time.sleep(wait_time)
                        continue
                    
                    logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                 max_retries, command[:200], type(e).__name__, e, effective_task_id, env_type)
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": f"Command execution failed: {type(e).__name__}: {str(e)}"
                    }, ensure_ascii=False)
                
                # Got a result
                break
            
            # Extract output
            output = result.get("output", "")
            returncode = result.get("returncode", 0)
            
            # Add helpful message for sudo failures in messaging context
            output = _handle_sudo_failure(output, env_type)
            
            # Truncate output if too long, keeping both head and tail
            MAX_OUTPUT_CHARS = 50000
            if len(output) > MAX_OUTPUT_CHARS:
                head_chars = int(MAX_OUTPUT_CHARS * 0.4)  # 40% head (error messages often appear early)
                tail_chars = MAX_OUTPUT_CHARS - head_chars  # 60% tail (most recent/relevant output)
                omitted = len(output) - head_chars - tail_chars
                truncated_notice = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted "
                    f"out of {len(output)} total] ...\n\n"
                )
                output = output[:head_chars] + truncated_notice + output[-tail_chars:]

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
        elif env_type == "ssh":
            from tools.environments.ssh import SSHEnvironment
            # Check that host and user are configured
            return bool(config.get("ssh_host")) and bool(config.get("ssh_user"))
        elif env_type == "modal":
            from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
            # Check for modal token
            return os.getenv("MODAL_TOKEN_ID") is not None or Path.home().joinpath(".modal.toml").exists()
        else:
            return False
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e)
        return False


if __name__ == "__main__":
    # Simple test when run directly
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
    default_img = "nikolaik/python-nodejs:python3.11-nodejs20"
    print(f"  TERMINAL_ENV: {os.getenv('TERMINAL_ENV', 'local')} (local/docker/singularity/modal/ssh)")
    print(f"  TERMINAL_DOCKER_IMAGE: {os.getenv('TERMINAL_DOCKER_IMAGE', default_img)}")
    print(f"  TERMINAL_SINGULARITY_IMAGE: {os.getenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_img}')}")
    print(f"  TERMINAL_MODAL_IMAGE: {os.getenv('TERMINAL_MODAL_IMAGE', default_img)}")
    print(f"  TERMINAL_CWD: {os.getenv('TERMINAL_CWD', os.getcwd())}")
    print(f"  TERMINAL_SANDBOX_DIR: {os.getenv('TERMINAL_SANDBOX_DIR', '~/.hermes/sandboxes')}")
    print(f"  TERMINAL_TIMEOUT: {os.getenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {os.getenv('TERMINAL_LIFETIME_SECONDS', '300')}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute on the VM"
            },
            "background": {
                "type": "boolean",
                "description": "Whether to run the command in the background (default: false)",
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": "Command timeout in seconds (optional)",
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path). Defaults to the session working directory."
            },
            "check_interval": {
                "type": "integer",
                "description": "Seconds between automatic status checks for background processes (gateway/messaging only, minimum 30). When set, I'll proactively report progress.",
                "minimum": 30
            },
            "pty": {
                "type": "boolean",
                "description": "Run in pseudo-terminal (PTY) mode for interactive CLI tools like Codex, Claude Code, or Python REPL. Only works with local and SSH backends. Default: false.",
                "default": False
            }
        },
        "required": ["command"]
    }
}


def _handle_terminal(args, **kw):
    return terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        workdir=args.get("workdir"),
        check_interval=args.get("check_interval"),
        pty=args.get("pty", False),
    )


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
)
