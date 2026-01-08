#!/usr/bin/env python3
"""
Simple Terminal Tool Module

A simplified terminal tool that executes commands on MorphCloud VMs without tmux.
No session persistence, no interactive app support - just simple command execution.

Features:
- Direct SSH command execution
- Background task support
- VM lifecycle management with TTL
- Automatic cleanup after inactivity

Usage:
    from simple_terminal_tool import simple_terminal_tool

    # Execute a simple command
    result = simple_terminal_tool("ls -la")

    # Execute in background
    result = simple_terminal_tool("python server.py", background=True)
"""

import json
import os
import time
import threading
import atexit
from typing import Optional, Dict, Any

# Tool description for LLM
SIMPLE_TERMINAL_TOOL_DESCRIPTION = """Execute commands on a secure Linux VM environment.

**Environment:**
- Minimal Debian-based OS with internet access
- Automatic VM lifecycle management (creates on-demand, reuses, cleans up)
- Filesystem is persisted between tool calls but environment variables, venvs, etc are reset.

**Command Execution:**
- Simple commands: Just provide the 'command' parameter
- Background processes: Set 'background': True for servers/long-running tasks
- Command timeout: Optional 'timeout' parameter in seconds

**Examples:**
- Run command: `{"command": "ls -la"}`
- Background task: `{"command": "source path/to/my/venv/bin/activate && python server.py", "background": True}`
- With timeout: `{"command": "long_task.sh", "timeout": 300}`

**Best Practices:**
- Run servers/long processes in background
- Monitor disk usage for large tasks
- Install whatever tools you need with sudo apt-get
- Do not be afraid to run pip with --break-system-packages

**Things to avoid**
- Do NOT use interactive tools such as tmux, vim, nano, python repl - you will get stuck. Even git sometimes becomes interactive if the output is large. If you're not sure pipe to cat.
"""

# Global state for VM lifecycle management
_active_instances: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_instance_lock = threading.Lock()
_cleanup_thread = None
_cleanup_running = False


def _cleanup_inactive_vms(vm_lifetime_seconds: int = 300):
    """Clean up VMs that have been inactive for longer than vm_lifetime_seconds."""
    global _active_instances, _last_activity

    current_time = time.time()
    tasks_to_cleanup = []

    with _instance_lock:
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > vm_lifetime_seconds:
                tasks_to_cleanup.append(task_id)

        for task_id in tasks_to_cleanup:
            try:
                if task_id in _active_instances:
                    instance = _active_instances[task_id]
                    if hasattr(instance, 'terminate'):
                        instance.terminate()
                    elif hasattr(instance, 'stop'):
                        instance.stop()
                    elif hasattr(instance, 'delete'):
                        instance.delete()

                    del _active_instances[task_id]
                    print(f"[VM Cleanup] Terminated inactive VM for task: {task_id}")

                if task_id in _last_activity:
                    del _last_activity[task_id]

            except Exception as e:
                # 404 errors are benign - VM already cleaned up by TTL
                error_str = str(e)
                if "404" in error_str or "InstanceNotFoundError" in error_str or "not found" in error_str.lower():
                    print(f"[VM Cleanup] VM for task {task_id} already cleaned up (likely TTL expiration)")
                else:
                    print(f"[VM Cleanup] Error cleaning up VM for task {task_id}: {e}")


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive VMs."""
    global _cleanup_running

    while _cleanup_running:
        try:
            vm_lifetime = int(os.getenv("HECATE_VM_LIFETIME_SECONDS", "300"))
            _cleanup_inactive_vms(vm_lifetime)
        except Exception as e:
            print(f"[VM Cleanup] Error in cleanup thread: {e}")

        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _instance_lock:
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


def cleanup_vm(task_id: str):
    """Manually clean up a specific VM by task_id."""
    global _active_instances, _last_activity

    with _instance_lock:
        try:
            if task_id in _active_instances:
                instance = _active_instances[task_id]
                if hasattr(instance, 'terminate'):
                    instance.terminate()
                elif hasattr(instance, 'stop'):
                    instance.stop()
                elif hasattr(instance, 'delete'):
                    instance.delete()

                del _active_instances[task_id]
                print(f"[VM Cleanup] Manually terminated VM for task: {task_id}")

            if task_id in _last_activity:
                del _last_activity[task_id]

        except Exception as e:
            # 404 errors are benign - VM already cleaned up by TTL
            error_str = str(e)
            if "404" in error_str or "InstanceNotFoundError" in error_str or "not found" in error_str.lower():
                print(f"[VM Cleanup] VM for task {task_id} already cleaned up (likely TTL expiration)")
            else:
                print(f"[VM Cleanup] Error manually cleaning up VM for task {task_id}: {e}")


atexit.register(_stop_cleanup_thread)


def _execute_ssh_command(instance, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute a command via SSH on the VM instance.

    Args:
        instance: MorphVM instance
        command: Command to execute
        timeout: Optional timeout in seconds

    Returns:
        dict with stdout, stderr, returncode
    """
    ssh_context_manager = None
    try:
        # Use the instance's SSH context manager
        ssh_context_manager = instance.ssh()
        ssh_context = ssh_context_manager.__enter__()

        # Execute the command
        result = ssh_context.run(command, get_pty=False, timeout=timeout or 120)

        # Close the SSH connection
        if ssh_context_manager:
            try:
                ssh_context_manager.__exit__(None, None, None)
            except:
                pass

        return {
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "returncode": result.returncode
        }

    except Exception as e:
        # Close connection on error
        if ssh_context_manager:
            try:
                ssh_context_manager.__exit__(None, None, None)
            except:
                pass

        # Check if it's a timeout
        error_str = str(e).lower()
        if "timeout" in error_str:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout or 120} seconds",
                "returncode": 124
            }

        return {
            "stdout": "",
            "stderr": f"SSH execution failed: {str(e)}",
            "returncode": -1
        }


def simple_terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None
) -> str:
    """
    Execute a command on a MorphCloud VM without session persistence.

    Args:
        command: The command to execute
        background: Whether to run in background (default: False)
        timeout: Command timeout in seconds (default: 120)
        task_id: Unique identifier for VM isolation (optional)

    Returns:
        str: JSON string with output, exit_code, and error fields

    Examples:
        # Execute a simple command
        >>> result = simple_terminal_tool(command="ls -la /tmp")

        # Run a background task
        >>> result = simple_terminal_tool(command="python server.py", background=True)

        # With custom timeout
        >>> result = simple_terminal_tool(command="long_task.sh", timeout=300)
    """
    global _active_instances, _last_activity

    try:
        # Import required modules
        try:
            from morphcloud.api import MorphCloudClient
        except ImportError as import_error:
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": f"Terminal tool disabled: {import_error}",
                "status": "disabled"
            }, ensure_ascii=False)

        # Get configuration
        vm_ttl_seconds = int(os.getenv("HECATE_VM_TTL_SECONDS", "1200"))
        snapshot_id = os.getenv("HECATE_DEFAULT_SNAPSHOT_ID", "snapshot_defv9tjg")

        # Check API key
        morph_api_key = os.getenv("MORPH_API_KEY")
        if not morph_api_key:
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": "MORPH_API_KEY environment variable not set",
                "status": "disabled"
            }, ensure_ascii=False)

        # Use task_id for VM isolation
        effective_task_id = task_id or "default"

        # Start cleanup thread
        _start_cleanup_thread()

        # Get or create VM instance
        with _instance_lock:
            if effective_task_id not in _active_instances:
                morph_client = MorphCloudClient(api_key=morph_api_key)
                _active_instances[effective_task_id] = morph_client.instances.start(
                    snapshot_id=snapshot_id,
                    ttl_seconds=vm_ttl_seconds,
                    ttl_action="stop"
                )

            # Update last activity time
            _last_activity[effective_task_id] = time.time()
            instance = _active_instances[effective_task_id]

        # Wait for instance to be ready
        instance.wait_until_ready()

        # Prepare command for execution
        if background:
            # Run in background with nohup and redirect output
            exec_command = f"nohup {command} > /tmp/bg_output.log 2>&1 &"
            result = _execute_ssh_command(instance, exec_command, timeout=10)

            # For background tasks, return immediately with info
            if result["returncode"] == 0:
                return json.dumps({
                    "output": "Background task started successfully",
                    "exit_code": 0,
                    "error": None
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "output": result["stdout"],
                    "exit_code": result["returncode"],
                    "error": result["stderr"]
                }, ensure_ascii=False)
        else:
            # Run foreground command
            result = _execute_ssh_command(instance, command, timeout=timeout)

            # Combine stdout and stderr for output
            output = result["stdout"]
            if result["stderr"] and result["returncode"] != 0:
                output = f"{output}\n{result['stderr']}" if output else result["stderr"]

            return json.dumps({
                "output": output.strip(),
                "exit_code": result["returncode"],
                "error": result["stderr"] if result["returncode"] != 0 else None
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Failed to execute command: {str(e)}",
            "status": "error"
        }, ensure_ascii=False)


def check_requirements() -> bool:
    """Check if all requirements for the simple terminal tool are met."""
    required_vars = ["MORPH_API_KEY"]
    missing_required = [var for var in required_vars if not os.getenv(var)]

    if missing_required:
        print(f"Missing required environment variables: {', '.join(missing_required)}")
        return False

    try:
        from morphcloud.api import MorphCloudClient
        return True
    except Exception as e:
        print(f"MorphCloud not available: {e}")
        return False


if __name__ == "__main__":
    """Simple test when run directly."""
    print("Simple Terminal Tool Module")
    print("=" * 40)

    if not check_requirements():
        print("Requirements not met. Please check the messages above.")
        exit(1)

    print("All requirements met!")
    print("\nAvailable Tool:")
    print("  - simple_terminal_tool: Execute commands without session persistence")

    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = simple_terminal_tool(command='ls -la')")
    print("  ")
    print("  # Run a background task")
    print("  result = simple_terminal_tool(command='python server.py', background=True)")

    print("\nEnvironment Variables:")
    print(f"  MORPH_API_KEY: {'Set' if os.getenv('MORPH_API_KEY') else 'Not set'}")
    print(f"  HECATE_VM_TTL_SECONDS: {os.getenv('HECATE_VM_TTL_SECONDS', '1200')} (default: 1200 / 20 minutes)")
    print(f"  HECATE_VM_LIFETIME_SECONDS: {os.getenv('HECATE_VM_LIFETIME_SECONDS', '300')} (default: 300 / 5 minutes)")
    print(f"  HECATE_DEFAULT_SNAPSHOT_ID: {os.getenv('HECATE_DEFAULT_SNAPSHOT_ID', 'snapshot_defv9tjg')}")
