#!/usr/bin/env python3
"""
Terminal Tool Module

This module provides a single terminal tool using Hecate's VM infrastructure.
It wraps Hecate's functionality to provide a simple interface for executing commands
on Morph VMs with automatic lifecycle management.

VM Lifecycle:
- VMs have a TTL (time to live) set at creation (default: 20 minutes)
- VMs are also cleaned up locally after 5 minutes of inactivity
- Timer resets with each use

Available tool:
- terminal_tool: Execute commands with optional interactive session support

Usage:
    from terminal_tool import terminal_tool

    # Execute a single command
    result = terminal_tool("ls -la")

    # Execute in an interactive session
    result = terminal_tool("python", input_keys="print('hello')\\nexit()\\n")
"""

import json
import os
import uuid
import threading
import time
import atexit
from typing import Optional, Dict, Any

# Detailed description for the terminal tool based on Hermes Terminal system prompt
TERMINAL_TOOL_DESCRIPTION = """Execute commands on a secure, persistent Linux VM environment with full interactive application support.

**Environment:** 
- Minimal Debian-based OS with internet access
- Automatic VM lifecycle management (creates on-demand, reuses, cleans up)
- **Full state persistence across tool calls**: current directory (pwd), environment variables, activated virtual environments (conda/venv), running processes, and command history all persist between consecutive tool calls
- Session state managed automatically via tmux

**Command Execution:**
- Simple commands: Just provide the 'command' parameter
- Background processes: Set 'background': True for servers/long-running tasks
- Interactive applications automatically detected and handled

**Interactive Applications (TUIs/Pagers/Prompts):**
When commands enter interactive mode (vim, nano, less, git prompts, package managers, etc.), you'll receive screen content with "frozen" status. This is NORMAL - the session is still active and waiting for input.

**To interact with frozen sessions:**
1. Use 'input_keys' parameter with keystrokes to send
2. System auto-detects and uses the active session
3. Session stays active until application exits

**Special Key Syntax for input_keys:**
- `<ESC>`: Escape key
- `<ENTER>`: Enter/Return  
- `<CTRL+C>`, `<CTRL+D>`, `<CTRL+Z>`: Control combinations
- `<UP>`, `<DOWN>`, `<LEFT>`, `<RIGHT>`: Arrow keys
- `<TAB>`, `<BACKSPACE>`: Tab and Backspace
- `<F1>` through `<F12>`: Function keys
- `<SHIFT+TAB>`: Shift+Tab
- Uppercase letters for Shift+letter (e.g., 'V' for Shift+V)
- Symbols for Shift+number (e.g., '!' for Shift+1, ':' for Shift+;)

**Examples:**
- Start vim: `{"command": "vim file.txt"}`
- Type in vim: `{"input_keys": "iHello World<ESC>"}`  
- Save and quit: `{"input_keys": ":wq<ENTER>"}`
- Navigate in less: `{"input_keys": "j"}`
- Quit less: `{"input_keys": "q"}`

**Best Practices:**
- Run servers/long processes in background with separate tool calls
- Chain multiple foreground commands in single call if needed
- Monitor disk usage for large tasks, clean up to free space
- Test components incrementally with mock inputs
- Install whatever tools needed - full system access provided"""

# Global state for VM lifecycle management
# These persist across tool calls to enable session continuity
# Changed to dictionaries keyed by task_id to prevent leakage between concurrent tasks
_active_instances: Dict[str, Any] = {}
_active_contexts: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}  # Track last activity time for each VM
_instance_lock = threading.Lock()
_cleanup_thread = None
_cleanup_running = False

def _cleanup_inactive_vms(vm_lifetime_seconds: int = 300):
    """
    Clean up VMs that have been inactive for longer than vm_lifetime_seconds.
    This function should be called periodically by a background thread.

    Args:
        vm_lifetime_seconds: Maximum lifetime in seconds for inactive VMs (default: 300)
    """
    global _active_instances, _active_contexts, _last_activity

    current_time = time.time()
    instances_to_cleanup = []

    # Find and extract instances to cleanup while holding lock
    with _instance_lock:
        # Find all VMs that have been inactive for too long
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > vm_lifetime_seconds:
                if task_id in _active_instances:
                    instances_to_cleanup.append((task_id, _active_instances[task_id]))
                    # Remove from tracking dictionaries immediately
                    del _active_instances[task_id]

                if task_id in _active_contexts:
                    del _active_contexts[task_id]

                if task_id in _last_activity:
                    del _last_activity[task_id]

    # Terminate outside the lock so we don't block other operations
    for task_id, instance in instances_to_cleanup:
        try:
            if hasattr(instance, 'terminate'):
                instance.terminate()
            elif hasattr(instance, 'stop'):
                instance.stop()
            elif hasattr(instance, 'delete'):
                instance.delete()
            print(f"[VM Cleanup] Terminated inactive VM for task: {task_id}")
        except Exception as e:
            print(f"[VM Cleanup] Error cleaning up VM for task {task_id}: {e}")

def _cleanup_thread_worker():
    """
    Background thread worker that periodically cleans up inactive VMs.
    Runs every 60 seconds.
    """
    global _cleanup_running

    while _cleanup_running:
        try:
            vm_lifetime = int(os.getenv("HECATE_VM_LIFETIME_SECONDS", "300"))
            _cleanup_inactive_vms(vm_lifetime)
        except Exception as e:
            print(f"[VM Cleanup] Error in cleanup thread: {e}")

        # Sleep for 60 seconds, but check every second if we should stop
        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)

def _start_cleanup_thread():
    """
    Start the background cleanup thread if it's not already running.
    """
    global _cleanup_thread, _cleanup_running

    with _instance_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()

def _stop_cleanup_thread():
    """
    Stop the background cleanup thread.
    """
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        _cleanup_thread.join(timeout=5)

def cleanup_vm(task_id: str):
    """
    Manually clean up a specific VM by task_id.
    This should be called when a task is completed.

    Args:
        task_id: The task ID of the VM to clean up
    """
    global _active_instances, _active_contexts, _last_activity

    # Extract instance from dict while holding lock, but don't terminate yet
    instance_to_cleanup = None
    with _instance_lock:
        if task_id in _active_instances:
            instance_to_cleanup = _active_instances[task_id]
            # Remove from tracking dictionaries immediately
            del _active_instances[task_id]

        if task_id in _active_contexts:
            del _active_contexts[task_id]

        if task_id in _last_activity:
            del _last_activity[task_id]

    # Terminate outside the lock so multiple cleanups can run concurrently
    if instance_to_cleanup:
        try:
            if hasattr(instance_to_cleanup, 'terminate'):
                instance_to_cleanup.terminate()
            elif hasattr(instance_to_cleanup, 'stop'):
                instance_to_cleanup.stop()
            elif hasattr(instance_to_cleanup, 'delete'):
                instance_to_cleanup.delete()
            print(f"[VM Cleanup] Manually terminated VM for task: {task_id}")
        except Exception as e:
            print(f"[VM Cleanup] Error manually cleaning up VM for task {task_id}: {e}")

# Register cleanup on program exit
atexit.register(_stop_cleanup_thread)

def terminal_tool(
    command: Optional[str] = None,
    input_keys: Optional[str] = None,
    session_id: Optional[str] = None,
    background: bool = False,
    idle_threshold: float = 5.0,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None
) -> str:
    """
    Execute a command on a Morph VM with optional interactive session support.

    This tool uses Hecate's VM lifecycle management to automatically create
    and manage VMs. VMs are reused within the configured lifetime window
    and automatically cleaned up after inactivity.

    Args:
        command: The command to execute (optional if continuing existing session)
        input_keys: Keystrokes to send to interactive session (e.g., "hello\\n")
        session_id: ID of existing session to continue (optional)
        background: Whether to run the command in the background (default: False)
        idle_threshold: Seconds to wait for output before considering session idle (default: 5.0)
        timeout: Command timeout in seconds (optional)
        task_id: Unique identifier for this task to isolate VMs between concurrent tasks (optional)

    Returns:
        str: JSON string containing command output, session info, exit code, and any errors
    
    Examples:
        # Execute a simple command
        >>> result = terminal_tool(command="ls -la /tmp")
        
        # Start an interactive Python session
        >>> result = terminal_tool(command="python3")
        >>> session_data = json.loads(result)
        >>> session_id = session_data["session_id"]
        
        # Send input to the session
        >>> result = terminal_tool(input_keys="print('Hello')\\n", session_id=session_id)
        
        # Run a background task
        >>> result = terminal_tool(command="sleep 60", background=True)
    """
    global _active_instances, _active_contexts

    try:
        # Import required modules lazily so this module can be imported
        # even when hecate is not installed
        try:
            from morphcloud._llm import ToolCall
            from morphcloud.api import MorphCloudClient
            from hecate.cli import run_tool, ExecutionContext
            from rich.console import Console
            import io
        except ImportError as import_error:
            return json.dumps({
                "output": "",
                "screen": "",
                "exit_code": -1,
                "error": f"Terminal tool is disabled due to import error: {import_error}",
                "status": "disabled"
            }, ensure_ascii=False)


        # Get configuration from environment
        vm_lifetime_seconds = int(os.getenv("HECATE_VM_LIFETIME_SECONDS", "300"))
        vm_ttl_seconds = int(os.getenv("HECATE_VM_TTL_SECONDS", "1200"))  # 20 minutes default
        snapshot_id = os.getenv("HECATE_DEFAULT_SNAPSHOT_ID", "snapshot_defv9tjg")

        # Check API key
        morph_api_key = os.getenv("MORPH_API_KEY")
        if not morph_api_key:
            return json.dumps({
                "output": "",
                "screen": "",
                "exit_code": -1,
                "error": "MORPH_API_KEY environment variable not set",
                "status": "disabled"
            }, ensure_ascii=False)

        # Use task_id to isolate VMs between concurrent tasks
        # If no task_id provided, use "default" for backward compatibility
        effective_task_id = task_id or "default"

        # Start the cleanup thread if not already running
        _start_cleanup_thread()

        # Get or create VM instance and execution context per task
        # This is critical for interactive session support - the context must persist!
        with _instance_lock:
            if effective_task_id not in _active_instances:
                morph_client = MorphCloudClient(api_key=morph_api_key)
                _active_instances[effective_task_id] = morph_client.instances.start(
                    snapshot_id=snapshot_id,
                    ttl_seconds=vm_ttl_seconds,
                    ttl_action="stop"
                )

            # Get or create persistent execution context per task
            if effective_task_id not in _active_contexts:
                _active_contexts[effective_task_id] = ExecutionContext()

            # Update last activity time for this VM (resets the inactivity timer)
            _last_activity[effective_task_id] = time.time()

            instance = _active_instances[effective_task_id]
            ctx = _active_contexts[effective_task_id]

        # Build tool input based on provided parameters
        tool_input = {}

        if command:
            tool_input["command"] = command
        if input_keys:
            tool_input["input_keys"] = input_keys
        if session_id:
            tool_input["session_id"] = session_id
        if background:
            tool_input["background"] = background
        if idle_threshold != 5.0:
            tool_input["idle_threshold"] = idle_threshold
        if timeout is not None:
            tool_input["timeout"] = timeout

        tool_call = ToolCall(
            name="run_command",
            input=tool_input
        )

        # Create a console for output (redirect to string buffer to avoid printing)
        console_output = io.StringIO()
        console = Console(file=console_output, force_terminal=False, legacy_windows=False)

        # Generate unique tool block ID
        tool_block_id = f"tool_{uuid.uuid4().hex[:8]}"

        # Execute the tool with hecate
        result = run_tool(
            tool_call=tool_call,
            instance=instance,
            console=console,
            tool_block_id=tool_block_id,
            ctx=ctx
        )

        # Format the result with only essential fields for the LLM
        # Map hecate's "stdout" to "output" for compatibility
        formatted_result = {
            "output": result.get("stdout", result.get("output", "")),
            "screen": result.get("screen", ""),
            "exit_code": result.get("returncode", result.get("exit_code", -1)),
            "error": result.get("error")
        }

        return json.dumps(formatted_result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "output": "",
            "screen": "",
            "exit_code": -1,
            "error": f"Failed to execute terminal command: {str(e)}",
            "status": "error"
        }, ensure_ascii=False)

def check_hecate_requirements() -> bool:
    """
    Check if all requirements for terminal tools are met.
    
    Returns:
        bool: True if all requirements are met, False otherwise
    """
    # Check for required environment variables
    required_vars = ["MORPH_API_KEY"]
    optional_vars = ["OPENAI_API_KEY"]  # Needed for Hecate's LLM features
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    
    if missing_required:
        print(f"Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"Warning: Missing optional environment variables: {', '.join(missing_optional)}")
        print("   (Some Hecate features may be limited)")
    
    # Check if Hecate and required modules are importable
    try:
        from morphcloud._llm import ToolCall
        from morphcloud.api import MorphCloudClient
        from hecate.cli import run_tool, ExecutionContext
        from rich.console import Console
        return True
    except Exception as e:
        print(f"Hecate not available: {e}")
        print(f"Make sure hecate is installed and MORPH_API_KEY is set.")
        return False

# Module-level initialization check
_requirements_met = check_hecate_requirements()

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("Terminal Tool Module")
    print("=" * 40)
    
    if not _requirements_met:
        print("Requirements not met. Please check the messages above.")
        exit(1)
    
    print("All requirements met!")
    print("\nAvailable Tool:")
    print("  - terminal_tool: Execute commands with optional interactive session support")
    
    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = terminal_tool(command='ls -la')")
    print("  ")
    print("  # Start an interactive session")
    print("  result = terminal_tool(command='python3')")
    print("  session_data = json.loads(result)")
    print("  session_id = session_data['session_id']")
    print("  ")
    print("  # Send input to the session")
    print("  result = terminal_tool(")
    print("      input_keys='print(\"Hello\")\\\\n',")
    print("      session_id=session_id")
    print("  )")
    print("  ")
    print("  # Run a background task")
    print("  result = terminal_tool(command='sleep 60', background=True)")
    
    print("\nEnvironment Variables:")
    print(f"  MORPH_API_KEY: {'Set' if os.getenv('MORPH_API_KEY') else 'Not set'}")
    print(f"  OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set (optional)'}")
    print(f"  HECATE_VM_TTL_SECONDS: {os.getenv('HECATE_VM_TTL_SECONDS', '1200')} (default: 1200 / 20 minutes)")
    print(f"  HECATE_VM_LIFETIME_SECONDS: {os.getenv('HECATE_VM_LIFETIME_SECONDS', '300')} (default: 300 / 5 minutes)")
    print(f"  HECATE_DEFAULT_SNAPSHOT_ID: {os.getenv('HECATE_DEFAULT_SNAPSHOT_ID', 'snapshot_defv9tjg')} (default: snapshot_defv9tjg)")
