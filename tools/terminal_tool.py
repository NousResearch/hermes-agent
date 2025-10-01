#!/usr/bin/env python3
"""
Terminal Tool Module

This module provides a single terminal tool using Hecate's VM infrastructure.
It wraps Hecate's functionality to provide a simple interface for executing commands
on Morph VMs with automatic lifecycle management.

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
from typing import Optional, Dict, Any
from hecate import run_tool_with_lifecycle_management
from morphcloud._llm import ToolCall

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

def terminal_tool(
    command: Optional[str] = None,
    input_keys: Optional[str] = None,
    session_id: Optional[str] = None,
    background: bool = False,
    idle_threshold: float = 5.0,
    timeout: Optional[int] = None
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
    try:
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
        
        # Execute with lifecycle management
        result = run_tool_with_lifecycle_management(tool_call)
        
        # Format the result with all possible fields
        # Map hecate's "stdout" to "output" for compatibility
        formatted_result = {
            "output": result.get("stdout", result.get("output", "")),
            "screen": result.get("screen", ""),
            "session_id": result.get("session_id"),
            "exit_code": result.get("returncode", result.get("exit_code", -1)),
            "error": result.get("error"),
            "status": "active" if result.get("session_id") else "ended"
        }
        
        return json.dumps(formatted_result)
        
    except Exception as e:
        return json.dumps({
            "output": "",
            "screen": "",
            "session_id": None,
            "exit_code": -1,
            "error": f"Failed to execute terminal command: {str(e)}",
            "status": "error"
        })

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
    
    # Check if Hecate is importable
    try:
        import hecate
        return True
    except ImportError:
        print("Hecate is not installed. Please install it with: pip install hecate")
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
    print(f"  HECATE_VM_LIFETIME_SECONDS: {os.getenv('HECATE_VM_LIFETIME_SECONDS', '300')} (default: 300)")
    print(f"  HECATE_DEFAULT_SNAPSHOT_ID: {os.getenv('HECATE_DEFAULT_SNAPSHOT_ID', 'snapshot_p5294qxt')} (default: snapshot_p5294qxt)")