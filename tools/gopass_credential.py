#!/usr/bin/env python3
"""
Retrieve credentials from a gopass store.
"""
import os
import subprocess
from tools.registry import register
from hermes_tools import ToolResult

def gopass_credential(path: str) -> ToolResult:
    """
    Retrieve a secret from gopass.
    
    Args:
        path: The gopass secret path (e.g., 'hermes-agent/openrouter')
    
    Returns:
        The decrypted secret as a string.
    """
    try:
        # Run gopass show command
        result = subprocess.run(
            ["gopass", "show", path],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"}
        )
        return ToolResult(output=result.stdout.strip())
    except subprocess.CalledProcessError as e:
        return ToolResult(
            output=f"Error retrieving credential '{path}': {e.stderr or 'Unknown error'}",
            succeeded=False
        )
    except Exception as e:
        return ToolResult(
            output=f"Unexpected error retrieving credential '{path}': {str(e)}",
            succeeded=False
        )

# Register the tool
register(
    name="gopass_credential",
    toolset="credentials",
    schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The gopass secret path (e.g., 'hermes-agent/openrouter')"
            }
        },
        "required": ["path"]
    },
    handler=gopass_credential,
    description="Retrieve a credential from the gopass password store.",
    emoji="🔑"
)
