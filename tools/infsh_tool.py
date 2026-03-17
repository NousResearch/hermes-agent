#!/usr/bin/env python3
"""
Inference.sh Tool Module

A simple tool for running AI apps via the inference.sh CLI (infsh).
Provides two functions:
  - infsh_install: Install the infsh CLI
  - infsh: Run any infsh command

This is a lightweight wrapper that gives agents direct access to 150+ AI apps
including image generation (FLUX, Reve), video (Veo, Wan), LLMs, search, and more.

Usage:
    from tools.infsh_tool import infsh_tool, infsh_install

    # Install the CLI
    result = infsh_install()

    # Search for apps first (always do this!)
    result = infsh_tool("app list --search flux")

    # Run an app
    result = infsh_tool("app run falai/flux-dev-lora --input '{\"prompt\": \"a cat\"}' --json")
"""

import json
import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 300  # 5 minutes for long-running AI tasks
INSTALL_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_infsh_requirements() -> bool:
    """Check if infsh is available in PATH."""
    return shutil.which("infsh") is not None


def _get_infsh_path() -> Optional[str]:
    """Get the path to infsh binary."""
    return shutil.which("infsh")


# ---------------------------------------------------------------------------
# Install function
# ---------------------------------------------------------------------------

def infsh_install() -> str:
    """
    Install the inference.sh CLI.

    Downloads and installs the infsh binary using the official installer script.
    The installer detects OS/arch, downloads the correct binary, verifies checksum,
    and places it in PATH.

    Returns:
        JSON string with success/error status
    """
    try:
        # Check if already installed
        if check_infsh_requirements():
            infsh_path = _get_infsh_path()
            # Get version
            version_result = subprocess.run(
                ["infsh", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"
            return json.dumps({
                "success": True,
                "message": f"infsh is already installed at {infsh_path}",
                "version": version,
                "already_installed": True
            })

        # Run the installer
        result = subprocess.run(
            ["sh", "-c", "curl -fsSL https://cli.inference.sh | sh"],
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT,
            env={**os.environ, "NONINTERACTIVE": "1"}
        )

        if result.returncode != 0:
            return json.dumps({
                "success": False,
                "error": f"Installation failed: {result.stderr}",
                "stdout": result.stdout
            })

        # Verify installation
        if not check_infsh_requirements():
            return json.dumps({
                "success": False,
                "error": "Installation completed but infsh not found in PATH. You may need to restart your shell or add ~/.local/bin to PATH.",
                "stdout": result.stdout
            })

        return json.dumps({
            "success": True,
            "message": "infsh installed successfully",
            "stdout": result.stdout,
            "next_step": "Run 'infsh login' to authenticate, or set INFSH_API_KEY environment variable"
        })

    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": f"Installation timed out after {INSTALL_TIMEOUT}s"
        })
    except Exception as e:
        logger.exception("infsh_install error: %s", e)
        return json.dumps({
            "success": False,
            "error": f"Installation error: {type(e).__name__}: {e}"
        })


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------

def infsh_tool(
    command: str,
    timeout: Optional[int] = None,
) -> str:
    """
    Execute an infsh CLI command.

    Args:
        command: The infsh command to run (without the 'infsh' prefix).
                 Examples: "app list", "app run falai/flux-schnell --input '{}'"
        timeout: Command timeout in seconds (default: 300)

    Returns:
        JSON string with output, exit_code, and error fields
    """
    try:
        effective_timeout = timeout or DEFAULT_TIMEOUT

        # Check if infsh is installed
        if not check_infsh_requirements():
            return json.dumps({
                "success": False,
                "error": "infsh CLI is not installed. Use infsh_install to install it first.",
                "hint": "Call the infsh_install tool to set up the CLI"
            })

        # Build the full command
        full_command = f"infsh {command}"

        # Execute
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env=os.environ
        )

        output = result.stdout
        error = result.stderr

        # Try to parse JSON output if present
        parsed_output = None
        if output.strip():
            try:
                parsed_output = json.loads(output)
            except json.JSONDecodeError:
                pass  # Not JSON, keep as string

        response = {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "output": parsed_output if parsed_output is not None else output,
        }

        if error:
            response["stderr"] = error

        return json.dumps(response, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": f"Command timed out after {effective_timeout}s",
            "hint": "For long-running tasks, consider using --no-wait flag"
        })
    except Exception as e:
        logger.exception("infsh_tool error: %s", e)
        return json.dumps({
            "success": False,
            "error": f"Execution error: {type(e).__name__}: {e}"
        })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

INFSH_TOOL_DESCRIPTION = """Run AI apps via inference.sh CLI. Access 150+ apps for image generation, video, LLMs, search, 3D, and more.

One API key for everything - manage all AI services (FLUX, Veo, Claude, Tavily, etc.) with a single inference.sh account. You can also bring your own API keys.

IMPORTANT: Always use 'app list --search <query>' first to find the exact app ID before running. App names change frequently.

Commands:
- app list --search <query>: Find apps (ALWAYS DO THIS FIRST)
- app run <app-id> --input '<json>' --json: Run an app
- app get <app-id>: Get app schema before running

Verified app examples (use --search to confirm current names):
- Image: google/nano-banana, google/nano-banana-pro, google/nano-banana-2, falai/flux-dev-lora, bytedance/seedream-5-lite, falai/reve, xai/grok-imagine-image
- Video: google/veo-3-1-fast, bytedance/seedance-1-5-pro, falai/wan-2-5
- Upscale: falai/topaz-image-upscaler
- Search: tavily/search-assistant, exa/search
- LLM: openrouter/claude-sonnet-45

Workflow: ALWAYS search first, then run:
1. app list --search image
2. app run falai/flux-dev-lora --input '{"prompt": "a sunset"}' --json"""

INFSH_SCHEMA = {
    "name": "infsh",
    "description": INFSH_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The infsh command (without 'infsh' prefix). ALWAYS use 'app list --search <query>' first to find correct app IDs, then 'app run <id> --input <json> --json'"
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default: 300). AI tasks like video generation may take 1-2 minutes.",
                "minimum": 1
            }
        },
        "required": ["command"]
    }
}

INFSH_INSTALL_SCHEMA = {
    "name": "infsh_install",
    "description": "Install the inference.sh CLI (infsh). Downloads and installs the binary. Run this first if infsh is not available.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}


def _handle_infsh(args, **kw):
    return infsh_tool(
        command=args.get("command", ""),
        timeout=args.get("timeout"),
    )


def _handle_infsh_install(args, **kw):
    return infsh_install()


# Register both tools under the "inference" toolset
registry.register(
    name="infsh",
    toolset="inference",
    schema=INFSH_SCHEMA,
    handler=_handle_infsh,
    check_fn=check_infsh_requirements,
    requires_env=[],
)

registry.register(
    name="infsh_install",
    toolset="inference",
    schema=INFSH_INSTALL_SCHEMA,
    handler=_handle_infsh_install,
    check_fn=lambda: True,  # Always available - it's the installer
    requires_env=[],
)
