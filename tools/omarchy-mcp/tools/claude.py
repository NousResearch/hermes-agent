import asyncio
import os
import asyncio
from executor import run_command_async


def register(mcp):
    @mcp.tool()
    async def claude_execute(
        prompt: str,
        cwd: str | None = None,
        timeout: int = 300,
    ) -> str:
        """Execute a prompt with Claude Code CLI (YOLO mode,
        dangerously-skip-permissions active).

        Args:
            prompt: The full prompt to send to Claude Code via stdin
            cwd: Working directory on Omarchy
            timeout: Max seconds to wait (30-1800)
        """
        effective_timeout = min(max(timeout, 30), 1800)
        result = await run_command_async(
            ["claude", "--dangerously-skip-permissions", "-"],
            cwd=cwd, timeout=effective_timeout,
            stdin_data=prompt.encode("utf-8"),
        )
        output = result["output"]
        if not result["ok"] and result["error"] != "timeout":
            output += f"\n\n---\nExited {result['exit_code']}"
        return output
