from executor import run_command_async


def register(mcp):
    @mcp.tool()
    async def hermes_execute(
        prompt: str,
        cwd: str | None = None,
        timeout: int = 300,
    ) -> str:
        """Execute a prompt with Hermes Agent CLI on Omarchy.

        Args:
            prompt: The full prompt to send to Hermes Agent via stdin
            cwd: Working directory on Omarchy
            timeout: Max seconds to wait (30-1800)
        """
        effective_timeout = min(max(timeout, 30), 1800)
        result = await run_command_async(
            ["hermes", "chat", "-q", "-"],
            cwd=cwd, timeout=effective_timeout,
            stdin_data=prompt.encode("utf-8"),
        )
        output = result["output"]
        if not result["ok"] and result["error"] != "timeout":
            output += f"\n\n---\nExited {result['exit_code']}"
        return output
