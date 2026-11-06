from executor import run_command_async


def register(mcp):
    @mcp.tool()
    async def codex_execute(
        prompt: str,
        cwd: str | None = None,
        timeout: int = 120,
    ) -> str:
        """Execute a prompt with Codex CLI.

        Args:
            prompt: The full prompt to send to Codex via stdin
            cwd: Working directory on Omarchy
            timeout: Max seconds to wait (30-600)
        """
        effective_timeout = min(max(timeout, 30), 600)
        result = await run_command_async(
            ["codex", "exec", "--skip-git-repo-check", "-"],
            cwd=cwd, timeout=effective_timeout,
            stdin_data=prompt.encode("utf-8"),
        )
        output = result["output"]
        if not result["ok"] and result["error"] != "timeout":
            output += f"\n\n---\nExited {result['exit_code']}"
        return output
