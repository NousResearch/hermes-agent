import shlex
import json
import re

# Shell metacharacters that would bypass a whitelist on the base command alone.
# When present in a command string, they can chain, redirect, or substitute
# arbitrary commands regardless of which binary is named first.
_SHELL_METACHARS = re.compile(r"[;&|`$(){}]|\|\||&&")

WHITELIST = [
    "git", "python3", "ls", "cat", "head", "tail", "grep", "which",
    "hermes", "grok",
    "whoami", "hostname", "uname", "date", "pwd", "echo", "wc",
    "find", "du", "df", "ps", "uptime", "free",
]


def register(mcp):
    @mcp.tool()
    async def system_run(
        command: str,
        cwd: str | None = None,
        timeout: int = 60,
    ) -> str:
        """Execute a shell command on Omarchy (whitelisted commands only).

        Only the base command must be on the whitelist. Shell metacharacters
        (;, |, &, $, backtick, (), {}, ||, &&) are rejected to prevent
        chaining arbitrary commands onto a whitelisted binary.

        Args:
            command: Shell command to run (no metacharacters allowed)
            cwd: Working directory
            timeout: Max seconds to wait (5-300)
        """
        parts = shlex.split(command)
        if not parts:
            return json.dumps({"error": "empty command"})
        base = parts[0]
        if base not in WHITELIST:
            return json.dumps({
                "error": f"Command '{base}' is not in the whitelist",
                "whitelist": WHITELIST,
            })

        # Reject shell metacharacters that would let an attacker bypass the
        # whitelist check — the base command passes, but ; rm -rf / would not.
        if _SHELL_METACHARS.search(command):
            return json.dumps({
                "error": "Shell metacharacters are not allowed in system_run",
                "command": command,
            })

        from executor import run_command_async
        # Run as a list (no shell) so shlex splitting is authoritative and
        # shell metacharacters in arguments are passed literally, not executed.
        result = await run_command_async(
            parts,
            cwd=cwd,
            timeout=min(max(timeout, 5), 300),
        )
        return json.dumps(result)
