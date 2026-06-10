#!/usr/bin/env python3
"""
Remote Run Tool — SSH Command Execution via Paramiko

Executes commands on remote hosts over SSH using Paramiko. Returns stdout,
stderr, and exit code. Supports password and key-based authentication,
sudo, custom ports, working directories, and environment variables.

Design:
- Gated on Paramiko being installed (optional dependency)
- Every connection uses a fresh SSHClient (no persistent connection pooling)
  to avoid stale-channel bugs across tool calls
- Key-based auth preferred; password auth supported as fallback
- sudo mode pipes the password via channel stdin (never in process cmdline)
"""

import json
import logging
import os
import re
import shlex
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Connection timeout (seconds) — separate from command execution timeout
SSH_CONNECT_TIMEOUT = 15

# Maximum combined stdout/stderr output before truncation (matching terminal tool)
MAX_RESULT_SIZE_CHARS = 100_000

# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


def check_remote_run_requirements() -> bool:
    """Return True if Paramiko is available."""
    try:
        import paramiko  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REMOTE_RUN_SCHEMA = {
    "name": "remote_run",
    "description": (
        "Execute a command on a remote host via SSH. "
        "Returns stdout, stderr, and exit code. "
        "Use this to run commands on servers, containers, or any SSH-accessible host.\n\n"
        "Authentication (in order of precedence):\n"
        "1. key_file (recommended) — path to an SSH private key\n"
        "2. password — password-based auth (less secure)\n"
        "3. SSH agent — uses any keys loaded in the local SSH agent\n\n"
        "Examples:\n"
        '  remote_run(host="myserver", command="whoami")\n'
        '  remote_run(host="myserver", command="systemctl status nginx", sudo=True)\n'
        '  remote_run(host="10.0.0.5", command="ls -la /opt", user="admin", port=2222)\n'
        '  remote_run(host="db1", command="./deploy.sh", workdir="/app", timeout=120)\n\n'
        "Note: Each call opens a new SSH connection. "
        "For multiple commands on the same host, make sequential calls — "
        "Paramiko reuses the underlying TCP connection within a single "
        "exec_command() session but each tool call creates a new SSHClient."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "host": {
                "type": "string",
                "description": "Remote hostname or IP address.",
            },
            "command": {
                "type": "string",
                "description": "Command to execute on the remote host.",
            },
            "user": {
                "type": "string",
                "description": (
                    "SSH username. Defaults to the current local user if omitted."
                ),
            },
            "port": {
                "type": "integer",
                "description": "SSH port (default: 22).",
                "default": 22,
            },
            "key_file": {
                "type": "string",
                "description": (
                    "Path to an SSH private key file for authentication. "
                    "E.g., '/home/user/.ssh/example_key'. "
                    "If not provided, attempts agent- and password-based auth."
                ),
            },
            "password": {
                "type": "string",
                "description": (
                    "SSH password. Only use when key-based auth is not available. "
                    "For sudo commands, this password is also used for sudo elevation."
                ),
            },
            "sudo": {
                "type": "boolean",
                "description": "Run the command with sudo (default: false).",
                "default": False,
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Working directory on the remote host. "
                    "The command is prefixed with 'cd <workdir> && '."
                ),
            },
            "env": {
                "type": "object",
                "description": (
                    "Environment variables to set on the remote host, "
                    "passed as KEY: VALUE pairs. Keys must be valid shell "
                    "identifiers matching [A-Za-z_][A-Za-z0-9_]*."
                ),
                "additionalProperties": {"type": "string"},
            },
            "timeout": {
                "type": "integer",
                "description": "Command execution timeout in seconds (default: 60). "
                               "The TCP connection timeout is fixed at 15s.",
                "default": 60,
            },
            "max_result_size_chars": {
                "type": "integer",
                "description": "Maximum combined stdout+stderr output size (default: 100000). "
                               "Output beyond this is truncated with a '[truncated]' marker.",
                "default": 100000,
            },
        },
        "required": ["host", "command"],
    },
}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _build_env_export(env: Optional[Dict[str, str]]) -> str:
    """Build shell 'export' preamble from env dict.

    Validates that all keys are valid shell identifiers to prevent
    shell injection via env key names.
    """
    if not env:
        return ""
    parts = []
    for k, v in env.items():
        # Validate env var key — reject anything that's not a valid shell identifier
        if not k or not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
            raise ValueError(
                f"Invalid environment variable key: {k!r}. "
                "Keys must be valid shell identifiers matching [A-Za-z_][A-Za-z0-9_]*"
            )
        escaped = v.replace("'", "'\\''")
        parts.append(f"export {k}='{escaped}'")
    return "; ".join(parts) + "; " if parts else ""


def _build_command(command: str, workdir: Optional[str],
                   env: Optional[Dict[str, str]],
                   sudo: bool) -> str:
    """Build the final command string to execute over SSH.

    All shell-interpolated values use shlex.quote() to prevent injection.
    The sudo wrapper uses shlex.quote() on the inner command so that
    commands containing single quotes or other shell metacharacters work
    correctly.
    """
    prefix = ""

    if workdir:
        prefix += f"cd {shlex.quote(workdir)} && "

    env_export = _build_env_export(env)
    if env_export:
        prefix += env_export

    full_cmd = f"{prefix}{command}"

    if sudo:
        # Wrap in sudo. The inner command is shlex-quoted so that single quotes
        # and other shell metacharacters in the command are preserved correctly.
        full_cmd = f"sudo bash -c {shlex.quote(full_cmd)}"

    return full_cmd


def _send_sudo_password(stdin_chan, password: str) -> None:
    """Send the sudo password to the remote process via channel stdin.

    This avoids embedding the password in the command string, which would
    make it visible in process listings on the remote host.
    """
    if not password:
        return
    # Brief delay to allow the sudo password prompt to appear
    time.sleep(0.5)
    stdin_chan.send(password + "\n")
    stdin_chan.shutdown_write()


def remote_run_handler(args: Dict[str, Any], **kwargs) -> str:
    """Execute a command on a remote host via SSH using Paramiko.

    Args:
        args: Dictionary with keys matching REMOTE_RUN_SCHEMA parameters.

    Returns:
        JSON string: {"stdout": "...", "stderr": "...", "exit_code": N}
    """
    import paramiko

    host = args["host"]
    command = args["command"]
    user = args.get("user")
    port = args.get("port", 22)
    key_file = args.get("key_file")
    password = args.get("password")
    sudo = args.get("sudo", False)
    workdir = args.get("workdir")
    env = args.get("env")
    timeout = args.get("timeout", 60)
    max_result_chars = args.get("max_result_size_chars", MAX_RESULT_SIZE_CHARS)

    client = paramiko.SSHClient()

    # Use WarningPolicy — logs a warning for unknown/changed host keys
    # but does not abort. This matches the existing SSH environment's
    # StrictHostKeyChecking=accept-new pattern.
    # For stricter verification, users can switch to RejectPolicy and
    # pre-configure known_hosts.
    client.set_missing_host_key_policy(paramiko.WarningPolicy())

    # Try to load system known_hosts — non-fatal if missing
    try:
        client.load_system_host_keys()
    except Exception:
        pass

    try:
        # Build connect kwargs
        connect_kwargs: Dict[str, Any] = {
            "hostname": host,
            "port": port,
            "timeout": SSH_CONNECT_TIMEOUT,  # connection timeout only
        }
        if user:
            connect_kwargs["username"] = user
        if key_file:
            connect_kwargs["key_filename"] = os.path.expanduser(key_file)
        if password:
            connect_kwargs["password"] = password

        # Also try SSH agent
        connect_kwargs["allow_agent"] = True
        connect_kwargs["look_for_keys"] = True

        logger.info(
            "remote_run: connecting to %s@%s:%s (connect_timeout=%s)",
            connect_kwargs.get("username", "default"), host, port,
            SSH_CONNECT_TIMEOUT,
        )
        client.connect(**connect_kwargs)
        logger.info("remote_run: connected, executing command")

        # Build the command string
        get_pty = sudo
        full_cmd = _build_command(command, workdir, env, sudo)

        # The max size for the combined command
        stdin, stdout, stderr = client.exec_command(
            full_cmd,
            timeout=timeout,
            get_pty=get_pty,  # PTY needed for sudo password prompt
        )

        # If sudo mode with password, send it via channel stdin
        # instead of embedding it in the command string
        if sudo and password:
            _send_sudo_password(stdin.channel, password)

        # Read output with truncation protection
        out_text = stdout.read().decode("utf-8", errors="replace")
        err_text = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()

        # Truncate if combined output exceeds limit
        combined_size = len(out_text) + len(err_text)
        if combined_size > max_result_chars:
            # Truncate proportionally
            out_max = max(0, max_result_chars - len(err_text) - 100)
            if len(out_text) > out_max:
                out_text = out_text[:out_max] + "\n... [stdout truncated]"
            if combined_size > max_result_chars:
                err_max = max_result_chars - len(out_text) - 100
                if len(err_text) > err_max:
                    err_text = err_text[:err_max] + "\n... [stderr truncated]"

        result = {
            "stdout": out_text,
            "stderr": err_text,
            "exit_code": exit_code,
            "host": host,
        }

        logger.info(
            "remote_run: exit_code=%s, output=%s bytes (trunc=%s)",
            exit_code, len(out_text) + len(err_text),
            combined_size > MAX_RESULT_SIZE_CHARS,
        )

        return json.dumps(result)

    except paramiko.AuthenticationException as e:
        return json.dumps({
            "error": f"Authentication failed: {e}",
            "host": host,
            "exit_code": 1,
        })
    except paramiko.SSHException as e:
        return json.dumps({
            "error": f"SSH connection failed: {e}",
            "host": host,
            "exit_code": 1,
        })
    except ValueError as e:
        # Raised by _build_env_export for invalid env keys
        return json.dumps({
            "error": str(e),
            "host": host,
            "exit_code": 1,
        })
    except OSError as e:
        return json.dumps({
            "error": f"Network error: {e}",
            "host": host,
            "exit_code": 1,
        })
    except Exception as e:
        logger.error("remote_run: unexpected error: %s", e, exc_info=True)
        return json.dumps({
            "error": f"Unexpected error: {e}",
            "host": host,
            "exit_code": 1,
        })
    finally:
        try:
            client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

registry.register(
    name="remote_run",
    toolset="ssh",
    schema=REMOTE_RUN_SCHEMA,
    handler=remote_run_handler,
    check_fn=check_remote_run_requirements,
    emoji="🔗",
)
