"""``headers_command`` for remote MCP servers: a command whose stdout is the headers to send.

For servers fronted by short-lived tokens (cloud IAM, ``client_credentials`` grants, vault-minted
bearers), a static ``headers:`` value expires mid-session and every reconnect replays it. The
command runs on the first request, its headers are cached, and a 401 re-runs it once and retries
that request, so a rotated token is picked up without a reconnect or a restart.

Inspired by Devin's "header scripts for MCP servers" (docs.devin.ai release notes, Oct 30 2026);
salvages the intent of #52418 (``bearer_refresh_cmd``, harrisali0101), generalised to any header.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from hermes_cli._subprocess_compat import bounded_probe_run, split_command_line

logger = logging.getLogger("tools.mcp_tool")

HEADERS_COMMAND_TIMEOUT_SECONDS = 30
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")  # RFC 9110 token


class HeadersCommandError(RuntimeError):
    """The command failed, timed out, or printed nothing usable."""


def parse_headers_output(server_name: str, stdout: str) -> dict[str, str]:
    """A JSON object (``{"Authorization": "Bearer …"}``) or ``Name: value`` lines → headers."""
    text = stdout.strip()
    if text.startswith("{"):
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise HeadersCommandError(f"MCP server '{server_name}': headers_command printed invalid JSON ({exc})")
        pairs = list(raw.items()) if isinstance(raw, dict) else []
    else:
        pairs = [line.split(":", 1) for line in text.splitlines() if line.strip()]
    headers: dict[str, str] = {}
    for pair in pairs:
        if len(pair) != 2:
            raise HeadersCommandError(f"MCP server '{server_name}': headers_command output must be a JSON "
                                      "object or 'Name: value' lines")
        name, value = str(pair[0]).strip(), str(pair[1]).strip()
        if not _HEADER_NAME.match(name) or not value or "\r" in value or "\n" in value:
            raise HeadersCommandError(f"MCP server '{server_name}': headers_command printed an invalid "
                                      f"header {name[:40]!r}")
        headers[name] = value
    if not headers:
        raise HeadersCommandError(f"MCP server '{server_name}': headers_command printed no headers")
    return headers


def headers_command_argv(command) -> list[str]:
    """*command* (one string or an argv list) as argv, ``~`` expanded in the executable. Raises
    ValueError on unbalanced quotes. Shared with the config security scan, so both read one shape."""
    argv = [str(part) for part in command] if isinstance(command, list) else split_command_line(str(command))
    return [os.path.expanduser(argv[0]), *argv[1:]] if argv else []


def run_headers_command(server_name: str, command) -> dict[str, str]:
    """Run *command* (string or argv list) and return the headers it printed; values are registered
    for redaction before they can reach a log or the model."""
    from agent.redact import register_vault_redaction_value, redact_sensitive_text
    from tools.environments.local import served_profile_child_env

    try:
        argv = headers_command_argv(command)
    except ValueError as exc:
        raise HeadersCommandError(f"MCP server '{server_name}': headers_command cannot be parsed: {exc}")
    if not argv:
        raise HeadersCommandError(f"MCP server '{server_name}': headers_command is empty")
    try:
        # It mints FOR the served profile: that profile's secrets, never the launch profile's.
        # Bounded: a plain run() can hang past its timeout on Windows when a grandchild holds the pipes.
        proc = bounded_probe_run(argv, timeout=HEADERS_COMMAND_TIMEOUT_SECONDS,
                                 env=served_profile_child_env(inherit_credentials=True),
                                 raise_on_spawn_failure=True)
    except OSError as exc:
        raise HeadersCommandError(f"MCP server '{server_name}': headers_command could not start: {exc}")
    if proc is None:
        raise HeadersCommandError(f"MCP server '{server_name}': headers_command timed out after "
                                  f"{HEADERS_COMMAND_TIMEOUT_SECONDS}s")
    if proc.returncode != 0:
        tail = redact_sensitive_text(" ".join((proc.stderr or "").split())[-300:])
        raise HeadersCommandError(f"MCP server '{server_name}': headers_command exited {proc.returncode}"
                                  + (f": {tail}" if tail else ""))
    headers = parse_headers_output(server_name, proc.stdout or "")
    for value in headers.values():
        register_vault_redaction_value(value)
        register_vault_redaction_value(value.split(" ", 1)[-1])  # the bare token of "Bearer <token>"
    return headers


def build_headers_command_auth(httpx_mod, server_name: str, command):
    """An ``httpx_mod.Auth`` that stamps the command's headers on every request and, on a 401,
    re-runs the command once and retries that request. One instance per connection attempt.
    ``minted_names`` (lowercase, live) is what the redirect stripper drops on a cross-origin hop."""

    class _HeadersCommandAuth(httpx_mod.Auth):
        def __init__(self) -> None:
            self._headers: dict[str, str] | None = None
            self._generation = 0
            self._lock = asyncio.Lock()
            self.minted_names: set[str] = set()

        async def _current(self, stale_generation: int | None = None) -> tuple[dict[str, str], int]:
            async with self._lock:
                # A concurrent 401 already refreshed past the generation this request used: reuse it.
                if self._headers is None or stale_generation == self._generation:
                    self._headers = await asyncio.to_thread(run_headers_command, server_name, command)
                    self._generation += 1
                    self.minted_names.update(name.lower() for name in self._headers)
                return self._headers, self._generation

        async def async_auth_flow(self, request):
            headers, generation = await self._current()
            request.headers.update(headers)
            response = yield request
            if response.status_code != 401:
                return
            logger.info("MCP server '%s': 401 with headers_command headers; re-running it once", server_name)
            stale = headers
            headers, _ = await self._current(stale_generation=generation)
            for name in stale:  # a credential the fresh output dropped must not ride along
                request.headers.pop(name, None)
            request.headers.update(headers)
            yield request

    return _HeadersCommandAuth()
