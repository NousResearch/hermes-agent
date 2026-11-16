"""``bearer_refresh_cmd`` for remote MCP servers: re-mint a static ``Authorization`` bearer on 401.

Some HTTP MCP servers sit behind a short-lived bearer (an OAuth ``client_credentials`` grant, a
vault-minted service token) that the operator puts in ``headers.Authorization`` from an env var.
The header is rendered once, so when the token expires every request answers 401. With
``bearer_refresh_cmd`` set, a 401 runs the operator's command, installs the token it prints as
``Authorization: Bearer <token>`` and retries that one request: a rotation never surfaces as a
connection failure, so it never reaches the reconnect ladder or the initial-auth park in
``mcp_tool_server_run`` (a refresh that cannot help returns the 401 unchanged, and the normal
auth-failure handling applies exactly as before).

The refresh lives in an httpx ``Auth`` for every request of the connection; the minted token is
kept on the server task (memory only, never on disk) so a later reconnect starts from it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("tools.mcp_tool")

BEARER_REFRESH_TIMEOUT_SECONDS = 10.0
BEARER_REFRESH_COOLDOWN_SECONDS = 60.0
_MIN_TOKEN_CHARS = 16
_MAX_TOKEN_CHARS = 8192
# Output a script prints when it failed but still exited 0 (or a token endpoint's error page).
_ERROR_PREFIXES = ("error", "fail", "{", "<", "[")


class BearerRefreshError(RuntimeError):
    """The command could not start, failed, timed out, or printed something that is not a token.
    Messages carry the server name and the reason, never the command's stdout."""


def bearer_refresh_argv(command) -> list[str]:
    """*command* (one string, split like a shell would but never run by one, or an argv list) as
    argv, ``~`` expanded in the executable. Raises ValueError on unbalanced quotes."""
    from hermes_cli._subprocess_compat import split_command_line

    argv = [str(part) for part in command] if isinstance(command, (list, tuple)) else split_command_line(str(command))
    return [os.path.expanduser(argv[0]), *argv[1:]] if argv else []


def parse_bearer_output(server_name: str, stdout: str) -> str:
    """The bare token the command printed, or BearerRefreshError. Rejected: empty, shorter than 16
    or longer than 8192 characters, any whitespace or non-printable/non-ASCII character inside (a
    ``Bearer `` prefix, two values smeared together, a CR/LF header injection), and output that looks
    like an error message or an HTML/JSON error body."""
    token = (stdout or "").strip()
    reason = None
    if not token:
        reason = "printed nothing"
    elif len(token) < _MIN_TOKEN_CHARS:
        reason = f"printed {len(token)} characters (a token is at least {_MIN_TOKEN_CHARS})"
    elif len(token) > _MAX_TOKEN_CHARS:
        reason = f"printed {len(token)} characters (more than {_MAX_TOKEN_CHARS})"
    elif any(not ("!" <= ch <= "~") for ch in token):
        reason = "printed whitespace or non-printable characters (print only the token, without 'Bearer ')"
    elif token.lower().startswith(_ERROR_PREFIXES):
        reason = "printed what looks like an error message, not a token"
    if reason:
        raise BearerRefreshError(f"MCP server '{server_name}': bearer_refresh_cmd {reason}; keeping the current bearer")
    return token


def run_bearer_refresh_command(server_name: str, command) -> str:
    """Run *command* without a shell, bounded by the timeout, with the serving profile's own
    secrets; return the validated token (registered for redaction before it is returned)."""
    from agent.redact import redact_sensitive_text, register_vault_redaction_value
    from hermes_cli._subprocess_compat import bounded_probe_run
    from tools.environments.local import served_profile_child_env

    try:
        argv = bearer_refresh_argv(command)
    except ValueError as exc:
        raise BearerRefreshError(f"MCP server '{server_name}': bearer_refresh_cmd cannot be parsed: {exc}") from None
    if not argv:
        raise BearerRefreshError(f"MCP server '{server_name}': bearer_refresh_cmd is empty")
    try:
        # It mints FOR the served profile: that profile's secrets, never a sibling profile's.
        env = served_profile_child_env(inherit_credentials=True)
        proc = bounded_probe_run(argv, timeout=BEARER_REFRESH_TIMEOUT_SECONDS, env=env, raise_on_spawn_failure=True)
    except Exception as exc:
        raise BearerRefreshError(f"MCP server '{server_name}': bearer_refresh_cmd could not start: "
                                 f"{type(exc).__name__}: {exc}") from None
    if proc is None:  # bounded_probe_run already tree-killed it
        raise BearerRefreshError(f"MCP server '{server_name}': bearer_refresh_cmd timed out after "
                                 f"{BEARER_REFRESH_TIMEOUT_SECONDS:.0f}s")
    if proc.returncode != 0:
        tail = redact_sensitive_text(" ".join((proc.stderr or "").split())[-200:])
        raise BearerRefreshError(f"MCP server '{server_name}': bearer_refresh_cmd exited {proc.returncode}"
                                 + (f": {tail}" if tail else ""))
    token = parse_bearer_output(server_name, proc.stdout or "")
    register_vault_redaction_value(token)
    return token


class BearerRefreshState:
    """Per-server refresh state, kept on the server task across reconnects (memory only): the
    current minted token, a generation counter so concurrent 401s share one run, and the time of
    the last attempt for the cooldown."""

    def __init__(self, command) -> None:
        self.command = command
        self.token: Optional[str] = None
        self.generation = 0
        self.last_attempt: Optional[float] = None
        self._lock = asyncio.Lock()

    async def refresh(self, server_name: str, seen_generation: int) -> Optional[str]:
        """A token newer than *seen_generation*: reuse one a concurrent 401 already minted, else
        run the command (at most once per cooldown window). None when nothing usable came out."""
        async with self._lock:
            if self.token is not None and self.generation != seen_generation:
                return self.token
            now = time.monotonic()
            if self.last_attempt is not None and now - self.last_attempt < BEARER_REFRESH_COOLDOWN_SECONDS:
                logger.debug("MCP server '%s': bearer_refresh_cmd skipped, last attempt %.0fs ago (cooldown %.0fs)",
                             server_name, now - self.last_attempt, BEARER_REFRESH_COOLDOWN_SECONDS)
                return None
            self.last_attempt = now
            try:
                token = await asyncio.to_thread(run_bearer_refresh_command, server_name, self.command)
            except BearerRefreshError as exc:
                logger.warning("%s", exc)
                return None
            self.token = token
            self.generation += 1
            logger.info("MCP server '%s': bearer refreshed via bearer_refresh_cmd", server_name)
            return token


def build_bearer_refresh_auth(httpx_mod, server_name: str, state: BearerRefreshState):
    """An ``httpx_mod.Auth`` that sends the latest minted bearer (once there is one) and, on a 401,
    refreshes through *state* and retries that request once. Without a configured Authorization
    header the command runs before the first request. A refresh that yields nothing leaves the 401
    to the caller, so the normal auth-failure handling is unchanged."""

    def _stamp(request, token: str) -> None:
        request.headers["Authorization"] = f"Bearer {token}"  # httpx headers: case-insensitive replace

    class _BearerRefreshAuth(httpx_mod.Auth):
        async def async_auth_flow(self, request):
            generation = state.generation
            token = state.token
            if token is None and "authorization" not in request.headers:
                token = await state.refresh(server_name, generation)
                generation = state.generation
            if token is not None:
                _stamp(request, token)
            response = yield request
            if response.status_code != 401:
                return
            fresh = await state.refresh(server_name, generation)
            if fresh is None:
                return
            _stamp(request, fresh)
            yield request

    return _BearerRefreshAuth()
