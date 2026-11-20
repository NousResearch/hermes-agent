"""One connector tool call made outside a turn: ``hermes connectors call`` for cron gate scripts.

A script runs unattended, so a call is read-only unless the caller opts in: the tool's ``facet`` in
the Portal tools listing (the gateway's own read/write/destructive classification) must be ``read``;
any other facet, an unknown one included, needs ``allow_write``. The account is picked the way the
model picks it, with the ``connector_alias`` argument the gateway resolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

from tools.connectors.gateway.names import format_connector_name, parse_connector_name

__all__ = ["CallExit", "CallOutcome", "call_connector_tool", "resolve_tool"]

_ACCOUNT_ARGUMENT = "connector_alias"


class CallExit(IntEnum):
    ok = 0
    bad_args = 2
    unavailable = 3
    tool_error = 4
    needs_allow_write = 5


@dataclass(frozen=True)
class CallOutcome:
    exit: CallExit
    # The tool's response on success, else an error document ``{"code", "message", ...}``.
    document: Any


@dataclass(frozen=True)
class ToolRequest:
    app: str
    tool: str


def resolve_tool(spec: str) -> Optional[ToolRequest]:
    """``connectors__<app>__<tool>`` or ``<app>.<tool>``; ``None`` when it is neither."""
    parsed = parse_connector_name(spec)
    if parsed is not None:
        return ToolRequest(parsed.connector.lower(), parsed.tool)
    app, dot, tool = spec.partition(".")
    if not dot or not app.strip() or not tool.strip() or "." in tool:
        return None
    return ToolRequest(app.strip().lower(), tool.strip())


def _listed_tool(request: ToolRequest, tools: list[Any]) -> Optional[Any]:
    """The listing entry the request names: the app-prefixed vendor slug first, like execute does."""
    by_slug = {str(t.slug).upper(): t for t in tools}
    wanted = request.tool.upper()
    return by_slug.get(f"{request.app.upper()}_{wanted}") or by_slug.get(wanted)


def _error(exit_code: CallExit, code: str, message: str) -> CallOutcome:
    return CallOutcome(exit_code, {"code": code, "message": message})


def call_connector_tool(
    spec: str, arguments: dict[str, Any], *, account: Optional[str] = None, allow_write: bool = False,
) -> CallOutcome:
    """Run one connector tool; a non-read tool without ``allow_write`` never reaches the gateway."""
    from tools.connectors.gateway.bridge import SIGN_IN_FAILED_CODE, run_remote
    from tools.connectors.gateway.errors import GatewayAuthError, ToolGatewayError
    from tools.connectors.gateway.merge import partition_calls
    from tools.connectors.portal.client import PortalConnectorClient
    from tools.connectors.portal.errors import InvalidConnectorSlug
    from tools.connectors.portal.tools_cache import read_tools

    request = resolve_tool(spec)
    if request is None:
        return _error(CallExit.bad_args, "BAD_TOOL_NAME",
                      f"{spec!r} is not a connector tool. Use <app>.<tool> or connectors__<app>__<tool>.")
    try:
        listing = read_tools(request.app, client=PortalConnectorClient())
    except InvalidConnectorSlug:
        return _error(CallExit.bad_args, "BAD_TOOL_NAME", f"{request.app!r} is not a connector name.")
    except GatewayAuthError:
        return _error(CallExit.unavailable, "SIGNED_OUT", "Run `hermes auth upgrade` to sign in.")
    except ToolGatewayError as exc:
        return _error(CallExit.unavailable, "TOOLS_UNAVAILABLE", f"Could not list {request.app} tools: {exc}")
    listed = _listed_tool(request, listing.tools)
    if listed is None:
        return _error(CallExit.bad_args, "TOOL_NOT_FOUND", f"{request.app} has no tool {request.tool!r}.")
    if listed.facet != "read" and not allow_write:
        return _error(CallExit.needs_allow_write, "NEEDS_ALLOW_WRITE",
                      f"{listed.slug} is classified {listed.facet!r}, not read-only; pass --allow-write to run it.")
    if account:
        arguments = {**arguments, _ACCOUNT_ARGUMENT: account}
    name = format_connector_name(request.app, str(listed.slug))
    partition = partition_calls([{"name": name, "arguments": arguments}])
    entry = run_remote(partition.remote, None, availability=None, client_factory=None)[0]
    if "error" in entry:
        error = entry["error"] if isinstance(entry["error"], dict) else {"message": str(entry["error"])}
        signed_out = error.get("code") == SIGN_IN_FAILED_CODE
        return CallOutcome(CallExit.unavailable if signed_out else CallExit.tool_error, error)
    return CallOutcome(CallExit.ok, entry.get("response"))
