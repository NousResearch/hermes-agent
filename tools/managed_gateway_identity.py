"""Caller identity for requests to the first-party gateways: who is calling, and which turn asked.

``user_agent()`` goes on every request. The correlation headers name the conversation, turn and tool
call, so they go only to connector routes and only when the profile has opted in to sending
telemetry. The gateway strips every ``x-nous-`` header before it calls a vendor.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "CallRef",
    "connector_headers",
    "correlation_headers",
    "log_connector_outcome",
    "request_surface",
    "response_request_id",
    "user_agent",
]

SESSION_HEADER = "x-nous-hermes-session"
TURN_HEADER = "x-nous-hermes-turn"
TOOL_CALL_HEADER = "x-nous-hermes-tool-call"
REQUEST_ID_HEADER = "x-request-id"

# sys.platform -> the shared-metrics OS family.
_OS_FAMILY = {"darwin": "macos", "linux": "linux", "win32": "windows"}

# The account route answers 404 with this code until the vendor has created the account.
_EXPECTED_MISSES = frozenset({"connection_not_found"})


@dataclass(frozen=True)
class CallRef:
    """The model's ids for the call a request serves. Absent fields are not sent."""

    turn_id: Optional[str] = None
    tool_call_id: Optional[str] = None


def _platform_name() -> str:
    """The session's platform: the running turn's agent platform, else the bound session vars.

    Off a turn (a desktop page action, an account operation thread) only the session vars exist.
    The classic CLI and one-shot runs bind neither, so they read as ``cli``."""
    from agent.relay_runtime import current_turn
    from gateway.session_context import get_session_env
    from tools.connectors.gateway.config import session_platform
    from utils import is_truthy_value

    turn = current_turn()
    if turn is not None and turn.lease.platform:
        return turn.lease.platform
    if platform := session_platform():
        return platform
    return "cron" if is_truthy_value(get_session_env("HERMES_CRON_SESSION", "")) else "cli"


def request_surface() -> str:
    from hermes_cli.observability.shared_metrics_contract import execution_surface

    surface = execution_surface({"execution_surface": _platform_name()})
    return "other" if surface == "unknown" else surface


def user_agent() -> str:
    from hermes_cli.version_info import get_version_info
    from hermes_platform.host.facts import os_family

    os_name = _OS_FAMILY.get(os_family(), "unknown")
    return f"hermes-agent/{get_version_info().base_version} ({request_surface()}; {os_name})"


def _send_consented() -> bool:
    from hermes_cli.config import read_raw_config_readonly
    from hermes_cli.observability.shared_metrics_send_config import resolve_send_config

    return resolve_send_config(read_raw_config_readonly() or {}).send is True


def _conversation_id() -> Optional[str]:
    from agent.portal_tags import get_conversation_context
    from gateway.session_context import get_session_env

    return get_conversation_context() or get_session_env("HERMES_SESSION_ID", "") or None


def correlation_headers(call: Optional[CallRef] = None) -> dict[str, str]:
    """``x-request-id`` (fresh per HTTP attempt) and the conversation, turn and tool call ids, or
    ``{}`` when the profile has not opted in to sending telemetry."""
    if not _send_consented():
        return {}
    headers = {REQUEST_ID_HEADER: str(uuid.uuid4())}
    ids = {
        SESSION_HEADER: _conversation_id(),
        TURN_HEADER: call.turn_id if call else None,
        TOOL_CALL_HEADER: call.tool_call_id if call else None,
    }
    headers.update({name: str(value) for name, value in ids.items() if value})
    return headers


def connector_headers(call: Optional[CallRef] = None) -> dict[str, str]:
    """Every identity header for one connector HTTP attempt. Call it once per attempt."""
    return {"User-Agent": user_agent(), **correlation_headers(call)}


def response_request_id(response: object) -> Optional[str]:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    value = headers.get(REQUEST_ID_HEADER) or headers.get("X-Request-Id")
    return str(value) if value else None


def log_connector_outcome(
    route: str, status: Optional[int], *, code: Optional[str] = None, request_id: Optional[str] = None,
    call: Optional[CallRef] = None,
) -> None:
    """One line per connector HTTP outcome: failures at WARNING, success and expected misses at DEBUG."""
    failed = status is None or not 200 <= status < 300
    level = logging.WARNING if failed and code not in _EXPECTED_MISSES else logging.DEBUG
    logger.log(
        level, "connector %s %s %s request_id=%s dispatch=%s",
        route, status if status is not None else "transport", code or "-", request_id or "-",
        (call.tool_call_id if call else None) or "-",
    )
