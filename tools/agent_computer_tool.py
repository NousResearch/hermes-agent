"""Opt-in Agent Computer tools.

Ordinary chat does not load this toolset. Enabling it still cannot start
a computer until the agent explicitly calls ``computer_ensure`` /
``computer_wake``. Takeover remains owner-only on the authenticated
gateway contract.
"""

from __future__ import annotations

import json
import os
from typing import Any

from gateway.agent_computer import get_contract
from gateway.agent_computer.contract import agent_from_profile, error_payload
from gateway.agent_computer.errors import AgentComputerError
from tools.registry import registry, tool_error


def _session_profile() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _profile_id(explicit: str | None = None) -> str:
    """Authorization uses the process/session profile only.

    A model-supplied profile_id that names a different permanent agent is
    rejected rather than trusted.
    """
    session = _session_profile()
    if explicit and explicit.strip() and explicit.strip() != session:
        raise AgentComputerError("profile_id does not match this session profile")
    return session


def _principal(profile_id: str) -> str:
    return agent_from_profile(profile_id)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def computer_ensure(profile_id: str = "") -> str:
    try:
        pid = _profile_id(profile_id)
        return _dump(get_contract().ensure(pid, _principal(pid)))
    except AgentComputerError as exc:
        return tool_error(error_payload(exc)["message"])


def computer_status(computer_id: str = "", profile_id: str = "") -> str:
    try:
        pid = _profile_id(profile_id)
        contract = get_contract()
        if not computer_id:
            return _dump(contract.ensure(pid, _principal(pid)))
        return _dump(contract.status(computer_id, _principal(pid)))
    except AgentComputerError as exc:
        return tool_error(error_payload(exc)["message"])


def computer_wake(computer_id: str, profile_id: str = "") -> str:
    try:
        pid = _profile_id(profile_id)
        return _dump(get_contract().wake(computer_id, _principal(pid)))
    except AgentComputerError as exc:
        return tool_error(error_payload(exc)["message"])


def computer_observe(
    computer_id: str,
    lease_id: str,
    fencing_epoch: int,
    profile_id: str = "",
) -> str:
    try:
        pid = _profile_id(profile_id)
        return _dump(
            get_contract().observe(
                computer_id,
                _principal(pid),
                lease_id=lease_id,
                fencing_epoch=int(fencing_epoch),
            )
        )
    except AgentComputerError as exc:
        return tool_error(error_payload(exc)["message"])


def computer_act(
    computer_id: str,
    lease_id: str,
    fencing_epoch: int,
    kind: str,
    target: str = "",
    text: str = "",
    action_class: str = "",
    profile_id: str = "",
    x: float | None = None,
    y: float | None = None,
    key: str = "",
    code: str = "",
    delta_x: float = 0,
    delta_y: float = 0,
) -> str:
    try:
        pid = _profile_id(profile_id)
        return _dump(
            get_contract().act(
                computer_id,
                _principal(pid),
                lease_id=lease_id,
                fencing_epoch=int(fencing_epoch),
                kind=kind,
                target=target,
                text=text,
                action_class=action_class,
                x=x,
                y=y,
                key=key,
                code=code,
                delta_x=delta_x,
                delta_y=delta_y,
            )
        )
    except AgentComputerError as exc:
        return tool_error(error_payload(exc)["message"])


def check_agent_computer_requirements() -> bool:
    # Opt-in toolset only. Presence in schema still requires the toolset to
    # be enabled; this check keeps it off accidental core inclusion.
    return os.environ.get("HERMES_AGENT_COMPUTER_TOOLS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_ENSURE_SCHEMA = {
    "name": "computer_ensure",
    "description": (
        "Ensure the durable AgentComputer for this permanent Hermes profile. "
        "Does not launch a browser by itself."
    ),
    "parameters": {
        "type": "object",
        "properties": {"profile_id": {"type": "string"}},
        "required": [],
    },
}

_STATUS_SCHEMA = {
    "name": "computer_status",
    "description": "Read AgentComputer lifecycle, control authority, and workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "computer_id": {"type": "string"},
            "profile_id": {"type": "string"},
        },
        "required": [],
    },
}

_WAKE_SCHEMA = {
    "name": "computer_wake",
    "description": "Wake the AgentComputer runtime and receive a fenced agent lease.",
    "parameters": {
        "type": "object",
        "properties": {
            "computer_id": {"type": "string"},
            "profile_id": {"type": "string"},
        },
        "required": ["computer_id"],
    },
}

_OBSERVE_SCHEMA = {
    "name": "computer_observe",
    "description": (
        "Observe the current live environment. Required after wake and after "
        "the owner returns control. Never guess hidden page state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "computer_id": {"type": "string"},
            "lease_id": {"type": "string"},
            "fencing_epoch": {"type": "integer"},
            "profile_id": {"type": "string"},
        },
        "required": ["computer_id", "lease_id", "fencing_epoch"],
    },
}

_ACT_SCHEMA = {
    "name": "computer_act",
    "description": (
        "Apply one fenced input (navigate/click/type or pointer/text/key/scroll) "
        "to the attached BrowserIdentity. Stale leases are rejected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "computer_id": {"type": "string"},
            "lease_id": {"type": "string"},
            "fencing_epoch": {"type": "integer"},
            "kind": {"type": "string"},
            "target": {"type": "string"},
            "text": {"type": "string"},
            "action_class": {"type": "string"},
            "x": {"type": "number"},
            "y": {"type": "number"},
            "key": {"type": "string"},
            "code": {"type": "string"},
            "delta_x": {"type": "number"},
            "delta_y": {"type": "number"},
            "profile_id": {"type": "string"},
        },
        "required": ["computer_id", "lease_id", "fencing_epoch", "kind"],
    },
}

registry.register(
    name="computer_ensure",
    toolset="agent_computer",
    schema=_ENSURE_SCHEMA,
    handler=lambda args, **kw: computer_ensure(args.get("profile_id") or ""),
    check_fn=check_agent_computer_requirements,
    emoji="💻",
)
registry.register(
    name="computer_status",
    toolset="agent_computer",
    schema=_STATUS_SCHEMA,
    handler=lambda args, **kw: computer_status(
        args.get("computer_id") or "", args.get("profile_id") or ""
    ),
    check_fn=check_agent_computer_requirements,
    emoji="💻",
)
registry.register(
    name="computer_wake",
    toolset="agent_computer",
    schema=_WAKE_SCHEMA,
    handler=lambda args, **kw: computer_wake(
        args.get("computer_id") or "", args.get("profile_id") or ""
    ),
    check_fn=check_agent_computer_requirements,
    emoji="💻",
)
registry.register(
    name="computer_observe",
    toolset="agent_computer",
    schema=_OBSERVE_SCHEMA,
    handler=lambda args, **kw: computer_observe(
        args.get("computer_id") or "",
        args.get("lease_id") or "",
        int(args.get("fencing_epoch") or 0),
        args.get("profile_id") or "",
    ),
    check_fn=check_agent_computer_requirements,
    emoji="💻",
)
registry.register(
    name="computer_act",
    toolset="agent_computer",
    schema=_ACT_SCHEMA,
    handler=lambda args, **kw: computer_act(
        args.get("computer_id") or "",
        args.get("lease_id") or "",
        int(args.get("fencing_epoch") or 0),
        args.get("kind") or "",
        args.get("target") or "",
        args.get("text") or "",
        args.get("action_class") or "",
        args.get("profile_id") or "",
        args.get("x"),
        args.get("y"),
        args.get("key") or "",
        args.get("code") or "",
        float(args.get("delta_x") or 0),
        float(args.get("delta_y") or 0),
    ),
    check_fn=check_agent_computer_requirements,
    emoji="💻",
)
