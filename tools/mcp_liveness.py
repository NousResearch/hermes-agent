"""Application-backed MCP liveness parsing and user-facing status descriptions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from hermes_platform import declaration
from hermes_platform.host import facts
from hermes_platform.resolver.app import AppDef, AppResolver
from hermes_platform.resolver.availability import Availability, availability
from hermes_platform.resolver.base import Effort
from hermes_platform.resolver.core import CheckState

LivenessKind = Literal["static", "server_json", "interactive_session"]
LivenessState = Literal[
    "app_not_running",
    "endpoint_unavailable",
    "no_interactive_session",
    "version_too_old",
    "missing_app",
]
Retry = Literal["after_user_action", "never_here"]


@dataclass(frozen=True)
class Liveness:
    kind: LivenessKind
    path: str = ""
    url_field: str = "http"
    token_field: str = "token"
    pid_field: str = "pid"

    def app_definition(self, definition: AppDef) -> AppDef:
        if self.kind != "server_json":
            return definition
        return replace(
            definition,
            liveness_kind="server_json",
            liveness_path=self.path,
            liveness_pid_key=self.pid_field,
            liveness_url_key=self.url_field,
            liveness_token_key=self.token_field,
        )


@dataclass(frozen=True)
class Status:
    state: LivenessState
    availability: Availability
    liveness: Liveness
    user_action: str
    retry: Retry


def parse_liveness(raw: Any) -> Liveness:
    """Parse one portable-plugin liveness declaration."""
    if not isinstance(raw, dict):
        raise ValueError("liveness must be an object")
    kind = raw.get("kind")
    if kind == "static":
        if set(raw) != {"kind"}:
            raise ValueError("static liveness only accepts 'kind'")
        return Liveness("static")
    if kind == "interactive_session":
        if set(raw) != {"kind"}:
            raise ValueError("interactive_session liveness only accepts 'kind'")
        return Liveness("interactive_session")
    if kind != "server_json":
        raise ValueError(f"unknown liveness kind: {kind!r}")
    if set(raw) - {"kind", "path", "fields"}:
        raise ValueError("server_json liveness has unknown fields")
    path = raw.get("path")
    fields = raw.get("fields")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("server_json liveness requires a non-empty path")
    if not isinstance(fields, dict) or set(fields) != {"url", "token", "pid"}:
        raise ValueError("server_json liveness fields must name url, token, and pid")
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        raise ValueError("server_json liveness field names must be non-empty strings")
    return Liveness(
        "server_json",
        path=path.strip(),
        url_field=fields["url"].strip(),
        token_field=fields["token"].strip(),
        pid_field=fields["pid"].strip(),
    )


def liveness_for(server_name: str) -> Liveness:
    """Return a server's registered liveness declaration, defaulting to static."""
    try:
        from hermes_cli.agent_plugins import liveness_for as registered_liveness
    except ImportError:
        return Liveness("static")
    raw = registered_liveness(server_name)
    return parse_liveness(raw) if raw is not None else Liveness("static")


def _action(state: LivenessState, app_name: str) -> tuple[str, Retry]:
    actions: dict[LivenessState, tuple[str, Retry]] = {
        "app_not_running": (f"Start {app_name}, then try again.", "after_user_action"),
        "endpoint_unavailable": (f"Open {app_name} and enable its local connection, then try again.", "after_user_action"),
        "no_interactive_session": (f"Open an interactive desktop session and start {app_name}, then try again.", "never_here"),
        "version_too_old": (f"Update {app_name}, then try again.", "after_user_action"),
        "missing_app": (f"Install {app_name}, then try again.", "after_user_action"),
    }
    return actions[state]


def status(server_name: str) -> Status | None:
    """Return the current unavailable state for a registered declaration."""
    decl = declaration.lookup(server_name)
    if decl is None:
        return None
    available = availability(decl)
    live = liveness_for(server_name)
    if available.state in {"missing_app", "unsupported_os"}:
        state: LivenessState = "missing_app"
    elif available.state == "version_too_old":
        state = "version_too_old"
    elif live.kind == "interactive_session" and not facts.interactive_session():
        state = "no_interactive_session"
    elif live.kind == "server_json":
        definition = decl.app_for(facts.os_family())
        if definition is None:
            state = "missing_app"
        else:
            probe = AppResolver(live.app_definition(definition)).probe(
                AppResolver(live.app_definition(definition)).locate(), effort=Effort.LOCAL
            )
            if probe.running.value is not True:
                state = "app_not_running"
            elif probe.endpoint.state is not CheckState.PRESENT:
                state = "endpoint_unavailable"
            else:
                state = "app_not_running"
    else:
        state = "app_not_running"
    action, retry = _action(state, decl.name)
    return Status(state, available, live, action, retry)


def describe(decl: declaration.Declaration, available: Availability, liveness_state: LivenessState) -> str:
    """Compose one unavailable-state sentence with one user action."""
    app_name = decl.name
    action, _retry = _action(liveness_state, app_name)
    if liveness_state == "missing_app":
        reason = f"{app_name} is not installed."
    elif liveness_state == "version_too_old":
        found = f" version {available.version}" if available.version else ""
        minimum = f"; version {available.min_version} or newer is required" if available.min_version else ""
        reason = f"{app_name}{found} is too old{minimum}."
    elif liveness_state == "no_interactive_session":
        reason = f"{app_name} needs an interactive desktop session."
    elif liveness_state == "endpoint_unavailable":
        reason = f"{app_name}'s local endpoint is unavailable."
    else:
        reason = f"{app_name} is not running."
    return f"{reason} {action}"


def unavailable_details(server_name: str) -> tuple[declaration.Declaration, Status, str] | None:
    """Return declaration, structured state, and its composed sentence."""
    decl = declaration.lookup(server_name)
    current = status(server_name)
    if decl is None or current is None:
        return None
    return decl, current, describe(decl, current.availability, current.state)
