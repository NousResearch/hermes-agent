"""Type-only JSON contracts shared by the TUI gateway and TypeScript clients.

These ``TypedDict`` definitions describe the wire payloads without changing the
runtime's deliberately lightweight plain-dict serialization.  ``ts-type``
generates the matching TypeScript definitions during frontend checks.
"""

from typing import Literal, TypedDict


class GatewayMcpServerStatusBase(TypedDict):
    connected: bool
    name: str
    tools: int
    transport: str


class GatewayMcpServerStatus(GatewayMcpServerStatusBase, total=False):
    disabled: bool
    status: str


class GatewayProjectInfoBase(TypedDict):
    id: str
    name: str
    slug: str


class GatewayProjectInfo(GatewayProjectInfoBase, total=False):
    primary_path: str | None


class GatewayUsage(TypedDict, total=False):
    calls: int
    context_max: int
    context_percent: int
    context_used: int
    cost_usd: float
    input: int
    output: int
    total: int


class GatewaySessionRuntimeInfo(TypedDict, total=False):
    approval_mode: Literal["manual", "off", "smart"]
    branch: str
    config_warning: str
    credential_warning: str
    cwd: str
    desktop_contract: int
    fast: bool
    install_warning: str
    mcp_servers: list[GatewayMcpServerStatus]
    model: str
    personality: str
    profile_name: str
    project: GatewayProjectInfo | None
    provider: str
    reasoning_effort: str
    release_date: str
    running: bool
    service_tier: str
    skills: dict[str, list[str]] | list[str]
    stored_session_id: str
    system_prompt: str
    title: str
    tools: dict[str, list[str]]
    update_behind: int | None
    update_command: str
    usage: GatewayUsage
    version: str
    yolo: bool
