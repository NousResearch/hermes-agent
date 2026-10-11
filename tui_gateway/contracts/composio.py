"""Composio hosted-MCP provider cards (account control only; app tools stay on native MCP)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Params, Result, WireEnum
from .common import ProfileParams
from .registry import method


class ComposioProviderStatus(WireEnum):
    connected = "connected"
    disconnected = "disconnected"
    pending = "pending"
    failed = "failed"
    expired = "expired"
    revoked = "revoked"
    inactive = "inactive"
    requires_mcp_auth = "requires_mcp_auth"
    unavailable = "unavailable"


class ComposioAccountStatus(WireEnum):
    active = "active"
    pending = "pending"
    failed = "failed"
    expired = "expired"
    revoked = "revoked"
    inactive = "inactive"
    unknown = "unknown"


class ComposioCatalogParams(ProfileParams):
    pass


class ComposioProviderRow(Result):
    provider: str
    label: str
    toolkit: str


class ComposioCatalogResult(Result):
    providers: list[ComposioProviderRow]


method(
    "composio.catalog",
    params=ComposioCatalogParams,
    result=ComposioCatalogResult,
    doc="The verified provider cards supported by the hosted Composio MCP server.",
)


class ComposioStatusParams(ProfileParams):
    provider: str | None = None


class ComposioAccountRow(Result):
    id: str
    alias: str | None = None
    status: ComposioAccountStatus
    is_default: bool = False
    account_type: str | None = None


class ComposioProviderState(Result):
    provider: str
    toolkit: str
    status: ComposioProviderStatus
    accounts: list[ComposioAccountRow] = Field(default_factory=list)


class ComposioStatusResult(Result):
    providers: list[ComposioProviderState]


method(
    "composio.status",
    params=ComposioStatusParams,
    result=ComposioStatusResult,
    doc="Read provider account status through Composio's allowlisted connection-management tools.",
)


class ComposioConnectParams(ProfileParams):
    provider: str
    alias: str | None = Field(default=None, min_length=1, max_length=120)


class ComposioConnectResult(Result):
    provider: str
    toolkit: str
    status: ComposioProviderStatus
    auth_url: str | None = None
    accounts: list[ComposioAccountRow] = Field(default_factory=list)


method(
    "composio.connect",
    params=ComposioConnectParams,
    result=ComposioConnectResult,
    doc="Start a new, user-requested provider account connection and return its Composio auth link.",
)


class ComposioRemoveParams(ProfileParams):
    provider: str
    account_id: str = Field(min_length=1)


class ComposioRemoveResult(Result):
    provider: str
    toolkit: str
    account_id: str
    status: Literal["removed", "not_found", "requires_mcp_auth"]


method(
    "composio.remove",
    params=ComposioRemoveParams,
    result=ComposioRemoveResult,
    doc="Remove only the exact account selected by the user from its provider toolkit.",
)
