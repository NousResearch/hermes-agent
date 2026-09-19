"""Contracts for connector catalog, authorization, and tool-list reads."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .base import Result, WireEnum
from .common import ConnectorOwner, OpenModel, ProfileParams
from .connectors_operation import ConnectionOperationStatus
from .registry import method

ConnectorSlug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]


class ConnectorErrorReason(WireEnum):
    invalid_params = "INVALID_PARAMS"
    not_owner = "NOT_OWNER"
    unsupported_runtime = "UNSUPPORTED_RUNTIME"
    connector_request_failed = "CONNECTOR_REQUEST_FAILED"
    invalid_connector_response = "INVALID_CONNECTOR_RESPONSE"
    unknown_target = "UNKNOWN_TARGET"
    link_still_valid = "LINK_STILL_VALID"
    reissue_refused = "REISSUE_REFUSED"
    unknown_operation = "UNKNOWN_OPERATION"
    invalid_answer = "INVALID_ANSWER"
    needs_nous_auth = "NEEDS_NOUS_AUTH"
    connector_not_found = "CONNECTOR_NOT_FOUND"
    tools_unavailable = "TOOLS_UNAVAILABLE"
    connectors_unavailable = "CONNECTORS_UNAVAILABLE"
    catalog_unavailable = "CATALOG_UNAVAILABLE"
    accounts_unavailable = "ACCOUNTS_UNAVAILABLE"
    connection_not_found = "CONNECTION_NOT_FOUND"
    policy_unavailable = "POLICY_UNAVAILABLE"
    policy_conflict = "POLICY_CONFLICT"
    forbidden_scope = "FORBIDDEN_SCOPE"
    invalid_policy = "INVALID_POLICY"


class ConnectorsListParams(ProfileParams):
    owner: ConnectorOwner


class ConnectorRow(OpenModel):
    """One ``manage_connections`` status entry after ``connector_ui_payload`` redaction; the
    connector service owns the closed key set, so unknown metadata passes through."""

    connector: str = ""
    connected: bool | None = None
    enabled: bool | None = None
    connectionStatus: str | None = None
    name: str | None = None
    description: str | None = None


class ConnectorsListResult(Result):
    available: bool
    connectors: list[ConnectorRow]


method(
    "connectors.list",
    params=ConnectorsListParams,
    result=ConnectorsListResult,
    doc="Connector catalog + connection state for one session or profile account owner.",
)


class ConnectorsConnectParams(ProfileParams):
    owner: ConnectorOwner
    connectors: list[ConnectorSlug] = Field(min_length=1)
    reconnect: bool = False


class ConnectorsConnectResult(ConnectionOperationStatus):
    """``methods_connectors._reissue`` / ``managed._off_desktop_result``: the operation the connect opened; ``status``/``note`` ride along from the tool result."""

    status: str | None = None
    note: str | None = None


method(
    "connectors.connect",
    params=ConnectorsConnectParams,
    result=ConnectorsConnectResult,
    doc="Start or re-initiate authorization for named connectors on a session or account operation.",
)


class ConnectorToolsParams(ProfileParams):
    slug: str
    refresh: bool = False


class ConnectorToolFacet(WireEnum):
    read = "read"
    write = "write"
    destructive = "destructive"
    unclassified = "unclassified"


class ConnectorToolRow(Result):
    model_config = Result.model_config | {"from_attributes": True}

    slug: str
    name: str
    description: str
    facet: ConnectorToolFacet
    hints: list[str]
    categories: list[str]
    deprecated: bool


class ConnectorToolsSource(WireEnum):
    cache = "cache"
    network = "network"
    revalidated = "revalidated"


class ConnectorToolsResult(Result):
    model_config = Result.model_config | {"from_attributes": True}

    connector: str
    toolkit_version: str
    etag: str
    fetched_at: float
    source: ConnectorToolsSource
    stale: bool
    tools: list[ConnectorToolRow]


method(
    "connectors.tools",
    params=ConnectorToolsParams,
    result=ConnectorToolsResult,
    doc="The scoped profile's cached or current tool list for one connector.",
)


class ConnectorCatalogRow(Result):
    model_config = Result.model_config | {"from_attributes": True}

    slug: str
    name: str
    description: str
    category: str
    logo_url: str | None = None


class ConnectorsCatalogResult(Result):
    model_config = Result.model_config | {"from_attributes": True}

    connectors: list[ConnectorCatalogRow]


method(
    "connectors.catalog",
    params=ProfileParams,
    result=ConnectorsCatalogResult,
    doc="The hosted connector catalog available to the scoped member.",
)


class ConnectorAccountStatus(WireEnum):
    pending = "pending"
    active = "active"
    failed = "failed"
    expired = "expired"
    revoked = "revoked"
    inactive = "inactive"


class ConnectorAccountsParams(ProfileParams):
    connector: ConnectorSlug | None = None


class ConnectorAccountRow(Result):
    connection_id: str
    connector: str
    status: ConnectorAccountStatus
    status_reason: str | None = None
    label: str
    alias: str | None = None
    active: bool
    created_at: str
    updated_at: str


class ConnectorAccountsResult(Result):
    accounts: list[ConnectorAccountRow]


method(
    "connectors.accounts",
    params=ConnectorAccountsParams,
    result=ConnectorAccountsResult,
    doc="The scoped member's hosted connector accounts, optionally filtered by connector slug.",
)


class ConnectorAccountsRemoveParams(ProfileParams):
    connection_id: str = Field(min_length=1)


class ConnectorAccountsRemoveResult(Result):
    connection_id: str
    status: Literal["removed"]


method(
    "connectors.accounts.remove",
    params=ConnectorAccountsRemoveParams,
    result=ConnectorAccountsRemoveResult,
    doc="Remove one hosted connector account owned by the scoped member.",
)


class ConnectorPolicyLayerKind(WireEnum):
    org = "org"
    role = "role"
    member = "member"


class ConnectorPolicyTags(Result):
    enable: list[str] | None = None
    disable: list[str] | None = None


class ConnectorPolicyUnrestrictedBody(Result):
    mode: Literal["unrestricted"]


class ConnectorPolicyDenyAllBody(Result):
    mode: Literal["deny-all"]


class ConnectorPolicyAllowBody(Result):
    mode: Literal["allow"]
    connectors: list[str]
    tools: dict[str, list[str]]
    tags: ConnectorPolicyTags | None = None


class ConnectorPolicyDenyBody(Result):
    mode: Literal["deny"]
    disabled_connectors: list[str]
    tools: dict[str, list[str]]
    tags: ConnectorPolicyTags | None = None


ConnectorPolicyBody = (
    ConnectorPolicyUnrestrictedBody
    | ConnectorPolicyDenyAllBody
    | ConnectorPolicyAllowBody
    | ConnectorPolicyDenyBody
)


class ConnectorPolicyLayer(Result):
    kind: ConnectorPolicyLayerKind
    revision: str
    body: ConnectorPolicyBody = Field(discriminator="mode")


class ConnectorPolicyGetResult(Result):
    layers: list[ConnectorPolicyLayer]


method(
    "connectors.policy.get",
    params=ProfileParams,
    result=ConnectorPolicyGetResult,
    doc="Policy layers for the scoped member, from organization to member scope.",
)


class ToolsChange(Result):
    type: Literal["tools"]
    connector: str
    disabled_tools: list[str] = Field(max_length=2000)


class ConnectorChange(Result):
    type: Literal["connector"]
    connector: str
    enabled: bool


ConnectorPolicyChange = ToolsChange | ConnectorChange


class ConnectorPolicySetParams(ProfileParams):
    change: ConnectorPolicyChange = Field(discriminator="type")
    expected_revision: str | None = None


class ConnectorPolicySetResult(Result):
    revision: str


method(
    "connectors.policy.set",
    params=ConnectorPolicySetParams,
    result=ConnectorPolicySetResult,
    doc="Apply one scoped member connector or tool-list policy change.",
)
