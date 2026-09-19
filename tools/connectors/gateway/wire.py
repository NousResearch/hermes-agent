"""Wire models for the gateway connector routes.

Tolerate added gateway fields; callers receive plain dictionaries. Correlate results by position, never wire ``index``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONNECTORS_PATH = "v1/connectors"
CONNECTOR_SEARCH_PATH = f"{CONNECTORS_PATH}/search"
CONNECTOR_SCHEMAS_PATH = f"{CONNECTORS_PATH}/schemas"
CONNECTOR_EXECUTE_PATH = f"{CONNECTORS_PATH}/execute"
CONNECTOR_CONNECTIONS_PATH = f"{CONNECTORS_PATH}/connections"
CONNECTOR_ACCOUNTS_PATH = f"{CONNECTORS_PATH}/accounts"

# The gateway's six-state account status (contract `CONNECTOR_CONNECTION_STATUSES`; the vendor's
# INITIALIZING and INITIATED both arrive as `pending`). Present only once the session binds an
# account; a value outside this set is a contract break and fails validation.
ConnectionStatus = Literal["pending", "active", "failed", "expired", "revoked", "inactive"]
# Where the vendor's done page sends the browser after consent; the dev desktop registers hermes-dev://.
ConnectorReturnTarget = Literal["hermes-desktop", "hermes-desktop-dev", "portal"]

# Hermes dispatch caps batches lower, so client-side chunking is deliberately absent.
WIRE_BATCH_MAX = 25

# A 200 execute envelope carries per-tool failures as results, not HTTP errors.
ConnectorErrorCode = Literal[
    "TOOL_NOT_ALLOWED",
    "CONNECTION_REQUIRED",
    "TOOL_NOT_FOUND",
    "PROVIDER_ERROR",
]


class _Wire(BaseModel):

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ConnectorSearchQuery(_Wire):
    use_case: str = Field(alias="useCase")
    known_fields: Optional[str] = Field(default=None, alias="knownFields")


class ConnectorSearchRequest(_Wire):
    queries: list[ConnectorSearchQuery]


class ConnectorToolSchema(_Wire):
    connector: str
    tool: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="inputSchema")


class ConnectorSearchResult(_Wire):
    # Vendor passthrough; result correlation uses array position.
    index: int
    use_case: str = Field(default="", alias="useCase")
    tools: list[str] = Field(default_factory=list)
    related_tools: list[str] = Field(default_factory=list, alias="relatedTools")
    connectors: list[str] = Field(default_factory=list)
    guidance: Optional[str] = None
    plan_steps: Optional[list[str]] = Field(default=None, alias="planSteps")
    pitfalls: Optional[list[str]] = None
    error: Optional[str] = None


class ConnectorConnectionStatus(_Wire):
    connector: str
    connected: bool = False
    description: str = ""


class ConnectorSearchResponse(_Wire):
    results: list[ConnectorSearchResult] = Field(default_factory=list)
    schemas: dict[str, ConnectorToolSchema] = Field(default_factory=dict)
    connections: list[ConnectorConnectionStatus] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list, alias="nextSteps")


class ConnectorSchemasRequest(_Wire):
    tools: list[str]


class ConnectorSchemasResponse(_Wire):
    schemas: dict[str, ConnectorToolSchema] = Field(default_factory=dict)
    not_found: list[str] = Field(default_factory=list, alias="notFound")
    suggestions: dict[str, list[str]] = Field(default_factory=dict)


class ConnectorToolError(_Wire):
    code: ConnectorErrorCode
    message: str
    connector: Optional[str] = None
    connect_url: Optional[str] = Field(default=None, alias="connectUrl")
    # The account the link was minted for; absent when the link mint failed. Absent means nothing to watch.
    connection_id: Optional[str] = Field(default=None, alias="connectionId")
    hint: Optional[str] = None


class ConnectorExecuteCall(_Wire):
    connector: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # On the wire for the multi-account switch; never sent by hermes today, because the vendor answers
    # 400 to any value while multi-account is off (contract probe F2).
    account: Optional[str] = None


class ConnectorExecuteRequest(_Wire):
    tools: list[ConnectorExecuteCall]
    # Ride any CONNECTION_REQUIRED link the call mints back to the surface that asked.
    return_to: Optional[ConnectorReturnTarget] = Field(default=None, alias="returnTo")
    op: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ConnectorExecuteResult(_Wire):
    # Never correlate results by wire index.
    index: int = 0
    connector: str = ""
    tool: str = ""
    data: Any = None
    error: Optional[ConnectorToolError] = None


class ConnectorExecuteResponse(_Wire):
    results: list[ConnectorExecuteResult] = Field(default_factory=list)
    success_count: int = Field(default=0, alias="successCount")
    error_count: int = Field(default=0, alias="errorCount")
    total_count: int = Field(default=0, alias="totalCount")


class ConnectorConnectionsRequest(_Wire):
    connectors: list[str]
    reinitiate: bool = False
    alias: Optional[str] = None
    return_to: Optional[ConnectorReturnTarget] = Field(default=None, alias="returnTo")
    # The caller's operation id, echoed on the hermes://connections/done link.
    op: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ConnectorConnectionResult(_Wire):
    connector: str
    status: Literal["active", "initiated", "failed"]
    connect_url: Optional[str] = Field(default=None, alias="connectUrl")
    # The vendor account the mint created or observed. Optional by vendor semantics (a no-auth toolkit
    # answers active with none); a target without one has nothing to watch.
    connection_id: Optional[str] = Field(default=None, alias="connectionId")
    alias: Optional[str] = None
    instruction: Optional[str] = None
    # Vendor error_message on ``failed``; the list route never carries it.
    status_reason: Optional[str] = Field(default=None, alias="statusReason")
    reinitiated: bool = False


class ConnectorListItem(_Wire):
    connector: str
    enabled: bool = True
    connected: bool = False
    connection_status: Optional[ConnectionStatus] = Field(default=None, alias="connectionStatus")
    status_reason: Optional[str] = Field(default=None, alias="statusReason")
    disabled_tools: list[str] = Field(default_factory=list, alias="disabledTools")


class ConnectorListResponse(_Wire):
    items: list[ConnectorListItem]
    next_cursor: Optional[str] = Field(alias="nextCursor")


class ConnectorAccount(_Wire):
    connection_id: str = Field(alias="connectionId")
    connector: str
    status: ConnectionStatus
    status_reason: Optional[str] = Field(default=None, alias="statusReason")
    label: str = Field(min_length=1)
    alias: Optional[str] = None
    # The newest active account for this connector: the one the vendor executes with.
    active: bool
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")


class ConnectorAccountsResponse(_Wire):
    accounts: list[ConnectorAccount]


class RemovedConnectorAccount(_Wire):
    connection_id: str = Field(alias="connectionId")
    status: Literal["removed"]


class ConnectorConnectionsSummary(_Wire):
    total: int = 0
    active: int = 0
    initiated: int = 0
    failed: int = 0


class ConnectorConnectionsResponse(_Wire):
    results: list[ConnectorConnectionResult] = Field(default_factory=list)
    summary: ConnectorConnectionsSummary = Field(
        default_factory=ConnectorConnectionsSummary
    )
