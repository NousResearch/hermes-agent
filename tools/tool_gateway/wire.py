"""Wire models for the tool gateway's connector routes.

Hand-written pydantic v2 models for the four routes hermes-agent consumes
(search, schemas, execute, connections), mirroring the gateway's frozen
contract. Field names are snake_case attributes with camelCase wire aliases;
requests serialize with ``model_dump(by_alias=True)``.

Rules:

- ``extra="ignore"`` everywhere: the gateway may add fields; we must not
  break when it does.
- These models never escape the wire/client layer. ``merge.py`` and above
  operate on plain dicts produced by the client.
- The ``index`` field on execute results is parsed but NEVER used for
  correlation — execute is 0-based while search is 1-based, and the safe
  rule is array position only.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Route paths, relative to the gateway origin.
CONNECTORS_PATH = "v1/connectors"
CONNECTOR_SEARCH_PATH = f"{CONNECTORS_PATH}/search"
CONNECTOR_SCHEMAS_PATH = f"{CONNECTORS_PATH}/schemas"
CONNECTOR_EXECUTE_PATH = f"{CONNECTORS_PATH}/execute"
CONNECTOR_CONNECTIONS_PATH = f"{CONNECTORS_PATH}/connections"

# The gateway refuses execute/connections batches larger than this. Unreachable
# in practice: the hermes-side dispatch cap (config.MAX_CALLS_PER_DISPATCH) is
# lower by design, so there is deliberately no chunking code.
WIRE_BATCH_MAX = 25

# Per-tool error codes inside a 200 execute envelope. A per-tool failure is a
# result, never an HTTP error; HTTP-level failures use the error envelope
# handled by errors.parse_gateway_error.
ConnectorErrorCode = Literal[
    "TOOL_NOT_ALLOWED",
    "CONNECTION_REQUIRED",
    "TOOL_NOT_FOUND",
    "PROVIDER_ERROR",
]


class _Wire(BaseModel):
    """Base for all wire models: tolerant parsing, alias-aware both ways."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# --- POST v1/connectors/search ---------------------------------------------


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
    # NOTE: 1-based vendor passthrough on this route; do not correlate on it.
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


# --- POST v1/connectors/schemas ---------------------------------------------


class ConnectorSchemasRequest(_Wire):
    tools: list[str]


class ConnectorSchemasResponse(_Wire):
    schemas: dict[str, ConnectorToolSchema] = Field(default_factory=dict)
    not_found: list[str] = Field(default_factory=list, alias="notFound")
    suggestions: dict[str, list[str]] = Field(default_factory=dict)


# --- POST v1/connectors/execute ---------------------------------------------


class ConnectorToolError(_Wire):
    code: ConnectorErrorCode
    message: str
    connector: Optional[str] = None
    connect_url: Optional[str] = Field(default=None, alias="connectUrl")
    hint: Optional[str] = None


class ConnectorExecuteCall(_Wire):
    connector: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ConnectorExecuteRequest(_Wire):
    tools: list[ConnectorExecuteCall]


class ConnectorExecuteResult(_Wire):
    # NOTE: 0-based on this route (unlike search); still never correlated on.
    index: int = 0
    connector: str = ""
    tool: str = ""
    data: Any = None
    error: Optional[ConnectorToolError] = None


class ConnectorExecuteResponse(_Wire):
    # 200 even when every tool failed; per-tool errors ride inside results.
    results: list[ConnectorExecuteResult] = Field(default_factory=list)
    success_count: int = Field(default=0, alias="successCount")
    error_count: int = Field(default=0, alias="errorCount")
    total_count: int = Field(default=0, alias="totalCount")


# --- POST v1/connectors/connections ------------------------------------------


class ConnectorConnectionsRequest(_Wire):
    connectors: list[str]
    reinitiate: bool = False


class ConnectorConnectionResult(_Wire):
    connector: str
    status: Literal["active", "initiated", "failed"]
    connect_url: Optional[str] = Field(default=None, alias="connectUrl")
    instruction: Optional[str] = None
    reinitiated: bool = False


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
