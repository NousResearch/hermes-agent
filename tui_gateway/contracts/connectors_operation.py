"""The connection operation (``manage_connections`` card): the request event that opens a card,
the update frames that drive it, and the RPCs the card answers through.

Shapes are typed from ``tools/connectors/operation.py`` (``Target.snapshot``,
``ConnectionOperation.request_payload`` / ``_snapshot_locked``) and
``tui_gateway/methods_connectors.py`` (``_operation_view``, ``_connection_update``). The card is a
projection: every frame carries the full target snapshot, and the renderer never derives state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hermes_cli.plugin_install_phase import InstallPhase

from .base import Params, Payload, Result, WireEnum
from .common import ConnectorOwner, ProfileParams
from .registry import event, method


class ConnectionTargetKind(WireEnum):
    connector = "connector"
    mcp = "mcp"
    # ``manage_catalog`` rows: a catalog plugin (may bring MCP tools and/or skills) or a skill.
    plugin = "plugin"
    skill = "skill"


class ConnectionTargetAction(WireEnum):
    authorize = "authorize"
    connect = "connect"
    enable = "enable"
    install = "install"
    reconnect = "reconnect"


class ConnectionTargetState(WireEnum):
    """``tools/connectors/contract.py::TargetState``."""

    pending = "pending"
    initiated = "initiated"
    connected = "connected"
    skipped = "skipped"
    failed = "failed"
    expired = "expired"
    not_connected = "not_connected"


class ConnectionActor(WireEnum):
    """``tools/connectors/contract.py::Actor``."""

    user = "user"
    backend_watcher = "backend_watcher"
    clock = "clock"


class ConnectionSettleReason(WireEnum):
    """``tools/connectors/contract.py::SettleReason``."""

    all_resolved = "all_resolved"
    continue_ = "continue"
    deadline = "deadline"
    interrupt = "interrupt"


class ConnectionTargetEnvField(Payload):
    """One credential an MCP install still needs; the card renders a field per entry and sends the
    values back with the approval."""

    name: str
    required: bool
    secret: bool
    default: str
    prompt: str | None = None


class CatalogTier(WireEnum):
    official = "official"
    community = "community"


class CatalogAppState(WireEnum):
    """The desktop app a catalog plugin drives, from its ``hermes_platform`` declaration."""

    present = "present"
    missing_app = "missing_app"
    app_not_running = "app_not_running"
    unknown = "unknown"


class CatalogScanStatus(WireEnum):
    passed = "passed"
    warnings = "warnings"
    failed = "failed"


class CatalogScan(Payload):
    """The catalog's security scan of the pinned commit; read-only on the card."""

    status: CatalogScanStatus
    summary: str


class CatalogApproved(Payload):
    """The non-secret Advanced choices the user approved on a catalog row; a Try again after the
    operation settled repeats them."""

    force: bool
    enable: bool
    ref: str | None = None


class CatalogServerError(Payload):
    """An MCP server an installed plugin brought that did not connect, with the raw reason."""

    name: str
    error: str


class ConnectionOperationTarget(Payload):
    """``Target.snapshot``: the link minted up front rides here, never in the model result. ``extra``
    keys a leg records (``tools``, ``hint``) are typed here as they appear."""

    name: str
    kind: ConnectionTargetKind
    action: ConnectionTargetAction
    state: ConnectionTargetState
    # ``Target.resolved``: the row needs nothing more from the user (a failed catalog row counts).
    resolved: bool | None = None
    detail: str | None = None
    instructions: str | None = None
    discovery_error: str | None = None
    connect_url: str | None = None
    # The vendor account a managed mint created or observed; never the desktop transport's id.
    connection_id: str | None = None
    # The name of the one hosted account this row connects or repairs.
    alias: str | None = None
    attempt: str | None = None
    # Present only on an MCP install that is waiting for credentials.
    required_env: list[ConnectionTargetEnvField] | None = None
    tools: list[str] | None = None
    hint: str | None = None
    # Catalog rows (kind ``plugin`` / ``skill``) only; field set agreed in CATALOG-ROW-CONTRACT.md.
    display: str | None = None
    description: str | None = None
    tier: CatalogTier | None = None
    platforms: list[str] | None = None
    repo: str | None = None
    sha: str | None = None
    subdir: str | None = None
    scan: CatalogScan | None = None
    # The Hermes version range the plugin needs; its env vars ride in ``required_env``.
    requires_hermes: str | None = None
    has_desktop_half: bool | None = None
    # The profile the row installs into; absent when the chat's home is no named profile.
    target_profile: str | None = None
    app_state: CatalogAppState | None = None
    # On an installed skill row: the qualified skill name the model can now load.
    skill: str | None = None
    phase: InstallPhase | None = None
    approved: CatalogApproved | None = None
    # Facts on an installed row, drawn by the card (``detail`` words them for the model).
    enabled: bool | None = None
    missing_env: list[str] | None = None
    server_errors: list[CatalogServerError] | None = None
    already_installed: bool | None = None


class ConnectionRequestPayload(Payload):
    """``ConnectionOperation.request_payload``: opens the card; also the ``pending_connection`` resume
    snapshot so a client that missed the event restores the card with the server's deadline."""

    op_id: str
    # Monotonic write counter for the operation; a frame whose seq is not higher than the one the
    # renderer holds is older and moves no row.
    seq: int
    deadline_at: float
    timeout_seconds: float
    targets: list[ConnectionOperationTarget]
    # The model's id for the call that opened the operation; the card binds to that tool row only.
    tool_call_id: str | None = None


event("connection.request", ConnectionRequestPayload,
      doc="A connection operation opened on this session; the desktop renders its card.")


class ConnectionOperationStatus(Result):
    """``methods_connectors._operation_view``: the operation's full snapshot."""

    op_id: str
    seq: int
    deadline_at: float
    settled: bool
    settled_at: float | None = None
    settled_by: ConnectionSettleReason | None = None
    targets: list[ConnectionOperationTarget]


class ConnectionUpdatePayload(ConnectionOperationStatus, Payload):
    """``methods_connectors._connection_update``: one target transition (``target``/``from``/``to``/
    ``actor``) or the settlement (none of those), with the full snapshot."""

    owner: ConnectorOwner
    target: str | None = None
    from_: ConnectionTargetState | None = Field(default=None, alias="from")  # ``from`` is a keyword
    to: ConnectionTargetState | None = None
    actor: ConnectionActor | None = None
    detail: str | None = None

event("connection.update", ConnectionUpdatePayload,
      doc="One transition or the settlement of an open connection operation.")


class ConnectionOperationParams(ProfileParams):
    owner: ConnectorOwner
    op_id: str


method("connectors.operation.status", params=ConnectionOperationParams, result=ConnectionOperationStatus,
       doc="The current snapshot of one open session or account operation.")


class ConnectionWakeResult(Result):
    status: Literal["ok"]


method("connectors.operation.wake", params=ConnectionOperationParams, result=ConnectionWakeResult,
       doc="The browser leg came back (hermes://connections/done): read the accounts now, not at the next tick.")


class ConnectionAnswerStatus(WireEnum):
    """What the card says about one row: ``tools/connectors/mcp.py::apply_answer``."""

    approved = "approved"
    skipped = "skipped"


class ConnectionAnswerTarget(Params):
    """One row's answer from the card. ``env`` carries the credential values an install asked for
    through ``required_env``."""

    name: str
    status: ConnectionAnswerStatus
    detail: str | None = None
    env: dict[str, str] | None = None


class ConnectionAnswer(Params):
    """The card's answer: per-target outcomes and an optional Continue
    (``settled_by: "continue"``). Settlement is derived from target states afterwards."""

    targets: list[ConnectionAnswerTarget] = Field(default_factory=list)
    settled_by: ConnectionSettleReason | None = None


class ConnectionRespondParams(ConnectionOperationParams):
    result: ConnectionAnswer


class ConnectionRespondResult(Result):
    status: Literal["ok"]
    settled: bool


method("connection.respond", params=ConnectionRespondParams, result=ConnectionRespondResult,
       doc="Per-target outcomes from the card, and an optional Continue.")
