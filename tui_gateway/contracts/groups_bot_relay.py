"""Hosted Group Chat rooms (``groups.*``), cross-connection bot relay (``bot_relay.*``) and the
dashboard browser controller (``browser.controller.*``).

Handlers: ``tui_gateway/methods_groups.py``, ``tui_gateway/methods_bot_relay.py``,
``tui_gateway/methods_browser_control.py``. Room / event / page shapes are produced by
``gateway/hosted_rooms.py`` (``_room_from_row`` / ``_event_from_row`` / ``read_events``) and
``gateway/hosted_room_replicas.py``; the RoomLink catalog by ``gateway/hosted_room_peer.py``.
"""

from __future__ import annotations

from typing import Literal

from .base import JsonValue, MethodParams, Params, Result
from .common import OkResult
from .registry import method
from .server_requests import ApprovalChoice

# ── shared room shapes ────────────────────────────────────────────────────────────────────────


class RoomMemberTargetLocal(Result):
    kind: Literal["local"]
    profile: str


class RoomMemberTargetPeer(Result):
    kind: Literal["peer"]
    peer_id: str
    installation_id: str
    profile: str
    capability_digest: str


class RoomMember(Result):
    """``HostedRoomService.create_room`` serializes the validated Discussion roster."""

    member_id: str
    profile: str
    handle: str
    display_name: str | None = None
    target: RoomMemberTargetLocal | RoomMemberTargetPeer | None = None


class RoomActor(Result):
    """Closed hosted-room actor shape; optional labels are omitted by ``_validate_actor``."""

    kind: Literal["user", "member", "gateway", "system"]
    id: str
    display_name: str | None = None
    profile: str | None = None
    connection_id: str | None = None


class RoomActorInput(Params):
    kind: Literal["user", "member", "gateway", "system"]
    id: str
    display_name: str | None = None
    connection_id: str | None = None


class RoomEvent(Result):
    """``gateway/hosted_rooms.py::_event_from_row``."""

    room_id: str
    seq: int
    event_id: str
    kind: str
    actor: RoomActor
    authority_epoch: int | None = None
    # gateway/hosted_rooms.py:493 preserves payloads for every room-event kind, including future ones.
    payload: JsonValue
    created_at: float
    idempotent: bool = False


class RoomEventInput(Params):
    room_id: str
    seq: int
    event_id: str
    kind: str
    actor: RoomActorInput
    authority_epoch: int | None = None
    payload: JsonValue
    created_at: float
    idempotent: bool = False

    def as_mapping(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_unset=True)


class Room(Result):
    """``gateway/hosted_rooms.py::_room_from_row`` plus handler-added lineage and rename events."""

    room_id: str
    name: str
    members: list[RoomMember]
    authority_gateway_id: str
    authority_epoch: int
    revision: int
    created_at: float
    updated_at: float
    idempotent: bool = False
    disbanded_at: float | None = None
    latest_seq: int | None = None
    adopted: bool | None = None
    claim_event: RoomEvent | None = None
    authority_claim: RoomEvent | None = None
    event: RoomEvent | None = None


class RoomAuthority(Result):
    gateway_id: str
    epoch: int


class RoomMemberInputTargetLocal(Params):
    kind: Literal["local"]
    profile: str


class RoomMemberInputTargetPeer(Params):
    kind: Literal["peer"]
    peer_id: str
    installation_id: str
    profile: str
    capability_digest: str


class RoomMemberInput(Params):
    """A roster row as the client proposes it; ``validate_roster`` owns the exact rules."""

    member_id: str
    profile: str
    handle: str
    display_name: str | None = None
    target: RoomMemberInputTargetLocal | RoomMemberInputTargetPeer | None = None

    def as_mapping(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_unset=True)


class RoomParams(MethodParams):
    """Any method addressed at one hosted room."""

    room_id: str


# ── RoomLink catalog ──────────────────────────────────────────────────────────────────────────


class RoomExecutionPolicy(Result):
    """``gateway/hosted_room_execution_policy.py::execution_policy_mapping``."""

    version: int
    target_profile: str
    enabled_toolsets: list[str]
    approval_mode: str
    max_iterations: int
    policy_digest: str


class RoomLinkEndpointAvailable(Result):
    available: Literal[True]
    url: str
    transport_security: str


class RoomLinkEndpointUnavailable(Result):
    available: Literal[False]
    reason: str


RoomLinkEndpoint = RoomLinkEndpointAvailable | RoomLinkEndpointUnavailable


class RoomLinkCatalog(Result):
    """``gateway/hosted_room_peer.py::GatewayRoomCatalog.as_mapping``."""

    installation_id: str
    protocol_versions: list[int]
    link_modes: list[str]
    persistent_process: bool
    text: bool
    attachments: bool
    execution_policy: RoomExecutionPolicy
    catalog_digest: str
    endpoint: RoomLinkEndpoint | None = None

    def as_mapping(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")


class RoomLinkCatalogInput(Params):
    """Inbound RoomLink catalog; it becomes a gateway peer catalog before probing."""

    installation_id: str
    protocol_versions: list[int]
    link_modes: list[str]
    persistent_process: bool
    text: bool
    attachments: bool
    execution_policy: RoomExecutionPolicy
    catalog_digest: str
    endpoint: RoomLinkEndpoint | None = None

    def as_mapping(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_unset=True)


class RoomLinkEnabled(Result):
    enabled: Literal[True]
    profile: str
    catalog: RoomLinkCatalog
    endpoint: RoomLinkEndpoint


class RoomLinkDisabled(Result):
    enabled: Literal[False]
    reason: str


RoomLinkStatus = RoomLinkEnabled | RoomLinkDisabled


# ── groups.capabilities ───────────────────────────────────────────────────────────────────────


class GroupsCapabilitiesParams(MethodParams):
    pass


class GroupsCapabilitiesResult(Result):
    protocol_version: int
    driver: bool
    persistent_process: bool
    authority_gateway_id: str
    room_link: RoomLinkStatus
    features: list[str]
    methods: list[str]
    max_log_limit: int


method("groups.capabilities", params=GroupsCapabilitiesParams, result=GroupsCapabilitiesResult,
       doc="Describe the hosted-room protocol implemented by this gateway.")


# ── groups.list / create / state ──────────────────────────────────────────────────────────────


class GroupsListParams(MethodParams):
    include_disbanded: bool | None = None
    limit: int | None = None
    offset: int | None = None


class GroupsListResult(Result):
    rooms: list[Room]
    next_offset: int | None = None


method("groups.list", params=GroupsListParams, result=GroupsListResult,
       doc="List rooms hosted by this gateway, most recently changed first.")


class GroupsCreateParams(MethodParams):
    room_id: str
    name: str
    members: list[RoomMemberInput]
    # tui_gateway/methods_groups.py:365 ignores this compatibility field; the server owns authority.
    authority_gateway_id: str | None = None


class GroupsCreateResult(Result):
    room: Room


method("groups.create", params=GroupsCreateParams, result=GroupsCreateResult,
       doc="Create a hosted room idempotently; authority is this gateway's stable install identity.")


class GroupsStateParams(RoomParams):
    include_disbanded: bool | None = None


class PeerRouteStatus(Result):
    room_id: str
    member_id: str
    status: str


class RoomRetryAction(Result):
    kind: Literal["retry"]
    task_id: str


class RoomApprovalAction(Result):
    kind: Literal["approval"]
    task_id: str
    execution_generation: int
    run_id: str | None = None
    session_id: str
    request_id: str | None = None
    # tui_gateway/hosted_room_driver.py:399 preserves the approval tool's nested, evolving payload.
    approval: JsonValue
    member_id: str


class RoomDriverStatus(Result):
    """``HostedRoomService.status(room_id)``."""

    running: bool
    working: bool
    blocked: bool
    counts: dict[str, int]
    pending_actions: list[RoomRetryAction | RoomApprovalAction]
    peer_routes: list[PeerRouteStatus]


class GroupsStateResult(Result):
    room: Room
    driver_status: RoomDriverStatus | None = None


method("groups.state", params=GroupsStateParams, result=GroupsStateResult,
       doc="One hosted room's replay cursor and fenced authority state, plus live driver status.")


# ── groups.send / rename / log ────────────────────────────────────────────────────────────────


class GroupsSendPayload(Params):
    """``HostedRoomService.send`` accepts exactly the Discussion ``message.user`` payload."""

    text: str
    thread_id: str


class GroupsSendParams(RoomParams):
    event_id: str | None = None
    # tui_gateway/hosted_room_service.py:452 owns future payload admission; preserve recursive JSON passthrough.
    payload: GroupsSendPayload | JsonValue


class GroupsSendResult(Result):
    event: RoomEvent
    client_event_id: str | None
    accepted: bool
    driver_started: bool


method("groups.send", params=GroupsSendParams, result=GroupsSendResult,
       doc="Append one inert message.user event idempotently; the actor is server-owned.")


class GroupsRenameParams(RoomParams):
    event_id: str
    name: str


class GroupsRenameResult(Result):
    room: Room


method("groups.rename", params=GroupsRenameParams, result=GroupsRenameResult,
       doc="Rename one hosted room atomically with its replay event.")


class GroupsLogParams(RoomParams):
    since_seq: int | None = None
    limit: int | None = None
    include_disbanded: bool | None = None


class GroupsLogResult(Result):
    """``gateway/hosted_rooms.py::read_events`` page."""

    events: list[RoomEvent]
    cursor: int
    latest_seq: int
    has_more: bool
    authority: RoomAuthority


class GroupsLogInput(Params):
    """Inbound replay page from a peer before replica persistence."""

    events: list[RoomEventInput]
    cursor: int
    latest_seq: int
    has_more: bool
    authority: RoomAuthority

    def as_mapping(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_unset=True)


class GroupsReplicateParams(RoomParams):
    room_name: str
    members: list[JsonValue]
    page: GroupsLogInput


class GroupsReplicateResult(Result):
    room_id: str
    stored_seq: int
    ingested: int
    authority: RoomAuthority
    caught_up: bool


method("groups.replicate", params=GroupsReplicateParams, result=GroupsReplicateResult,
       doc="Persist one authority-stamped replay page into the local replica store; idempotent.")


method("groups.log", params=GroupsLogParams, result=GroupsLogResult,
       doc="A monotonic room-log delta after since_seq, bounded by count and page bytes.")


# ── groups.disband / stop / approve / retry ───────────────────────────────────────────────────


class GroupsDisbandParams(RoomParams):
    cancel_id: str | None = None


class RoomTombstone(Result):
    room_id: str
    disbanded_at: float
    idempotent: bool
    history_expired: bool | None = None
    event: RoomEvent | None = None


class GroupsDisbandResult(Result):
    tombstone: RoomTombstone


method("groups.disband", params=GroupsDisbandParams, result=GroupsDisbandResult,
       doc="Permanently tombstone a hosted room id after stopping its work and revoking peer routes.")


class GroupsStopParams(RoomParams):
    cancel_id: str | None = None


class GroupsStopResult(Result):
    cancelled: int


method("groups.stop", params=GroupsStopParams, result=GroupsStopResult,
       doc="Durably cancel queued or running work for one hosted room.")


class GroupsApproveParams(RoomParams):
    member_id: str
    task_id: str
    execution_generation: int
    choice: ApprovalChoice
    request_id: str


class GroupsApproveResult(Result):
    """The local approval response and peer run-action receipts remain producer-owned JSON."""

    approved: bool
    # tui_gateway/hosted_room_service.py:517 returns local or peer approval receipts with distinct shapes.
    result: JsonValue


method("groups.approve", params=GroupsApproveParams, result=GroupsApproveResult,
       doc="Resolve one exact pending approval raised by a local or peer room member.")


class GroupsRetryParams(RoomParams):
    task_id: str


class RoomTaskReceipt(Result):
    room_id: str
    task_id: str
    thread_id: str
    turn_id: str
    status: str
    execution_generation: int
    cancel_generation: int


class GroupsRetryResult(Result):
    retried: bool
    task: RoomTaskReceipt


method("groups.retry", params=GroupsRetryParams, result=GroupsRetryResult,
       doc="Retry one indeterminate room task after explicit user confirmation.")


# ── replication / authority takeover ──────────────────────────────────────────────────────────


class GroupsReplicaStateParams(RoomParams):
    pass


class GroupsReplicaStateResult(Result):
    room_id: str
    name: str
    # gateway/hosted_room_replicas.py:192 returns verbatim persisted replica roster rows.
    members: list[JsonValue]
    authority: RoomAuthority
    last_seq: int
    latest_seq: int
    event_bytes: int
    created_at: float
    updated_at: float


method("groups.replica_state", params=GroupsReplicaStateParams, result=GroupsReplicaStateResult,
       doc="The local replica's coverage and authority lineage for one room.")


class GroupsPromoteParams(RoomParams):
    confirm: bool | None = None
    reason: str | None = None


class GroupsPromoteResult(Result):
    room_id: str
    authority_gateway_id: str
    authority_epoch: int
    previous_gateway_id: str
    previous_epoch: int
    claim_seq: int
    latest_seq: int


method("groups.promote", params=GroupsPromoteParams, result=GroupsPromoteResult,
       doc="Continue a replicated room on this gateway at epoch + 1; requires confirm=true.")


class GroupsDemoteParams(RoomParams):
    observed_gateway_id: str
    observed_epoch: int


class GroupsDemoteResult(Result):
    room_id: str
    authority_gateway_id: str
    authority_epoch: int
    idempotent: bool


method("groups.demote", params=GroupsDemoteParams, result=GroupsDemoteResult,
       doc="Fence this gateway's stale room authority against a proven newer epoch.")


# ── peer routes (RoomLink) ────────────────────────────────────────────────────────────────────


class GroupsPeerInviteParams(MethodParams):
    room_id: str | None = None
    home_install_id: str | None = None
    authority_gateway_id: str | None = None
    authority_epoch: int | None = None
    member_id: str | None = None
    grant_id: str | None = None
    ttl_seconds: float | None = None


class GroupsPeerInviteResult(Result):
    grant: str
    target_profile: str
    catalog: RoomLinkCatalog
    endpoint: RoomLinkEndpoint


method("groups.peer.invite", params=GroupsPeerInviteParams, result=GroupsPeerInviteResult,
       doc="Mint one target-issued room/profile grant for a prospective room home.")


class GroupsPeerRevokeParams(MethodParams):
    grant: str


class GroupsPeerRevokeResult(Result):
    revoked: bool


method("groups.peer.revoke", params=GroupsPeerRevokeParams, result=GroupsPeerRevokeResult,
       doc="Revoke one target-issued grant using its exact profile scope.")


class GroupsPeerRegisterParams(RoomParams):
    member_id: str
    target_url: str
    target_profile: str
    grant: str
    catalog: RoomLinkCatalogInput
    cancellation_scope_id: str | None = None
    trace_id: str | None = None


class GroupsPeerRegisterResult(Result):
    registered: bool
    mode: str
    transport_security: str
    target_install_id: str
    target_profile: str


method("groups.peer.register", params=GroupsPeerRegisterParams, result=GroupsPeerRegisterResult,
       doc="Register and probe one scoped peer route on the room home.")


# ── bot relay ─────────────────────────────────────────────────────────────────────────────────


class RelayAgentRow(Params):
    """One Desktop roster row normalized by ``tools/bot_relay.py::_normalize_roster_row``."""

    handle: str | None = None
    connection_id: str | None = None
    connection_label: str | None = None
    title: str | None = None
    description: str | None = None
    online: bool | None = None


class BotRelayRosterSyncParams(MethodParams):
    agents: list[RelayAgentRow] | None = None


class BotRelayRosterSyncResult(Result):
    count: int


method("bot_relay.roster.sync", params=BotRelayRosterSyncParams, result=BotRelayRosterSyncResult,
       doc="Replace this gateway's view of agents on other connections; answers the accepted row count.")


class BotRelayOutboxDrainParams(MethodParams):
    pass


class RelayEnvelope(Result):
    """``tools/bot_relay.py::enqueue_envelope``."""

    id: str
    created_at: int
    from_profile: str
    from_handle: str
    target_connection: str
    target_profile: str
    target_handle: str
    message: str


class BotRelayOutboxDrainResult(Result):
    envelopes: list[RelayEnvelope]


method("bot_relay.outbox.drain", params=BotRelayOutboxDrainParams, result=BotRelayOutboxDrainResult,
       doc="Atomically claim every pending cross-connection envelope queued on this gateway.")


class BotRelayDeliverParams(MethodParams):
    """``profile`` here is the TARGET profile on this gateway (also what the desktop route wrapper adds)."""

    profile: str
    message: str
    from_profile: str | None = None
    from_handle: str | None = None
    from_connection: str | None = None


class BotRelayDeliverResult(Result):
    reply: str


method("bot_relay.deliver", params=BotRelayDeliverParams, result=BotRelayDeliverResult,
       doc="Deliver a relayed DM into a Bot Chat on this gateway and return the one-turn reply (blocking).")


class BotRelayReplyParams(MethodParams):
    id: str
    reply: str | None = None
    error: str | None = None
    reason: str | None = None


method("bot_relay.reply", params=BotRelayReplyParams, result=OkResult,
       doc="Write a relayed reply and/or typed error for an envelope so the sender-side waiter resolves.")


# ── browser controller ────────────────────────────────────────────────────────────────────────


class BrowserControllerParams(MethodParams):
    """Every controller call names the session the controller is attached to."""

    session_id: str


class BrowserControllerRegisterParams(BrowserControllerParams):
    controller_id: str
    browser_profile_id: str
    capabilities: list[str] | None = None
    protocol_version: int | None = None
    # Ignored: the principal is derived from the server-minted identity, never client-supplied.
    principal_id: str | None = None


class ControllerScope(Result):
    principal_id: str
    profile_id: str
    session_id: str
    controller_id: str
    browser_profile_id: str
    transport_family: str
    capabilities: list[str]


class BrowserControllerRegisterResult(Result):
    scope: ControllerScope


method("browser.controller.register", params=BrowserControllerRegisterParams,
       result=BrowserControllerRegisterResult,
       doc="Attach this connection as the browser controller for one session; fails closed (4403).")


class BrowserControllerResultParams(BrowserControllerParams):
    command_id: str
    ok: bool | None = None
    # gateway/browser_control_broker.py:392 stores arbitrary controller success or error values.
    result: JsonValue | None = None
    error: JsonValue | None = None


class BrowserControllerResultResult(Result):
    accepted: bool


method("browser.controller.result", params=BrowserControllerResultParams,
       result=BrowserControllerResultResult,
       doc="Deliver one command result to the broker; accepted is false for unknown or settled command ids.")


method("browser.controller.heartbeat", params=BrowserControllerParams, result=OkResult,
       doc="Acknowledge a heartbeat only for this transport's own attached controller.")


class BrowserControllerDetachResult(Result):
    detached: bool


method("browser.controller.detach", params=BrowserControllerParams, result=BrowserControllerDetachResult,
       doc="Hard-detach only the controller owned by this authenticated transport.")


__all__ = [
    "GroupsLogResult", "RelayEnvelope", "Room", "RoomAuthority", "RoomEvent", "RoomLinkCatalog",
    "RoomMember", "RoomMemberInput",
]
