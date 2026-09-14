"""Hosted Group Chat rooms (``groups.*``), cross-connection bot relay (``bot_relay.*``) and the
dashboard browser controller (``browser.controller.*``).

Handlers: ``tui_gateway/methods_groups.py``, ``tui_gateway/methods_bot_relay.py``,
``tui_gateway/methods_browser_control.py``. Room / event / page shapes are produced by
``gateway/hosted_rooms.py`` (``_room_from_row`` / ``_event_from_row`` / ``read_events``) and
``gateway/hosted_room_replicas.py``; the RoomLink catalog by ``gateway/hosted_room_peer.py``.
"""

from __future__ import annotations

from .base import JsonValue, Params, Result, WireEnum
from .common import OkResult, OpenModel, ProfileParams
from .registry import method

# ── shared room shapes ────────────────────────────────────────────────────────────────────────


class RoomMember(OpenModel):
    """One roster row (``hosted_room_discussion.validate_roster``); legacy rooms may carry
    pre-normalisation rows, so the set stays open."""

    member_id: str | None = None
    profile: str | None = None
    handle: str | None = None
    display_name: str | None = None
    target: dict[str, JsonValue] | None = None


class RoomActor(Result):
    kind: str
    id: str


class RoomEvent(Result):
    """``gateway/hosted_rooms.py::_event_from_row``."""

    room_id: str
    seq: int
    event_id: str
    kind: str
    actor: RoomActor
    authority_epoch: int | None = None
    payload: dict[str, JsonValue]
    created_at: float
    idempotent: bool = False


class Room(Result):
    """``gateway/hosted_rooms.py::_room_from_row`` plus the branch-only keys ``create`` (legacy
    adoption), ``state`` (``authority_claim``) and ``rename`` (``event``) add."""

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


class RoomMemberInput(Params):
    """A roster row as the client proposes it; ``validate_roster`` owns the exact rules."""

    member_id: str | None = None
    profile: str | None = None
    handle: str | None = None
    display_name: str | None = None
    target: dict[str, JsonValue] | None = None
    model_config = Params.model_config | {"extra": "allow"}


class RoomParams(ProfileParams):
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


class RoomLinkEndpoint(Result):
    """``GatewayRoomCatalog.endpoint_mapping``: ``url``/``transport_security`` when available,
    ``reason`` when not."""

    available: bool
    url: str | None = None
    transport_security: str | None = None
    reason: str | None = None


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


class RoomLinkStatus(Result):
    """``enabled`` with ``profile``/``catalog``/``endpoint``, or disabled with a ``reason``."""

    enabled: bool
    profile: str | None = None
    catalog: RoomLinkCatalog | None = None
    endpoint: RoomLinkEndpoint | None = None
    reason: str | None = None


# ── groups.capabilities ───────────────────────────────────────────────────────────────────────


class GroupsCapabilitiesParams(ProfileParams):
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


class GroupsListParams(ProfileParams):
    include_disbanded: bool | None = None
    limit: int | None = None
    offset: int | None = None


class GroupsListResult(Result):
    rooms: list[Room]
    next_offset: int | None = None


method("groups.list", params=GroupsListParams, result=GroupsListResult,
       doc="List rooms hosted by this gateway, most recently changed first.")


class GroupsCreateParams(ProfileParams):
    room_id: str
    name: str
    members: list[RoomMemberInput]
    # Ignored: authority is always this gateway's install identity (a client cannot spoof it).
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


class RoomDriverStatus(Result):
    """``HostedRoomService.status(room_id)``; ``pending_actions`` rows are ``{kind: retry, task_id}``
    or the driver's approval action (``kind: approval`` + run/session/approval context)."""

    running: bool
    working: bool
    blocked: bool
    counts: dict[str, int]
    pending_actions: list[dict[str, JsonValue]]
    peer_routes: list[PeerRouteStatus]


class GroupsStateResult(Result):
    room: Room
    driver_status: RoomDriverStatus | None = None


method("groups.state", params=GroupsStateParams, result=GroupsStateResult,
       doc="One hosted room's replay cursor and fenced authority state, plus live driver status.")


# ── groups.send / rename / log ────────────────────────────────────────────────────────────────


class GroupsSendParams(RoomParams):
    event_id: str | None = None
    payload: dict[str, JsonValue]


class GroupsSendResult(Result):
    event: RoomEvent
    client_event_id: str | None = None
    accepted: bool = True
    driver_started: bool = True


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
    """``gateway/hosted_rooms.py::read_events`` page — also the ``page`` ``groups.replicate`` ingests."""

    events: list[RoomEvent]
    cursor: int
    latest_seq: int
    has_more: bool
    authority: RoomAuthority


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


class ApprovalChoice(WireEnum):
    once = "once"
    deny = "deny"


class GroupsApproveParams(RoomParams):
    member_id: str
    task_id: str
    execution_generation: int
    choice: ApprovalChoice
    request_id: str


class GroupsApproveResult(Result):
    """``result`` is the local ``approval.respond`` answer or the peer's run-action receipt."""

    approved: bool = True
    result: dict[str, JsonValue]


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
    retried: bool = True
    task: RoomTaskReceipt


method("groups.retry", params=GroupsRetryParams, result=GroupsRetryResult,
       doc="Retry one indeterminate room task after explicit user confirmation.")


# ── replication / authority takeover ──────────────────────────────────────────────────────────


class GroupsReplicateParams(RoomParams):
    room_name: str
    members: list[RoomMemberInput]
    page: dict[str, JsonValue]  # a verbatim ``groups.log`` result


class GroupsReplicateResult(Result):
    room_id: str
    stored_seq: int
    ingested: int
    authority: RoomAuthority
    caught_up: bool


method("groups.replicate", params=GroupsReplicateParams, result=GroupsReplicateResult,
       doc="Persist one authority-stamped replay page into the local replica store; idempotent.")


class GroupsReplicaStateParams(RoomParams):
    pass


class GroupsReplicaStateResult(Result):
    room_id: str
    name: str
    members: list[RoomMember]
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


class GroupsPeerInviteParams(ProfileParams):
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


class GroupsPeerRevokeParams(ProfileParams):
    grant: str


class GroupsPeerRevokeResult(Result):
    revoked: bool = True


method("groups.peer.revoke", params=GroupsPeerRevokeParams, result=GroupsPeerRevokeResult,
       doc="Revoke one target-issued grant using its exact profile scope.")


class GroupsPeerRegisterParams(RoomParams):
    member_id: str
    target_url: str
    target_profile: str
    grant: str
    catalog: dict[str, JsonValue]  # a RoomLinkCatalog mapping; ``GatewayRoomCatalog.from_mapping`` is exact
    cancellation_scope_id: str | None = None
    trace_id: str | None = None


class GroupsPeerRegisterResult(Result):
    registered: bool = True
    mode: str
    transport_security: str
    target_install_id: str
    target_profile: str


method("groups.peer.register", params=GroupsPeerRegisterParams, result=GroupsPeerRegisterResult,
       doc="Register and probe one scoped peer route on the room home.")


# ── bot relay ─────────────────────────────────────────────────────────────────────────────────


class RelayAgentRow(Params):
    """A roster row the Desktop pushes (``tools/bot_relay.py::_normalize_roster_row``); invalid
    rows are dropped server-side, so the shape stays open."""

    profile: str | None = None
    handle: str | None = None
    connection_id: str | None = None
    connection_label: str | None = None
    title: str | None = None
    description: str | None = None
    online: bool | None = None
    model_config = Params.model_config | {"extra": "allow"}


class BotRelayRosterSyncParams(ProfileParams):
    agents: list[RelayAgentRow] | None = None


class BotRelayRosterSyncResult(Result):
    count: int


method("bot_relay.roster.sync", params=BotRelayRosterSyncParams, result=BotRelayRosterSyncResult,
       doc="Replace this gateway's view of agents on other connections; answers the accepted row count.")


class BotRelayOutboxDrainParams(ProfileParams):
    pass


class RelayEnvelope(OpenModel):
    """``tools/bot_relay.py::enqueue_envelope``."""

    id: str
    created_at: int | float
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


class BotRelayDeliverParams(Params):
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


class BotRelayReplyParams(ProfileParams):
    id: str
    reply: str | None = None
    error: str | None = None
    reason: str | None = None


method("bot_relay.reply", params=BotRelayReplyParams, result=OkResult,
       doc="Write a relayed reply and/or typed error for an envelope so the sender-side waiter resolves.")


# ── browser controller ────────────────────────────────────────────────────────────────────────


class BrowserControllerParams(Params):
    """Every controller call names the session the controller is attached to."""

    session_id: str


class BrowserControllerRegisterParams(BrowserControllerParams):
    controller_id: str
    browser_profile_id: str
    capabilities: list[str] | None = None
    protocol_version: JsonValue | None = None  # checked exactly by the handler (an int today)
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
    ok: JsonValue | None = None  # only the exact ``true`` counts as success
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
    detached: bool = True


method("browser.controller.detach", params=BrowserControllerParams, result=BrowserControllerDetachResult,
       doc="Hard-detach only the controller owned by this authenticated transport.")


__all__ = [
    "GroupsLogResult", "RelayEnvelope", "Room", "RoomAuthority", "RoomEvent", "RoomLinkCatalog",
    "RoomMember", "RoomMemberInput",
]
