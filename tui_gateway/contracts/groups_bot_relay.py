"""Hosted Group Chat rooms (``groups.*``), cross-connection bot relay (``bot_relay.*``) and the
dashboard browser controller (``browser.controller.*``).

Handlers: ``tui_gateway/methods_groups.py``, ``tui_gateway/methods_bot_relay.py``,
``tui_gateway/methods_browser_control.py``. Room / event / page shapes are produced by
``gateway/hosted_rooms.py`` (``_room_from_row`` / ``_event_from_row`` / ``read_events``) and
``gateway/hosted_room_replicas.py``; the RoomLink catalog by ``gateway/hosted_room_peer.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .common import OkResult, OpenModel, ProfileParams
from .registry import method
from .server_requests import ApprovalChoice

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
    #: ``authority_quarantined`` (``groups.list`` only) when the room's history records an unproven
    #: takeover; ``safety_reason`` names it. Such a room stays readable and refuses new events.
    safety_status: str | None = None
    safety_reason: str | None = None
    #: ``true`` for a copy of another gateway's Group Chat held here: read-only, never driven
    #: (``revision`` is 0), shown to this installation's operator or the room's recorded owner.
    copy: bool | None = None
    #: A copy's custody summary (``groups.state``): ``{configuration_seq, at_risk_after_seq, custodians}``.
    custody: dict[str, JsonValue] | None = None


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
    authentication: Literal['proof-v2'] | None = None
    profile: str | None = None
    catalog: RoomLinkCatalog | None = None
    endpoint: RoomLinkEndpoint | None = None
    reason: str | None = None


# ── groups.capabilities ───────────────────────────────────────────────────────────────────────


class GroupsCapabilitiesParams(ProfileParams):
    pass


class GatewayRoomIdentity(Result):
    """How this installation shows up in other installations' Group Chats."""

    install_id: str
    #: ``gateway.display_name``, else the host name.
    name: str | None = None
    #: ``gateway.owner_name``: the person who runs this installation; null when unset.
    operator_name: str | None = None
    #: No battery, unless ``group_chat.always_on`` says otherwise.
    always_on: bool


class GroupsCapabilitiesResult(Result):
    protocol_version: int
    driver: bool
    persistent_process: bool
    authority_gateway_id: str
    room_link: RoomLinkStatus
    features: list[str]
    methods: list[str]
    max_log_limit: int
    room_identity: GatewayRoomIdentity | None = None


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
    peer_cleanup: list[dict[str, JsonValue]] | None = None
    retiring: bool | None = None
    replication: dict[str, JsonValue] | None = None
    #: Turns waiting for another computer: ``{task_id, member_id, state: "waiting_for_host", resource,
    #: host_name}``; ``resource`` is ``bot`` or ``file`` (``tool``/``secret`` reserved).
    tasks: list[dict[str, JsonValue]] | None = None


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
    #: Majority mode only: the send is stored on a majority of the room's voters, so an automatic move
    #: keeps it. The gateway waits briefly for it; a send that is not protected stays in the room,
    #: inside the tail at risk, and clients offer it again after a move. Absent in other modes, where
    #: dispatch doesn't wait for copies.
    protected: bool | None = None


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
    #: Required to disband a quarantined room, which then only ends on this gateway (its history is kept).
    confirm_quarantined: bool | None = None


class RoomTombstone(Result):
    room_id: str
    disbanded_at: float
    idempotent: bool
    history_expired: bool | None = None
    event: RoomEvent | None = None


class GroupsDisbandResult(Result):
    tombstone: RoomTombstone


method("groups.disband", params=GroupsDisbandParams, result=GroupsDisbandResult,
       doc="Permanently tombstone a hosted room id after stopping its work and revoking peer routes. "
           "A quarantined room needs confirm_quarantined=true and only ends on this gateway, history kept.")


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
    """``result`` is the local ``approval.respond`` answer or the peer's run-action receipt."""

    approved: bool = True
    result: dict[str, JsonValue]


method("groups.approve", params=GroupsApproveParams, result=GroupsApproveResult,
       doc="Resolve one exact pending approval raised by a local or peer room member.")


class GroupsRetryParams(RoomParams):
    task_id: str
    # Canonical controls bind the member and exact generation; legacy uses task_id.
    member_id: str | None = None
    execution_generation: int | None = None


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
       doc="Retry one eligible room task; canonical controls require exact proven nonadmission.")


class GroupsDiscardParams(RoomParams):
    member_id: str
    task_id: str
    execution_generation: int


class GroupsDiscardResult(Result):
    discarded: bool
    task: RoomTaskReceipt


method("groups.discard", params=GroupsDiscardParams, result=GroupsDiscardResult,
       doc="Discard one exact canonically proven-unaccepted attempt; accepted or unknown work requires Stop.")


class GroupsAttachmentUploadParams(RoomParams):
    upload_id: str
    kind: str
    name: str
    mime: str
    data_base64: str


class GroupsAttachmentResult(Result):
    attachment_id: str
    kind: str
    name: str
    size: int
    mime: str
    sha256: str
    state: str
    created_at: float
    idempotent: bool
    event_id: str | None = None


method("groups.attachment.upload", params=GroupsAttachmentUploadParams, result=GroupsAttachmentResult,
       doc="Upload owner-authorized bytes for a canonical room message.")


class GroupsAttachmentDownloadParams(RoomParams):
    event_id: str
    attachment_id: str


class GroupsAttachmentDownloadResult(GroupsAttachmentResult):
    data_base64: str


method("groups.attachment.download", params=GroupsAttachmentDownloadParams, result=GroupsAttachmentDownloadResult,
       doc="Read bytes bound to a canonical room event, subject to current viewer authorization.")


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
       doc="Persist one authority-stamped replay page into the local replica store; idempotent. "
           "Refused (4116, reason replica_provenance_required) until exclusive-authority recovery exists.")


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
    disbanded_at: float | None = None
    #: ``passive``, or ``quarantined`` when the stored lineage failed the replica audit
    #: (``safety_reason`` names the first defect).
    safety_status: str | None = None
    safety_reason: str | None = None
    work_records: dict[str, JsonValue] | None = None
    copy_retired_at: float | None = None


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
       doc="Continue a replicated room on this gateway at epoch + 1; requires confirm=true. "
           "Refused (4118, reason authority_takeover_disabled) until exclusive-authority recovery exists.")


class GroupsDemoteParams(RoomParams):
    observed_gateway_id: str
    observed_epoch: int


class GroupsDemoteResult(Result):
    room_id: str
    authority_gateway_id: str
    authority_epoch: int
    idempotent: bool


method("groups.demote", params=GroupsDemoteParams, result=GroupsDemoteResult,
       doc="Fence this gateway's stale room authority against a proven newer epoch. "
           "Refused (4119, reason authority_takeover_disabled) until exclusive-authority recovery exists.")


# ── peer routes (RoomLink) ────────────────────────────────────────────────────────────────────


class GroupsPeerInviteParams(ProfileParams):
    room_id: str | None = None
    home_install_id: str | None = None
    authority_gateway_id: str | None = None
    authority_epoch: int | None = None
    member_id: str | None = None
    grant_id: str | None = None
    ttl_seconds: float | None = None
    # How long the room's gateway may keep renewing the grant (canonical surface); defaults to
    # ``ttl_seconds``, so nothing is renewed unless the operator chooses a longer horizon.
    status_ttl_seconds: float | None = None
    # The installation keeps the room's history as a custodian unless this is ``false``.
    replication: bool | None = None
    work_records: bool | None = None
    passive_only: bool | None = None
    # The operator's consent that the room owner may continue the group on this installation.
    successor: bool | None = None
    # A copy-only grant for an installation without a Bot in the room (no ``member_id``).
    custody_only: bool | None = None
    # The same grant is re-issued to a verified successor of the room unless this is ``false``.
    continuation: bool | None = None


class GroupsPeerInviteResult(Result):
    grant: str
    target_profile: str
    catalog: RoomLinkCatalog
    endpoint: RoomLinkEndpoint


method("groups.peer.invite", params=GroupsPeerInviteParams, result=GroupsPeerInviteResult,
       doc="Mint one target-issued room/profile grant for a prospective room home; the installation keeps "
           "the room's history unless replication is false.")


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


class ReplicaRetirementEnrollment(Result):
    enrollment_id: str
    room_id: str
    authority_gateway_id: str
    authority_epoch: int
    target_install_id: str
    roster_sha256: str
    commitment: str


class GroupsReplicationPrepareParams(RoomParams):
    target_install_id: str
    endpoint: str
    enrollment_id: str | None = None
    replace_enrollment_id: str | None = None


class GroupsReplicationPrepareResult(Result):
    enrollment: ReplicaRetirementEnrollment


class GroupsReplicationEnrollParams(ProfileParams):
    enrollment: ReplicaRetirementEnrollment
    expected_enrollment_id: str | None = None
    expected_state: str | None = None


class GroupsReplicationEnrollResult(ReplicaRetirementEnrollment):
    state: str


class GroupsReplicationRevokeParams(RoomParams):
    enrollment_id: str


class GroupsReplicationRevokeResult(Result):
    room_id: str
    enrollment_id: str
    state: str


method('groups.replication.prepare', params=GroupsReplicationPrepareParams, result=GroupsReplicationPrepareResult,
       doc='Prepare public retirement verification material for an opted-in participant copy.')
method('groups.replication.enroll', params=GroupsReplicationEnrollParams, result=GroupsReplicationEnrollResult,
       doc='The participant operator enrolls retirement of one exact passive copy.')
method('groups.replication.revoke', params=GroupsReplicationRevokeParams, result=GroupsReplicationRevokeResult,
       doc='The participant operator withdraws one copy-retirement enrollment.')


# ── custody ───────────────────────────────────────────────────────────────────────────────────


class CustodyWatermark(Result):
    epoch: int
    seq: int
    event_hash: str


class CustodyHead(Result):
    """A head the host signs: the chain hash of its room's prefix ``1..seq`` at its ``epoch``."""

    room_id: str
    host: str
    epoch: int
    seq: int
    chain_hash: str
    signature: str


class CustodyCustodian(Result):
    """One entry of a ``custody.configured`` event."""

    install_id: str
    public_key: str
    endpoint: str | None = None
    #: ``authority`` (the current host), ``custodian`` (a member installation) or ``custodian_only``.
    role: str
    #: May continue the group: its operator allowed it and the room owner designated it.
    successor: bool
    #: Reported by the installation: no battery, unless its operator says otherwise.
    always_on: bool
    #: Votes on moving the group by itself: the host, or an always-on successor (at most seven).
    voter: bool
    name: str | None = None
    operator_name: str | None = None


class CustodyConfiguration(Result):
    configuration_seq: int
    custodians: list[CustodyCustodian]
    owner_name: str | None = None
    #: The owner lets the group move by itself (``groups.custody.automatic``).
    automatic: bool = True
    #: The voters in the owner's order, the host first.
    voters: list[str] = []


class CustodyWaiting(Result):
    """A task held back until a majority of voters stores its ``task.admitted``."""

    task_id: str
    seq: int


class CustodyCustodianStatus(Result):
    install_id: str
    role: str | None = None
    #: ``active``, ``opted_out``, ``unsupported`` (an older Hermes: never counted) or ``withdrawn``.
    state: str
    name: str | None = None
    operator_name: str | None = None
    successor: bool
    allowed: bool | None = None
    designated: bool | None = None
    opted_out: bool
    voter: bool
    always_on: bool
    watermark: CustodyWatermark | None = None
    acknowledged_at: float | None = None
    #: When the host last heard from it (any acknowledgment, idle heartbeats included).
    last_seen: float | None = None
    divergent: bool


class GroupsCustodyStatusParams(RoomParams):
    pass


class GroupsCustodyStatusResult(Result):
    room_id: str
    #: ``authority`` on the room's host, ``custodian`` on an installation holding a copy.
    role: str
    custodians: list[CustodyCustodianStatus]
    #: The highest seq an eligible successor durably holds; later events are at risk.
    at_risk_after_seq: int
    #: The highest seq a majority of voters durably holds, counting the host.
    protected_seq: int
    automatic: bool
    #: The voters in the owner's order, the host first.
    voters: list[str]
    #: The voter sets whose majorities protection needs now: two while a change is not settled.
    voter_sets: list[list[str]]
    #: ``majority`` (3+ voters), ``careful`` (exactly 2) or ``ask``.
    mode: str
    waiting_for_copies: CustodyWaiting | None = None
    configuration_seq: int
    configuration: CustodyConfiguration
    watermark: CustodyWatermark | None = None
    #: The host-signed head that vouches for the history held here (signed now on the host).
    head: CustodyHead | None = None


class GroupsCustodyDesignateParams(RoomParams):
    install_id: str
    successor: bool


class GroupsCustodyDesignateResult(Result):
    room_id: str
    install_id: str
    successor: bool
    configuration_seq: int


class GroupsCustodyAddParams(RoomParams):
    target_url: str
    catalog: dict[str, JsonValue]  # a RoomLinkCatalog mapping
    grant: str  # minted with ``groups.peer.invite`` and ``custody_only: true``
    successor: bool | None = None


class GroupsCustodyChangeResult(Result):
    room_id: str
    install_id: str
    configuration_seq: int


class GroupsCustodyRemoveParams(RoomParams):
    install_id: str


class GroupsCustodyAllowParams(RoomParams):
    successor: bool


class GroupsCustodyAllowResult(Result):
    room_id: str
    install_id: str
    allowed: bool
    #: The host recorded the same choice (from its report beside the next page).
    confirmed: bool


class GroupsCustodyAutomaticParams(RoomParams):
    enabled: bool


class GroupsCustodyAutomaticResult(Result):
    room_id: str
    #: The value requested.
    automatic: bool
    #: The latest configuration; the switch rides in the next one once earlier changes settle.
    configuration_seq: int
    #: True until the switch is in force: in a configuration stored on a majority of the voters.
    pending: bool


method("groups.custody.status", params=GroupsCustodyStatusParams, result=GroupsCustodyStatusResult,
       doc="Who keeps this Group Chat's history, how far each copy reaches, and the tail at risk.")
method("groups.custody.designate", params=GroupsCustodyDesignateParams, result=GroupsCustodyDesignateResult,
       doc="The room owner designates (or not) one custodian to continue the group; its operator must allow it.")
method("groups.custody.add", params=GroupsCustodyAddParams, result=GroupsCustodyChangeResult,
       doc="Add an installation that keeps the room's history without a Bot, after a live scoped probe.")
method("groups.custody.remove", params=GroupsCustodyRemoveParams, result=GroupsCustodyChangeResult,
       doc="Stop keeping a copy on one custodian-only installation.")
method("groups.custody.allow", params=GroupsCustodyAllowParams, result=GroupsCustodyAllowResult,
       doc="On a member installation: allow (or not) the room owner to continue the group here.")
method("groups.custody.automatic", params=GroupsCustodyAutomaticParams, result=GroupsCustodyAutomaticResult,
       doc="On the host: the room owner (or the operator) lets the group move by itself, or asks first.")


# ── succession (a group's host is lost) ──────────────────────────────────────────────────────────


class SuccessionComputer(Result):
    install_id: str | None = None
    name: str | None = None


class SuccessionHost(Result):
    install_id: str | None = None
    name: str | None = None
    reachable: bool
    #: When the host was last heard from, once it counts as offline (Unix seconds).
    since: float | None = None
    #: The end of a restart the host announced.
    restarting_until: float | None = None


class SuccessionThisInstall(Result):
    install_id: str
    name: str | None = None
    #: ``host``, ``backup`` (keeps a copy), ``member`` (a Bot without a copy) or ``none``.
    role: str


class SuccessionOwner(Result):
    name: str | None = None


class SuccessionBackup(Result):
    install_id: str
    name: str | None = None
    #: May continue the group: ``allowed`` (its operator) and ``designated`` (the owner).
    successor: bool
    #: ``caught_up``, ``behind``, ``offline``, ``unknown``, ``unsupported`` (an older Hermes) or, on the host,
    #: ``needs_reauthorization`` (its copy is refused until that computer renews its grant).
    readiness: str
    behind_by: int | None = None
    last_seen: float | None = None
    allowed: bool
    designated: bool
    #: ``member`` (has a Bot in the group) or ``backup`` (keeps a copy only).
    kind: str
    operator_name: str | None = None
    #: Votes on moving the group by itself (an always-on successor).
    voter: bool
    #: Reports having no battery, or its operator says it is always on.
    always_on: bool


class SuccessionAtRisk(Result):
    count: int


class SuccessionMoving(Result):
    to: SuccessionComputer
    #: ``fencing``, ``catching_up``, ``reconciling`` or ``finishing``; on the host, ``waiting_for_turns``
    #: while the owner's move waits for the replies in progress (``actions`` offers ``move_now``).
    step: str
    started_at: float | None = None
    #: ``manual`` (the owner continued it), ``automatic`` or ``handover`` (the host handed it over).
    reason: str | None = None
    #: With ``waiting_for_turns``: how many replies are still in progress.
    running: int | None = None


class SuccessionConflictHost(SuccessionComputer):
    since: float | None = None


class SuccessionConflict(Result):
    hosts: list[SuccessionConflictHost]
    #: When the two lost contact, after a careful move (the standby's evidence); else None.
    start: float | None = None
    #: When this computer found the group running in two places.
    end: float | None = None
    #: The host the group keeps running on meanwhile (the higher epoch; on a tie certified, then
    #: evidence, then attested, then the lower install id). The other one keeps its messages apart.
    running_on: SuccessionComputer | None = None


class SuccessionMoved(Result):
    to: SuccessionComputer
    at: float | None = None
    #: Events this computer wrote while cut off, kept apart (``groups.succession.branch_log``).
    separate_events: int
    branch_id: str | None = None


class SuccessionWork(Result):
    completed: int
    elsewhere: int
    unknown: int
    waiting_for_host: int


class SuccessionPreviousHost(SuccessionComputer):
    offline_since: float | None = None


class SuccessionBotPlace(SuccessionComputer):
    #: That computer answered lately: moving the group back there (a planned handover) brings the Bot back.
    reachable: bool


class SuccessionBot(Result):
    member_id: str | None = None
    name: str | None = None
    #: The computer this Bot runs on: the group's original home for its local Bots, the old host for
    #: its own peer members.
    on: SuccessionBotPlace | None = None


class SuccessionAttempt(Result):
    to: SuccessionComputer | None = None
    error: str
    at: float | None = None


class SuccessionAutomatic(Result):
    """Whether the group moves by itself if its host goes offline."""

    #: ``majority`` (three or more voters), ``careful`` (exactly two) or ``ask``.
    mode: str
    #: ``ready``, ``not_ready`` (reason ``voters_offline``), ``unavailable`` (reason ``needs_computers``)
    #: or ``off`` (the owner chose to be asked first).
    state: str
    standby: SuccessionComputer | None = None
    voters: list[SuccessionComputer]
    #: The owner's switch as the group's configuration holds it.
    enabled: bool
    #: The value the owner asked for while that change still settles with the voters; else None.
    pending: bool | None = None
    reason: str | None = None
    offline: list[SuccessionComputer] | None = None
    #: How many more always-on computers would make the group move by itself.
    needed: int | None = None


class SuccessionPaused(Result):
    """The host executes and appends nothing, to stay safe."""

    #: ``lost_majority`` (no lease from a majority of the voters), ``isolated`` (careful mode: cut off),
    #: ``no_lease_layer`` (its lease layer isn't running; the owner may continue it anyway) or
    #: ``step_not_taken`` (its next step was promised to a computer that never took it, and
    #: ``waiting_for`` can't yet confirm that nothing else happened; the owner may continue it anyway).
    reason: str
    since: float | None = None
    waiting_for: list[SuccessionComputer]


class SuccessionMovedIn(Result):
    """On a new host after an automatic move or a handover, until the old host is a copy again."""

    from_: SuccessionComputer = Field(alias="from")  # ``from`` is a keyword
    at: float | None = None
    #: ``certified``, ``evidence`` (a careful move: ``actions`` offers going back) or ``handover``.
    proof_kind: str


class GroupsSuccessionStatusParams(RoomParams):
    pass


class GroupsSuccessionStatusResult(Result):
    """What this computer knows about the group's host. Codes and parameters only."""

    #: ``ok``, ``paused``, ``host_unreachable``, ``host_restarting``, ``moving``, ``continued_on_two`` or
    #: ``moved_away``.
    state: str
    host: SuccessionHost
    this_install: SuccessionThisInstall
    owner: SuccessionOwner
    backups: list[SuccessionBackup]
    at_risk: SuccessionAtRisk
    moving: SuccessionMoving | None = None
    conflict: SuccessionConflict | None = None
    moved: SuccessionMoved | None = None
    work: SuccessionWork | None = None
    #: ``{action: continue|keep|designate|remove_backup|move, targets}``, ``{action: open_on, target}``,
    #: ``{action: add_backup}``, ``{action: continue_anyway, turns_off_automatic?}`` (true when continuing
    #: also turns automatic moves off: the host's lease layer isn't running), ``{action: move_now}`` or
    #: ``{action: automatic, enabled}``; ``continue`` and ``move`` targets come best placed first,
    #: ``keep`` targets the host the group runs on first (keeping it is "keep going"), and
    #: ``designate`` lists every computer that keeps a copy (a switch each).
    actions: list[dict[str, JsonValue]]
    #: ``not_owner``, ``no_successor``, ``successor_behind_offline``, ``host_reachable`` or
    #: ``takeover_waiting`` (a reachable majority should move the group by itself; after five more
    #: minutes ``continue`` is offered too).
    unavailable_reason: str | None = None
    previous_host: SuccessionPreviousHost | None = None
    unavailable_bots: list[SuccessionBot]
    last_attempt: SuccessionAttempt | None = None
    automatic: SuccessionAutomatic
    paused: SuccessionPaused | None = None
    moved_in: SuccessionMovedIn | None = None


class GroupsSuccessionPrepareParams(RoomParams):
    #: Runs on that computer's own gateway; any other answers ``target_not_local``.
    target_install_id: str


class SuccessionTarget(SuccessionComputer):
    operator_name: str | None = None


class GroupsSuccessionPrepareResult(Result):
    preview_id: str
    target: SuccessionTarget
    owner: SuccessionOwner
    behind_by: int
    at_risk: SuccessionAtRisk
    work: SuccessionWork
    unavailable_bots: list[SuccessionBot]
    #: ``{code: "host_may_be_running"}``, ``{code: "participant_not_fenced", names, count}`` or, in
    #: majority mode, ``{code: "voters_unreachable", names, count}``.
    cautions: list[dict[str, JsonValue]]


class GroupsSuccessionPromoteParams(RoomParams):
    target_install_id: str
    preview_id: str
    confirm: bool


class GroupsSuccessionKeepParams(RoomParams):
    #: The computer to keep; runs on either of the two.
    install_id: str


class GroupsSuccessionBranchLogParams(RoomParams):
    branch_id: str
    after_seq: int | None = None
    limit: int | None = None


class GroupsSuccessionBranchLogResult(Result):
    room_id: str
    branch_id: str
    events: list[dict[str, JsonValue]]
    cursor: int
    latest_seq: int
    has_more: bool


class GroupsSuccessionLearnParams(RoomParams):
    #: The ``authority.transition`` and ``custody.configured`` events after this computer's epoch, in
    #: log order, with the event just before the first transition. The proofs are the authority.
    events: list[dict[str, JsonValue]]


class GroupsSuccessionLearnResult(Result):
    room_id: str
    learned: bool
    #: ``not_superseded`` or ``already_following`` when nothing changed.
    reason: str | None = None
    #: ``continued_on_two`` when this host kept writing after an automatic move.
    state: str | None = None


class GroupsSuccessionMoveParams(RoomParams):
    #: On the host: hand the group over to this successor now.
    target_install_id: str


class GroupsSuccessionMoveNowParams(RoomParams):
    pass


class GroupsSuccessionContinueAnywayParams(RoomParams):
    pass


class GroupsSuccessionHandoverAllParams(Params):
    #: ``sleep``, ``stop`` or ``quit``.
    reason: Literal["sleep", "stop", "quit"]


class SuccessionSkipped(Result):
    room_id: str
    reason: str


class GroupsSuccessionHandoverAllResult(Result):
    moved: list[str]
    skipped: list[SuccessionSkipped]
    reason: str


method("groups.succession.status", params=GroupsSuccessionStatusParams, result=GroupsSuccessionStatusResult,
       doc="Whether the group's host can be reached from this computer, and what the owner may do.")
method("groups.succession.prepare", params=GroupsSuccessionPrepareParams, result=GroupsSuccessionPrepareResult,
       doc="On the target computer: what continuing the group there would mean. Changes nothing.")
method("groups.succession.promote", params=GroupsSuccessionPromoteParams, result=GroupsSuccessionStatusResult,
       doc="On the target computer: continue the group there, for its owner; returns the status (poll while moving).")
method("groups.succession.keep", params=GroupsSuccessionKeepParams, result=GroupsSuccessionStatusResult,
       doc="Resolve a group continued on two computers, from either one.")
method("groups.succession.branch_log", params=GroupsSuccessionBranchLogParams, result=GroupsSuccessionBranchLogResult,
       doc="Messages this computer wrote while cut off, kept apart after the group moved on (groups.log page shape).")
method("groups.succession.learn", params=GroupsSuccessionLearnParams, result=GroupsSuccessionLearnResult,
       doc="Hand this computer the chain of later hosts; it verifies it with pinned keys and steps down if replaced.")
method("groups.succession.move", params=GroupsSuccessionMoveParams, result=GroupsSuccessionStatusResult,
       doc="On the host, for the owner: hand the group over to a successor (signed handover), once the "
           "replies in progress finish.")
method("groups.succession.move_now", params=GroupsSuccessionMoveNowParams, result=GroupsSuccessionStatusResult,
       doc="On the host, for the owner: hand over a group waiting for its replies at once; those show as unknown.")
method("groups.succession.continue_anyway", params=GroupsSuccessionContinueAnywayParams,
       result=GroupsSuccessionStatusResult,
       doc="On a host paused to stay safe, for the owner: continue it here anyway.")
method("groups.succession.handover_all", params=GroupsSuccessionHandoverAllParams,
       result=GroupsSuccessionHandoverAllResult,
       doc="Hand every group this computer hosts to its best reachable standby (Desktop's sleep hook).")


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
