"""Secret-free peer selection and server-owned mixed RoomLink setup."""
from typing import Literal

from pydantic import Field

from .base import Params, Result
from .common import ProfileParams
from .groups_bot_relay import RoomExecutionPolicy, Room
from .registry import canonical_method


class PeerCatalog(Result):
    installation_id: str
    protocol_versions: list[int]
    link_modes: list[str]
    persistent_process: bool
    text: bool
    attachments: bool
    execution_policy: RoomExecutionPolicy
    catalog_digest: str


class PeerBot(Result):
    kind: Literal['peer']
    route_id: str
    alias: str
    profile: str
    reachability: Literal['reachable', 'unreachable', 'unauthenticated', 'unsupported']
    installation_id: str | None = None
    room_link: PeerCatalog | None = None


class PeersRosterResult(Result):
    bots: list[PeerBot]


class PeersDMParams(ProfileParams):
    route_id: str
    message: str = Field(min_length=1, max_length=262144)
    request_id: str = Field(min_length=1, max_length=256)


class PeersDMResult(Result):
    route_id: str
    session_id: str
    status: Literal['completed', 'queued', 'unknown']
    reply: str
    idempotency_supported: bool
    idempotency_durable: bool = False
    run_id: str | None = None


class LocalBotMember(Params):
    kind: Literal['local']
    profile: str
    member_id: str
    handle: str
    display_name: str = ''


class PeerBotMember(Params):
    kind: Literal['peer']
    route_id: str
    member_id: str
    handle: str
    display_name: str = ''


class GroupsCreateMixedParams(ProfileParams):
    room_id: str
    name: str
    members: list[LocalBotMember | PeerBotMember] = Field(min_length=2, max_length=6)


class MixedLocalTarget(Result):
    kind: Literal['local']
    profile: str


class MixedPeerTarget(Result):
    kind: Literal['peer']
    peer_id: str
    installation_id: str
    profile: str
    capability_digest: str


class MixedRoomMember(Result):
    member_id: str
    profile: str
    handle: str
    display_name: str | None = None
    target: MixedLocalTarget | MixedPeerTarget


class MixedRoom(Room):
    members: list[MixedRoomMember]


class GroupsCreateMixedResult(Result):
    room: MixedRoom


canonical_method('peers.roster', params=ProfileParams, result=PeersRosterResult,
                 doc='Probe this profile\'s authenticated peer bots without exposing credentials or endpoints.')
canonical_method('peers.dm', params=PeersDMParams, result=PeersDMResult,
                 doc='Deliver to an opaque peer route\'s actual canonical Bot Chat.')
canonical_method('groups.create_mixed', params=GroupsCreateMixedParams, result=GroupsCreateMixedResult,
                 doc='Invite peer members, create a local room and register verified RoomLinks; compensate failures.')
