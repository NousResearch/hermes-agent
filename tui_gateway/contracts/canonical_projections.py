"""Canonical projections with demonstrated differences from standalone wire."""
from typing import Literal

from .base import JsonValue, Params, Result
from .common import ProfileParams
from .canonical_sessions import CanonicalSessionParams
from .config_free_tier_control import ConfigGetParams, ConfigGetResult, ConfigSetResult
from .groups_bot_relay import GroupsApproveResult, GroupsCreateResult, GroupsRetryResult, RoomMemberInput, RoomParams
from .prompt_voice import AttachedImageResult, ImageAttachBytesParams
from .profiles_vault_complete_foreign_subagents import ProfileCanonicalSession, ProfileRow, ProfilesListResult
from .registry import canonical_method


class CanonicalConfigGetResult(ConfigGetResult):
    key: str | None = None
    scope: str | None = None


class CanonicalConfigSetParams(CanonicalSessionParams):
    """``model`` is recognised only to be refused with ``use_session_mutation_model``: a model pick is
    the revision-fenced ``session.mutate`` operation, never a config write."""

    key: Literal['busy', 'verbose', 'yolo', 'model']
    value: JsonValue = None
    confirm_expensive_model: bool | None = None


class CanonicalAttachedImageResult(AttachedImageResult):
    mime: str


class CanonicalProfileRow(ProfileRow):
    last_session: ProfileCanonicalSession | None = None


class CanonicalProfilesResult(ProfilesListResult):
    profiles: list[CanonicalProfileRow]


class CanonicalProfilesParams(ProfileParams):
    include_sessions: bool = True


class CanonicalRoomCreateParams(RoomParams):
    name: str
    members: list[RoomMemberInput]


class CanonicalRoomAttemptParams(RoomParams):
    member_id: str
    task_id: str
    execution_generation: int


class CanonicalRoomApproveParams(CanonicalRoomAttemptParams):
    choice: Literal['once', 'deny']
    request_id: str


class CanonicalBotDeliverParams(Params):
    profile: str | None = None
    message: str
    id: str
    session_id: str | None = None
    author: dict[str, JsonValue] | None = None
    notification_category: str | None = None


class CanonicalBotDeliverResult(Result):
    status: str
    delivery_id: str
    profile_home: str
    session_id: str
    admission_id: str
    message: str
    reply: str | None = None
    retry_admission_id: str | None = None
    error: str | None = None
    reason: str | None = None


for _name, _params, _result in (
    ('config.get', ConfigGetParams, CanonicalConfigGetResult),
    ('config.set', CanonicalConfigSetParams, ConfigSetResult),
    ('profiles.list', CanonicalProfilesParams, CanonicalProfilesResult),
    ('image.attach_bytes', ImageAttachBytesParams, CanonicalAttachedImageResult),
    ('groups.retry', CanonicalRoomAttemptParams, GroupsRetryResult),
    ('groups.create', CanonicalRoomCreateParams, GroupsCreateResult),
    ('groups.approve', CanonicalRoomApproveParams, GroupsApproveResult),
    ('bot_relay.deliver', CanonicalBotDeliverParams, CanonicalBotDeliverResult),
):
    canonical_method(_name, params=_params, result=_result)
