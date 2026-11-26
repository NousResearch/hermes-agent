"""Canonical projections with demonstrated differences from standalone wire."""
from typing import Literal

from .base import JsonValue, Params, Result
from .common import ProfileParams
from .canonical_sessions import CanonicalSessionParams
from .config_free_tier_control import ConfigGetParams, ConfigGetResult, ConfigSetResult
from .groups_bot_relay import (
    BotRelayDeliverResult, GroupsApproveResult, GroupsCreateResult, GroupsRetryResult, RoomMemberInput, RoomParams,
)
from .prompt_voice import AttachedImageResult, ImageAttachBytesParams
from .profiles_vault_complete_foreign_subagents import ProfileCanonicalSession, ProfileRow, ProfilesListResult
from .registry import canonical_method


class CanonicalConfigGetResult(ConfigGetResult):
    key: str | None = None
    scope: str | None = None
    options: list[Literal['normal', 'fast', 'auto', 'cold', 'ultrafast']] | None = None


class CanonicalConfigSetParams(ProfileParams):
    """Session controls (``busy`` / ``verbose`` / ``yolo`` / ``show_reasoning`` / ``service_tier``, which
    need ``session_id``) and the TUI's profile preferences (``gateway/session_tui_config.py``;
    ``session_id`` optional). ``model`` is recognised only to be refused with
    ``use_session_mutation_model``: a model pick is the revision-fenced ``session.mutate`` operation,
    never a config write."""

    session_id: str | None = None
    key: str
    value: JsonValue = None
    confirm_expensive_model: bool | None = None
    # ``reasoning`` (``global`` = ``/reasoning --global``) and ``yolo`` (``global`` = approvals.mode).
    scope: Literal['session', 'global'] | None = None


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


class CanonicalBotCancelParams(Params):
    profile: str | None = None
    id: str
    author_id: str | None = None


class CanonicalBotCancelResult(Result):
    """``gateway/session_bot_cancel.py``: ``cancelled`` (withdrawn before it ran, or its pending
    automatic retry withdrawn), ``stopping`` (its running turn was told to stop), or ``finished``
    (nothing left to cancel); ``receipt_status`` is the delivery receipt's status after it."""

    delivery_id: str
    status: Literal['cancelled', 'stopping', 'finished']
    receipt_status: str


for _name, _params, _result in (
    ('config.get', ConfigGetParams, CanonicalConfigGetResult),
    ('config.set', CanonicalConfigSetParams, ConfigSetResult),
    ('profiles.list', CanonicalProfilesParams, CanonicalProfilesResult),
    ('image.attach_bytes', ImageAttachBytesParams, CanonicalAttachedImageResult),
    ('groups.retry', CanonicalRoomAttemptParams, GroupsRetryResult),
    ('groups.create', CanonicalRoomCreateParams, GroupsCreateResult),
    ('groups.approve', CanonicalRoomApproveParams, GroupsApproveResult),
    ('bot_relay.deliver', CanonicalBotDeliverParams, BotRelayDeliverResult),
    # The sender's finalize of a refusal marked ``admitted: false``: the same identity, answered
    # with the id's terminal not-sent receipt or its existing admission (``gateway/session_bot.py``).
    ('bot_relay.settle', CanonicalBotDeliverParams, BotRelayDeliverResult),
    # The sender's own cancel by delivery id, decided on the owner beside its one retry.
    ('bot_relay.cancel', CanonicalBotCancelParams, CanonicalBotCancelResult),
):
    canonical_method(_name, params=_params, result=_result)
