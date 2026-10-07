"""Contracts: delegation controls, handoff, message reactions, pet generation
and ``project.facts`` (handlers in ``tui_gateway/methods_session.py``).
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .common import MessageReaction, OpenModel, ProfileParams, SessionParams
from .registry import method

# ── delegation / subagent.steer ───────────────────────────────────────────────────────────────


class ActiveSubagent(OpenModel):
    """One live child from ``tools/delegate_tool_registry.py::list_active_subagents`` (the record
    is extended by the child runner — ``missed_steer`` etc. — so it stays open)."""

    subagent_id: str
    parent_id: str | None = None
    depth: int | None = None
    goal: str | None = None
    delegation_id: str | None = None
    model: str | None = None
    started_at: float | None = None
    status: str | None = None
    tool_count: int | None = None
    owner_agent_session_id: str | None = None


class DelegationStatusResult(Result):
    active: list[ActiveSubagent]
    paused: bool
    max_spawn_depth: int
    max_concurrent_children: int


method("delegation.status", params=ProfileParams, result=DelegationStatusResult,
       doc="Running subagent tree plus the spawn pause flag and limits.")


class DelegationPauseParams(ProfileParams):
    paused: bool = True


class DelegationPauseResult(Result):
    paused: bool


method("delegation.pause", params=DelegationPauseParams, result=DelegationPauseResult,
       doc="Block/unblock NEW spawns globally (active children keep running); returns the new state.")


class SubagentSteerParams(SessionParams):
    subagent_id: str
    text: str


class SteerStatus(WireEnum):
    queued = "queued"
    rejected = "rejected"


class SubagentSteerResult(Result):
    """``queued`` is not ``delivered``: a child past its final tool batch surfaces ``missed_steer``."""

    status: SteerStatus
    subagent_id: str
    text: str


method("subagent.steer", params=SubagentSteerParams, result=SubagentSteerResult,
       doc="Queue steering text into a live delegated child owned by this session.")


# ── handoff ───────────────────────────────────────────────────────────────────────────────────


class HandoffRequestParams(SessionParams):
    platform: str


class HandoffRequestResult(Result):
    queued: bool
    session_key: str
    platform: str
    home_name: str


method("handoff.request", params=HandoffRequestParams, result=HandoffRequestResult,
       doc="Queue a handoff to a messaging platform's home channel; the gateway watcher claims it.")


class HandoffStateResult(Result):
    """``state`` is pending | running | completed | failed, or '' when nothing was requested."""

    state: str
    platform: str
    error: str


method("handoff.state", params=SessionParams, result=HandoffStateResult,
       doc="Poll the handoff row for this session.")


class HandoffFailParams(SessionParams):
    error: str | None = None


class HandoffFailResult(Result):
    """``failed`` false when the watcher already claimed the row; ``state`` is what it is now."""

    failed: bool
    state: str


method("handoff.fail", params=HandoffFailParams, result=HandoffFailResult,
       doc="Fail a not-yet-claimed handoff (client poll timeout); CAS against the watcher.")


# ── message.react ─────────────────────────────────────────────────────────────────────────────


class ReactionAuthor(WireEnum):
    user = "user"
    agent = "agent"


class MessageReactParams(SessionParams):
    """``row_id`` is ``messages.id``; a not-yet-persisted live message names ``newest_role`` instead.
    ``emoji`` null clears; the same emoji again retracts."""

    row_id: int | None = None
    newest_role: str | None = None
    emoji: str | None = None
    author: ReactionAuthor | None = None


class MessageReactResult(Result):
    row_id: int
    reactions: list[MessageReaction]


method("message.react", params=MessageReactParams, result=MessageReactResult,
       doc="Set/clear one author's emoji reaction on a message; returns the row's full reaction list.")


# ── pets: generate / hatch / cancel / status ──────────────────────────────────────────────────


class PetCancelParams(ProfileParams):
    token: str | None = None


class PetCancelResult(Result):
    ok: bool


method("pet.cancel", params=PetCancelParams, result=PetCancelResult,
       doc="Stop an in-flight pet generate/hatch by token (idempotent).")


class PetGenProvider(Result):
    """``agent/pet/generate/imagegen.py::list_sprite_providers`` row."""

    name: str
    label: str
    default: bool


class PetGenerateStatusResult(Result):
    available: bool
    providers: list[PetGenProvider]


method("pet.generate.status", params=ProfileParams, result=PetGenerateStatusResult,
       doc="Whether pet generation is possible (a reference-capable image backend) and which providers.")


class PetGenerateParams(ProfileParams):
    """``prompt`` or a ``referenceImage`` data URL is required (the handler answers 4004 without one)."""

    prompt: str | None = None
    referenceImage: str | None = None  # noqa: N815 - wire key
    count: int | None = None
    style: str | None = None
    provider: str | None = None


class PetDraft(Result):
    index: int
    dataUri: str  # noqa: N815 - wire key


class PetGenerateResult(Result):
    ok: bool
    token: str
    drafts: list[PetDraft]


method("pet.generate", params=PetGenerateParams, result=PetGenerateResult,
       doc="Candidate base looks for a new pet (draft step); drafts also stream via pet.generate.progress.")


class PetHatchParams(ProfileParams):
    token: str
    name: str
    cancelToken: str | None = None  # noqa: N815 - wire key
    index: int | None = None
    description: str | None = None
    prompt: str | None = None
    style: str | None = None
    provider: str | None = None


# TODO(common): ``tui_gateway/server.py::_pet_sprite_payload`` is one shape for ``pet.info`` and
# ``pet.hatch``; consolidate with the pets contract module. Every field optional: ``pet.hatch``
# emits ``{}`` when the installed pet cannot be reloaded.
class PetSpritePayload(Result):
    slug: str | None = None
    displayName: str | None = None  # noqa: N815 - wire key
    mime: str | None = None
    spritesheetBase64: str | None = None  # noqa: N815 - wire key
    spritesheetRevision: str | None = None  # noqa: N815 - wire key
    frameW: int | None = None  # noqa: N815 - wire key
    frameH: int | None = None  # noqa: N815 - wire key
    framesPerState: int | None = None  # noqa: N815 - wire key
    framesByState: dict[str, int] | None = None  # noqa: N815 - wire key
    framesByRow: dict[str, int] | None = None  # noqa: N815 - wire key
    loopMs: int | None = None  # noqa: N815 - wire key
    scale: float | None = None
    stateRows: list[str] | None = None  # noqa: N815 - wire key


class PetHatchResult(Result):
    """The hatched pet is installed but NOT active (``pet.select`` adopts, ``pet.remove`` discards)."""

    ok: bool
    slug: str
    displayName: str  # noqa: N815 - wire key
    warnings: list[JsonValue] = Field(default_factory=list)
    pet: PetSpritePayload


method("pet.hatch", params=PetHatchParams, result=PetHatchResult,
       doc="Turn a base draft into a full spritesheet pet; progress streams via pet.hatch.progress.")


# ── project.facts ─────────────────────────────────────────────────────────────────────────────


class ProjectFactsParams(ProfileParams):
    cwd: str | None = None


class ProjectFacts(Result):
    """``agent/coding_context.py::project_facts_for`` — the system prompt's coding-context detection."""

    root: str
    manifests: list[str]
    packageManagers: list[str]  # noqa: N815 - wire key
    verifyCommands: list[str]  # noqa: N815 - wire key
    contextFiles: list[str]  # noqa: N815 - wire key


class ProjectFactsResult(Result):
    """``facts`` null outside a workspace (or when detection failed)."""

    facts: ProjectFacts | None = None


method("project.facts", params=ProjectFactsParams, result=ProjectFactsResult,
       doc="Structured project facts for a cwd so UIs don't re-sniff the workspace.")
