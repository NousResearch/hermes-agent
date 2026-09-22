"""Bot Marketplace catalog and atomic profile-install wire contracts."""

from __future__ import annotations

from pydantic import Field

from .base import Params, Result, WireEnum
from .common import ProfileParams
from .registry import method


class BotTier(WireEnum):
    official = "official"
    community = "community"


class BotCredentialPolicy(WireEnum):
    none = "none"
    copy_api_keys = "copy_api_keys"


class BotSetupState(WireEnum):
    ready = "ready"
    needs_setup = "needs_setup"


class BotCatalogProfile(Result):
    suggested_name: str
    description: str
    soul: str
    starter_prompt: str


class BotCatalogCapabilities(Result):
    skills: list[str] = Field(default_factory=list)
    toolsets: list[str] = Field(default_factory=list)


class BotRequirementKind(WireEnum):
    toolset = "toolset"
    command = "command"
    connector = "connector"
    plugin = "plugin"


class BotRequirementStatus(WireEnum):
    ready = "ready"
    needs_setup = "needs_setup"


class BotSetupRequirementResult(Result):
    kind: BotRequirementKind
    id: str
    required: bool
    purpose: str


class BotCatalogSetup(Result):
    requirements: list[BotSetupRequirementResult] = Field(default_factory=list)


class BotCatalogRoutine(Result):
    id: str
    name: str
    prompt: str
    schedule: str


class BotCatalogPresentation(Result):
    emoji: str
    color: str


class BotCatalogEntryResult(Result):
    name: str
    version: str
    maintainer: str
    tier: BotTier
    category: str
    tags: list[str] = Field(default_factory=list)
    title: str
    summary: str
    profile: BotCatalogProfile
    capabilities: BotCatalogCapabilities
    setup: BotCatalogSetup
    routines: list[BotCatalogRoutine] = Field(default_factory=list)
    presentation: BotCatalogPresentation


class RemovedBotResult(Result):
    name: str
    reason: str


class BotsCatalogParams(ProfileParams):
    pass


class BotsCatalogResult(Result):
    entries: list[BotCatalogEntryResult] = Field(default_factory=list)
    removed: list[RemovedBotResult] = Field(default_factory=list)


method(
    "bots.catalog",
    params=BotsCatalogParams,
    result=BotsCatalogResult,
    doc="Return the reviewed Bot Marketplace catalog and removal list resolved by this backend.",
)


class BotsInstalledParams(ProfileParams):
    pass


class InstalledBotPresentation(Result):
    emoji: str
    color: str


class InstalledBotResult(Result):
    profile: str
    catalog_name: str
    title: str
    summary: str
    setup_state: BotSetupState
    presentation: InstalledBotPresentation


class BotsInstalledResult(Result):
    bots: list[InstalledBotResult] = Field(default_factory=list)


method(
    "bots.installed",
    params=BotsInstalledParams,
    result=BotsInstalledResult,
    doc="List installed bot profiles from local immutable manifests without runtime readiness probes.",
)


class BotsInstallParams(ProfileParams):
    catalog_name: str
    name: str
    source_profile: str
    credentials: BotCredentialPolicy = BotCredentialPolicy.none


class BotsInstallResult(Result):
    ok: bool = True
    committed: bool
    name: str
    path: str
    catalog_name: str
    catalog_version: str
    source_profile: str
    copied_credentials: list[str] = Field(default_factory=list)
    oauth_setup_required: list[str] = Field(default_factory=list)
    setup_state: BotSetupState
    setup_requirements: list[str] = Field(default_factory=list)
    post_publish_warnings: list[str] = Field(default_factory=list)


method(
    "bots.install",
    params=BotsInstallParams,
    result=BotsInstallResult,
    doc="Resolve a reviewed catalog name server-side and atomically publish one complete bot profile.",
)


class BotRequirementReadiness(BotSetupRequirementResult):
    status: BotRequirementStatus
    action: str | None = None
    detail: str | None = None


class BotRuntimeReadiness(Result):
    ok: bool
    provider: str | None = None
    model: str | None = None
    source: str | None = None
    error: str | None = None
    action: str | None = None
    reused_sign_in: bool = False


class BotFirstTaskStatus(WireEnum):
    pending = "pending"
    complete = "complete"


class BotFirstTaskReadiness(Result):
    status: BotFirstTaskStatus
    session_id: str | None = None


class BotsStatusParams(ProfileParams):
    pass


class BotsStatusResult(Result):
    profile: str
    catalog_name: str
    setup_state: BotSetupState
    requirements: list[BotRequirementReadiness] = Field(default_factory=list)
    runtime: BotRuntimeReadiness
    first_task: BotFirstTaskReadiness
    starter_prompt: str | None = None
    can_start_first_task: bool
    can_activate_routines: bool


method(
    "bots.status",
    params=BotsStatusParams,
    result=BotsStatusResult,
    doc="Probe an installed bot profile's real runtime, declared dependencies, and first Bot Chat task proof.",
)
method(
    "bots.setup",
    params=BotsStatusParams,
    result=BotsStatusResult,
    doc="Resume Finish setup by re-probing authoritative state; client readiness claims are not accepted.",
)


class BotRoutineState(WireEnum):
    paused = "paused"
    active = "active"


class BotRoutineResult(Result):
    id: str
    state: BotRoutineState
    job_id: str | None = None
    schedule: str | None = None
    timezone: str | None = None
    destination: str | None = None


class BotRoutineListItem(Result):
    id: str
    name: str
    prompt: str
    schedule: str
    state: BotRoutineState
    job_id: str | None = None
    timezone: str | None = None
    destination: str | None = None


class BotsRoutinesListParams(ProfileParams):
    pass


class BotsRoutinesListResult(Result):
    routines: list[BotRoutineListItem] = Field(default_factory=list)


class BotsRoutineActivateParams(ProfileParams):
    routine_id: str
    schedule: str
    timezone: str
    destination: str


class BotsRoutinePauseParams(ProfileParams):
    routine_id: str


method(
    "bots.routines.list",
    params=BotsRoutinesListParams,
    result=BotsRoutinesListResult,
    doc="List reviewed routine blueprints and their authoritative cron activation state.",
)
method(
    "bots.routines.activate",
    params=BotsRoutineActivateParams,
    result=BotRoutineResult,
    doc="Explicitly activate one ready bot routine with confirmed schedule, IANA timezone, and destination.",
)
method(
    "bots.routines.pause",
    params=BotsRoutinePauseParams,
    result=BotRoutineResult,
    doc="Pause an activated bot routine through the canonical cron API.",
)
