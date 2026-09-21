"""Contracts: system / process / slash-command / rollback / cron / browser / config RPCs
(handlers in ``tui_gateway/methods_tools.py``, browser helpers in ``methods_browser.py``).

Several results here are pass-throughs of dicts another module owns (``tools/process_registry.py``,
``tools/checkpoint_manager.py``, ``tools/cronjob_tools.py``): those declare every key the producer is
known to emit plus ``extra="allow"`` so a new upstream key never trips the strict gate.
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .registry import method


class _Open(Result):
    """A pass-through row whose closed set is owned by another module."""

    model_config = Result.model_config | {"extra": "allow"}


# ── system.battery ────────────────────────────────────────────────────────────────────────────


class BatteryCategory(WireEnum):
    """``agent/battery.py::battery_category`` colour bucket."""

    good = "good"
    warn = "warn"
    bad = "bad"
    critical = "critical"
    dim = "dim"


class SystemBatteryParams(Params):
    profile: str | None = None


class SystemBatteryResult(Result):
    available: bool
    percent: int | None = None
    plugged: bool | None = None
    category: BatteryCategory = BatteryCategory.dim


method("system.battery", params=SystemBatteryParams, result=SystemBatteryResult,
       doc="Host battery for the status bar; always resolves, ``available: false`` when unreadable.")


# ── process.* / agents.list ───────────────────────────────────────────────────────────────────


class ProcessStopParams(Params):
    session_id: str | None = None
    profile: str | None = None


class ProcessStopResult(Result):
    killed: int


method("process.stop", params=ProcessStopParams, result=ProcessStopResult,
       doc="Kill every background process in the registry (``/stop``), answering the count killed.")


class AgentsListParams(Params):
    profile: str | None = None


class AgentProcessRow(Result):
    session_id: str
    command: str
    status: str
    uptime: int


class AgentsListResult(Result):
    processes: list[AgentProcessRow] = Field(default_factory=list)


method("agents.list", params=AgentsListParams, result=AgentsListResult,
       doc="Registry-wide background process summary for ``/agents``.")


class ProcessListParams(Params):
    session_id: str
    profile: str | None = None


class ProcessEntry(_Open):
    """``tools/process_registry.py::list_sessions`` row plus the gateway's ``output_tail``."""

    session_id: str
    command: str = ""
    cwd: str | None = None
    pid: int | None = None
    owner_task_id: str | None = None
    started_at: str | None = None
    uptime_seconds: int | None = None
    status: str = "running"
    output_preview: str = ""
    output_tail: str | None = None
    session_scoped: bool | None = None
    watch_patterns: list[str] | None = None
    watch_hit: bool | None = None
    notify_on_complete: bool | None = None
    exit_code: int | None = None
    exited_at: float | None = None
    completion_reason: str | None = None
    detached: bool | None = None


class ProcessListResult(Result):
    processes: list[ProcessEntry] = Field(default_factory=list)


method("process.list", params=ProcessListParams, result=ProcessListResult,
       doc="Background processes owned by the caller's session (desktop status stack poll).")


class ProcessKillParams(Params):
    session_id: str
    process_id: str
    profile: str | None = None


class ProcessKillStatus(WireEnum):
    killed = "killed"
    already_exited = "already_exited"
    not_found = "not_found"
    error = "error"


class ProcessKillResult(_Open):
    """``tools/process_registry.py::kill_process`` snapshot; ``error`` rides on the failure statuses."""

    status: ProcessKillStatus
    session_id: str | None = None
    command: str | None = None
    exit_code: int | None = None
    completion_reason: str | None = None
    termination_source: str | None = None
    output: str | None = None
    error: str | None = None


method("process.kill", params=ProcessKillParams, result=ProcessKillResult,
       doc="Kill one background process the caller's session owns and return its output snapshot.")


# ── shell.exec / cli.exec ─────────────────────────────────────────────────────────────────────


class ShellExecParams(Params):
    command: str
    profile: str | None = None


class ShellExecResult(Result):
    stdout: str
    stderr: str
    code: int


method("shell.exec", params=ShellExecParams, result=ShellExecResult,
       doc="Run a safe (non-dangerous) shell command captured for ``!cmd`` / inline substitution.")


class CliExecParams(Params):
    argv: list[str]
    timeout: int | None = None
    profile: str | None = None


class CliExecResult(Result):
    blocked: bool
    code: int
    output: str
    hint: str | None = None


method("cli.exec", params=CliExecParams, result=CliExecResult,
       doc="Run ``hermes <argv>`` non-interactively and capture its output; ``blocked`` explains a refusal.")


# ── command catalog / resolve / dispatch / slash.exec ─────────────────────────────────────────


class CommandsCatalogParams(Params):
    session_id: str | None = None
    profile: str | None = None


class ArgumentMode(WireEnum):
    options = "options"
    text = "text"
    mixed = "mixed"


class CommandCatalogMeta(Result):
    argument_mode: ArgumentMode | None = None
    desktop: str | None = None


class CommandCategory(Result):
    name: str
    pairs: list[list[str]] = Field(default_factory=list)


class SkillCatalogEntry(Result):
    usage: int = 0
    origin: str = "local"


class CommandsCatalogResult(Result):
    pairs: list[list[str]] = Field(default_factory=list)
    sub: dict[str, list[str]] = Field(default_factory=dict)
    canon: dict[str, str] = Field(default_factory=dict)
    commands: dict[str, CommandCatalogMeta] = Field(default_factory=dict)
    categories: list[CommandCategory] = Field(default_factory=list)
    skills: dict[str, SkillCatalogEntry] = Field(default_factory=dict)
    skill_count: int = 0
    warning: str = ""


method("commands.catalog", params=CommandsCatalogParams, result=CommandsCatalogResult,
       doc="Categorized slash metadata (registry, quick, plugin, skill) for completion menus.")


class CommandResolveParams(Params):
    name: str | None = None
    profile: str | None = None


class CommandResolveResult(Result):
    canonical: str
    description: str
    category: str


method("command.resolve", params=CommandResolveParams, result=CommandResolveResult,
       doc="Canonical registry command for a name or alias.")


class DispatchType(WireEnum):
    """``apps/shared/src/slash.ts::parseCommandDispatch`` branches on this."""

    exec = "exec"
    alias = "alias"
    plugin = "plugin"
    send = "send"
    skill = "skill"
    prefill = "prefill"


class CommandDispatchParams(Params):
    name: str
    arg: str | None = None
    session_id: str | None = None
    profile: str | None = None


class CommandDispatchResult(Result):
    """One structured directive: ``exec``/``plugin`` carry ``output``; ``alias`` a ``target``;
    ``send``/``prefill``/``skill`` a ``message`` (UIs render ``display``, never ``message``)."""

    type: DispatchType
    output: str | None = None
    target: str | None = None
    message: str | None = None
    notice: str | None = None
    display: str | None = None
    name: str | None = None
    status: str | None = None


method("command.dispatch", params=CommandDispatchParams, result=CommandDispatchResult,
       doc="Run a quick/plugin/bundle/skill/built-in slash command and answer a structured directive.")


class SlashExecParams(Params):
    session_id: str
    command: str
    profile: str | None = None


class SlashExecResult(Result):
    """Plain worker/plugin text in ``output`` (+ ``warning``), or — when the command was rerouted to
    ``command.dispatch`` — that method's directive fields with ``type`` set."""

    output: str | None = None
    warning: str | None = None
    type: DispatchType | None = None
    target: str | None = None
    message: str | None = None
    notice: str | None = None
    display: str | None = None
    name: str | None = None
    status: str | None = None


method("slash.exec", params=SlashExecParams, result=SlashExecResult,
       doc="Execute a slash command against the session's slash worker (or a live/plugin shortcut).")


# ── insights.get / config.show ────────────────────────────────────────────────────────────────


class InsightsGetParams(Params):
    days: int | None = None
    profile: str | None = None


class InsightsGetResult(Result):
    days: int
    sessions: int
    messages: int


method("insights.get", params=InsightsGetParams, result=InsightsGetResult,
       doc="Session/message counts over the last ``days`` for the (optionally scoped) profile store.")


class ConfigShowParams(Params):
    profile: str | None = None


class ConfigSection(Result):
    title: str
    rows: list[list[str]] = Field(default_factory=list)


class ConfigShowResult(Result):
    model_config = Result.model_config | {"extra": "allow"}

    sections: list[ConfigSection] = Field(default_factory=list)


method("config.show", params=ConfigShowParams, result=ConfigShowResult,
       doc="Masked, display-ready config summary (model / agent / environment rows).")


# ── rollback.* ────────────────────────────────────────────────────────────────────────────────


class RollbackListParams(Params):
    session_id: str
    profile: str | None = None


class RollbackCheckpoint(Result):
    hash: str = ""
    timestamp: str = ""
    message: str = ""


class RollbackListResult(Result):
    enabled: bool
    checkpoints: list[RollbackCheckpoint] = Field(default_factory=list)


method("rollback.list", params=RollbackListParams, result=RollbackListResult,
       doc="Checkpoints for the session's cwd; ``enabled: false`` when checkpointing is off.")


class RollbackRestoreParams(Params):
    session_id: str
    hash: str
    file_path: str | None = None
    profile: str | None = None


class RollbackRestoreResult(_Open):
    """``tools/checkpoint_manager.py::restore`` outcome; ``history_removed`` is added for a full
    (non-file) restore that also rewound the live transcript."""

    success: bool
    restored_to: str | None = None
    reason: str | None = None
    directory: str | None = None
    file: str | None = None
    restored_files: list[str] | None = None
    skipped_user_edits: list[str] | None = None
    skipped_oversize: list[str] | None = None
    failed_deletes: list[str] | None = None
    history_removed: int | None = None
    error: str | None = None
    debug: JsonValue | None = None


method("rollback.restore", params=RollbackRestoreParams, result=RollbackRestoreResult,
       doc="Restore the working tree (or one file) to a checkpoint by hash or 1-based index.")


class RollbackDiffParams(Params):
    session_id: str
    hash: str
    profile: str | None = None


class RollbackDiffResult(Result):
    stat: str = ""
    diff: str = ""
    rendered: str | None = None


method("rollback.diff", params=RollbackDiffParams, result=RollbackDiffResult,
       doc="Diff between a checkpoint and the working tree, with an ANSI rendering sized to the TUI.")


# ── cron.manage ───────────────────────────────────────────────────────────────────────────────


class CronAction(WireEnum):
    list = "list"
    add = "add"
    remove = "remove"
    pause = "pause"
    resume = "resume"


class CronManageParams(Params):
    action: CronAction = CronAction.list
    name: str | None = None
    include_disabled: bool | str | None = None
    schedule: str | None = None
    prompt: str | None = None
    repeat: int | str | None = None
    continuity: bool | str | None = None
    deliver: str | None = None
    profile: str | None = None


class CronJobRow(_Open):
    """``tools/cronjob_job_args.py::_format_job``."""

    job_id: str
    name: str = ""
    skill: str | None = None
    skills: list[str] = Field(default_factory=list)
    prompt_preview: str = ""
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    schedule: str = "?"
    repeat: int | str | None = None
    deliver: str | None = None
    next_run_at: str | None = None
    last_run_at: str | None = None
    last_status: str | None = None
    last_delivery_error: str | None = None
    last_delivery_unverified: bool | None = None
    last_fire_error: str | None = None
    last_error: str | None = None
    enabled: bool = True
    state: str | None = None
    paused_at: str | None = None
    paused_reason: str | None = None
    workdir: str | None = None
    script: str | None = None
    reasoning_effort: str | None = None
    monitor_script: str | None = None
    monitor_url: str | None = None
    monitor_state: JsonValue | None = None
    no_agent: bool | None = None
    enabled_toolsets: list[str] | None = None
    continuity: bool | None = None
    context_from: list[str] | None = None
    attach_to_session: bool | None = None


class CronRemovedJob(Result):
    id: str
    name: str = ""
    schedule: str | None = None


class CronManageResult(_Open):
    """Pass-through of ``tools/cronjob_tools.py::cronjob`` JSON: ``list`` → ``jobs``/``count``
    (+ ``scoped`` when profile-scoped); ``add`` → the created job's summary + ``job``; ``remove`` →
    ``removed_job``; ``pause``/``resume`` → ``job``. A tool-level failure lands in ``error``."""

    success: bool | None = None
    error: str | None = None
    count: int | None = None
    jobs: list[CronJobRow] | None = None
    scoped: str | None = None
    gateway_running: bool | None = None
    warning: str | None = None
    job_id: str | None = None
    name: str | None = None
    skill: str | None = None
    skills: list[str] | None = None
    schedule: str | None = None
    repeat: int | str | None = None
    deliver: str | None = None
    next_run_at: str | None = None
    job: CronJobRow | None = None
    message: str | None = None
    guidance: JsonValue | None = None
    removed_job: CronRemovedJob | None = None


method("cron.manage", params=CronManageParams, result=CronManageResult,
       doc="List/add/remove/pause/resume cron jobs in the (optionally profile-scoped) cron store.")


# ── browser.manage ────────────────────────────────────────────────────────────────────────────


class BrowserAction(WireEnum):
    status = "status"
    connect = "connect"
    disconnect = "disconnect"


class BrowserManageParams(Params):
    action: BrowserAction = BrowserAction.status
    url: str | None = None
    session_id: str | None = None
    profile: str | None = None


class BrowserManageResult(Result):
    connected: bool
    url: str | None = None
    messages: list[str] | None = None


method("browser.manage", params=BrowserManageParams, result=BrowserManageResult,
       doc="Inspect, attach to, or drop the CDP browser the tools use; ``messages`` narrate a connect.")
