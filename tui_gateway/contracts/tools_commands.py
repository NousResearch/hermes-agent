"""Contracts: system / process / slash-command / rollback / cron / browser / config RPCs
(handlers in ``tui_gateway/methods_tools.py``, browser helpers in ``methods_browser.py``)."""

from __future__ import annotations

from pydantic import Field

from .base import Params, Result, WireEnum
from .registry import method


# ── system.battery ────────────────────────────────────────────────────────────────────────────


class BatteryCategory(WireEnum):
    """``agent/battery.py::battery_category`` colour bucket."""

    good = "good"
    warn = "warn"
    bad = "bad"
    critical = "critical"
    dim = "dim"


class SystemBatteryParams(Params):
    pass


class SystemBatteryResult(Result):
    available: bool
    percent: int | None
    plugged: bool | None
    category: BatteryCategory


method("system.battery", params=SystemBatteryParams, result=SystemBatteryResult,
       doc="Host battery for the status bar; always resolves, ``available: false`` when unreadable.")


# ── process.* / agents.list ───────────────────────────────────────────────────────────────────


class ProcessStopParams(Params):
    session_id: str | None = None


class ProcessStopResult(Result):
    killed: int


method("process.stop", params=ProcessStopParams, result=ProcessStopResult,
       doc="Kill every background process in the registry (``/stop``), answering the count killed.")


class AgentsListParams(Params):
    pass


class AgentProcessRow(Result):
    session_id: str
    command: str
    status: str
    uptime: int


class AgentsListResult(Result):
    processes: list[AgentProcessRow]


method("agents.list", params=AgentsListParams, result=AgentsListResult,
       doc="Registry-wide background process summary for ``/agents``.")


class ProcessListParams(Params):
    session_id: str


class ProcessEntry(Result):
    """``tools/process_registry.py:2011-2034`` plus ``server.py:3154``."""

    session_id: str
    command: str
    cwd: str | None
    pid: int | None
    owner_task_id: str | None
    started_at: str
    uptime_seconds: int
    status: str
    output_preview: str
    output_tail: str
    session_scoped: bool | None
    watch_patterns: list[str] | None
    watch_hit: bool | None
    notify_on_complete: bool | None
    exit_code: int | None
    detached: bool | None


class ProcessListResult(Result):
    processes: list[ProcessEntry]


method("process.list", params=ProcessListParams, result=ProcessListResult,
       doc="Background processes owned by the caller's session (desktop status stack poll).")


class ProcessKillParams(Params):
    session_id: str
    process_id: str


class ProcessKillStatus(WireEnum):
    killed = "killed"
    already_exited = "already_exited"
    not_found = "not_found"
    error = "error"


class ProcessKillResult(Result):
    """``tools/process_registry.py:1808-1907`` kill outcome."""

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
    command: str = ""


class ShellExecResult(Result):
    stdout: str
    stderr: str
    code: int


method("shell.exec", params=ShellExecParams, result=ShellExecResult,
       doc="Run a safe (non-dangerous) shell command captured for ``!cmd`` / inline substitution.")


class CliExecParams(Params):
    argv: list[str] = Field(default_factory=list)
    timeout: int = 240


class CliExecResult(Result):
    blocked: bool
    code: int
    output: str
    hint: str | None


method("cli.exec", params=CliExecParams, result=CliExecResult,
       doc="Run ``hermes <argv>`` non-interactively and capture its output; ``blocked`` explains a refusal.")


# ── command catalog / resolve / dispatch / slash.exec ─────────────────────────────────────────


class CommandsCatalogParams(Params):
    session_id: str | None = None


class ArgumentMode(WireEnum):
    options = "options"
    text = "text"
    mixed = "mixed"


class CommandCatalogMeta(Result):
    argument_mode: ArgumentMode | None
    desktop: str | None


class CommandCategory(Result):
    name: str
    pairs: list[list[str]]


class SkillCatalogEntry(Result):
    usage: int
    origin: str


class CommandsCatalogResult(Result):
    pairs: list[list[str]]
    sub: dict[str, list[str]]
    canon: dict[str, str]
    commands: dict[str, CommandCatalogMeta]
    categories: list[CommandCategory]
    skills: dict[str, SkillCatalogEntry]
    skill_count: int
    warning: str


method("commands.catalog", params=CommandsCatalogParams, result=CommandsCatalogResult,
       doc="Categorized slash metadata (registry, quick, plugin, skill) for completion menus.")


class CommandResolveParams(Params):
    name: str | None = None


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


class CommandDispatchResult(Result):
    """One structured directive: ``exec``/``plugin`` carry ``output``; ``alias`` a ``target``;
    ``send``/``prefill``/``skill`` a ``message`` (UIs render ``display``, never ``message``)."""

    type: DispatchType
    output: str | None
    target: str | None
    message: str | None
    notice: str | None
    display: str | None
    name: str | None
    status: str | None


method("command.dispatch", params=CommandDispatchParams, result=CommandDispatchResult,
       doc="Run a quick/plugin/bundle/skill/built-in slash command and answer a structured directive.")


class SlashExecParams(Params):
    session_id: str
    command: str


class SlashExecResult(Result):
    """Plain worker/plugin text in ``output`` (+ ``warning``), or — when the command was rerouted to
    ``command.dispatch`` — that method's directive fields with ``type`` set."""

    output: str | None
    warning: str | None
    type: DispatchType | None
    target: str | None
    message: str | None
    notice: str | None
    display: str | None
    name: str | None
    status: str | None


method("slash.exec", params=SlashExecParams, result=SlashExecResult,
       doc="Execute a slash command against the session's slash worker (or a live/plugin shortcut).")


# ── insights.get / config.show ────────────────────────────────────────────────────────────────


class InsightsGetParams(Params):
    days: int | None = None


class InsightsGetResult(Result):
    days: int
    sessions: int
    messages: int


method("insights.get", params=InsightsGetParams, result=InsightsGetResult,
       doc="Session/message counts over the last ``days`` for the (optionally scoped) profile store.")


class ConfigShowParams(Params):
    pass


class ConfigSection(Result):
    title: str
    rows: list[list[str]]


class ConfigShowResult(Result):
    sections: list[ConfigSection]


method("config.show", params=ConfigShowParams, result=ConfigShowResult,
       doc="Masked, display-ready config summary (model / agent / environment rows).")


# ── rollback.* ────────────────────────────────────────────────────────────────────────────────


class RollbackListParams(Params):
    session_id: str


class RollbackCheckpoint(Result):
    hash: str
    timestamp: str
    message: str


class RollbackListResult(Result):
    enabled: bool
    checkpoints: list[RollbackCheckpoint]


method("rollback.list", params=RollbackListParams, result=RollbackListResult,
       doc="Checkpoints for the session's cwd; ``enabled: false`` when checkpointing is off.")


class RollbackRestoreParams(Params):
    session_id: str
    hash: str
    file_path: str | None = None


class RollbackRestoreResult(Result):
    """``tools/checkpoint_manager.py:725-776`` plus ``methods_tools.py:935-948``."""

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
    debug: str | None = None


method("rollback.restore", params=RollbackRestoreParams, result=RollbackRestoreResult,
       doc="Restore the working tree (or one file) to a checkpoint by hash or 1-based index.")


class RollbackDiffParams(Params):
    session_id: str
    hash: str


class RollbackDiffResult(Result):
    stat: str
    diff: str
    rendered: str | None


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


class CronMonitorState(Result):
    """``cron/monitor.py:178-181`` persisted monitor state."""

    last_output_hash: str
    last_changed_at: str


class CronJobRow(Result):
    """Closed row from ``tools/cronjob_job_args.py:346-394``."""

    job_id: str
    name: str
    skill: str | None
    skills: list[str]
    prompt_preview: str
    model: str | None
    provider: str | None
    base_url: str | None
    schedule: str
    repeat: int | str | None
    deliver: str | None
    next_run_at: str | None
    last_run_at: str | None
    last_status: str | None
    last_delivery_error: str | None
    last_delivery_unverified: bool | None
    last_fire_error: str | None
    last_error: str | None
    enabled: bool
    state: str | None
    paused_at: str | None
    paused_reason: str | None
    script: str | None
    reasoning_effort: str | None
    monitor_script: str | None
    monitor_url: str | None
    monitor_state: CronMonitorState | None
    no_agent: bool | None
    enabled_toolsets: list[str] | None
    workdir: str | None
    continuity: bool | None
    context_from: list[str] | None
    attach_to_session: bool | None


class CronRemovedJob(Result):
    id: str
    name: str
    schedule: str | None


class CronManageResult(Result):
    """``methods_tools.py:1059-1083`` adapts the listed ``cronjob`` action outcomes."""

    success: bool
    error: str | None
    count: int | None
    jobs: list[CronJobRow] | None
    scoped: str | None
    gateway_running: bool | None
    warning: str | None
    job_id: str | None
    name: str | None
    skill: str | None
    skills: list[str] | None
    schedule: str | None
    repeat: int | str | None
    deliver: str | None
    next_run_at: str | None
    job: CronJobRow | None
    message: str | None
    guidance: list[str] | None
    removed_job: CronRemovedJob | None


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


class BrowserManageResult(Result):
    connected: bool
    url: str | None = None
    messages: list[str] | None = None


method("browser.manage", params=BrowserManageParams, result=BrowserManageResult,
       doc="Inspect, attach to, or drop the CDP browser the tools use; ``messages`` narrate a connect.")
