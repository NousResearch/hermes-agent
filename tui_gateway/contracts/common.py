"""Value shapes shared by several methods and events (declared once, imported everywhere)."""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, MethodParams, Params, Payload, Result, WireEnum


class Usage(Result):
    """``tui_gateway/server.py::_get_usage`` + ``agent/context_breakdown.py::context_usage_fields``."""

    model: str = ""
    input: int = 0
    output: int = 0
    reasoning: int = 0
    prompt: int = 0
    completion: int = 0
    total: int = 0
    calls: int = 0
    compressions: int | None = None
    context_used: int | None = None
    context_max: int | None = None
    context_percent: int | None = None
    context_source: str | None = None
    context_estimated: bool | None = None
    cache_hit_pct: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    avg_latency_s: float | None = None
    avg_tps: float | None = None
    active_subagents: int | None = None
    dev_credits_spent_micros: int | None = None
    cost_usd: float | None = None
    cost_status: str | None = None


class ProjectRef(Result):
    """``tui_gateway/server.py::_project_info_for_cwd``."""

    id: str
    slug: str
    name: str
    primary_path: str | None = None


class McpServerStatus(Result):
    """``tools/mcp_tool_discovery.py:get_mcp_status`` status row."""

    name: str
    transport: str
    tools: int
    connected: bool
    disabled: bool
    status: str
    error: str | None = None
    sampling: dict[str, int] | None = None


class SessionLiveInfo(Result):
    """``tui_gateway/server.py::_session_info`` — the ``session.info`` event and the ``info`` field of
    ``session.create`` / ``session.resume`` / ``session.activate`` results."""

    model: str | None = None
    provider: str = ""
    reasoning_effort: str = ""
    service_tier: str = ""
    fast: bool = False
    yolo: bool = False
    approval_mode: str = "manual"
    # Lazy and compute-host mirror paths omit these until the live agent is available.
    tools: dict[str, list[str]] | None = None
    skills: dict[str, list[str]] | None = None
    cwd: str = ""
    branch: str | None = None
    project: ProjectRef | None = None
    terminal_backend: str = ""
    personality: str = ""
    running: bool = False
    turn_started_at: float | None = None
    title: str = ""
    stored_session_id: str = ""
    desktop_contract: int | str | None = None
    version: str = ""
    release_date: str = ""
    update_behind: int | None = None  # ``hermes_cli.banner.get_update_result`` returns commits behind or None.
    update_command: str = ""
    usage: Usage | None = None
    profile_name: str | None = None
    mcp_servers: list[McpServerStatus] = Field(default_factory=list)
    system_prompt: str | None = None
    credential_warning: str | None = None
    lazy: bool | None = None


class StoredSessionRow(Result):
    """One ``sessions`` row as ``hermes_state`` lists it (``session.list`` / ``session.info`` rows /
    ``sessions.changed``)."""

    id: str
    title: str | None = None
    preview: str | None = None
    source: str | None = None
    model: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    last_active: float | None = None
    message_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    is_active: bool = False
    cwd: str | None = None
    git_branch: str | None = None
    git_repo_root: str | None = None
    parent_session_id: str | None = None
    pinned: bool | None = None
    unread: bool | None = None
    archived: bool | None = None
    actual_cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    handoff_platform: str | None = None
    handoff_state: str | None = None
    lineage_root_id: str | None = Field(default=None, alias="_lineage_root_id")
    lineage_ids: list[str] | None = Field(default=None, alias="_lineage_ids")


class TranscriptMessage(Result):
    """One ``session_history._history_to_messages`` display projection."""

    role: str
    text: str | None = None
    timestamp: float | None = None
    row_id: int | None = None
    display_kind: str | None = None
    # Display-kind metadata is producer-defined timeline data, so it remains recursive JSON.
    display_metadata: JsonValue | None = None
    name: str | None = None
    context: str | None = None
    # Tool arguments originate in arbitrary tool schemas, so the projection carries recursive JSON.
    args: dict[str, JsonValue] | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: JsonValue | None = None
    codex_reasoning_items: JsonValue | None = None
    codex_message_items: JsonValue | None = None


class SubagentStatus(WireEnum):
    """Lifecycle of one delegated child (``tools/delegate_tool_child_run.py``); ``failed`` /
    ``error`` / ``timeout`` / ``interrupted`` / ``completed`` are terminal."""

    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    error = "error"
    timeout = "timeout"
    interrupted = "interrupted"


TERMINAL_SUBAGENT_STATUSES = frozenset({
    SubagentStatus.completed, SubagentStatus.failed, SubagentStatus.error,
    SubagentStatus.timeout, SubagentStatus.interrupted,
})


class ApprovalChoice(WireEnum):
    once = "once"
    session = "session"
    always = "always"
    deny = "deny"


class PendingApproval(Result):
    """One unresolved ``tools/approval.py`` gateway queue entry as ``server._approval_request_payload``
    renders it (command redacted; ``choices`` precomputed)."""

    request_id: str | None = None
    command: str | None = None
    description: str | None = None
    pattern_key: str | None = None
    pattern_keys: list[str] | None = None
    allow_permanent: bool | None = None
    allow_session: bool | None = None
    smart_denied: bool | None = None
    choices: list[ApprovalChoice] | None = None
    tool_name: str | None = None


class MessageReaction(Result):
    """One persisted reaction row (``hermes_state_messages.set_message_reaction``); ``seen`` is
    stamped once announced."""

    emoji: str
    author: str
    at: float | None = None
    seen: bool | None = None


class SessionParams(MethodParams):
    """A method addressed at one live session."""

    session_id: str


class OkResult(Result):
    ok: bool = True


class StatusResult(Result):
    status: str


class EmptyResult(Result):
    pass


class EmptyPayload(Payload):
    pass


__all__ = [
    "TERMINAL_SUBAGENT_STATUSES", "EmptyPayload", "EmptyResult", "McpServerStatus", "MessageReaction", "OkResult", "PendingApproval",
    "ProjectRef", "SessionLiveInfo", "SessionParams", "StatusResult", "StoredSessionRow",
    "SubagentStatus", "TranscriptMessage", "Usage",
]
