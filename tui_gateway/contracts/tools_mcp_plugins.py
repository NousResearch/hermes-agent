"""Tools / toolsets / MCP servers / plugins / skills / learning graph / reload contracts
(``tui_gateway/methods_tools.py``).

Every ``mcp.servers.*``, ``mcp.catalog``, ``skills.manage`` and ``plugins.manage`` handler runs under
``_profile_scoped_rpc`` and the desktop routes them via ``requestGatewayForProfile`` /
``requestForBot``, so all of them accept the optional ``profile`` key.
"""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Params, Result, WireEnum
from .common import OpenModel, ProfileParams, SessionLiveInfo
from .registry import method


class _SessionScoped(Params):
    """Handlers that look a live session up with ``_sessions.get(params.get("session_id"))``: an
    absent / unknown id falls back to the launch profile's config, so it is never required."""

    session_id: str | None = None


# ── tools / toolsets ──────────────────────────────────────────────────────────────────────────


class ToolsetRow(Result):
    """One row of ``methods_tools._toolset_rows``; ``tools`` only when the caller asked for them
    (``tools.list``)."""

    name: str
    description: str
    tool_count: int
    enabled: bool
    tools: list[str] | None = None


class ToolsetsListResult(Result):
    toolsets: list[ToolsetRow]


method("tools.list", params=_SessionScoped, result=ToolsetsListResult,
       doc="Every toolset with its resolved tool names, flagged against the session's (or config's) enabled set.")

method("toolsets.list", params=_SessionScoped, result=ToolsetsListResult,
       doc="Toolset summaries (no tool names) for the desktop Toolsets tab.")


class ToolShowRow(Result):
    name: str
    description: str


class ToolShowSection(Result):
    name: str
    tools: list[ToolShowRow]


class ToolsShowResult(Result):
    sections: list[ToolShowSection]
    total: int


method("tools.show", params=_SessionScoped, result=ToolsShowResult,
       doc="The /tools listing grouped by toolset, including tools deferred behind the tool_search bridge.")


class ToolsAction(WireEnum):
    enable = "enable"
    disable = "disable"


class ToolsConfigureParams(Params):
    """``names`` are toolset keys or ``server:tool`` MCP targets; with ``session_id`` the live session's
    profile is authoritative and its agent is rebuilt."""

    action: ToolsAction
    names: list[str]
    session_id: str | None = None
    profile: str | None = None


class ToolsConfigureResult(Result):
    changed: list[str]
    enabled_toolsets: list[str]
    info: SessionLiveInfo | None = None
    missing_servers: list[str]
    reset: bool
    unknown: list[str]


method("tools.configure", params=ToolsConfigureParams, result=ToolsConfigureResult,
       doc="Persist a toolset / MCP enable-disable change and rebuild the session agent so it takes effect now.")


# ── reload ────────────────────────────────────────────────────────────────────────────────────


class ReloadEnvParams(Params):
    pass


class ReloadEnvResult(Result):
    updated: int


method("reload.env", params=ReloadEnvParams, result=ReloadEnvResult,
       doc="Re-read ~/.hermes/.env (CLI /reload parity); built agents keep their pool until /new.")


class ReloadMcpParams(Params):
    """Without ``confirm`` the handler may answer ``confirm_required`` (per ``approvals.mcp_reload_confirm``);
    ``always`` persists the opt-out; ``rev`` is the config revision the caller wants loaded (coalescing)."""

    session_id: str | None = None
    confirm: bool = False
    always: bool = False
    rev: str | None = None


class ReloadMcpStatus(WireEnum):
    confirm_required = "confirm_required"
    reloaded = "reloaded"


class ReloadMcpResult(Result):
    status: ReloadMcpStatus
    message: str | None = None
    loaded_rev: str | None = None
    coalesced: bool | None = None
    turn_isolation: bool | None = None
    host_ack: JsonValue | None = None


method("reload.mcp", params=ReloadMcpParams, result=ReloadMcpResult,
       doc="Tear down and rediscover MCP servers for every live session (prompt cache is invalidated).")


# ── skills ────────────────────────────────────────────────────────────────────────────────────


class SkillsAction(WireEnum):
    list = "list"
    search = "search"
    install = "install"
    browse = "browse"
    inspect = "inspect"


class SkillsManageParams(ProfileParams):
    """``query`` is the search text / hub identifier / browse page (digits); ``page`` / ``page_size``
    apply to ``browse``."""

    action: SkillsAction = SkillsAction.list
    query: str | None = None
    page: int | None = None
    page_size: int | None = None


class SkillHubHit(Result):
    name: str
    description: str


class SkillBrowseItem(OpenModel):
    """``hermes_cli.skills_hub.browse_skills`` row."""

    name: str = ""
    description: str = ""
    source: str = ""
    trust: str | None = None
    identifier: str | None = None


class SkillInspectInfo(OpenModel):
    """``hermes_cli.skills_hub.inspect_skill``; ``{}`` when the identifier resolves nowhere."""

    name: str | None = None
    description: str | None = None
    source: str | None = None
    identifier: str | None = None
    tags: list[str] | None = None
    skill_md_preview: str | None = None


class SkillsManageResult(Result):
    """Shape follows the action: ``list`` → ``skills`` (category → names); ``search`` → ``results``;
    ``install`` → ``installed`` + ``name``; ``browse`` → ``items`` + paging; ``inspect`` → ``info``."""

    skills: dict[str, list[str]] | None = None
    results: list[SkillHubHit] | None = None
    installed: bool | None = None
    name: str | None = None
    items: list[SkillBrowseItem] | None = None
    page: int | None = None
    total_pages: int | None = None
    total: int | None = None
    info: SkillInspectInfo | None = None


method("skills.manage", params=SkillsManageParams, result=SkillsManageResult,
       doc="Skills hub backend: list the profile's skills or search / browse / inspect / install from the hub.")


class SkillsReloadParams(Params):
    """``session_id`` binds the rescan to that session's profile and workspace (project skills)."""

    session_id: str | None = None


class SkillCommandRef(Result):
    name: str
    description: str = ""


class SkillsReloadDiff(OpenModel):
    """``agent.skill_commands.reload_skills``."""

    added: list[SkillCommandRef] = Field(default_factory=list)
    removed: list[SkillCommandRef] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    total: int = 0
    commands: int = 0


class SkillsReloadResult(Result):
    output: str
    result: SkillsReloadDiff


method("skills.reload", params=SkillsReloadParams, result=SkillsReloadResult,
       doc="Re-scan skill dirs; the pre-rendered ``output`` is what /reload-skills prints.")


# ── learning graph (/journey) ─────────────────────────────────────────────────────────────────


class LearningFramesParams(Params):
    cols: int | None = None
    rows: int | None = None
    frames: int | None = None


class LearningFrame(Result):
    """``agent.learning_graph_render.render_graph`` projection; ``grid`` rows are lists of
    ``[text, styleKey, alpha?, hexOverride?]`` runs."""

    reveal: float
    date: str
    visible: int
    grid: list[list[JsonValue]]
    labels: list[dict[str, JsonValue]] = Field(default_factory=list)


class LearningLegendItem(Result):
    glyph: str
    label: str
    style: str | None = None
    color: str | None = None


class LearningNodeRow(Result):
    id: str
    glyph: str
    label: str
    fullLabel: str  # noqa: N815 — wire key from learning_graph_render._bucket_rows
    meta: str
    body: str
    style: str


class LearningBucketRow(Result):
    index: int
    label: str
    date: str
    skills: int
    memories: int
    total: int
    category: str | None = None
    color: str | None = None
    nodes: list[LearningNodeRow]


class LearningAxis(Result):
    start: str
    end: str


class LearningFramesResult(Result):
    frames: list[LearningFrame]
    legend: list[LearningLegendItem]
    categories: list[LearningLegendItem]
    buckets: list[LearningBucketRow]
    summary: list[str]
    axis: LearningAxis
    count: int
    cols: int
    rows: int


method("learning.frames", params=LearningFramesParams, result=LearningFramesResult,
       doc="Pre-render the /journey timeline (frames + legend/summary) so the TUI walks it locally.")


class LearningNodeParams(Params):
    id: str | None = None


class LearningEditParams(LearningNodeParams):
    content: str | None = None


class LearningMutationResult(Result):
    """``agent.learning_mutations`` — ``ok: false`` carries the reason in ``message``."""

    ok: bool
    message: str | None = None


class LearningDetailResult(LearningMutationResult):
    kind: str | None = None
    id: str | None = None
    label: str | None = None
    content: str | None = None


method("learning.detail", params=LearningNodeParams, result=LearningDetailResult,
       doc="Node content (SKILL.md or memory chunk) for an edit prefill.")

method("learning.delete", params=LearningNodeParams, result=LearningMutationResult,
       doc="Archive a skill (restorable via curator) or remove a memory chunk.")

method("learning.edit", params=LearningEditParams, result=LearningMutationResult,
       doc="Rewrite a node's content (SKILL.md or memory chunk).")


# ── MCP catalog + per-profile server lifecycle ────────────────────────────────────────────────


class McpCatalogEntry(Result):
    name: str
    description: str
    connector_slug: str | None = None
    installed: bool
    enabled: bool
    requires: list[str]
    transport: str


class McpCatalogResult(Result):
    servers: list[McpCatalogEntry]


method("mcp.catalog", params=ProfileParams, result=McpCatalogResult,
       doc="Curated MCP presets with per-profile installed/enabled state and the env keys each needs.")


class McpServerSource(WireEnum):
    config = "config"
    plugin = "plugin"


class McpServerSummary(Result):
    """``tui_gateway/mcp_rpc_helpers.summarize_server`` — a server's config without secret values."""

    name: str
    transport: str
    url: str | None = None
    command: str | None = None
    args: list[str]
    env: list[str]
    auth: str | None = None
    oauth_tokens_present: bool | None = None
    enabled: bool
    tools: JsonValue | None = None
    source: McpServerSource
    plugin: str | None = None


class McpServersListResult(Result):
    servers: list[McpServerSummary]


method("mcp.servers.list", params=ProfileParams, result=McpServersListResult,
       doc="Configured MCP servers for the (scoped) profile, secrets redacted to env-key names.")


class McpRuntimeStatus(WireEnum):
    connected = "connected"
    disabled = "disabled"
    connecting = "connecting"
    failed = "failed"
    lazy = "lazy"
    configured = "configured"


class McpServerRuntimeRow(Result):
    """Safe projection of ``tools.mcp_tool_discovery.get_mcp_status`` rows."""

    name: str
    transport: str
    tools: int
    connected: bool
    disabled: bool
    status: McpRuntimeStatus
    source: McpServerSource
    plugin: str | None = None


class McpServersStatusResult(Result):
    servers: list[McpServerRuntimeRow]
    checked_at: int


method("mcp.servers.status", params=ProfileParams, result=McpServersStatusResult,
       doc="Cached runtime state per configured server; never connects, probes, or starts auth.")


class McpServerNameParams(ProfileParams):
    name: str


class McpServersAddParams(McpServerNameParams):
    """``preset`` (catalog id) and/or ``config`` (url/command/args/env/headers/auth/tools); a
    ``bearer_token`` is written to the profile's .env, only the header template persists."""

    preset: str | None = None
    config: dict[str, JsonValue] | None = None
    bearer_token: str | None = None


class McpServersAddResult(Result):
    ok: bool
    name: str
    server: McpServerSummary


method("mcp.servers.add", params=McpServersAddParams, result=McpServersAddResult,
       doc="Add a server to the profile's config from a catalog preset and/or an explicit config.")


class McpServersSetApiKeyParams(McpServerNameParams):
    value: str
    env_var: str | None = None


class McpServersSetApiKeyResult(Result):
    ok: bool
    name: str
    env_var: str
    server: McpServerSummary


method("mcp.servers.set_api_key", params=McpServersSetApiKeyParams, result=McpServersSetApiKeyResult,
       doc="Store a credential in the profile's .env and reference it from the server config (header or env).")


class McpProbeTool(Result):
    name: str
    description: str


class McpServersTestResult(Result):
    """``ok: false`` carries ``error``; ``prompts`` / ``resources`` are only counted on success."""

    ok: bool
    tools: list[McpProbeTool]
    error: str | None = None
    prompts: int | None = None
    resources: int | None = None
    oauth_needed: bool
    oauth_tokens_present: bool | None = None


method("mcp.servers.test", params=McpServerNameParams, result=McpServersTestResult,
       doc="Connect, list tools, disconnect — an OAuth server with no token on disk is reported as not ok.")


class McpServersRemoveResult(Result):
    ok: bool
    removed: bool


method("mcp.servers.remove", params=McpServerNameParams, result=McpServersRemoveResult,
       doc="Drop a server from the profile's config.yaml.")


class McpOauthStartParams(McpServerNameParams):
    """With ``client_redirect_uri`` the CLIENT hosts the loopback and relays the code via
    ``mcp.servers.oauth.callback``."""

    client_redirect_uri: str | None = None


class McpOauthStartResult(Result):
    ok: bool
    session_id: str
    auth_url: str
    flow: str


method("mcp.servers.oauth.start", params=McpOauthStartParams, result=McpOauthStartResult,
       doc="Begin a PKCE OAuth flow; the client opens auth_url and polls mcp.servers.oauth.poll.")


class McpOauthFlowParams(McpServerNameParams):
    """``session_id`` is the OAuth flow id returned by ``oauth.start`` (not a gateway session)."""

    session_id: str


class McpOauthPollStatus(WireEnum):
    pending = "pending"
    approved = "approved"
    error = "error"


class McpOauthPollResult(Result):
    ok: bool
    status: McpOauthPollStatus
    session_id: str | None = None
    error_message: str | None = None
    auth_url: str | None = None
    tools: list[McpProbeTool] | None = None


method("mcp.servers.oauth.poll", params=McpOauthFlowParams, result=McpOauthPollResult,
       doc="Poll a flow; approved persists tokens for the profile and returns the probed tools.")


class McpOauthCancelResult(Result):
    ok: bool
    status: str | None = None
    error_message: str | None = None


method("mcp.servers.oauth.cancel", params=McpOauthFlowParams, result=McpOauthCancelResult,
       doc="Cancel a flow owned by the resolved profile, waking its callback worker.")


class McpOauthCallbackParams(McpOauthFlowParams):
    code: str | None = None
    state: str | None = None
    error: str | None = None
    # RFC 9207 issuer; extra="forbid" would otherwise 4000 the desktop relay that always sends it.
    iss: str | None = None


class McpOauthCallbackResult(Result):
    ok: bool
    session_id: str | None = None
    error_message: str | None = None


method("mcp.servers.oauth.callback", params=McpOauthCallbackParams, result=McpOauthCallbackResult,
       doc="Relay a client-captured redirect into a client_redirect_uri flow.")


# ── plugins ───────────────────────────────────────────────────────────────────────────────────


class PluginsListParams(Params):
    pass


class LegacyPluginRow(Result):
    name: str
    version: str
    enabled: bool


class PluginsListResult(Result):
    plugins: list[LegacyPluginRow]


method("plugins.list", params=PluginsListParams, result=PluginsListResult,
       doc="Loaded plugin manager entries (legacy flat view); the Plugins Hub uses plugins.manage list.")


class PluginsAction(WireEnum):
    list = "list"
    toggle = "toggle"
    install = "install"
    update = "update"
    remove = "remove"
    settings = "settings"


class PluginsManageParams(ProfileParams):
    """``toggle``: ``key``/``name`` + ``enable``; ``install``: ``identifier``/``repo`` or ``catalog_name``
    (+ ``force``, ``enable``, ``ref``); ``update``: ``name`` (+ ``accept_capabilities`` to apply a re-pin
    that widened the plugin after the user confirmed the ``delta``); ``remove``: ``name`` (user installs only);
    ``settings``: ``key`` + ``values`` (``{setting_key: value}``, non-secret schema keys only)."""

    action: PluginsAction = PluginsAction.list
    key: str | None = None
    name: str | None = None
    enable: bool | None = None
    identifier: str | None = None
    repo: str | None = None
    catalog_name: str | None = None
    force: bool | None = None
    ref: str | None = None
    accept_capabilities: bool | None = None
    values: dict[str, JsonValue] | None = None


class PluginSettingFieldType(WireEnum):
    string = "string"
    number = "number"
    boolean = "boolean"
    enum = "enum"
    secret = "secret"
    json = "json"


class PluginSettingField(Result):
    """One ``config_schema`` key of a plugin manifest, rendered by the Plugins hub
    (``hermes_cli.plugins_settings.plugin_settings_fields``). ``secret`` fields carry no value: ``env``
    names the ``.env`` variable and ``has_value`` whether it is set."""

    key: str
    type: PluginSettingFieldType
    label: str
    description: str
    required: bool
    value: JsonValue | None = None
    default: JsonValue | None = None
    choices: list[str] | None = None
    env: str | None = None
    has_value: bool | None = None


class PluginServerState(WireEnum):
    connected = "connected"
    app_not_running = "app_not_running"
    endpoint_unavailable = "endpoint_unavailable"
    no_interactive_session = "no_interactive_session"
    version_too_old = "version_too_old"
    missing_app = "missing_app"
    unknown = "unknown"


class PluginServerRow(Result):
    name: str
    state: PluginServerState
    sentence: str


class AgentPluginRow(Result):
    """``methods_tools._plugin_rows`` + ``plugins_cmd_catalog.catalog_row_fields`` provenance."""

    name: str
    key: str
    version: str
    description: str
    source: str
    status: str
    portable: bool
    install_dir: str
    has_desktop_half: bool
    servers: list[PluginServerRow]
    catalog_name: str | None = None
    catalog_tier: str | None = None
    installed_sha: str | None = None
    catalog_sha: str | None = None
    catalog_version: str | None = None
    update_available: bool | None = None
    pinned_sha: str | None = None
    settings_schema: list[PluginSettingField] | None = None


class PluginActivation(Result):
    """What a plugin loaded mid-run does NOW vs later (``hermes_cli.plugins_activation``), ``{kind: [names]}``
    with only non-empty kinds present. ``activated_now`` kinds: ``gateway_commands`` (slash names),
    ``gateway_transforms`` / ``hooks`` (hook names), ``callbacks`` (platforms / ``slack:<action_id>``) — live in
    the running gateway once it reloaded (``gateway_reloaded``). ``deferred`` kinds: ``tools`` (tool names) and
    ``prompt`` (section ids) apply from the next session; ``mcp_servers`` lists the plugin's mcp.json server
    names (``<namespace>__<server>``, as ``mcp.servers.*`` know them) — not connected until ``mcp.reload``.
    The Desktop "Installed. Connect its servers now" card reads exactly ``deferred.mcp_servers``."""

    name: str
    key: str
    activated_now: dict[str, list[str]] = Field(default_factory=dict)
    deferred: dict[str, list[str]] = Field(default_factory=dict)


class PluginsManageResult(Result):
    """``list`` → ``plugins`` + counts; ``toggle`` → ``ok``/``unchanged``/``restart_required``/``name``
    (the canonical key written)/``plugin``; ``install`` → ``hermes_cli.plugins_cmd.dashboard_install_plugin``'s
    ok payload; ``toggle``/``install``/``update`` that loaded a plugin also carry ``gateway_reloaded`` (the
    running gateway picked it up and re-wired its handlers) and ``activation`` — the honest split of what is
    live now vs deferred, so ``restart_required`` is True only when no gateway answered; ``update`` → ``ok``/``unchanged``/``sha``, or ``ok=false`` + ``consent_required`` with the
    ``delta`` (``{surface: [added...]}``) / ``delta_lines`` a widened pin adds — nothing changed until the
    client retries with ``accept_capabilities``; ``remove`` → ``ok``/``name`` plus
    ``cleared_memory_provider`` when the removed plugin was the live ``memory.provider``."""

    plugins: list[AgentPluginRow] | None = None
    user_count: int | None = None
    bundled_count: int | None = None
    ok: bool | None = None
    unchanged: bool | None = None
    restart_required: bool | None = None
    gateway_reloaded: bool | None = None
    activation: PluginActivation | None = None
    cleared_memory_provider: bool | None = None
    name: str | None = None
    plugin: AgentPluginRow | None = None
    plugin_name: str | None = None
    warnings: list[str] | None = None
    missing_env: list[str] | None = None
    # ``install`` → the manifest's ``python_dependencies`` the installer applied (``[]`` when none).
    python_dependencies: list[str] | None = None
    after_install_path: str | None = None
    enabled: bool | None = None
    sha: str | None = None
    consent_required: bool | None = None
    delta: dict[str, list[str]] | None = None
    delta_lines: list[str] | None = None
    error: str | None = None
    written: list[str] | None = None


method("plugins.manage", params=PluginsManageParams, result=PluginsManageResult,
       doc="Plugins Hub backend: list installed plugins, toggle, git-install, re-pin a catalog install, "
           "or remove a user install.")
