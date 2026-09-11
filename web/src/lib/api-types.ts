import type { DashboardTheme } from "@/themes/types";

/**
 * Data-transfer types shared by the dashboard API client and consumers.
 * Runtime requests, authentication, profile scoping and WebSocket behavior
 * remain in api.ts. Keep this module type-only.
 */
/** Identity payload returned by ``GET /api/auth/me`` (Phase 7).
 *
 * Returned by the dashboard's gated middleware when a valid session cookie
 * is attached. ``email`` and ``display_name`` are empty strings under the
 * Nous Portal contract V1 (the access token has no email/name claims —
 * see Contract Anchor C4 in the plan). The AuthWidget surfaces a
 * truncated ``user_id`` instead.
 */
export interface AuthMeResponse {
  user_id: string;
  email: string;
  display_name: string;
  org_id: string;
  provider: string;
  expires_at: number;
}

export interface ActionResponse {
  archive?: string;
  name: string;
  ok: boolean;
  pid: number | null;
  error?: string;
  message?: string;
  uploaded_bytes?: number;
  update_command?: string;
}

export interface DebugShareResponse {
  ok: boolean;
  // label -> paste URL, e.g. { Report: "https://paste.rs/abc", "agent.log": "..." }
  urls: Record<string, string>;
  // "label: error" strings for optional full-log uploads that failed.
  failures: string[];
  redacted: boolean;
  auto_delete_seconds: number;
}

export interface SessionStoreStats {
  total: number;
  active_store: number;
  archived: number;
  messages: number;
  by_source: Record<string, number>;
}

export interface SessionImportResponse {
  ok: boolean;
  imported: number;
  skipped: number;
  detached: number;
  imported_ids: string[];
  skipped_ids: string[];
  errors: Array<Record<string, unknown>>;
}

export interface SkillHubResult {
  name: string;
  description: string;
  source: string;
  identifier: string;
  trust_level: string;
  repo: string | null;
  tags: string[];
}

/** Lock-entry summary for an already-installed hub skill (keyed by identifier). */
export interface SkillHubInstalledEntry {
  name: string | null;
  trust_level: string | null;
  scan_verdict: string | null;
}

export interface SkillHubSearchResponse {
  results: SkillHubResult[];
  /** source_id -> number of results returned by that source. */
  source_counts: Record<string, number>;
  /** source ids that didn't return within the parallel-search timeout. */
  timed_out: string[];
  /** identifier -> installed lock entry (for "already installed" badges). */
  installed: Record<string, SkillHubInstalledEntry>;
}

export interface SkillHubSource {
  id: string;
  label: string;
  /** GitHub only: whether the API is currently rate-limited. */
  rate_limited?: boolean;
  /** hermes-index only: whether the centralized index loaded. */
  available?: boolean;
}

export interface SkillHubSourcesResponse {
  sources: SkillHubSource[];
  index_available: boolean;
  /** Featured/popular skills from the centralized index (zero extra API calls). */
  featured: SkillHubResult[];
  installed: Record<string, SkillHubInstalledEntry>;
}

export interface SkillHubPreview {
  name: string;
  description: string;
  source: string;
  identifier: string;
  trust_level: string;
  repo: string | null;
  tags: string[];
  /** Rendered SKILL.md content (the actual skill text). */
  skill_md: string;
  /** Relative paths of every file in the bundle. */
  files: string[];
}

export interface SkillHubScanFinding {
  severity: string;
  category: string;
  file: string;
  line: number;
  description: string;
}

export interface SkillHubScan {
  name: string;
  identifier: string;
  source: string;
  trust_level: string;
  /** "safe" | "caution" | "dangerous". */
  verdict: string;
  summary: string;
  /** Install-policy decision for this trust+verdict combo. */
  policy: "allow" | "ask" | "block";
  policy_reason: string;
  findings: SkillHubScanFinding[];
  severity_counts: Record<string, number>;
}

// ── Admin types ───────────────────────────────────────────────────────

export interface McpServer {
  name: string;
  transport: "http" | "stdio" | "unknown";
  url: string | null;
  command: string | null;
  args: string[];
  env: Record<string, string>;
  auth: "header" | "oauth" | null;
  enabled: boolean;
  tools: string[] | null;
}

export interface McpCatalogEntry {
  name: string;
  description: string;
  source: string;
  transport: "http" | "stdio";
  auth_type: "api_key" | "oauth" | "none";
  required_env: Array<{ name: string; prompt: string; required: boolean }>;
  // Transport details — what actually connects (http) or runs (stdio).
  command: string | null;
  args: string[];
  url: string | null;
  // Git bootstrap (only set for entries that clone + build locally).
  install_url: string | null;
  install_ref: string | null;
  bootstrap: string[];
  // Default tool pre-selection (null = all tools pre-checked) + guidance text.
  default_enabled: string[] | null;
  post_install: string;
  needs_install: boolean;
  installed: boolean;
  enabled: boolean;
}

export interface McpCatalogDiagnostic {
  name: string;
  kind: string;
  message: string;
}


export type McpHttpAuth = "none" | "header" | "oauth";

export interface McpServerCreate {
  name: string;
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  auth?: McpHttpAuth;
  bearer_token?: string;
}

export interface McpTestResult {
  ok: boolean;
  error?: string;
  tools: Array<{ name: string; description: string }>;
}

export interface McpOAuthFlow {
  flow_id: string;
  server_name: string;
  status: "starting" | "authorization_required" | "approved" | "error";
  authorization_url: string | null;
  error: string | null;
  tools?: Array<{ name: string; description: string }>;
}

export interface MessagingPlatformEnvVar {
  key: string;
  required: boolean;
  is_set: boolean;
  redacted_value: string | null;
  description: string;
  prompt: string;
  help: string;
  url: string | null;
  is_password: boolean;
  advanced: boolean;
}

export interface MessagingPlatform {
  id: string;
  name: string;
  description: string;
  docs_url: string;
  enabled: boolean;
  configured: boolean;
  gateway_running: boolean;
  /**
   * "connected" | "disabled" | "not_configured" | "pending_restart" |
   * "gateway_stopped" | "startup_failed" | "disconnected" | "fatal" | string
   */
  state: string;
  error_code: string | null;
  error_message: string | null;
  updated_at: string | null;
  home_channel: { platform: string; chat_id: string; name: string; thread_id?: string } | null;
  whatsapp_setup?: {
    mode?: string;
    allowed_users_set?: boolean;
    home_channel_set?: boolean;
  } | null;
  env_vars: MessagingPlatformEnvVar[];
}

export interface MessagingPlatformsResponse {
  env_path: string;
  gateway_start_command: string;
  platforms: MessagingPlatform[];
}

export interface MessagingPlatformUpdate {
  enabled?: boolean;
  env?: Record<string, string>;
  clear_env?: string[];
}

export interface MessagingPlatformTestResult {
  ok: boolean;
  state: string;
  message: string;
}

export interface PairingUser {
  platform: string;
  user_id: string;
  user_name?: string;
  request_id?: string;
  age_minutes?: number;
}

export interface PairingResponse {
  pending: PairingUser[];
  approved: PairingUser[];
}

export interface WebhookRoute {
  name: string;
  description: string;
  events: string[];
  deliver: string;
  deliver_only: boolean;
  prompt: string;
  skills: string[];
  created_at: string | null;
  url: string;
  secret_set: boolean;
  enabled: boolean;
}

export interface WebhooksResponse {
  enabled: boolean;
  base_url: string;
  subscriptions: WebhookRoute[];
}

export interface WebhookEnableResponse {
  ok: boolean;
  platform: "webhook";
  enabled: true;
  needs_restart: boolean;
  restart_started?: boolean;
  restart_action?: string;
  restart_pid?: number | null;
  restart_error?: string;
}

export interface WebhookCreate {
  name: string;
  description?: string;
  events?: string[];
  prompt?: string;
  skills?: string[];
  deliver?: string;
  deliver_only?: boolean;
  deliver_chat_id?: string;
}

export interface CredentialPoolEntry {
  index: number;
  id: string | null;
  label: string | null;
  auth_type: string | null;
  source: string | null;
  priority: number;
  last_status: string | null;
  request_count: number;
  token_preview: string;
  has_refresh: boolean;
}

export interface CredentialPoolProvider {
  provider: string;
  entries: CredentialPoolEntry[];
}

export interface MemoryProviderInfo {
  name: string;
  description: string;
  available: boolean;
  configured: boolean;
  status: "ready" | "needs_config" | "unavailable" | "missing";
  setup?: MemoryProviderSetupInfo;
}

export interface MemoryStatus {
  active: string;
  providers: MemoryProviderInfo[];
  builtin_files: { memory: number; user: number };
}

export interface MemoryProviderExternalDependency {
  name: string;
  install: string;
  check: string;
}

export interface MemoryProviderSetupInfo {
  pip_dependencies: string[];
  external_dependencies: MemoryProviderExternalDependency[];
  required_env: string[];
  dependencies_installed: boolean;
}

export interface MemoryProviderSetupResult {
  kind: string;
  name: string;
  status: string;
  command: string;
  returncode: number | null;
  stdout: string;
  stderr: string;
}

export interface MemoryProviderSetupResponse {
  ok: boolean;
  provider: string;
  results: MemoryProviderSetupResult[];
  status?: MemoryProviderInfo | null;
}

export interface MemoryProviderFieldOption {
  value: string;
  label: string;
  description?: string;
}

export interface MemoryProviderField {
  key: string;
  label: string;
  kind: "text" | "secret" | "select" | "boolean" | "integer" | "number";
  description: string;
  placeholder: string;
  required: boolean;
  value: string | boolean | number;
  is_set: boolean;
  options: MemoryProviderFieldOption[];
  url: string;
  minimum?: number | null;
  maximum?: number | null;
  step?: number | null;
  when?: Record<string, string | boolean | number> | null;
}

export interface MemoryProviderConfig {
  name: string;
  label: string;
  fields: MemoryProviderField[];
  setup?: MemoryProviderSetupInfo;
}

export interface HookEntry {
  event: string;
  matcher: string | null;
  command: string | null;
  timeout: number | null;
  allowed: boolean;
  approved_at?: string | null;
  executable?: boolean;
}

export interface HooksResponse {
  hooks: HookEntry[];
  valid_events: string[];
}

export interface HookCreate {
  event: string;
  command: string;
  matcher?: string;
  timeout?: number;
  approve?: boolean;
}

export interface UpdateCheckResponse {
  install_method: string;
  current_version: string;
  // commits behind: >=1 known count, 0 up to date, -1 behind by unknown
  // count (nix/pypi), or null when the check could not run.
  behind: number | null;
  update_available: boolean;
  can_apply: boolean;
  update_command: string;
  message: string | null;
}

export interface SystemStats {
  os: string;
  os_release: string;
  os_version: string;
  platform: string;
  arch: string;
  hostname: string;
  python_version: string;
  python_impl: string;
  hermes_version: string;
  cpu_count: number | null;
  psutil: boolean;
  cpu_percent?: number;
  load_avg?: number[];
  uptime_seconds?: number;
  memory?: { total: number; available: number; used: number; percent: number };
  disk?: { total: number; used: number; free: number; percent: number };
  process?: { pid: number; rss: number; create_time: number; num_threads: number };
}

export interface CuratorStatus {
  enabled: boolean;
  paused: boolean;
  interval_hours: number | null;
  last_run_at: string | null;
  min_idle_hours: number | null;
  stale_after_days: number | null;
  archive_after_days: number | null;
}

export interface PortalFeature {
  label: string;
  state: string;
}

export interface PortalStatus {
  logged_in: boolean;
  portal_url: string | null;
  inference_url: string | null;
  provider: string;
  subscription_url: string;
  features: PortalFeature[];
}

export interface CheckpointSession {
  session: string;
  files: number;
  bytes: number;
}

export interface CheckpointsResponse {
  sessions: CheckpointSession[];
  total_bytes: number;
}

/** Per-call overrides for {@link fetchJSON}. */
export interface FetchJSONOptions {
  /** When true, a 401 response is surfaced as a normal thrown error rather
   *  than triggering the loopback stale-token page reload. Use for probes
   *  whose 401 is an expected signal (e.g. /api/auth/me in non-gated mode)
   *  rather than evidence of a rotated session token. */
  allowUnauthorized?: boolean;
}

export interface ActionStatusResponse {
  exit_code: number | null;
  lines: string[];
  name: string;
  pid: number | null;
  running: boolean;
}

export interface PlatformStatus {
  error_code?: string;
  error_message?: string;
  state: string;
  updated_at: string;
}

export interface StatusResponse {
  active_sessions: number;
  /** Phase 7: ``true`` when the dashboard's OAuth gate is engaged
   * (public bind, no ``--insecure``). Read alongside ``auth_providers``
   * to render a "gated / loopback" badge. */
  auth_required?: boolean;
  /** Phase 7: registered ``DashboardAuthProvider`` names (e.g. ``["nous"]``).
   * Empty in loopback mode; empty + ``auth_required=true`` is a
   * fail-closed state (the dashboard will refuse to bind). */
  auth_providers?: string[];
  /** Supported dashboard auth flows for the client to choose from. In gated
   * mode always includes ``"cookie"``; includes ``"native_pkce"`` when any
   * interactive session provider is registered (OAuth providers broker the
   * IDP redirect; password providers complete at /login in the system
   * browser), signalling that the desktop can use the RFC 8252
   * system-browser + loopback + PKCE flow (no embedded webview, no session
   * cookies). Absent / missing ``"native_pkce"`` ⇒ an older gateway ⇒ the
   * desktop falls back to the embedded-webview flow. */
  auth_flows?: string[];
  /** False when the dashboard is running in a hosted/managed layout where
   * updates are handled by the outer launcher instead of ``hermes update``. */
  can_update_hermes?: boolean;
  config_path: string;
  config_version: number;
  env_path: string;
  gateway_exit_reason: string | null;
  gateway_health_url: string | null;
  gateway_pid: number | null;
  gateway_platforms: Record<string, PlatformStatus>;
  gateway_running: boolean;
  gateway_state: string | null;
  gateway_updated_at: string | null;
  hermes_home: string;
  latest_config_version: number;
  /** NS-656: memory-pressure rollup from the gateway heartbeat +
   * lifecycle ledger. Absent on older gateways. */
  memory?: MemoryPressureStatus;
  /** NS-656: disk-usage rollup for the HERMES_HOME volume. Absent on
   * older gateways. */
  disk?: DiskPressureStatus;
  release_date: string;
  version: string;
}

/** NS-656: coarse memory telemetry served by /api/status. */
export interface MemoryPressureStatus {
  pressure: "ok" | "elevated" | "critical" | "unknown";
  gateway_rss_mb?: number | null;
  system_total_mb?: number | null;
  system_available_mb?: number | null;
  swap_used_mb?: number | null;
  sampled_at?: string | null;
  /** Previous gateway life died without running any exit path. */
  last_boot_unclean?: boolean;
  /** ...and its final heartbeat showed near-exhausted memory. Heuristic —
   * strong evidence of an OOM kill, not proof the kernel OOM killer acted. */
  last_boot_suspected_oom?: boolean;
  /** Identity of the current gateway life (sentinel started_at). Changes on
   * every restart; keys per-incident banner dismissal. */
  boot_id?: string | null;
}

/** NS-656: coarse disk telemetry served by /api/status. Live statvfs
 * sample of the HERMES_HOME volume — no staleness dimension, so no
 * sampled_at. */
export interface DiskPressureStatus {
  pressure: "ok" | "elevated" | "critical" | "unknown";
  total_mb?: number | null;
  free_mb?: number | null;
  used_percent?: number | null;
}

export interface SessionInfo {
  id: string;
  source: string | null;
  model: string | null;
  title: string | null;
  started_at: number;
  ended_at: number | null;
  last_active: number;
  is_active: boolean;
  message_count: number;
  tool_call_count: number;
  input_tokens: number;
  output_tokens: number;
  preview: string | null;
  parent_session_id?: string | null;
  /** Owning profile stamped by the list/detail endpoints (the store the row
   * was read from). Absent on search-endpoint rows, which carry no stamp. */
  profile?: string;
}

export interface SessionLatestDescendantResponse {
  requested_session_id: string;
  session_id: string;
  path: string[];
  changed: boolean;
}

export interface PaginatedSessions {
  sessions: SessionInfo[];
  total: number;
  limit: number;
  offset: number;
}

export interface EnvVarInfo {
  is_set: boolean;
  redacted_value: string | null;
  description: string;
  url: string | null;
  category: string;
  is_password: boolean;
  tools: string[];
  advanced: boolean;
  /** True when this var is a messaging-platform credential owned by the Channels page. */
  channel_managed?: boolean;
  /** True when this key is set in .env but not in any catalog (user-added custom key). */
  custom?: boolean;
}

export interface TelegramOnboardingStartResponse {
  pairing_id: string;
  suggested_username: string;
  deep_link: string;
  qr_payload: string;
  expires_at: string;
}

export type TelegramOnboardingStatusResponse =
  | { status: "waiting"; expires_at: string }
  | {
      status: "ready";
      bot_username: string;
      owner_user_id?: string;
      expires_at: string;
    };

export interface TelegramOnboardingApplyResponse {
  ok: boolean;
  platform: "telegram";
  bot_username?: string;
  needs_restart: boolean;
  restart_started?: boolean;
  restart_action?: string;
  restart_pid?: number | null;
  restart_error?: string;
}

export interface WhatsAppOnboardingStartResponse {
  pairing_id: string;
  status:
    | "starting"
    | "installing"
    | "waiting"
    | "connected"
    | "error"
    | "expired"
    | "cancelled";
  qr_payload?: string | null;
  expires_at: string;
  mode: "bot" | "self-chat";
  allowed_users: string;
  account_id?: string | null;
  account_name?: string | null;
  account_phone?: string | null;
  error?: string | null;
}

export type WhatsAppOnboardingStatusResponse = WhatsAppOnboardingStartResponse;

export interface WhatsAppOnboardingApplyResponse {
  ok: boolean;
  platform: "whatsapp";
  needs_restart: boolean;
  restart_started?: boolean;
  restart_action?: string;
  restart_pid?: number | null;
  restart_error?: string;
}

export interface SessionMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | null;
  tool_calls?: Array<{
    id: string;
    function: { name: string; arguments: string };
  }>;
  tool_name?: string;
  tool_call_id?: string;
  timestamp?: number;
}

export interface SessionMessagesResponse {
  session_id: string;
  messages: SessionMessage[];
  pagination?: {
    limit: number;
    offset: number;
    order: "latest" | "oldest";
    returned: number;
  };
}

export interface LogsResponse {
  file: string;
  lines: string[];
}

export interface ManagedFileEntry {
  name: string;
  path: string;
  is_directory: boolean;
  size: number | null;
  mtime: number;
  mime_type: string | null;
}

export interface ManagedFilesResponse {
  root: string | null;
  path: string;
  parent: string | null;
  locked_root: string | null;
  can_change_path: boolean;
  entries: ManagedFileEntry[];
}

export interface ManagedFileReadResponse {
  name: string;
  path: string;
  size: number;
  mime_type: string;
  data_url: string;
  root: string | null;
  locked_root: string | null;
  can_change_path: boolean;
}

export interface ManagedFileWriteResponse {
  ok: boolean;
  path: string;
  entry: ManagedFileEntry;
  root: string | null;
  locked_root: string | null;
  can_change_path: boolean;
}

export interface AnalyticsDailyEntry {
  day: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  reasoning_tokens: number;
  estimated_cost: number;
  actual_cost: number;
  sessions: number;
  api_calls: number;
}

export interface AnalyticsModelEntry {
  model: string;
  input_tokens: number;
  output_tokens: number;
  estimated_cost: number;
  sessions: number;
  api_calls: number;
}

export interface AnalyticsSkillEntry {
  skill: string;
  view_count: number;
  manage_count: number;
  total_count: number;
  percentage: number;
  last_used_at: number | null;
}

export interface AnalyticsSkillsSummary {
  total_skill_loads: number;
  total_skill_edits: number;
  total_skill_actions: number;
  distinct_skills_used: number;
}

export interface AnalyticsResponse {
  daily: AnalyticsDailyEntry[];
  by_model: AnalyticsModelEntry[];
  totals: {
    total_input: number;
    total_output: number;
    total_cache_read: number;
    total_reasoning: number;
    total_estimated_cost: number;
    total_actual_cost: number;
    total_sessions: number;
    total_api_calls: number;
  };
  skills: {
    summary: AnalyticsSkillsSummary;
    top_skills: AnalyticsSkillEntry[];
  };
}

export interface ActiveProfileInfo {
  active: string;
  current: string;
}

export interface ProfileDescribeAutoResult {
  ok: boolean;
  reason: string;
  description: string | null;
  description_auto: boolean;
}

export interface ProfileInfo {
  name: string;
  path: string;
  is_default: boolean;
  model: string | null;
  provider: string | null;
  has_env: boolean;
  skill_count: number;
  gateway_running: boolean;
  description: string;
  description_auto: boolean;
  display_name?: string;
  distribution_name: string | null;
  distribution_version: string | null;
  distribution_source: string | null;
  has_alias: boolean;
}

export interface ModelsAnalyticsModelEntry {
  model: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  reasoning_tokens: number;
  estimated_cost: number;
  actual_cost: number;
  sessions: number;
  api_calls: number;
  tool_calls: number;
  last_used_at: number;
  avg_tokens_per_session: number;
  capabilities: {
    supports_tools?: boolean;
    supports_vision?: boolean;
    supports_reasoning?: boolean;
    context_window?: number;
    max_output_tokens?: number;
    model_family?: string;
  };
}

export interface ModelsAnalyticsResponse {
  models: ModelsAnalyticsModelEntry[];
  totals: {
    distinct_models: number;
    total_input: number;
    total_output: number;
    total_cache_read: number;
    total_reasoning: number;
    total_estimated_cost: number;
    total_actual_cost: number;
    total_sessions: number;
    total_api_calls: number;
  };
  period_days: number;
}

export interface CronJobRepeat {
  times: number | null;
  completed?: number;
}

export interface CronJobMutation {
  name?: string;
  prompt?: string;
  schedule?: string;
  deliver?: string;
  skills?: string[];
  provider?: string | null;
  model?: string | null;
  base_url?: string | null;
  script?: string | null;
  no_agent?: boolean;
  context_from?: string[] | null;
  enabled_toolsets?: string[] | null;
  workdir?: string | null;
}

export interface CronJob {
  id: string;
  profile?: string | null;
  profile_name?: string | null;
  hermes_home?: string | null;
  is_default_profile?: boolean;
  name?: string | null;
  prompt?: string | null;
  script?: string | null;
  skills?: string[] | null;
  schedule?: { kind?: string; expr?: string; run_at?: string; display?: string };
  schedule_display?: string | null;
  repeat?: CronJobRepeat | null;
  enabled: boolean;
  state?: string | null;
  deliver?: string | null;
  model?: string | null;
  provider?: string | null;
  base_url?: string | null;
  no_agent?: boolean | null;
  context_from?: string[] | string | null;
  enabled_toolsets?: string[] | null;
  workdir?: string | null;
  last_run_at?: string | null;
  next_run_at?: string | null;
  last_status?: string | null;
  last_error?: string | null;
  last_delivery_error?: string | null;
  last_fire_error?: { at?: string | null; detail?: string | null } | null;
}

export interface CronDeliveryTarget {
  id: string;
  name: string;
  home_target_set: boolean;
  home_env_var: string | null;
}

export interface AutomationBlueprintField {
  name: string;
  type: "time" | "enum" | "text" | "weekdays";
  label: string;
  default: string | null;
  options: string[];
  optional: boolean;
  /** When false, options are suggestions — any value is accepted. */
  strict?: boolean;
  help: string;
}

export interface AutomationBlueprint {
  key: string;
  title: string;
  description: string;
  category: string;
  tags: string[];
  fields: AutomationBlueprintField[];
  command: string;
  appUrl: string;
}

export interface SkillInfo {
  name: string;
  description: string;
  category: string;
  enabled: boolean;
}

export interface SkillContent {
  name: string;
  content: string;
  path: string;
}

export interface SkillWriteResult {
  success: boolean;
  message?: string;
  path?: string;
  error?: string;
}

export interface ToolsetInfo {
  name: string;
  label: string;
  description: string;
  platform: string;
  platform_label: string;
  enabled: boolean;
  configured: boolean;
  tools: string[];
}

export interface ToolsetProviderEnvVar {
  key: string;
  prompt: string;
  url: string | null;
  default: string | null;
  is_set: boolean;
}

export interface ToolsetProvider {
  name: string;
  badge: string;
  tag: string;
  env_vars: ToolsetProviderEnvVar[];
  post_setup: string | null;
  requires_nous_auth: boolean;
  is_active: boolean;
}

export interface ToolsetConfig {
  name: string;
  has_category: boolean;
  providers: ToolsetProvider[];
  active_provider: string | null;
}

export interface ToolsetEnvResult {
  ok: boolean;
  name: string;
  saved: string[];
  skipped: string[];
  is_set: Record<string, boolean>;
}

export interface SessionSearchResult extends SessionInfo {
  session_id: string;
  snippet: string;
  role: string | null;
  session_started: number | null;
  lineage_root?: string;
}

export interface SessionSearchResponse {
  results: SessionSearchResult[];
}

// ── Model info types ──────────────────────────────────────────────────

export interface ModelInfoResponse {
  model: string;
  provider: string;
  auto_context_length: number;
  config_context_length: number;
  effective_context_length: number;
  capabilities: {
    supports_tools?: boolean;
    supports_vision?: boolean;
    supports_reasoning?: boolean;
    context_window?: number;
    max_output_tokens?: number;
    model_family?: string;
  };
}

// ── Model options / assignment types ──────────────────────────────────

export interface ModelOptionProvider {
  name: string;
  slug: string;
  models?: string[];
  total_models?: number;
  is_current?: boolean;
  is_user_defined?: boolean;
  source?: string;
  warning?: string;
  authenticated?: boolean;
}

export interface ModelOptionsResponse {
  model?: string;
  provider?: string;
  providers?: ModelOptionProvider[];
}

export interface AuxiliaryTaskAssignment {
  task: string;
  provider: string;
  model: string;
  base_url: string;
}

export interface AuxiliaryModelsResponse {
  tasks: AuxiliaryTaskAssignment[];
  main: { provider: string; model: string };
}

export interface MoaModelSlot {
  provider: string;
  model: string;
  /** Optional per-slot reasoning effort — round-tripped, not edited here. */
  reasoning_effort?: string;
  enabled?: boolean;
}

export interface MoaConfigResponse {
  default_preset: string;
  active_preset: string;
  presets: Record<string, {
    reference_models: MoaModelSlot[];
    aggregator: MoaModelSlot;
    reference_temperature: number;
    aggregator_temperature: number;
    reference_timeout: number | null;
    degraded_reference_policy: "loud" | "silent";

    /** Fan-out cadence (user_turn default | per_iteration | every_n:N) — round-tripped. */
    fanout?: string;
    enabled: boolean;
  }>;
  reference_models: MoaModelSlot[];
  aggregator: MoaModelSlot;
  reference_temperature: number;
  aggregator_temperature: number;
  reference_timeout: number | null;
  degraded_reference_policy: "loud" | "silent";

  enabled: boolean;
}

export interface ModelAssignmentRequest {
  confirm_expensive_model?: boolean;
  scope: "main" | "auxiliary";
  provider: string;
  model: string;
  /** Optional OpenAI-compatible endpoint URL for custom/local main providers. */
  base_url?: string;
  /** For auxiliary: task slot name, "" for all, "__reset__" to reset all. */
  task?: string;
}

/** An auxiliary task still pinned to a provider that differs from the
 *  newly-selected main provider after a main-model switch. */
export interface StaleAuxAssignment {
  task: string;
  provider: string;
  model: string;
}

export interface ModelAssignmentResponse {
  confirm_message?: string;
  confirm_required?: boolean;
  ok: boolean;
  scope?: string;
  provider?: string;
  model?: string;
  tasks?: string[];
  reset?: boolean;
  /** Auxiliary slots still pinned to a different provider than the new main.
   *  Switching main never clears aux pins; this lets the UI warn the user
   *  their helper tasks aren't following the switch. Only set on scope:'main'. */
  stale_aux?: StaleAuxAssignment[];
}

// ── OAuth provider types ────────────────────────────────────────────────

export interface OAuthProviderStatus {
  logged_in: boolean;
  source?: string | null;
  source_label?: string | null;
  token_preview?: string | null;
  expires_at?: string | null;
  has_refresh_token?: boolean;
  last_refresh?: string | null;
  error?: string;
}

export interface OAuthProvider {
  id: string;
  name: string;
  /** "pkce" (browser redirect + paste code), "device_code" (show code + URL),
   *  or "external" (delegated to a separate CLI like Claude Code or Qwen). */
  flow: "pkce" | "device_code" | "external";
  cli_command: string;
  docs_url: string;
  status: OAuthProviderStatus;
}

export interface OAuthProvidersResponse {
  providers: OAuthProvider[];
}

/** Discriminated union — the shape of /start depends on the flow. */
export type OAuthStartResponse =
  | {
      session_id: string;
      flow: "pkce";
      auth_url: string;
      expires_in: number;
    }
  | {
      session_id: string;
      flow: "device_code";
      user_code: string;
      verification_url: string;
      expires_in: number;
      poll_interval: number;
    };

export interface OAuthSubmitResponse {
  ok: boolean;
  status: "approved" | "error";
  message?: string;
}

export interface OAuthPollResponse {
  session_id: string;
  status: "pending" | "approved" | "denied" | "expired" | "error";
  error_message?: string | null;
  expires_at?: number | null;
}

// ── Dashboard theme types ──────────────────────────────────────────────

export interface DashboardThemeSummary {
  description: string;
  label: string;
  name: string;
  /** Full theme definition for user themes; undefined for built-ins
   *  (which the frontend already has locally). */
  definition?: DashboardTheme;
}

export interface DashboardThemesResponse {
  active: string;
  themes: DashboardThemeSummary[];
}

export interface DashboardFontResponse {
  /** Active font-override id, or "theme" when no override is set. */
  font: string;
}

// ── Dashboard plugin types ─────────────────────────────────────────────

export interface PluginManifestResponse {
  name: string;
  label: string;
  description: string;
  icon: string;
  version: string;
  tab: {
    path: string;
    position?: string;
    override?: string;
    hidden?: boolean;
  };
  slots?: string[];
  entry: string;
  css?: string | null;
  has_api: boolean;
  source: string;
}

export interface HubAgentPluginRow {
  name: string;
  version: string;
  description: string;
  source: string;
  runtime_status: "disabled" | "enabled" | "inactive";
  has_dashboard_manifest: boolean;
  dashboard_manifest: PluginManifestResponse | null;
  path: string;
  can_remove: boolean;
  can_update_git: boolean;
  auth_required: boolean;
  auth_command: string;
  user_hidden: boolean;
  /** Reason string when this plugin is on the catalog removed blocklist. */
  removed_reason?: string | null;
}

export interface PluginsHubProviders {
  memory_provider: string;
  memory_options: MemoryProviderInfo[];
  context_engine: string;
  context_options: Array<{ name: string; description: string }>;
}

export interface PluginsHubResponse {
  plugins: HubAgentPluginRow[];
  orphan_dashboard_plugins: PluginManifestResponse[];
  providers: PluginsHubProviders;
}

export interface AgentPluginInstallRequest {
  identifier: string;
  force?: boolean;
  enable?: boolean;
  /** Install by curated-catalog name (resolves repo + pinned SHA server-side). */
  catalog_name?: string;
}

export interface AgentPluginInstallResponse {
  ok: boolean;
  plugin_name?: string;
  warnings?: string[];
  missing_env?: string[];
  after_install_path?: string | null;
  enabled?: boolean;
  error?: string;
}

// ── Plugin catalog types ───────────────────────────────────────────────

export interface CatalogCapabilities {
  provides_tools: string[];
  provides_hooks: string[];
  provides_middleware: string[];
  requires_env: string[];
}

export interface CatalogEntry {
  name: string;
  description: string;
  repo: string;
  sha: string;
  sha_short: string;
  tier: "official" | "community";
  maintainer: string;
  requires_hermes: string;
  platforms: string[];
  capabilities: CatalogCapabilities;
  docs_url: string;
  capability_summary: string;
  /** Installed-state merge (computed server-side). */
  installed: boolean;
  installed_sha: string | null;
  update_available: boolean;
  runtime_status: "disabled" | "enabled" | "inactive" | null;
}

export interface CatalogRemovedEntry {
  name: string;
  repo: string;
  reason: string;
  date: string;
}

export interface CatalogResponse {
  entries: CatalogEntry[];
  removed: CatalogRemovedEntry[];
  generated_at: string;
}

export interface AgentPluginUpdateResponse {
  ok: boolean;
  name?: string;
  output?: string;
  unchanged?: boolean;
  error?: string;
}

export interface PluginProvidersPutRequest {
  memory_provider?: string;
  context_engine?: string;
}
