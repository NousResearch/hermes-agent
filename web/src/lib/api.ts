// The dashboard can be served either at the root of its host (e.g.
// https://kanban.tilos.com/) or under a URL prefix when reverse-proxied
// (e.g. https://mission-control.tilos.com/hermes/). The Python backend
// injects ``window.__HERMES_BASE_PATH__`` into index.html based on the
// incoming ``X-Forwarded-Prefix`` header so the SPA can address its own
// ``/api/...`` and ``/dashboard-plugins/...`` URLs correctly without a
// rebuild. Empty string means "served at root".
function readBasePath(): string {
  if (typeof window === "undefined") return "";
  const raw = window.__HERMES_BASE_PATH__ ?? "";
  if (!raw) return "";
  // Normalise: ensure leading slash, strip trailing slash.
  const withLead = raw.startsWith("/") ? raw : `/${raw}`;
  return withLead.replace(/\/+$/, "");
}

export const HERMES_BASE_PATH = readBasePath();
const BASE = HERMES_BASE_PATH;

import type { DashboardTheme } from "@/themes/types";

// Ephemeral session token for protected endpoints.
// Injected into index.html by the server — never fetched via API.
declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string;
    __HERMES_BASE_PATH__?: string;
  }
}
let _sessionToken: string | null = null;
const SESSION_HEADER = "X-Hermes-Session-Token";

function setSessionHeader(headers: Headers, token: string): void {
  if (!headers.has(SESSION_HEADER)) {
    headers.set(SESSION_HEADER, token);
  }
}

export async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  // Inject the session token into all /api/ requests.
  const headers = new Headers(init?.headers);
  const token = window.__HERMES_SESSION_TOKEN__;
  if (token) {
    setSessionHeader(headers, token);
  }
  const res = await fetch(`${BASE}${url}`, { ...init, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

async function getSessionToken(): Promise<string> {
  if (_sessionToken) return _sessionToken;
  const injected = window.__HERMES_SESSION_TOKEN__;
  if (injected) {
    _sessionToken = injected;
    return _sessionToken;
  }
  throw new Error("Session token not available — page must be served by the Hermes dashboard server");
}

export const api = {
  getStatus: () => fetchJSON<StatusResponse>("/api/status"),
  getSessions: (limit = 20, offset = 0) =>
    fetchJSON<PaginatedSessions>(`/api/sessions?limit=${limit}&offset=${offset}`),
  getSessionMessages: (id: string) =>
    fetchJSON<SessionMessagesResponse>(`/api/sessions/${encodeURIComponent(id)}/messages`),
  getSessionLatestDescendant: (id: string) =>
    fetchJSON<SessionLatestDescendantResponse>(
      `/api/sessions/${encodeURIComponent(id)}/latest-descendant`,
    ),
  deleteSession: (id: string) =>
    fetchJSON<{ ok: boolean }>(`/api/sessions/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  getLogs: (params: { file?: string; lines?: number; level?: string; component?: string }) => {
    const qs = new URLSearchParams();
    if (params.file) qs.set("file", params.file);
    if (params.lines) qs.set("lines", String(params.lines));
    if (params.level && params.level !== "ALL") qs.set("level", params.level);
    if (params.component && params.component !== "all") qs.set("component", params.component);
    return fetchJSON<LogsResponse>(`/api/logs?${qs.toString()}`);
  },
  getAnalytics: (days: number) =>
    fetchJSON<AnalyticsResponse>(`/api/analytics/usage?days=${days}`),
  getModelsAnalytics: (days: number) =>
    fetchJSON<ModelsAnalyticsResponse>(`/api/analytics/models?days=${days}`),
  getConfig: () => fetchJSON<Record<string, unknown>>("/api/config"),
  getDefaults: () => fetchJSON<Record<string, unknown>>("/api/config/defaults"),
  getSchema: () => fetchJSON<{ fields: Record<string, unknown>; category_order: string[] }>("/api/config/schema"),
  getModelInfo: () => fetchJSON<ModelInfoResponse>("/api/model/info"),
  getModelOptions: () => fetchJSON<ModelOptionsResponse>("/api/model/options"),
  getAuxiliaryModels: () => fetchJSON<AuxiliaryModelsResponse>("/api/model/auxiliary"),
  setModelAssignment: (body: ModelAssignmentRequest) =>
    fetchJSON<ModelAssignmentResponse>("/api/model/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  saveConfig: (config: Record<string, unknown>) =>
    fetchJSON<{ ok: boolean }>("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    }),
  getConfigRaw: () => fetchJSON<{ yaml: string }>("/api/config/raw"),
  saveConfigRaw: (yaml_text: string) =>
    fetchJSON<{ ok: boolean }>("/api/config/raw", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_text }),
    }),
  getEnvVars: () => fetchJSON<Record<string, EnvVarInfo>>("/api/env"),
  setEnvVar: (key: string, value: string) =>
    fetchJSON<{ ok: boolean }>("/api/env", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, value }),
    }),
  deleteEnvVar: (key: string) =>
    fetchJSON<{ ok: boolean }>("/api/env", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    }),
  revealEnvVar: async (key: string) => {
    const token = await getSessionToken();
    return fetchJSON<{ key: string; value: string }>("/api/env/reveal", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [SESSION_HEADER]: token,
      },
      body: JSON.stringify({ key }),
    });
  },

  // Cron jobs
  getCronJobs: () => fetchJSON<CronJob[]>("/api/cron/jobs"),
  createCronJob: (job: { prompt: string; schedule: string; name?: string; deliver?: string }) =>
    fetchJSON<CronJob>("/api/cron/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(job),
    }),
  pauseCronJob: (id: string) =>
    fetchJSON<{ ok: boolean }>(`/api/cron/jobs/${id}/pause`, { method: "POST" }),
  resumeCronJob: (id: string) =>
    fetchJSON<{ ok: boolean }>(`/api/cron/jobs/${id}/resume`, { method: "POST" }),
  triggerCronJob: (id: string) =>
    fetchJSON<{ ok: boolean }>(`/api/cron/jobs/${id}/trigger`, { method: "POST" }),
  deleteCronJob: (id: string) =>
    fetchJSON<{ ok: boolean }>(`/api/cron/jobs/${id}`, { method: "DELETE" }),

  // Profiles (minimal)
  getProfiles: () =>
    fetchJSON<{ profiles: ProfileInfo[] }>("/api/profiles"),
  createProfile: (body: { name: string; clone_from_default: boolean }) =>
    fetchJSON<{ ok: boolean; name: string; path: string }>("/api/profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  renameProfile: (name: string, newName: string) =>
    fetchJSON<{ ok: boolean; name: string; path: string }>(
      `/api/profiles/${encodeURIComponent(name)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: newName }),
      },
    ),
  deleteProfile: (name: string) =>
    fetchJSON<{ ok: boolean }>(
      `/api/profiles/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  getProfileSetupCommand: (name: string) =>
    fetchJSON<{ command: string }>(
      `/api/profiles/${encodeURIComponent(name)}/setup-command`,
    ),
  getProfileSoul: (name: string) =>
    fetchJSON<{ content: string; exists: boolean }>(
      `/api/profiles/${encodeURIComponent(name)}/soul`,
    ),
  updateProfileSoul: (name: string, content: string) =>
    fetchJSON<{ ok: boolean }>(
      `/api/profiles/${encodeURIComponent(name)}/soul`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      },
    ),

  // Skills & Toolsets
  getSkills: () => fetchJSON<SkillInfo[]>("/api/skills"),
  toggleSkill: (name: string, enabled: boolean) =>
    fetchJSON<{ ok: boolean }>("/api/skills/toggle", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, enabled }),
    }),
  getToolsets: () => fetchJSON<ToolsetInfo[]>("/api/tools/toolsets"),

  // Session search (FTS5)
  searchSessions: (q: string) =>
    fetchJSON<SessionSearchResponse>(`/api/sessions/search?q=${encodeURIComponent(q)}`),

  // OAuth provider management
  getOAuthProviders: () =>
    fetchJSON<OAuthProvidersResponse>("/api/providers/oauth"),
  disconnectOAuthProvider: async (providerId: string) => {
    const token = await getSessionToken();
    return fetchJSON<{ ok: boolean; provider: string }>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}`,
      {
        method: "DELETE",
        headers: { [SESSION_HEADER]: token },
      },
    );
  },
  startOAuthLogin: async (providerId: string) => {
    const token = await getSessionToken();
    return fetchJSON<OAuthStartResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/start`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          [SESSION_HEADER]: token,
        },
        body: "{}",
      },
    );
  },
  submitOAuthCode: async (providerId: string, sessionId: string, code: string) => {
    const token = await getSessionToken();
    return fetchJSON<OAuthSubmitResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/submit`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          [SESSION_HEADER]: token,
        },
        body: JSON.stringify({ session_id: sessionId, code }),
      },
    );
  },
  pollOAuthSession: (providerId: string, sessionId: string) =>
    fetchJSON<OAuthPollResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/poll/${encodeURIComponent(sessionId)}`,
    ),
  cancelOAuthSession: async (sessionId: string) => {
    const token = await getSessionToken();
    return fetchJSON<{ ok: boolean }>(
      `/api/providers/oauth/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: "DELETE",
        headers: { [SESSION_HEADER]: token },
      },
    );
  },

  // Gateway / update actions
  restartGateway: () =>
    fetchJSON<ActionResponse>("/api/gateway/restart", { method: "POST" }),
  updateHermes: () =>
    fetchJSON<ActionResponse>("/api/hermes/update", { method: "POST" }),
  getActionStatus: (name: string, lines = 200) =>
    fetchJSON<ActionStatusResponse>(
      `/api/actions/${encodeURIComponent(name)}/status?lines=${lines}`,
    ),

  // Cockpit observer (read-only)
  getCockpitCapabilities: () =>
    fetchJSON<CockpitCapabilitiesResponse>("/api/cockpit/capabilities"),
  getCockpitLanes: (params: { limit?: number; offset?: number } = {}) => {
    const qs = new URLSearchParams();
    qs.set("limit", String(params.limit ?? 50));
    qs.set("offset", String(params.offset ?? 0));
    return fetchJSON<CockpitLanesResponse>(`/api/cockpit/lanes?${qs.toString()}`);
  },
  getCockpitSignals: (params: { limit?: number; message_limit?: number } = {}) => {
    const qs = new URLSearchParams();
    qs.set("limit", String(params.limit ?? 50));
    qs.set("message_limit", String(params.message_limit ?? 20));
    return fetchJSON<CockpitSignalsResponse>(`/api/cockpit/signals?${qs.toString()}`);
  },
  getCockpitAgentActivity: (params: { limit?: number; message_limit?: number } = {}) => {
    const qs = new URLSearchParams();
    qs.set("limit", String(params.limit ?? 50));
    qs.set("message_limit", String(params.message_limit ?? 8));
    return fetchJSON<CockpitAgentActivityResponse>(`/api/cockpit/agent-activity?${qs.toString()}`);
  },
  getCockpitLaneMessages: (
    laneId: string,
    params: { limit?: number; offset?: number } = {},
  ) => {
    const qs = new URLSearchParams();
    qs.set("limit", String(params.limit ?? 100));
    qs.set("offset", String(params.offset ?? 0));
    return fetchJSON<CockpitLaneMessagesResponse>(
      `/api/cockpit/lanes/${encodeURIComponent(laneId)}/messages?${qs.toString()}`,
    );
  },
  getCockpitN8nChecks: () =>
    fetchJSON<CockpitN8nChecksResponse>("/api/cockpit/n8n-checks"),
  getCockpitAutomationHealth: () =>
    fetchJSON<CockpitAutomationHealthResponse>("/api/cockpit/automation-health"),
  getCockpitDailyOpsRadar: () =>
    fetchJSON<CockpitDailyOpsRadarResponse>("/api/cockpit/daily-ops-radar"),
  getCockpitSelfWorkHandoff: () =>
    fetchJSON<CockpitSelfWorkHandoffResponse>("/api/cockpit/self-work-handoff"),
  sendCockpitIntent: (body: CockpitSendIntentRequest) =>
    fetchJSON<CockpitSendIntentResponse>("/api/cockpit/send-intents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  // Dashboard plugins
  getPlugins: () =>
    fetchJSON<PluginManifestResponse[]>("/api/dashboard/plugins"),
  rescanPlugins: () =>
    fetchJSON<{ ok: boolean; count: number }>("/api/dashboard/plugins/rescan"),

  getPluginsHub: () => fetchJSON<PluginsHubResponse>("/api/dashboard/plugins/hub"),

  installAgentPlugin: (body: AgentPluginInstallRequest) =>
    fetchJSON<AgentPluginInstallResponse>("/api/dashboard/agent-plugins/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body }),
    }),

  enableAgentPlugin: (name: string) =>
    fetchJSON<{ ok: boolean; name: string; unchanged?: boolean }>(
      `/api/dashboard/agent-plugins/${encodeURIComponent(name)}/enable`,
      { method: "POST" },
    ),

  disableAgentPlugin: (name: string) =>
    fetchJSON<{ ok: boolean; name: string; unchanged?: boolean }>(
      `/api/dashboard/agent-plugins/${encodeURIComponent(name)}/disable`,
      { method: "POST" },
    ),

  updateAgentPlugin: (name: string) =>
    fetchJSON<AgentPluginUpdateResponse>(
      `/api/dashboard/agent-plugins/${encodeURIComponent(name)}/update`,
      { method: "POST" },
    ),

  removeAgentPlugin: (name: string) =>
    fetchJSON<{ ok: boolean; name: string }>(
      `/api/dashboard/agent-plugins/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),

  savePluginProviders: (body: PluginProvidersPutRequest) =>
    fetchJSON<{ ok: boolean }>("/api/dashboard/plugin-providers", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  setPluginVisibility: (name: string, hidden: boolean) =>
    fetchJSON<{ ok: boolean; name: string; hidden: boolean }>(
      `/api/dashboard/plugins/${encodeURIComponent(name)}/visibility`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hidden }),
      },
    ),

  // Dashboard themes
  getThemes: () =>
    fetchJSON<DashboardThemesResponse>("/api/dashboard/themes"),
  setTheme: (name: string) =>
    fetchJSON<{ ok: boolean; theme: string }>("/api/dashboard/theme", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
};

export interface ActionResponse {
  name: string;
  ok: boolean;
  pid: number;
}

export interface ActionStatusResponse {
  exit_code: number | null;
  lines: string[];
  name: string;
  pid: number | null;
  running: boolean;
}

export interface CockpitCapabilitiesResponse {
  schema_version: number;
  read_only: boolean;
  input_enabled: boolean;
  control_enabled: boolean;
  external_send_enabled?: boolean;
  routing_enabled?: boolean;
  voice_enabled?: boolean;
  attachments_enabled?: boolean;
  local_chat?: {
    enabled: boolean;
    transport?: string;
    endpoint?: string;
    scope?: string;
  };
  risky_send?: {
    enabled: boolean;
    kill_switch_active: boolean;
    delivery_mode?: string;
    allowed_lanes?: CockpitResolvedLane[];
    requires_idempotency?: boolean;
    audit_enabled?: boolean;
  };
  endpoints: Record<string, unknown>;
  transcript_window?: CockpitTranscriptWindow;
  observer: {
    enabled: boolean;
    read_only: boolean;
    status?: string;
  };
}

export interface CockpitTranscriptWindow {
  kind: "recent" | string;
  bounded: boolean;
  window_limit: number;
  total_scope: "bounded_recent_window" | string;
}

export interface CockpitResolvedLane {
  lane_alias: string;
  lane_label: string;
  platform: string;
}

export interface CockpitDispatchResult {
  ok: boolean;
  message_ref_hash?: string;
  error_code?: string;
  warnings?: string[];
}

export interface CockpitSendRecord {
  schema_version: number;
  send_record_id: string;
  lane_alias: string;
  lane_label: string;
  platform: string;
  status: string;
  approval_policy: string;
  dispatch_count: number;
  gateway_request_id?: string | null;
  dispatch_result?: CockpitDispatchResult | null;
  error_code?: string | null;
  content_preview_redacted: string;
  created_at: string;
  updated_at: string;
}

export interface CockpitSendAuditEvent {
  event_type: string;
  send_record_id: string;
  idempotency_key_hash?: string;
  actor_user_id_hash?: string | null;
  approver_user_id_hash?: string | null;
  lane_alias: string;
  platform: string;
  message_digest?: string;
  content_preview_redacted: string;
  approval_policy: string;
  status: string;
  gateway_request_id?: string | null;
  dispatch_result?: CockpitDispatchResult | null;
  error_code?: string | null;
  created_at: string;
  updated_at: string;
}

export interface CockpitSendIntentRequest {
  lane_alias: string;
  idempotency_key: string;
  message_text: string;
}

export interface CockpitSendIntentResponse {
  schema_version?: number;
  ok: boolean;
  status: string;
  lane?: CockpitResolvedLane | null;
  record?: CockpitSendRecord | null;
  audit?: CockpitSendAuditEvent[];
  error_code?: string | null;
  idempotent_replay?: boolean;
}

export interface CockpitLaneAlias {
  alias: string;
  display_only: boolean;
  ambiguous?: boolean;
  canonical_ref?: null;
}

export interface CockpitLane {
  schema_version: number;
  lane_id: string;
  alias?: CockpitLaneAlias;
  platform?: string | null;
  chat_type?: string | null;
  title?: string | null;
  status?: string | null;
  updated_at?: string | number | null;
  completed_at?: string | number | null;
  agent_role?: "biff" | "forge" | "vex" | "quill" | "ranger" | string | null;
  role_label?: string | null;
  delegate_kind?: "delegate" | "continuation" | string | null;
  issue_identifier?: string | null;
  goal?: string | null;
  subagent_ref?: string | null;
  session_id?: string;
}

export interface CockpitLanesResponse {
  schema_version: number;
  lanes: CockpitLane[];
  total: number;
  limit: number;
  offset: number;
}

export interface CockpitMessage {
  role?: "user" | "assistant" | "system" | "tool" | string;
  content?: string | null;
  timestamp?: number;
  created_at?: string;
  status?: string;
  source?: string;
}

export interface CockpitLaneMessagesResponse {
  schema_version: number;
  lane_id: string;
  messages: CockpitMessage[];
  total: number;
  limit: number;
  offset: number;
  window?: CockpitTranscriptWindow;
  bounded?: boolean;
  window_limit?: number;
  total_scope?: "bounded_recent_window" | string;
}

export interface CockpitSelfWorkChecklistStep {
  label: string;
  status: "done" | "current" | "pending" | "blocked" | "skipped" | string;
}

export interface CockpitSelfWorkHandoffRecord {
  schema_version: number;
  handoff_id: string;
  created_at: string;
  updated_at: string;
  issue_identifier?: string | null;
  linear_url?: string | null;
  session_id?: string | null;
  parent_session_id?: string | null;
  goal?: string | null;
  current_phase?: string | null;
  last_action?: string | null;
  next_safe_step?: string | null;
  touched_files?: string[];
  touched_routes?: string[];
  completed_verification?: string[];
  pending_verification?: string[];
  known_failures?: string[];
  running_processes?: string[];
  operator_checklist?: {
    title?: string;
    current_index?: number | null;
    max_steps?: number;
    steps: CockpitSelfWorkChecklistStep[];
  };
  rendered_checklist?: string;
}

export interface CockpitSelfWorkHandoffResponse {
  schema_version: number;
  read_only: boolean;
  actions_enabled: boolean;
  mutation_enabled: boolean;
  source: string;
  has_handoff: boolean;
  handoff?: CockpitSelfWorkHandoffRecord | null;
}

export interface CockpitN8nCheck {
  id: string;
  name: string;
  status: string;
  last_run: string;
  next_schedule: string;
  delivery: string;
  auth: "configured" | "not_required" | "unknown" | string;
  action_needed: string;
  summary: string;
  live_source?: "live" | "live_missing_workflow" | "live_no_execution" | "fixture_fallback" | string;
  execution_status?: string;
  last_started?: string;
  last_completed?: string;
  output_summary?: string;
}

export interface CockpitN8nChecksResponse {
  schema_version: number;
  read_only: boolean;
  source: "n8n_live_latest_execution" | "fixture_bif_525_inventory" | string;
  generated_at: number;
  actions_enabled: boolean;
  external_delivery_enabled: boolean;
  live?: boolean;
  fallback?: boolean;
  stale?: boolean;
  live_error?: string;
  checks: CockpitN8nCheck[];
}

export interface CockpitAutomationHealthCard {
  title: string;
  bucket: "attention" | "healthy" | "stale_or_failing" | string;
  summary: string;
  last_checked: string;
  source: string;
  details?: string[];
}

export interface CockpitAutomationHealthResponse {
  schema_version: number;
  read_only: boolean;
  actions_enabled: boolean;
  mutation_enabled: boolean;
  external_delivery_enabled: boolean;
  generated_at: number;
  reading_model: {
    attention: string;
    healthy: string;
    stale_or_failing: string;
    last_checked_source: string;
  };
  summary: {
    attention: number;
    healthy: number;
    stale_or_failing: number;
    headline: string;
    last_checked: string;
    source: string;
  };
  cards: CockpitAutomationHealthCard[];
}

export interface CockpitDailyOpsRadarCommit {
  sha: string;
  title: string;
  files?: string;
  text: string;
}

export interface CockpitDailyOpsRadarCategory {
  label: string;
  summary: string;
  more_count?: number | null;
  items: CockpitDailyOpsRadarCommit[];
}

export interface CockpitDailyOpsRadarUpgradeBrief {
  question: string;
  headline: string;
  groups: Array<{
    label: string;
    summary: string;
    examples: string[];
  }>;
  why_this_matters: string;
}

export interface CockpitDailyOpsRadarResponse {
  schema_version: number;
  read_only: boolean;
  actions_enabled: boolean;
  external_delivery_enabled: boolean;
  generated_at: number;
  source: string;
  job: {
    id: string;
    name: string;
    script: string;
    enabled?: boolean | null;
    state?: string | null;
    status?: string | null;
    schedule?: string | null;
    last_run_at?: string | null;
    next_run_at?: string | null;
  };
  summary: {
    last_run?: string | null;
    behind_count?: number | null;
    relevant_change_count?: number | null;
    behind_ref?: string | null;
    repo?: string | null;
    categories: CockpitDailyOpsRadarCategory[];
    top_commits: CockpitDailyOpsRadarCommit[];
    compare_command?: string | null;
    source_file?: string;
    source_mtime?: number | null;
    upgrade_brief?: CockpitDailyOpsRadarUpgradeBrief | null;
  };
  raw_excerpt: string;
  upgrade: {
    brief?: CockpitDailyOpsRadarUpgradeBrief | null;
    recommendation: {
      question: "Should we upgrade?" | string;
      label: "Upgrade now" | "Prepare review first" | "Wait" | "Do not upgrade" | string;
      risk_level: "low" | "medium" | "high" | "unknown" | string;
      rationale: string;
      basis: string;
      freshness: "fresh" | "stale" | "unknown" | string;
      freshness_age_seconds?: number | null;
      freshness_detail: string;
      signals: string[];
      behind_count?: number | null;
      relevant_change_count?: number | null;
    };
    prepare_review: {
      label: string;
      enabled: boolean;
      mutates: boolean;
      method: string;
      endpoint?: string | null;
      status: string;
      description: string;
    };
  };
}

export interface CockpitSignalSource {
  lane_id?: string | null;
  lane_title?: string | null;
  platform?: string | null;
  status?: string | null;
  message_role?: string | null;
  agent_role?: string | null;
  role_label?: string | null;
  delegate_kind?: string | null;
  issue_identifier?: string | null;
  goal?: string | null;
  class?: "operational" | "sessiondb" | "event" | string;
  freshness?: "operational" | "recent_context" | "archive" | string;
}

export interface CockpitSignalFreshness {
  bucket: "operational" | "recent_context" | "archive" | string;
  age_seconds?: number | null;
  threshold_seconds?: number;
  archive_threshold_seconds?: number;
}

export interface CockpitSignal {
  schema_version: number;
  id: string;
  category: "now" | "needs_marco" | "stuck_failed" | "waiting" | "recently_completed" | "active_role_work" | string;
  lane_id: string;
  title: string;
  source: CockpitSignalSource;
  provenance: "lane_snapshot" | "lane_message" | "lane_event" | string;
  timestamp?: string | number | null;
  recency_label: string;
  freshness?: CockpitSignalFreshness;
  confidence: number;
  reason: string;
}

export interface CockpitSignalCategory {
  label: string;
  empty_state: string;
  signals: CockpitSignal[];
}

export interface CockpitSignalsResponse {
  schema_version: number;
  read_only: boolean;
  generated_at: number;
  categories: Record<string, CockpitSignalCategory>;
  total: number;
}

export interface CockpitAgentActivityItem {
  schema_version: number;
  id: string;
  lane_id: string;
  agent_role: "biff" | "forge" | "vex" | "quill" | "ranger" | string;
  role_label: string;
  status: "running" | "completed" | "failed" | "blocked" | "waiting" | "stale" | string;
  issue_identifier?: string | null;
  goal?: string | null;
  title?: string | null;
  updated_at?: string | number | null;
  completed_at?: string | number | null;
  latest_evidence: string;
  latest_evidence_role?: string | null;
  recency_label: string;
  freshness?: CockpitSignalFreshness;
  delegate_kind?: "delegate" | "continuation" | string | null;
  source?: { class?: string; freshness?: string };
}

export interface CockpitAgentActivityResponse {
  schema_version: number;
  read_only: boolean;
  actions_enabled: boolean;
  mutation_enabled: boolean;
  external_delivery_enabled: boolean;
  generated_at: number;
  items: CockpitAgentActivityItem[];
  total: number;
  counts: Record<string, number>;
  empty_state: string;
}

export interface PlatformStatus {
  error_code?: string;
  error_message?: string;
  state: string;
  updated_at: string;
}

export interface SessionQuotaRecommendation {
  threshold: number;
  level: string;
  prompt_tokens: number;
  text: string;
  dedupe_key: string;
  context_length?: number;
}

export interface BiffOperatingMode {
  name: string;
  label: string;
  description: string;
  max_iterations?: number | null;
  max_tool_output_chars?: number;
  max_message_content_chars?: number | null;
  preview_chars?: number;
}

export interface StatusResponse {
  active_sessions: number;
  biff_operating_mode?: BiffOperatingMode;
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
  release_date: string;
  session_quota_recommendation?: SessionQuotaRecommendation | null;
  version: string;
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
}

export interface LogsResponse {
  file: string;
  lines: string[];
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

export interface ProfileInfo {
  name: string;
  path: string;
  is_default: boolean;
  model: string | null;
  provider: string | null;
  has_env: boolean;
  skill_count: number;
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

export interface CronJob {
  id: string;
  name?: string | null;
  prompt?: string | null;
  script?: string | null;
  schedule?: { kind?: string; expr?: string; display?: string };
  schedule_display?: string | null;
  enabled: boolean;
  state?: string | null;
  deliver?: string | null;
  last_run_at?: string | null;
  next_run_at?: string | null;
  last_error?: string | null;
}

export interface SkillInfo {
  name: string;
  description: string;
  category: string;
  enabled: boolean;
}

export interface ToolsetInfo {
  name: string;
  label: string;
  description: string;
  enabled: boolean;
  configured: boolean;
  tools: string[];
}

export interface SessionSearchResult {
  session_id: string;
  snippet: string;
  role: string | null;
  source: string | null;
  model: string | null;
  session_started: number | null;
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

export interface ModelAssignmentRequest {
  scope: "main" | "auxiliary";
  provider: string;
  model: string;
  /** For auxiliary: task slot name, "" for all, "__reset__" to reset all. */
  task?: string;
}

export interface ModelAssignmentResponse {
  ok: boolean;
  scope?: string;
  provider?: string;
  model?: string;
  tasks?: string[];
  reset?: boolean;
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
}

export interface PluginsHubProviders {
  memory_provider: string;
  memory_options: Array<{ name: string; description: string }>;
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
