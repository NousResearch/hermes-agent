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
    /** Server-injected flag: ``true`` when the dashboard's OAuth gate is
     * engaged (public bind, no ``--insecure``). Toggles the SPA's
     * WS-upgrade path from legacy ``?token=`` to single-use ``?ticket=``
     * fetched via :func:`getWsTicket`. */
    __HERMES_AUTH_REQUIRED__?: boolean;
  }
}
let _sessionToken: string | null = null;
const SESSION_HEADER = "X-Hermes-Session-Token";

function setSessionHeader(headers: Headers, token: string): void {
  if (!headers.has(SESSION_HEADER)) {
    headers.set(SESSION_HEADER, token);
  }
}

// ── Global management-profile scope ──────────────────────────────────
// The dashboard is a machine-level management surface: one header switcher
// (ProfileProvider in App.tsx) decides which profile the management pages
// read/write, and fetchJSON transparently appends ?profile=<name> to the
// profile-scoped endpoint families below. "" = the dashboard process's own
// profile (legacy behavior). Calls that already carry an explicit profile
// (e.g. ProfileBuilder writes) are left untouched — explicit beats global.
let _managementProfile = "";

export function setManagementProfile(name: string): void {
  _managementProfile = (name || "").trim();
}

export function getManagementProfile(): string {
  return _managementProfile;
}

// Endpoint families that honor ?profile= on the backend (web_server.py
// _profile_scope or explicit per-profile DB opens). Anything else — ops,
// pairing, telegram onboarding, cron (which has its own per-job profile
// params), profiles themselves — is machine-global or self-scoped and must
// NOT be rewritten.
const PROFILE_SCOPED_PREFIXES = [
  "/api/analytics",
  "/api/skills",
  "/api/tools/toolsets",
  "/api/config",
  "/api/env",
  "/api/mcp",
  "/api/messaging/platforms",
  "/api/model/info",
  "/api/model/set",
  "/api/model/auxiliary",
  "/api/model/options",
];

function withManagementProfile(url: string): string {
  if (!_managementProfile) return url;
  if (url.includes("profile=")) return url; // explicit param wins
  const path = url.split("?")[0];
  if (!PROFILE_SCOPED_PREFIXES.some((p) => path.startsWith(p))) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}profile=${encodeURIComponent(_managementProfile)}`;
}

export async function fetchJSON<T>(
  url: string,
  init?: RequestInit,
  options?: FetchJSONOptions,
): Promise<T> {
  url = withManagementProfile(url);
  // Inject the session token into all /api/ requests.
  const headers = new Headers(init?.headers);
  const token = window.__HERMES_SESSION_TOKEN__;
  if (token) {
    setSessionHeader(headers, token);
  }
  const res = await fetch(`${BASE}${url}`, {
    ...init,
    headers,
    // ``credentials: 'include'`` so the cookie-auth path (gated mode) works
    // for any fetch routed through here. Loopback mode is unaffected — the
    // server doesn't read cookies and the legacy session-token header is
    // already attached above.
    credentials: init?.credentials ?? "include",
  });
  if (res.status === 401) {
    // Phase 6: the gated middleware emits a structured envelope so the
    // SPA can full-page-navigate to /login on session expiry. Parse it,
    // and only redirect on the known error codes — domain-level 401s
    // (e.g. "you don't have permission to read this monitor") bubble
    // up as regular errors so callers can handle them.
    let body: { error?: string; login_url?: string } = {};
    try {
      body = await res.clone().json();
    } catch {
      /* non-JSON 401 — let it fall through */
    }
    if (
      (body.error === "unauthenticated" || body.error === "session_expired") &&
      body.login_url
    ) {
      // Preserve where the user was so /auth/callback can land them back
      // after re-auth. The gate's login_url already carries a ``next=``
      // built from the request path, but the SPA may be deep inside a
      // SPA route the gate never saw — e.g. a hash route or a client-side
      // /sessions/<id> deep link. Save the current location as a
      // fallback the post-login handler can read.
      try {
        sessionStorage.setItem(
          "hermes.lastLocation",
          window.location.pathname + window.location.search,
        );
      } catch {
        /* SSR / privacy mode — ignore */
      }
      window.location.assign(body.login_url);
      // Never resolve — the page is about to unload.
      return new Promise<T>(() => {});
    }
    // Loopback mode: ``_SESSION_TOKEN`` rotates on every server restart
    // (``hermes update``, ``hermes gateway restart``, etc.). A tab kept
    // open across the restart holds the OLD token in
    // ``window.__HERMES_SESSION_TOKEN__`` from the previous HTML render,
    // so every fetch returns 401. The HTML is served ``Cache-Control:
    // no-store`` so a reload picks up the freshly-injected token. Trigger
    // that reload once on the first stale-token 401 — gated mode is
    // handled above, so reaching here in gated mode means a real
    // middleware failure that should not reload-loop.
    if (!window.__HERMES_AUTH_REQUIRED__ && !options?.allowUnauthorized) {
      let alreadyReloaded = false;
      try {
        alreadyReloaded =
          sessionStorage.getItem("hermes.tokenReloadAttempted") === "1";
      } catch {
        /* SSR / privacy mode — fall through to throw */
      }
      if (!alreadyReloaded) {
        try {
          sessionStorage.setItem("hermes.tokenReloadAttempted", "1");
        } catch {
          /* SSR / privacy mode — best effort */
        }
        window.location.reload();
        return new Promise<T>(() => {});
      }
    }
  }
  if (res.ok) {
    // Clear the stale-token reload guard: a successful 2xx proves the
    // current ``window.__HERMES_SESSION_TOKEN__`` is valid, so the next
    // 401 — if any — should be allowed to trigger its own reload cycle.
    try {
      sessionStorage.removeItem("hermes.tokenReloadAttempted");
    } catch {
      /* SSR / privacy mode — ignore */
    }
  }
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

/** Encode a plugin registry key for URL paths (preserves `/` segment separators). */
function pluginPath(name: string): string {
  return name.split("/").map(encodeURIComponent).join("/");
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

/**
 * Fetch a single-use ticket for a WebSocket upgrade in gated mode.
 *
 * The dashboard's gated-mode WS auth (``hermes_cli.web_server._ws_auth_ok``)
 * rejects the legacy ``?token=<_SESSION_TOKEN>`` path and only accepts
 * ``?ticket=<minted>`` consumed against the in-memory ticket store. Browsers
 * can't set ``Authorization`` on a WS upgrade, so this round-trip via the
 * authenticated REST endpoint is the bridge from cookie auth to WS auth.
 *
 * Tickets are single-use and TTL=30s — every WS connect attempt must
 * fetch a fresh ticket.
 */
export async function getWsTicket(): Promise<{ ticket: string; ttl_seconds: number }> {
  const res = await fetch(`${BASE}/api/auth/ws-ticket`, {
    method: "POST",
    credentials: "include",
  });
  if (!res.ok) {
    throw new Error(`/api/auth/ws-ticket: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Resolve the auth query-param pair (``[name, value]``) for a WebSocket
 * connect. In gated mode mints a fresh single-use ticket; in loopback
 * mode returns the injected session token.
 */
export async function buildWsAuthParam(): Promise<[string, string]> {
  if (window.__HERMES_AUTH_REQUIRED__) {
    const { ticket } = await getWsTicket();
    return ["ticket", ticket];
  }
  const token = window.__HERMES_SESSION_TOKEN__ ?? "";
  return ["token", token];
}

/**
 * Authenticated ``fetch`` for dashboard ``/api/...`` requests that aren't
 * plain JSON — file uploads (``FormData``), binary downloads (blobs), etc.
 * Mirrors ``fetchJSON``'s auth handling but returns the raw ``Response`` so
 * the caller can read ``.blob()`` / ``.formData()`` / stream it.
 *
 * Auth, in both modes, exactly as ``fetchJSON`` does it:
 *  - loopback / ``--insecure``: attach the ``X-Hermes-Session-Token`` header.
 *  - gated OAuth: no token header (it's absent by design); the
 *    ``hermes_session_at`` cookie rides along via ``credentials: 'include'``.
 *
 * Unlike ``fetchJSON`` this does NOT parse the body, does NOT throw on
 * non-2xx (the caller decides — a 404 on a download is meaningful), and
 * does NOT run the global 401 → /login redirect (binary endpoints aren't
 * navigation targets). Callers that want the redirect behaviour should use
 * ``fetchJSON``.
 */
export async function authedFetch(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const headers = new Headers(init?.headers);
  const token = window.__HERMES_SESSION_TOKEN__;
  if (token) {
    setSessionHeader(headers, token);
  }
  return fetch(`${BASE}${url}`, {
    ...init,
    headers,
    credentials: init?.credentials ?? "include",
  });
}

/**
 * Build an absolute ``ws(s)://`` URL for a dashboard WebSocket endpoint,
 * with the correct auth query param appended for the active mode (fresh
 * single-use ``ticket`` in gated mode, ``token`` in loopback). Plugins and
 * the SPA should use this instead of hand-assembling a WS URL + reading
 * ``window.__HERMES_SESSION_TOKEN__`` directly, so the gated-mode ticket
 * path can never be forgotten.
 *
 * ``path`` is the dashboard-relative path (e.g.
 * ``"/api/plugins/kanban/events"``); the base-path prefix and host are
 * applied here. Extra query params can be supplied via ``params`` and are
 * merged before the auth param.
 */
export async function buildWsUrl(
  path: string,
  params?: Record<string, string>,
): Promise<string> {
  const [authName, authValue] = await buildWsAuthParam();
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const qs = new URLSearchParams(params ?? {});
  qs.set(authName, authValue);
  return `${proto}//${window.location.host}${BASE}${path}?${qs}`;
}

/** Build a ``?profile=<name>`` query suffix, or "" when unset.
 *
 * Used by the skills/toolsets endpoints so the dashboard can manage a
 * profile other than the one the server process runs under. */
function profileQuery(profile?: string): string {
  return profile ? `?profile=${encodeURIComponent(profile)}` : "";
}

function appendProfileParam(url: string, profile?: string): string {
  if (!profile || url.includes("profile=")) return url;
  return `${url}${url.includes("?") ? "&" : "?"}profile=${encodeURIComponent(profile)}`;
}

export const api = {
  getStatus: () => fetchJSON<StatusResponse>("/api/status"),
  /**
   * Identity probe for the dashboard auth gate (Phase 7).
   *
   * Returns the verified Session as JSON when gated mode is active and a
   * valid cookie is attached. Loopback mode is unaffected — the endpoint
   * still exists but is never useful there (no Session, no cookie). The
   * AuthWidget component swallows 401s from this call: if the gate isn't
   * engaged, /api/auth/me returns 401 and the widget renders nothing.
   *
   * ``allowUnauthorized`` is load-bearing: in loopback mode this endpoint
   * 401s by design, and fetchJSON's default loopback behaviour treats a
   * 401 as a rotated session token and full-page-reloads to pick up a
   * fresh one. Because every *other* dashboard request succeeds (and so
   * clears the one-shot reload guard), that turns this expected 401 into
   * an infinite reload loop. Opting out keeps the 401 a plain throw the
   * widget can catch.
   */
  getAuthMe: () =>
    fetchJSON<AuthMeResponse>("/api/auth/me", undefined, {
      allowUnauthorized: true,
    }),
  logout: () =>
    fetch(`${BASE}/auth/logout`, {
      method: "POST",
      credentials: "include",
    }).then((r) => {
      // /auth/logout returns 302 → /login. Follow that with a full-page
      // navigation rather than letting fetch() opaquely consume the
      // redirect — the SPA needs to leave the protected area.
      window.location.assign("/login");
      return r;
    }),
  getSessions: (limit = 20, offset = 0, profile = getManagementProfile()) =>
    fetchJSON<PaginatedSessions>(
      appendProfileParam(`/api/sessions?limit=${limit}&offset=${offset}`, profile),
    ),
  getSessionMessages: (id: string, profile = getManagementProfile()) =>
    fetchJSON<SessionMessagesResponse>(
      appendProfileParam(`/api/sessions/${encodeURIComponent(id)}/messages`, profile),
    ),
  getSessionLatestDescendant: (id: string) =>
    fetchJSON<SessionLatestDescendantResponse>(
      `/api/sessions/${encodeURIComponent(id)}/latest-descendant`,
    ),
  deleteSession: (id: string, profile = getManagementProfile()) =>
    fetchJSON<{ ok: boolean }>(
      appendProfileParam(`/api/sessions/${encodeURIComponent(id)}`, profile),
      {
        method: "DELETE",
      },
    ),
  getEmptySessionsCount: (profile = getManagementProfile()) =>
    fetchJSON<{ count: number }>(
      appendProfileParam("/api/sessions/empty/count", profile),
    ),
  deleteEmptySessions: (profile = getManagementProfile()) =>
    fetchJSON<{ ok: boolean; deleted: number }>(
      appendProfileParam("/api/sessions/empty", profile),
      {
        method: "DELETE",
      },
    ),
  bulkDeleteSessions: (ids: string[], profile = getManagementProfile()) =>
    fetchJSON<{ ok: boolean; deleted: number }>("/api/sessions/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids, profile: profile || undefined }),
    }),
  renameSession: (id: string, title: string, profile = getManagementProfile()) =>
    fetchJSON<{ ok: boolean; title: string }>(
      `/api/sessions/${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, profile: profile || undefined }),
      },
    ),
  getSessionStats: (profile = getManagementProfile()) =>
    fetchJSON<SessionStoreStats>(appendProfileParam("/api/sessions/stats", profile)),
  exportSessionUrl: (id: string, profile = getManagementProfile()) =>
    appendProfileParam(`/api/sessions/${encodeURIComponent(id)}/export`, profile),
  pruneSessions: (
    older_than_days: number,
    source?: string,
    profile = getManagementProfile(),
  ) =>
    fetchJSON<{ ok: boolean; removed: number }>("/api/sessions/prune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ older_than_days, source, profile: profile || undefined }),
    }),
  listFiles: (path?: string) => {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    return fetchJSON<ManagedFilesResponse>(`/api/files${query}`);
  },
  readFile: (path: string) =>
    fetchJSON<ManagedFileReadResponse>(
      `/api/files/read?path=${encodeURIComponent(path)}`,
    ),
  uploadFile: (path: string, dataUrl: string, overwrite = true) =>
    fetchJSON<ManagedFileWriteResponse>("/api/files/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, data_url: dataUrl, overwrite }),
    }),
  createDirectory: (path: string) =>
    fetchJSON<ManagedFileWriteResponse>("/api/files/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }),
  deleteFile: (path: string, recursive = false) =>
    fetchJSON<{ ok: boolean; path: string }>("/api/files", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, recursive }),
    }),
  getLogs: (params: { file?: string; lines?: number; level?: string; component?: string }) => {
    const qs = new URLSearchParams();
    if (params.file) qs.set("file", params.file);
    if (params.lines) qs.set("lines", String(params.lines));
    if (params.level && params.level !== "ALL") qs.set("level", params.level);
    if (params.component && params.component !== "all") qs.set("component", params.component);
    return fetchJSON<LogsResponse>(`/api/logs?${qs.toString()}`);
  },
  getAnalytics: (days: number, profile = getManagementProfile()) =>
    fetchJSON<AnalyticsResponse>(
      appendProfileParam(`/api/analytics/usage?days=${days}`, profile),
    ),
  getModelsAnalytics: (days: number, profile = getManagementProfile()) =>
    fetchJSON<ModelsAnalyticsResponse>(
      appendProfileParam(`/api/analytics/models?days=${days}`, profile),
    ),
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
  getConfigRaw: () => fetchJSON<{ yaml: string; path?: string }>("/api/config/raw"),
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
  getCronJobs: (profile = "all") =>
    fetchJSON<CronJob[]>(`/api/cron/jobs?profile=${encodeURIComponent(profile)}`),
  getCronDeliveryTargets: () =>
    fetchJSON<{ targets: CronDeliveryTarget[] }>("/api/cron/delivery-targets"),
  createCronJob: (job: { prompt: string; schedule: string; name?: string; deliver?: string; skills?: string[] }, profile = "default") =>
    fetchJSON<CronJob>(`/api/cron/jobs?profile=${encodeURIComponent(profile)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(job),
    }),
  pauseCronJob: (id: string, profile = "default") =>
    fetchJSON<CronJob>(`/api/cron/jobs/${encodeURIComponent(id)}/pause?profile=${encodeURIComponent(profile)}`, { method: "POST" }),
  updateCronJob: (
    id: string,
    updates: { prompt?: string; schedule?: string; name?: string; deliver?: string; skills?: string[] },
    profile = "default",
  ) =>
    fetchJSON<CronJob>(
      `/api/cron/jobs/${encodeURIComponent(id)}?profile=${encodeURIComponent(profile)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates }),
      },
    ),
  resumeCronJob: (id: string, profile = "default") =>
    fetchJSON<CronJob>(`/api/cron/jobs/${encodeURIComponent(id)}/resume?profile=${encodeURIComponent(profile)}`, { method: "POST" }),
  triggerCronJob: (id: string, profile = "default") =>
    fetchJSON<CronJob>(`/api/cron/jobs/${encodeURIComponent(id)}/trigger?profile=${encodeURIComponent(profile)}`, { method: "POST" }),
  deleteCronJob: (id: string, profile = "default") =>
    fetchJSON<{ ok: boolean }>(`/api/cron/jobs/${encodeURIComponent(id)}?profile=${encodeURIComponent(profile)}`, { method: "DELETE" }),

  // Automation Blueprints — parameterized automation blueprints
  getAutomationBlueprints: () =>
    fetchJSON<{ blueprints: AutomationBlueprint[] }>("/api/cron/blueprints"),
  instantiateAutomationBlueprint: (
    body: { blueprint: string; values: Record<string, string> },
    profile = "default",
  ) =>
    fetchJSON<CronJob>(`/api/cron/blueprints/instantiate?profile=${encodeURIComponent(profile)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  // Profiles
  getProfiles: () =>
    fetchJSON<{ profiles: ProfileInfo[] }>("/api/profiles"),
  getActiveProfile: () =>
    fetchJSON<ActiveProfileInfo>("/api/profiles/active"),
  setActiveProfile: (name: string) =>
    fetchJSON<{ ok: boolean; active: string }>("/api/profiles/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  createProfile: (body: {
    name: string;
    clone_from?: string | null;
    clone_from_default?: boolean;
    clone_all?: boolean;
    no_skills?: boolean;
    description?: string;
    provider?: string;
    model?: string;
    mcp_servers?: McpServerCreate[];
    keep_skills?: string[];
    hub_skills?: string[];
  }) =>
    fetchJSON<{
      ok: boolean;
      name: string;
      path: string;
      model_set?: boolean;
      mcp_written?: number;
      skills_disabled?: number;
      hub_installs?: Array<{ identifier: string; pid: number | null }>;
    }>("/api/profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  updateProfileDescription: (name: string, description: string) =>
    fetchJSON<{ ok: boolean; description: string; description_auto: boolean }>(
      `/api/profiles/${encodeURIComponent(name)}/description`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description }),
      },
    ),
  describeProfileAuto: (name: string, overwrite = true) =>
    fetchJSON<ProfileDescribeAutoResult>(
      `/api/profiles/${encodeURIComponent(name)}/describe-auto`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ overwrite }),
      },
    ),
  setProfileModel: (name: string, provider: string, model: string) =>
    fetchJSON<{ ok: boolean; provider: string; model: string }>(
      `/api/profiles/${encodeURIComponent(name)}/model`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model }),
      },
    ),
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
  //
  // All calls accept an optional ``profile`` so the Skills page can manage
  // any profile's skills/toolsets — not just the one the dashboard process
  // runs under. Omitted/empty profile = the dashboard's own profile.
  getSkills: (profile?: string) =>
    fetchJSON<SkillInfo[]>(`/api/skills${profileQuery(profile)}`),
  toggleSkill: (name: string, enabled: boolean, profile?: string) =>
    fetchJSON<{ ok: boolean }>("/api/skills/toggle", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, enabled, profile: profile || undefined }),
    }),
  getSkillContent: (name: string, profile?: string) =>
    fetchJSON<SkillContent>(
      `/api/skills/content?name=${encodeURIComponent(name)}${profile ? `&profile=${encodeURIComponent(profile)}` : ""}`,
    ),
  createSkill: (skill: { name: string; content: string; category?: string }, profile?: string) =>
    fetchJSON<SkillWriteResult>("/api/skills", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...skill, profile: profile || undefined }),
    }),
  updateSkillContent: (name: string, content: string, profile?: string) =>
    fetchJSON<SkillWriteResult>("/api/skills/content", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, content, profile: profile || undefined }),
    }),
  getToolsets: (profile?: string) =>
    fetchJSON<ToolsetInfo[]>(`/api/tools/toolsets${profileQuery(profile)}`),
  toggleToolset: (name: string, enabled: boolean, profile?: string) =>
    fetchJSON<{ ok: boolean; name: string; enabled: boolean }>(
      `/api/tools/toolsets/${encodeURIComponent(name)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled, profile: profile || undefined }),
      },
    ),
  getToolsetConfig: (name: string, profile?: string) =>
    fetchJSON<ToolsetConfig>(
      `/api/tools/toolsets/${encodeURIComponent(name)}/config${profileQuery(profile)}`,
    ),
  selectToolsetProvider: (name: string, provider: string, profile?: string) =>
    fetchJSON<{ ok: boolean; name: string; provider: string }>(
      `/api/tools/toolsets/${encodeURIComponent(name)}/provider`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, profile: profile || undefined }),
      },
    ),
  saveToolsetEnv: (name: string, env: Record<string, string>, profile?: string) =>
    fetchJSON<ToolsetEnvResult>(
      `/api/tools/toolsets/${encodeURIComponent(name)}/env`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ env, profile: profile || undefined }),
      },
    ),
  runToolsetPostSetup: (name: string, key: string, profile?: string) =>
    fetchJSON<ActionResponse & { key: string }>(
      `/api/tools/toolsets/${encodeURIComponent(name)}/post-setup`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, profile: profile || undefined }),
      },
    ),

  // Session search (FTS5)
  searchSessions: (q: string, profile = getManagementProfile()) =>
    fetchJSON<SessionSearchResponse>(
      appendProfileParam(`/api/sessions/search?q=${encodeURIComponent(q)}`, profile),
    ),

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

  // Messaging platforms (gateway channels)
  getMessagingPlatforms: () =>
    fetchJSON<{ platforms: MessagingPlatform[] }>("/api/messaging/platforms"),
  updateMessagingPlatform: (id: string, body: MessagingPlatformUpdate) =>
    fetchJSON<{ ok: boolean; platform: string }>(
      `/api/messaging/platforms/${encodeURIComponent(id)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  testMessagingPlatform: (id: string) =>
    fetchJSON<MessagingPlatformTestResult>(
      `/api/messaging/platforms/${encodeURIComponent(id)}/test`,
      { method: "POST" },
    ),
  startTelegramOnboarding: (body: { bot_name?: string }) =>
    fetchJSON<TelegramOnboardingStartResponse>(
      "/api/messaging/telegram/onboarding/start",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  getTelegramOnboardingStatus: (pairingId: string) =>
    fetchJSON<TelegramOnboardingStatusResponse>(
      `/api/messaging/telegram/onboarding/${encodeURIComponent(pairingId)}`,
    ),
  applyTelegramOnboarding: (
    pairingId: string,
    body: { allowed_user_ids: string[] },
  ) =>
    fetchJSON<TelegramOnboardingApplyResponse>(
      `/api/messaging/telegram/onboarding/${encodeURIComponent(pairingId)}/apply`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  cancelTelegramOnboarding: (pairingId: string) =>
    fetchJSON<{ ok: boolean }>(
      `/api/messaging/telegram/onboarding/${encodeURIComponent(pairingId)}`,
      { method: "DELETE" },
    ),

  // Gateway / update actions
  restartGateway: () =>
    fetchJSON<ActionResponse>("/api/gateway/restart", { method: "POST" }),
  updateHermes: () =>
    fetchJSON<ActionResponse>("/api/hermes/update", { method: "POST" }),
  checkHermesUpdate: (force = false) =>
    fetchJSON<UpdateCheckResponse>(
      `/api/hermes/update/check${force ? "?force=true" : ""}`,
    ),
  getActionStatus: (name: string, lines = 200) =>
    fetchJSON<ActionStatusResponse>(
      `/api/actions/${encodeURIComponent(name)}/status?lines=${lines}`,
    ),

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
      `/api/dashboard/agent-plugins/${pluginPath(name)}/enable`,
      { method: "POST" },
    ),

  disableAgentPlugin: (name: string) =>
    fetchJSON<{ ok: boolean; name: string; unchanged?: boolean }>(
      `/api/dashboard/agent-plugins/${pluginPath(name)}/disable`,
      { method: "POST" },
    ),

  updateAgentPlugin: (name: string) =>
    fetchJSON<AgentPluginUpdateResponse>(
      `/api/dashboard/agent-plugins/${pluginPath(name)}/update`,
      { method: "POST" },
    ),

  removeAgentPlugin: (name: string) =>
    fetchJSON<{ ok: boolean; name: string }>(
      `/api/dashboard/agent-plugins/${pluginPath(name)}`,
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
      `/api/dashboard/plugins/${pluginPath(name)}/visibility`,
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
  getFontPref: () =>
    fetchJSON<DashboardFontResponse>("/api/dashboard/font"),
  setFontPref: (font: string) =>
    fetchJSON<{ ok: boolean; font: string }>("/api/dashboard/font", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ font }),
    }),

  // ── Admin: MCP servers ──────────────────────────────────────────────
  getMcpServers: () => fetchJSON<{ servers: McpServer[] }>("/api/mcp/servers"),
  addMcpServer: (body: McpServerCreate) =>
    fetchJSON<McpServer>("/api/mcp/servers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  removeMcpServer: (name: string) =>
    fetchJSON<{ ok: boolean }>(`/api/mcp/servers/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),
  testMcpServer: (name: string) =>
    fetchJSON<McpTestResult>(
      `/api/mcp/servers/${encodeURIComponent(name)}/test`,
      { method: "POST" },
    ),
  setMcpServerEnabled: (name: string, enabled: boolean) =>
    fetchJSON<{ ok: boolean; name: string; enabled: boolean }>(
      `/api/mcp/servers/${encodeURIComponent(name)}/enabled`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      },
    ),
  getMcpCatalog: () =>
    fetchJSON<{ entries: McpCatalogEntry[]; diagnostics: McpCatalogDiagnostic[] }>(
      "/api/mcp/catalog",
    ),
  installMcpCatalogEntry: (
    name: string,
    env: Record<string, string> = {},
    enable = true,
  ) =>
    fetchJSON<{ ok: boolean; name: string; background: boolean; action?: string }>(
      "/api/mcp/catalog/install",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, env, enable }),
      },
    ),

  // ── Admin: Pairing ──────────────────────────────────────────────────
  getPairing: () => fetchJSON<PairingResponse>("/api/pairing"),
  approvePairing: (platform: string, code: string) =>
    fetchJSON<{ ok: boolean; user: PairingUser }>("/api/pairing/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ platform, code }),
    }),
  revokePairing: (platform: string, user_id: string) =>
    fetchJSON<{ ok: boolean }>("/api/pairing/revoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ platform, user_id }),
    }),
  clearPendingPairing: () =>
    fetchJSON<{ ok: boolean; cleared: number }>("/api/pairing/clear-pending", {
      method: "POST",
    }),

  // ── Admin: Webhooks ─────────────────────────────────────────────────
  getWebhooks: () => fetchJSON<WebhooksResponse>("/api/webhooks"),
  enableWebhooks: () =>
    fetchJSON<WebhookEnableResponse>("/api/webhooks/enable", { method: "POST" }),
  createWebhook: (body: WebhookCreate) =>
    fetchJSON<WebhookRoute & { secret: string }>("/api/webhooks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteWebhook: (name: string) =>
    fetchJSON<{ ok: boolean }>(`/api/webhooks/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),
  setWebhookEnabled: (name: string, enabled: boolean) =>
    fetchJSON<{ ok: boolean; name: string; enabled: boolean }>(
      `/api/webhooks/${encodeURIComponent(name)}/enabled`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      },
    ),

  // ── Admin: Credential pool ──────────────────────────────────────────
  getCredentialPool: () =>
    fetchJSON<{ providers: CredentialPoolProvider[] }>("/api/credentials/pool"),
  addCredentialPoolEntry: (
    provider: string,
    api_key: string,
    label?: string,
  ) =>
    fetchJSON<{ ok: boolean; provider: string; count: number }>(
      "/api/credentials/pool",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, api_key, label }),
      },
    ),
  removeCredentialPoolEntry: (provider: string, index: number) =>
    fetchJSON<{ ok: boolean; provider: string; count: number }>(
      `/api/credentials/pool/${encodeURIComponent(provider)}/${index}`,
      { method: "DELETE" },
    ),

  // ── Admin: Memory provider ──────────────────────────────────────────
  getMemory: () => fetchJSON<MemoryStatus>("/api/memory"),
  setMemoryProvider: (provider: string) =>
    fetchJSON<{ ok: boolean; active: string }>("/api/memory/provider", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    }),
  resetMemory: (target: "all" | "memory" | "user") =>
    fetchJSON<{ ok: boolean; deleted: string[] }>("/api/memory/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    }),

  // ── Admin: Gateway lifecycle ────────────────────────────────────────
  startGateway: () =>
    fetchJSON<ActionResponse>("/api/gateway/start", { method: "POST" }),
  stopGateway: () =>
    fetchJSON<ActionResponse>("/api/gateway/stop", { method: "POST" }),

  // ── Admin: Operations ───────────────────────────────────────────────
  runDoctor: () =>
    fetchJSON<ActionResponse>("/api/ops/doctor", { method: "POST" }),
  runSecurityAudit: () =>
    fetchJSON<ActionResponse>("/api/ops/security-audit", { method: "POST" }),
  runBackup: (output?: string) =>
    fetchJSON<ActionResponse>("/api/ops/backup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output }),
    }),
  runImport: (archive: string, force = false) =>
    fetchJSON<ActionResponse>("/api/ops/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ archive, force }),
    }),
  getHooks: () => fetchJSON<HooksResponse>("/api/ops/hooks"),
  createHook: (body: HookCreate) =>
    fetchJSON<{ ok: boolean; event: string; command: string; approved: boolean }>(
      "/api/ops/hooks",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  deleteHook: (event: string, command: string) =>
    fetchJSON<{ ok: boolean }>("/api/ops/hooks", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, command }),
    }),
  getSystemStats: () => fetchJSON<SystemStats>("/api/system/stats"),

  // ── Admin: Curator ──────────────────────────────────────────────────
  getCurator: () => fetchJSON<CuratorStatus>("/api/curator"),
  setCuratorPaused: (paused: boolean) =>
    fetchJSON<{ ok: boolean; paused: boolean }>("/api/curator/paused", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paused }),
    }),
  runCurator: () =>
    fetchJSON<ActionResponse>("/api/curator/run", { method: "POST" }),

  // ── Admin: Portal ───────────────────────────────────────────────────
  getPortal: () => fetchJSON<PortalStatus>("/api/portal"),

  // ── Admin: Diagnostics (backgrounded) ───────────────────────────────
  runPromptSize: () =>
    fetchJSON<ActionResponse>("/api/ops/prompt-size", { method: "POST" }),
  runDump: () => fetchJSON<ActionResponse>("/api/ops/dump", { method: "POST" }),
  runConfigMigrate: () =>
    fetchJSON<ActionResponse>("/api/ops/config-migrate", { method: "POST" }),
  runDebugShare: (opts?: { redact?: boolean; lines?: number }) =>
    fetchJSON<DebugShareResponse>("/api/ops/debug-share", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        redact: opts?.redact ?? true,
        lines: opts?.lines ?? 200,
      }),
    }),


  getCheckpoints: () => fetchJSON<CheckpointsResponse>("/api/ops/checkpoints"),
  pruneCheckpoints: () =>
    fetchJSON<ActionResponse>("/api/ops/checkpoints/prune", { method: "POST" }),

  // ── Admin: Skills hub ───────────────────────────────────────────────
  // ``profile`` scopes install/uninstall/update and the installed-state
  // annotations to that profile (omitted = the dashboard's own profile).
  installSkillFromHub: (identifier: string, profile?: string) =>
    fetchJSON<ActionResponse>("/api/skills/hub/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, profile: profile || undefined }),
    }),
  uninstallSkillFromHub: (name: string, profile?: string) =>
    fetchJSON<ActionResponse>("/api/skills/hub/uninstall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, profile: profile || undefined }),
    }),
  updateSkillsFromHub: (profile?: string) =>
    fetchJSON<ActionResponse>("/api/skills/hub/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: profile || undefined }),
    }),
  searchSkillsHub: (q: string, source = "all", limit = 20, profile?: string) =>
    fetchJSON<SkillHubSearchResponse>(
      `/api/skills/hub/search?q=${encodeURIComponent(q)}&source=${encodeURIComponent(source)}&limit=${limit}${profile ? `&profile=${encodeURIComponent(profile)}` : ""}`,
    ),
  getSkillHubSources: (profile?: string) =>
    fetchJSON<SkillHubSourcesResponse>(
      `/api/skills/hub/sources${profileQuery(profile)}`,
    ),
  previewSkillFromHub: (identifier: string) =>
    fetchJSON<SkillHubPreview>(
      `/api/skills/hub/preview?identifier=${encodeURIComponent(identifier)}`,
    ),
  scanSkillFromHub: (identifier: string) =>
    fetchJSON<SkillHubScan>(
      `/api/skills/hub/scan?identifier=${encodeURIComponent(identifier)}`,
    ),
};

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
  name: string;
  ok: boolean;
  pid: number | null;
  error?: string;
  message?: string;
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
  auth: string | null;
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
  needs_install: boolean;
  installed: boolean;
  enabled: boolean;
}

export interface McpCatalogDiagnostic {
  name: string;
  kind: string;
  message: string;
}


export interface McpServerCreate {
  name: string;
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  auth?: string;
}

export interface McpTestResult {
  ok: boolean;
  error?: string;
  tools: Array<{ name: string; description: string }>;
}

export interface MessagingPlatformEnvVar {
  key: string;
  required: boolean;
  is_set: boolean;
  redacted_value: string | null;
  description: string;
  prompt: string;
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
   * "gateway_stopped" | "disconnected" | "fatal" | string
   */
  state: string;
  error_code: string | null;
  error_message: string | null;
  updated_at: string | null;
  home_channel: { platform: string; chat_id: string; name: string; thread_id?: string } | null;
  env_vars: MessagingPlatformEnvVar[];
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
  code?: string;
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
  configured: boolean;
}

export interface MemoryStatus {
  active: string;
  providers: MemoryProviderInfo[];
  builtin_files: { memory: number; user: number };
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

export interface PortalReadinessScope {
  selected_paid_routes: "pass" | "blocked";
  selected_tool_gateway_routes?: "pass" | "blocked";
  official_one_shot_bundle: "pass" | "partial";
  portal_setup_bundle?: "pass" | "partial";
  official_term_source?: string;
  model_provider_nous: boolean;
  browser_via_portal: boolean;
  official_one_shot_bundle_reasons: string[];
  portal_setup_bundle_reasons?: string[];
}

export interface McpKeepaliveRow {
  server: string;
  configuration_status: "enabled" | "disabled" | "unknown";
  recent_keepalive_warning_count: number;
  recent_registered_tools: number | null;
  keepalive_status: "partial" | "pending-smoke" | "blocked-safe" | "unknown";
  smoke_status: "pending";
  claim_not_allowed: string;
  next_action: string;
}

export interface McpLocalPreflightRow {
  server: string;
  preflight_status: "pass" | "partial" | "blocked";
  evidence_scope: string;
  detail: string;
  claim_not_allowed: string;
}

export interface PortalStatus {
  logged_in: boolean;
  portal_url: string | null;
  inference_url: string | null;
  provider: string;
  subscription_url: string;
  features: PortalFeature[];
  readiness_scope?: PortalReadinessScope;
  mcp_keepalive_matrix?: McpKeepaliveRow[];
  mcp_local_preflight_matrix?: McpLocalPreflightRow[];
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
interface FetchJSONOptions {
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
  release_date: string;
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
  /** True when this var is a messaging-platform credential owned by the Channels page. */
  channel_managed?: boolean;
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
  schedule?: { kind?: string; expr?: string; display?: string };
  schedule_display?: string | null;
  enabled: boolean;
  state?: string | null;
  deliver?: string | null;
  last_run_at?: string | null;
  next_run_at?: string | null;
  last_error?: string | null;
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
