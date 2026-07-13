import { buildHermesWebSocketUrl } from "@hermes/shared";

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
// pairing, cron (which has its own per-job profile params), profiles
// themselves — is machine-global or self-scoped and must NOT be rewritten.
const PROFILE_SCOPED_PREFIXES = [
  "/api/status",
  "/api/gateway",
  "/api/analytics",
  "/api/skills",
  "/api/tools/toolsets",
  "/api/config",
  "/api/env",
  "/api/mcp",
  "/api/messaging/platforms",
  "/api/messaging/telegram/onboarding",
  "/api/messaging/whatsapp/onboarding",
  "/api/model/info",
  "/api/model/set",
  "/api/model/auxiliary",
  "/api/model/moa",
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
  return buildHermesWebSocketUrl({
    authParam: await buildWsAuthParam(),
    basePath: BASE,
    params,
    path,
  });
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
  buildWsUrl,
  getStatus: () => fetchJSON<StatusResponse>("/api/status"),
  getTeamProposals: () => fetchJSON<TeamProposalsResponse>("/api/team-proposals"),
  getKanbanBoard: (board: string, includeArchived = false) =>
    fetchJSON<KanbanBoardResponse>(
      `/api/plugins/kanban/board?board=${encodeURIComponent(board)}&include_archived=${String(includeArchived)}`,
    ),
  getKanbanTask: (id: string, board = "co2farm-chief") =>
    fetchJSON<KanbanTaskDetailResponse>(`/api/plugins/kanban/tasks/${encodeURIComponent(id)}?board=${encodeURIComponent(board)}`),
  commentKanbanTask: (id: string, body: string, board: string, author = "command-center") =>
    fetchJSON<{ ok: boolean }>(`/api/plugins/kanban/tasks/${encodeURIComponent(id)}/comments?board=${encodeURIComponent(board)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body, author }),
    }),
  updateKanbanTask: (id: string, payload: { status?: string; block_reason?: string; title?: string; body?: string }, board: string) =>
    fetchJSON<{ task: KanbanTask | null }>(`/api/plugins/kanban/tasks/${encodeURIComponent(id)}?board=${encodeURIComponent(board)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getRadarHermes: () => fetchJSON<RadarHermesSnapshot>("/api/team-proposals/radar-hermes"),
  upsertTeamProposal: (proposal: TeamProposalUpsertRequest) =>
    fetchJSON<TeamProposalUpsertResponse>("/api/team-proposals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(proposal),
    }),
  updateTeamProposal: (id: string, proposal: TeamProposalUpdateRequest) =>
    fetchJSON<TeamProposalUpsertResponse>(`/api/team-proposals/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(proposal),
    }),
  submitSubagentProposal: (proposal: TeamProposalUpsertRequest) =>
    fetchJSON<TeamProposalUpsertResponse>("/api/team-proposals/subagent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(proposal),
    }),
  reviewTeamProposalAsChief: (id: string, action: "shortlist" | "defer" | "reject", note?: string) =>
    fetchJSON<TeamProposalReviewResponse>(
      `/api/team-proposals/${encodeURIComponent(id)}/chief-review`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, note }),
      },
    ),
  collectKanbanBlockerProposals: () =>
    fetchJSON<TeamProposalCollectorResponse>("/api/team-proposals/collectors/kanban-blockers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    }),
  generateAutonomousTeamProposals: (limit = 5, kind: "operative" | "evolution" = "evolution") =>
    fetchJSON<TeamProposalAutonomousGenerateResponse>("/api/team-proposals/autonomous/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit, kind }),
    }),
  getTeamProposalTaskPreview: (id: string) =>
    fetchJSON<TeamProposalTaskPreviewResponse>(
      `/api/team-proposals/${encodeURIComponent(id)}/task-preview`,
    ),
  getTeamProposalPlanPreview: (id: string) =>
    fetchJSON<TeamProposalPlanPreviewResponse>(
      `/api/team-proposals/${encodeURIComponent(id)}/plan-preview`,
    ),
  approveTeamProposalMinStep: (
    id: string,
    approval: TeamProposalApproveMinStepRequest,
  ) =>
    fetchJSON<TeamProposalApproveMinStepResponse>(
      `/api/team-proposals/${encodeURIComponent(id)}/approve-min-step`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(approval),
      },
    ),
  convertTeamProposalToPlan: (id: string, confirmed_preview_hash: string, board?: string) =>
    fetchJSON<TeamProposalPlanConvertResponse>(
      `/api/team-proposals/${encodeURIComponent(id)}/convert-to-plan`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board, confirmed_preview_hash }),
      },
    ),
  convertTeamProposalToTask: (id: string, confirmed_preview_hash: string, board?: string) =>
    fetchJSON<TeamProposalConvertResponse>(
      `/api/team-proposals/${encodeURIComponent(id)}/convert-to-task`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board, confirmed_preview_hash }),
      },
    ),
  setTeamProposalStatus: (id: string, status: TeamProposalStatus) =>
    fetchJSON<TeamProposalStatusResponse>(
      `/api/team-proposals/${encodeURIComponent(id)}/status`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      },
    ),
  getRemoteBrowserStatus: () => fetchJSON<RemoteBrowserStatus>("/api/remote-browser/status"),
  openRemoteBrowser: (url: string) =>
    fetchJSON<RemoteBrowserActionResponse>("/api/remote-browser/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }),
  getRemoteBrowserScreenshot: () =>
    fetchJSON<RemoteBrowserScreenshotResponse>("/api/remote-browser/screenshot"),
  clickRemoteBrowser: (x: number, y: number) =>
    fetchJSON<RemoteBrowserActionResponse>("/api/remote-browser/click", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: Math.round(x), y: Math.round(y) }),
    }),
  typeRemoteBrowser: (text: string) =>
    fetchJSON<RemoteBrowserActionResponse>("/api/remote-browser/type", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }),
  pressRemoteBrowserKey: (key: string) =>
    fetchJSON<RemoteBrowserActionResponse>("/api/remote-browser/key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    }),
  scrollRemoteBrowser: (dy: number) =>
    fetchJSON<RemoteBrowserActionResponse>("/api/remote-browser/scroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dy: Math.round(dy) }),
    }),
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
  getSessions: (
    limit = 20,
    offset = 0,
    profile = getManagementProfile(),
    order: "created" | "recent" = "created",
  ) =>
    fetchJSON<PaginatedSessions>(
      appendProfileParam(
        `/api/sessions?limit=${limit}&offset=${offset}&order=${order}`,
        profile,
      ),
    ),
  getSessionMessages: (id: string, profile = getManagementProfile()) =>
    fetchJSON<SessionMessagesResponse>(
      appendProfileParam(`/api/sessions/${encodeURIComponent(id)}/messages`, profile),
    ),
  getSessionDetail: (id: string, profile = getManagementProfile()) =>
    fetchJSON<SessionInfo>(
      appendProfileParam(`/api/sessions/${encodeURIComponent(id)}`, profile),
    ),
  getSessionLatestDescendant: (id: string, profile = getManagementProfile()) =>
    fetchJSON<SessionLatestDescendantResponse>(
      appendProfileParam(
        `/api/sessions/${encodeURIComponent(id)}/latest-descendant`,
        profile,
      ),
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
  importSessions: (
    sessions: Array<Record<string, unknown>>,
    profile = getManagementProfile(),
  ) =>
    fetchJSON<SessionImportResponse>("/api/sessions/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessions, profile: profile || undefined }),
    }),
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
  uploadFile: (path: string, file: File, overwrite = true) => {
    // Stream the raw bytes as multipart/form-data. Do NOT set Content-Type —
    // the browser adds the multipart boundary automatically. Sending the file
    // as base64 JSON (the old path) inflated the body ~33%, buffered the whole
    // file in memory, and 502'd on large backup archives behind the proxy
    // (NS-501).
    const form = new FormData();
    form.append("path", path);
    form.append("overwrite", String(overwrite));
    form.append("file", file, file.name);
    return fetchJSON<ManagedFileWriteResponse>("/api/files/upload-stream", {
      method: "POST",
      body: form,
    });
  },
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
  getConfig: (profile = getManagementProfile()) =>
    fetchJSON<Record<string, unknown>>(appendProfileParam("/api/config", profile)),
  getDefaults: () => fetchJSON<Record<string, unknown>>("/api/config/defaults"),
  getSchema: () => fetchJSON<{ fields: Record<string, unknown>; category_order: string[] }>("/api/config/schema"),
  getModelInfo: (profile = getManagementProfile()) =>
    fetchJSON<ModelInfoResponse>(appendProfileParam("/api/model/info", profile)),
  getModelOptions: (
    profileOrOptions?: string | { profile?: string; refresh?: boolean },
  ) => {
    const profile =
      typeof profileOrOptions === "string"
        ? profileOrOptions
        : profileOrOptions?.profile;
    const refresh =
      typeof profileOrOptions === "object" && !!profileOrOptions.refresh;
    const qs = new URLSearchParams();
    if (profile) qs.set("profile", profile);
    if (refresh) qs.set("refresh", "1");
    // Dashboard surfaces (Models page, profile builder, cron) are
    // management/setup UIs: keep the full provider universe with setup
    // affordances. The endpoint now defaults to the configured subset for
    // desktop chat pickers (#56974), so opt in explicitly here.
    qs.set("include_unconfigured", "1");
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return fetchJSON<ModelOptionsResponse>(`/api/model/options${suffix}`);
  },
  getAuxiliaryModels: (profile = getManagementProfile()) =>
    fetchJSON<AuxiliaryModelsResponse>(
      appendProfileParam("/api/model/auxiliary", profile),
    ),
  getMoaModels: () => fetchJSON<MoaConfigResponse>("/api/model/moa"),
  saveMoaModels: (body: MoaConfigResponse) =>
    fetchJSON<MoaConfigResponse & { ok: boolean }>("/api/model/moa", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  setModelAssignment: (
    body: ModelAssignmentRequest,
    profile = getManagementProfile(),
  ) =>
    fetchJSON<ModelAssignmentResponse>(
      appendProfileParam("/api/model/set", profile),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  saveConfig: (config: Record<string, unknown>, profile = getManagementProfile()) =>
    fetchJSON<{ ok: boolean }>(appendProfileParam("/api/config", profile), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    }),
  getConfigRaw: (profile = getManagementProfile()) =>
    fetchJSON<{ yaml: string; path?: string }>(
      appendProfileParam("/api/config/raw", profile),
    ),
  saveConfigRaw: (yaml_text: string, profile = getManagementProfile()) =>
    fetchJSON<{ ok: boolean }>(appendProfileParam("/api/config/raw", profile), {
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
  revealEnvVar: (key: string) =>
    fetchJSON<{ key: string; value: string }>("/api/env/reveal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    }),

  // Cron jobs
  getCronJobs: (profile = "all") =>
    fetchJSON<CronJob[]>(`/api/cron/jobs?profile=${encodeURIComponent(profile)}`),
  getCronDeliveryTargets: () =>
    fetchJSON<{ targets: CronDeliveryTarget[] }>("/api/cron/delivery-targets"),
  createCronJob: (job: CronJobMutation, profile = "default") =>
    fetchJSON<CronJob>(`/api/cron/jobs?profile=${encodeURIComponent(profile)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(job),
    }),
  pauseCronJob: (id: string, profile = "default") =>
    fetchJSON<CronJob>(`/api/cron/jobs/${encodeURIComponent(id)}/pause?profile=${encodeURIComponent(profile)}`, { method: "POST" }),
  updateCronJob: (
    id: string,
    updates: CronJobMutation,
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
    fetchJSON<{ ok: boolean; name: string; platform: string; enabled: boolean }>(
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
  disconnectOAuthProvider: (providerId: string) =>
    fetchJSON<{ ok: boolean; provider: string }>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}`,
      {
        method: "DELETE",
      },
    ),
  startOAuthLogin: (providerId: string) =>
    fetchJSON<OAuthStartResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/start`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      },
    ),
  submitOAuthCode: (providerId: string, sessionId: string, code: string) =>
    fetchJSON<OAuthSubmitResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/submit`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, code }),
      },
    ),
  pollOAuthSession: (providerId: string, sessionId: string) =>
    fetchJSON<OAuthPollResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/poll/${encodeURIComponent(sessionId)}`,
    ),
  cancelOAuthSession: (sessionId: string) =>
    fetchJSON<{ ok: boolean }>(
      `/api/providers/oauth/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: "DELETE",
      },
    ),

  // Messaging platforms (gateway channels)
  getMessagingPlatforms: () =>
    fetchJSON<MessagingPlatformsResponse>("/api/messaging/platforms"),
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
    body: { allowed_user_ids: string[]; profile?: string },
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
  startWhatsAppOnboarding: (body: {
    mode?: "bot" | "self-chat";
    allowed_users?: string;
  }) =>
    fetchJSON<WhatsAppOnboardingStartResponse>(
      "/api/messaging/whatsapp/onboarding/start",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  getWhatsAppOnboardingStatus: (pairingId: string) =>
    fetchJSON<WhatsAppOnboardingStatusResponse>(
      `/api/messaging/whatsapp/onboarding/${encodeURIComponent(pairingId)}`,
    ),
  applyWhatsAppOnboarding: (
    pairingId: string,
    body: { mode?: "bot" | "self-chat"; allowed_users?: string; profile?: string },
  ) =>
    fetchJSON<WhatsAppOnboardingApplyResponse>(
      `/api/messaging/whatsapp/onboarding/${encodeURIComponent(pairingId)}/apply`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  cancelWhatsAppOnboarding: (pairingId: string) =>
    fetchJSON<{ ok: boolean }>(
      `/api/messaging/whatsapp/onboarding/${encodeURIComponent(pairingId)}`,
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
  authMcpServer: (name: string) =>
    fetchJSON<McpOAuthFlow>(
      `/api/mcp/servers/${encodeURIComponent(name)}/auth`,
      { method: "POST" },
    ),
  getMcpOAuthFlow: (flowId: string) =>
    fetchJSON<McpOAuthFlow>(
      `/api/mcp/oauth/flows/${encodeURIComponent(flowId)}`,
    ),
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
  getMemoryProviderConfig: (provider: string) =>
    fetchJSON<MemoryProviderConfig>(
      `/api/memory/providers/${encodeURIComponent(provider)}/config`,
    ),
  updateMemoryProviderConfig: (provider: string, values: Record<string, unknown>) =>
    fetchJSON<{ ok: boolean; active: string }>(
      `/api/memory/providers/${encodeURIComponent(provider)}/config`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values }),
      },
    ),
  setupMemoryProvider: (provider: string, values: Record<string, unknown> = {}) =>
    fetchJSON<MemoryProviderSetupResponse>(
      `/api/memory/providers/${encodeURIComponent(provider)}/setup`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values }),
      },
    ),
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
  downloadBackup: (archive: string) =>
    authedFetch(
      `/api/ops/backup/download?archive=${encodeURIComponent(archive)}`,
    ),
  runImport: (archive: string, force = false) =>
    fetchJSON<ActionResponse>("/api/ops/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ archive, force }),
    }),
  runImportUpload: (file: File, force = false) => {
    const form = new FormData();
    form.append("force", String(force));
    form.append("file", file, file.name);
    return fetchJSON<ActionResponse>("/api/ops/import-upload", {
      method: "POST",
      body: form,
    });
  },
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
  archive?: string;
  name: string;
  ok: boolean;
  pid: number | null;
  error?: string;
  message?: string;
  uploaded_bytes?: number;
  update_command?: string;
}

export type TeamProposalStatus =
  | "proposta"
  | "raccomandata"
  | "approvata"
  | "standby"
  | "parcheggiata"
  | "scartata"
  | "trasformata_in_task"
  | "signal_detected"
  | "interpreting"
  | "challenging"
  | "synthesized"
  | "needs_reliability_check"
  | "ready_for_gate"
  | "approved_min_step"
  | "blocked_by_daniele"
  | "parked"
  | "rejected"
  | "converted_to_kanban"
  | "archived";


export interface KanbanTask {
  id: string;
  title: string;
  body?: string | null;
  status: string;
  assignee?: string | null;
  priority?: number | null;
  tenant?: string | null;
  created_at?: number | null;
  started_at?: number | null;
  completed_at?: number | null;
  latest_summary?: string | null;
  comment_count?: number;
  link_counts?: { parents: number; children: number };
  progress?: { done: number; total: number } | null;
}

export interface KanbanBoardResponse {
  columns: Array<{ name: string; tasks: KanbanTask[] }>;
  tenants: string[];
  assignees: string[];
  latest_event_id: number;
  now: number;
}

export interface KanbanTaskDetailResponse {
  task: {
    id: string;
    title: string;
    body?: string | null;
    assignee?: string | null;
    status: string;
    priority?: number | null;
    tenant?: string | null;
    workspace_path?: string | null;
    latest_summary?: string | null;
    result?: string | null;
    created_at?: number | null;
    started_at?: number | null;
    completed_at?: number | null;
  };
  comments: Array<{ id: number; task_id: string; author: string; body: string; created_at: number }>;
  events: Array<{ id: number; task_id: string; kind: string; payload?: unknown; created_at: number; run_id?: number | null }>;
  runs: Array<{ id: number; task_id: string; profile: string; status: string; outcome?: string | null; summary?: string | null; error?: string | null; started_at?: number | null; ended_at?: number | null }>;
  attachments?: Array<{ id: number; filename: string; stored_path?: string; created_at: number }>;
}

export type RadarHermesLevel = "high" | "medium" | "low";
export type RadarHermesApprovalState =
  | "candidate"
  | "needs_review"
  | "approved_for_spec"
  | "approved_for_kanban"
  | "parked"
  | "rejected"
  | "done";

export interface RadarHermesProposal {
  id: string;
  source: {
    kind: string;
    source_id: string;
    path: string;
    source_key?: string | null;
    excerpt: string;
  };
  title: string;
  rationale: string;
  evidence: Array<{ type: string; ref: string; summary: string; confidence: RadarHermesLevel }>;
  priority: {
    label: "P0" | "P1" | "P2" | "P3";
    score: number;
    impact: RadarHermesLevel;
    effort: RadarHermesLevel;
    risk: RadarHermesLevel;
    confidence: RadarHermesLevel;
  };
  flags: {
    controversial: boolean;
    parkable: boolean;
    source_gap: boolean;
    already_done: boolean;
    requires_review: boolean;
    no_auto_dispatch: boolean;
  };
  suggested_assignee: string;
  approval_state: RadarHermesApprovalState;
  approval: {
    allowed_actions: string[];
    preview_available: boolean;
    kanban_creation_available: boolean;
    requires_explicit_confirmation: boolean;
  };
  ranking: {
    block: "top" | "controversial" | "parkable";
    rank: number;
    score: number;
    score_breakdown: Record<string, number>;
  };
  timestamps: {
    created_at?: string | null;
    updated_at?: string | null;
    last_signal_at?: string | null;
    source_observed_at: string;
  };
  governance: {
    read_only_surface: boolean;
    no_cron_created: boolean;
    no_external_send: boolean;
    no_subagent_spawned: boolean;
    kanban_mutation_requires_approval: boolean;
  };
}

export interface RadarHermesSnapshot {
  version: "radar_hermes.v1";
  generated_at: string;
  source_summary: {
    sources_read: string[];
    proposals_seen: number;
    proposals_returned: number;
    side_effects: {
      kanban_mutated: boolean;
      cron_created: boolean;
      external_send: boolean;
      subagent_spawned: boolean;
    };
  };
  blocks: {
    top: RadarHermesProposal[];
    controversial: RadarHermesProposal | null;
    parkable: RadarHermesProposal | null;
  };
  controversy_state: {
    status: "has_controversy" | "insufficient_controversy";
    title: string;
    message: string;
  };
  approval_policy: {
    read_only_first: boolean;
    requires_explicit_approval_for_kanban: boolean;
    preview_does_not_dispatch: boolean;
    create_children_does_not_force_dispatch: boolean;
  };
  empty_state?: { title: string; message: string } | null;
}

export type AgentGrowthFieldState = "present" | "missing" | "unknown" | "redacted";
export type AgentGrowthConfidence = "high" | "medium" | "low";
export type AgentGrowthReadinessBand = "emerging" | "operational" | "trusted_for_preview" | "insufficient_data";

export interface AgentGrowthProvenanceRef {
  kind: string;
  source_id?: string;
  path?: string;
  proposal_id?: string;
  task_id?: string;
  run_id?: number;
  field_path?: string;
  observed_at?: string;
}

export interface AgentGrowthField<T = unknown> {
  state: AgentGrowthFieldState;
  value?: T;
  display: string;
  missing_reason?: string;
  confidence: AgentGrowthConfidence;
  provenance: AgentGrowthProvenanceRef[];
}

export interface AgentGrowthProfile {
  schema_version: "agent_growth.v1";
  generated_at: string;
  agent: {
    agent_id: string;
    display_name: string;
    logical_role?: string | null;
    aliases: string[];
    source_ids: string[];
    mapping_confidence: AgentGrowthConfidence;
    mapping_notes: string[];
    provenance: AgentGrowthProvenanceRef[];
  };
  role_foundation: AgentGrowthField;
  last_observed_signal: AgentGrowthField;
  own_proposal: AgentGrowthField;
  challenges_received: AgentGrowthField<Array<{ critic: string; challenge: string; status: string; veto_risk?: string }>>;
  learning_notes: AgentGrowthField<Array<{ note: string; trigger: string; tone: string; linked_quality_check?: string }>>;
  next_role_development: AgentGrowthField<{ recommendation: string; micro_capability: string; suggested_exercise: string; approval_gate: string; forbidden_actions: string[] }>;
  scoring: {
    state: "computed" | "not_computed";
    growth_score?: number | null;
    readiness_band: AgentGrowthReadinessBand;
    confidence: AgentGrowthConfidence;
    components: Array<{ key: string; points: number; max_points: number; state: string; reason_code: string; display: string; provenance: AgentGrowthProvenanceRef[] }>;
    explainers: string[];
    non_punitive_notes: string[];
    provenance: AgentGrowthProvenanceRef[];
  };
  privacy: {
    pii_detected: boolean;
    redactions_applied: string[];
    max_raw_chars_per_evidence: number;
    raw_logs_included: false;
    secrets_included: false;
  };
  invariants: {
    read_only: true;
    approval_gated: true;
    no_cron_created: true;
    no_external_send: true;
    no_auto_spawn: true;
    no_auto_dispatch: true;
    no_leaderboard: true;
  };
  growth_score: number | null;
  readiness: "high" | "medium" | "low";
  strengths: string[];
  learning_note_summaries?: string[];
  challenge_notes: string[];
  needs_context: string[];
  latest_proposal?: string | null;
  next_growth_step: string;
}

export interface TeamSpecialist {
  id: string;
  name: string;
  logical_role?: string | null;
  legacy_id?: string | null;
  aliases?: string[];
  profile_verified?: boolean;
  mapping_status?: string;
  mapping_reason?: string;
  status: "active" | "watching" | "dormant";
  mission: string;
  observes: string;
  currentSignal: string;
  nextProposal: string;
  confidence: "high" | "medium" | "low";
  metrics?: {
    proposal_count: number;
    approved_count: number;
    rejected_count: number;
    transformed_count: number;
    trust_score: number;
  };
  growth_profile?: AgentGrowthProfile;
}

export interface TeamProposalViewpoint {
  actor: string;
  rationale: string;
  method?: "rules_based_v1" | "rules_based_v2" | string;
  role?: "supporter" | "critic" | string;
  profile?: string;
  agent?: string;
  case_for?: string;
  case_against?: string;
  failure_mode?: string;
  mitigation?: string;
}

export interface TeamProposalGate {
  requires_daniele: boolean;
  decision_needed: string;
  safe_actions_without_approval: string[];
  forbidden_without_approval: string[];
  approved_by?: string | null;
  approved_at?: string | null;
  state?: string;
  no_auto_dispatch?: boolean;
}

export interface TeamProposalSuggestedProfile {
  profile: string;
  exists_verified: boolean;
  reason: string;
  role: string;
  confidence: "low" | "medium" | "high" | string;
}

export interface TeamProposalSignal {
  summary: string;
  source_type?: string;
  source_ref?: string;
  observed_at?: string;
}

export interface TeamProposalInterpretation {
  hypothesis: string;
  expected_benefit?: string;
  effort?: string;
  risk?: string;
}

export interface TeamProposalChiefSynthesis {
  recommendation?: "do_now" | "prepare" | "park" | "reject" | string;
  synthesis: string;
  decision_needed?: string;
  acceptance?: string;
  unresolved_questions?: string[];
}

export interface TeamProposalEvidenceContract {
  refs: string[];
  excerpts?: string[];
  confidence?: string;
}

export interface TeamProposalConversion {
  plan_task_id?: string | null;
  child_task_ids?: string[];
  created_by?: string | null;
  board?: string | null;
  initial_status?: "blocked" | "ready" | string | null;
  converted_at?: string | null;
}

export interface TeamProposal {
  id: string;
  title: string;
  kind: "operative" | "evolution";
  origin: string;
  category?: string | null;
  priority: "P0" | "P1" | "P2" | "P3";
  benefit: "high" | "medium" | "low";
  effort: "low" | "medium" | "high";
  risk: "low" | "medium" | "high";
  confidence: "high" | "medium" | "low";
  status: TeamProposalStatus;
  whyNow: string;
  evidence?: string | null;
  acceptance: string;
  recommendation: "do_now" | "prepare" | "park" | "reject";
  status_updated_at?: string;
  task_id?: string;
  plan_task_id?: string;
  plan_child_task_ids?: string[];
  transformed_at?: string;
  kanban_plan_status?: "active" | "closed" | string;
  kanban_task_statuses?: Record<string, string>;
  kanban_active_task_ids?: string[];
  kanban_synced_at?: string;
  kanban_closed_at?: string;
  auto_refill?: boolean;
  auto_refill_reason?: string;
  source_key?: string | null;
  source_agent?: string | null;
  source_agent_legacy?: string | null;
  source_agent_status?: "verified_profile" | "legacy_missing" | "normalized_from_legacy_persona" | string;
  suggested_next_action?: string | null;
  dedupe_key?: string;
  signals_count?: number;
  chief_review_score?: number;
  chief_review_status?: "pending" | "shortlisted" | "deferred" | "rejected";
  chief_review_reason?: string;
  chief_reviewed_at?: string;
  canonical_status?: TeamProposalStatus;
  approval_required_before_actions?: boolean;
  gate_status?: TeamProposalGateStatus;
  source_agent_details?: TeamProposalSourceAgentDetails;
  audit_log?: TeamProposalAuditEventV3[];
  formulated_at?: string;
  created_at?: string;
  updated_at?: string;
  last_signal_at?: string;
  autonomy_level?: string;
  autonomy_gate?: string;
  record_type?: "autonomous_proposal_candidate" | string;
  schema_version?: "autonomous_proposal.v1" | "autonomous_proposal.v2" | "autonomous_proposal.v3" | string;
  signal?: string | TeamProposalSignal;
  source_signal?: string;
  interpretation?: string | TeamProposalInterpretation;
  supporter?: TeamProposalViewpoint;
  critic?: TeamProposalViewpoint;
  supporter_view?: TeamProposalViewpoint;
  critic_view?: TeamProposalViewpoint;
  chief_synthesis?: string | TeamProposalChiefSynthesis;
  gate_state?: "review_required" | "needs_revision" | "parked" | string;
  gate?: TeamProposalGate;
  suggested_profiles?: TeamProposalSuggestedProfile[];
  evidence_refs?: string[];
  evidence_contract?: TeamProposalEvidenceContract;
  conversion?: TeamProposalConversion;
  engine?: {
    name: string;
    version: string;
    method: "rules_based_v1" | "rules_based_v2" | string;
    generated_at?: string;
    limits?: string[];
  };
  external_send?: boolean;
  auto_spawned?: boolean;
  cron_created?: boolean;
  no_auto_dispatch?: boolean;
  challenge?: {
    supporter: string;
    critic: string;
    support: string;
    challenge: string;
    chief_synthesis: string;
    veto_risk?: string;
  };
}

export type TeamProposalApprovedActionType = "task" | "plan" | "spec";

export interface TeamProposalGateStatus {
  approval_required_before_actions: true;
  ready_for_daniele: boolean;
  status: "pending" | "approved" | "blocked" | "parked" | "rejected" | "converted" | string;
  approved_by?: string | null;
  approved_at?: string | null;
  approved_action?: {
    type: TeamProposalApprovedActionType;
    board?: string | null;
    max_tasks?: number;
    assignees?: Array<string | null | undefined>;
    initial_status?: "blocked";
  } | null;
  approved_preview_hash?: string | null;
  allowed_next_actions: string[];
  forbidden_without_approval: string[];
}

export interface TeamProposalSourceAgentDetails {
  profile?: string | null;
  profile_verified: boolean;
  logical_role?: string | null;
  legacy_persona?: string | null;
  mapping_status: "verified_profile" | "mapped_from_logical_role" | "missing_blocked" | string;
  mapping_reason?: string | null;
}

export interface TeamProposalAuditEventV3 {
  at: string;
  actor_type?: "system" | "profile" | "ui" | "api" | string;
  actor_id?: string;
  event: string;
  detail?: string | null;
  before_status?: string | null;
  after_status?: string | null;
  side_effects?: {
    kanban_created?: boolean;
    tasks_created_count?: number;
    initial_status?: string | null;
    dispatch_started?: boolean;
    cron_created?: boolean;
    external_send?: boolean;
  };
  request_id?: string | null;
  idempotency_key?: string | null;
}

export interface TeamProposalUpsertRequest {
  title: string;
  kind: "operative" | "evolution";
  origin?: string;
  category?: string;
  whyNow?: string;
  evidence?: string;
  benefit?: "high" | "medium" | "low";
  effort?: "high" | "medium" | "low";
  risk?: "high" | "medium" | "low";
  priority?: "P0" | "P1" | "P2" | "P3";
  confidence?: "high" | "medium" | "low";
  recommendation?: "do_now" | "prepare" | "park" | "reject";
  acceptance?: string;
  source_key?: string;
  source_agent?: string;
  suggested_next_action?: string;
  signal?: string | Record<string, unknown>;
  source_signal?: string | Record<string, unknown>;
  interpretation?: string | Record<string, unknown>;
  supporter?: TeamProposalViewpoint;
  critic?: TeamProposalViewpoint;
  supporter_view?: TeamProposalViewpoint;
  critic_view?: TeamProposalViewpoint;
  chief_synthesis?: string;
  gate?: Partial<TeamProposalGate>;
  gate_state?: string;
  suggested_profiles?: TeamProposalSuggestedProfile[];
  evidence_refs?: string[];
  canonical_status?: TeamProposalStatus;
  gate_status?: TeamProposalGateStatus;
  source_agent_details?: TeamProposalSourceAgentDetails;
  approval_required_before_actions?: boolean;
}

export type TeamProposalUpdateRequest = Partial<Pick<TeamProposalUpsertRequest,
  | "title"
  | "whyNow"
  | "evidence"
  | "benefit"
  | "effort"
  | "risk"
  | "priority"
  | "confidence"
  | "recommendation"
  | "acceptance"
  | "suggested_next_action"
  | "signal"
  | "source_signal"
  | "interpretation"
  | "supporter"
  | "critic"
  | "supporter_view"
  | "critic_view"
  | "chief_synthesis"
  | "gate"
  | "gate_state"
  | "suggested_profiles"
  | "evidence_refs"
  | "canonical_status"
  | "gate_status"
  | "source_agent_details"
  | "approval_required_before_actions"
>>;

export interface TeamProposalTaskPreview {
  title: string;
  body: string;
  priority: number;
  assignee: string | null;
  tenant: string;
  workspace_kind: string;
  idempotency_key: string;
  initial_status: "ready" | "blocked";
}

export interface TeamProposalPlanPreview {
  title: string;
  body: string;
  priority: number;
  tenant: string;
  workspace_kind: string;
  idempotency_key: string;
  initial_status: "ready" | "blocked";
  assignee: string | null;
  assignment_strategy?: string;
  available_profiles?: string[];
  tasks: Array<{
    title: string;
    body: string;
    priority: number;
    tenant: string;
    workspace_kind: string;
    idempotency_key: string;
    initial_status: "ready" | "blocked";
    assignee: string | null;
    assignment_reason?: string;
    candidate_profiles?: string[];
  }>;
}

export interface TeamProposalsSource {
  kind: "dashboard_registry" | string;
  label: string;
  profile: string;
  tenant: string;
  freshness: "fresh" | "stale" | "missing" | "error" | string;
}

export interface TeamProposalsSafety {
  mode: "proposal_review" | string;
  no_auto_dispatch: boolean;
  preview_read_only: boolean;
  conversion_requires_confirmation: boolean;
  conversion_initial_status: "ready" | "blocked" | string;
}

export interface TeamConstitutionContract {
  team: string;
  lead: string;
  mission: string;
  north_star: string;
  must_not: string[];
  cycle_start_reads: string[];
  prompt_sources: string[];
  handoff: string;
  mode: string;
}

export interface TeamProposalsResponse {
  version: number;
  updated_at?: string;
  source?: TeamProposalsSource;
  safety?: TeamProposalsSafety;
  team_constitutions?: {
    operative?: TeamConstitutionContract;
    evolution?: TeamConstitutionContract;
  };
  specialists: TeamSpecialist[];
  proposals: TeamProposal[];
  chief_review?: {
    queue: TeamProposal[];
    summary: string;
  };
  strategic_review?: TeamProposalStrategicReview;
  team_pulse?: TeamPulseSummary;
  team_pulses?: {
    operative: TeamPulseSummary;
    evolution: TeamPulseSummary;
  };
}

export interface TeamPulseSummary {
  summary: string;
  kind?: "operative" | "evolution" | null;
  last_run_at?: string | null;
  last_cycle?: TeamPulseCycle | null;
  visibility_state?: "fresh" | "not_recorded" | string;
  active_count: number;
  autonomous_count: number;
  challenged_count: number;
  controversial_count: number;
  top_autonomous: TeamProposal[];
  mature_proposals?: TeamProposal[];
  standby_proposals?: TeamProposal[];
  controversial: TeamProposal[];
  controversy_lane?: TeamPulseControversyLane;
  guardrails: {
    external_send: boolean;
    auto_spawned: boolean;
    cron_created: boolean;
    approval_required: boolean;
  };
}

export interface TeamPulseCycle {
  kind: "operative" | "evolution" | string;
  cycle_type: string;
  ran_at: string;
  source: string;
  active_count: number;
  mature_count: number;
  parked_count: number;
  created_count: number;
  updated_count: number;
  summary: string;
  guardrails: {
    no_auto_dispatch: boolean;
    external_send: boolean;
    auto_spawned: boolean;
    cron_created: boolean;
    kanban_mutated: boolean;
    approval_required: boolean;
  };
}

export type TeamPulseControversyStatus =
  | "has_controversy"
  | "no_meaningful_controversy"
  | "insufficient_review_data";

export type TeamPulseControversyRiskLevel =
  | "L0_none"
  | "L1_objection"
  | "L2_material_risk"
  | "L3_blocking_veto"
  | "L4_stop_escalate";

export interface TeamPulseSourceRef {
  kind: "kanban_task" | "run_metadata" | "comment" | "document" | "system_event" | "manual_entry";
  id: string;
  url?: string;
  excerpt?: string;
}

export interface TeamPulseControversyLane {
  cycle_id: string;
  generated_at: string;
  status: TeamPulseControversyStatus;
  selection_policy_version: "controversy_lane_v1" | string;
  items: TeamPulseControversialProposal[];
  empty_state?: {
    title: string;
    message: string;
    reviewed_proposal_count?: number;
    review_completeness?: "complete" | "partial" | "unknown";
  };
}

export interface TeamPulseControversialProposal {
  proposal_id: string;
  proposal_title: string;
  proposal_summary?: string | null;
  proposer: TeamPulseActorRef;
  critic: TeamPulseActorRef;
  additional_critics?: Array<TeamPulseActorRef & { rationale_summary?: string }>;
  contested: {
    type: "claim" | "decision" | "assumption" | "evidence" | "implementation" | "priority" | "other";
    text: string;
    source_quote?: string | null;
    source_ref?: TeamPulseSourceRef;
  };
  critic_rationale: {
    summary: string;
    details?: string | null;
    source_refs: TeamPulseSourceRef[];
  };
  evidence_gap: {
    summary: string;
    required_evidence?: string[];
    current_evidence_refs?: TeamPulseSourceRef[];
  };
  risk: {
    level: TeamPulseControversyRiskLevel;
    label: string;
    rationale: string;
    domains: Array<"legal" | "claims" | "reliability" | "mrv" | "finance" | "ops" | "market" | "other">;
    veto_owner?: string | null;
  };
  chief_synthesis: {
    summary: string;
    decision_state: "unresolved" | "accepted_with_caveats" | "rejected" | "needs_more_evidence" | "deferred";
    unresolved_decision?: string | null;
    rationale?: string | null;
  };
  recommended_action: {
    action: "include_in_shortlist" | "include_with_caveat" | "revise_before_shortlist" | "request_evidence" | "route_to_specialist" | "defer" | "reject";
    owner?: string | null;
    next_step: string;
    due_by_cycle?: string;
  };
  provenance: {
    created_from: TeamPulseSourceRef[];
    last_updated_at: string;
    confidence: "high" | "medium" | "low";
    notes?: string;
  };
}

export interface TeamPulseActorRef {
  subagent_id: string;
  subagent_label: string;
  source_task_id?: string | null;
  source_run_id?: string | null;
}

export interface TeamProposalStrategicReview {
  summary: string;
  top_operative: TeamProposal[];
  top_evolution: TeamProposal[];
  parked: TeamProposal[];
  counts: {
    active: number;
    operative: number;
    evolution: number;
    parked: number;
    rejected: number;
  };
}

export interface TeamProposalStatusResponse {
  ok: boolean;
  proposal: TeamProposal;
}

export interface TeamProposalUpsertResponse {
  ok: boolean;
  created: boolean;
  proposal: TeamProposal;
}

export interface TeamProposalCollectorResponse {
  ok: boolean;
  collector: string;
  board: string;
  blocked_count: number;
  created?: boolean;
  proposals: TeamProposal[];
}

export interface TeamProposalAutonomousGenerateResponse {
  ok: boolean;
  created: TeamProposal[];
  updated: TeamProposal[];
  created_count: number;
  updated_count: number;
  team_pulse: TeamPulseSummary;
}

export interface TeamProposalReviewResponse {
  ok: boolean;
  proposal: TeamProposal;
  chief_review: {
    queue: TeamProposal[];
  };
}

export interface TeamProposalTaskPreviewResponse {
  proposal_id: string;
  task: TeamProposalTaskPreview;
  preview_hash: string;
}

export interface TeamProposalPlanPreviewResponse {
  proposal_id: string;
  plan: TeamProposalPlanPreview;
  preview_hash: string;
}

export interface TeamProposalApproveMinStepRequest {
  action_type: TeamProposalApprovedActionType;
  confirmed_preview_hash: string;
  board?: string;
  approved_by?: string;
  note?: string;
}

export interface TeamProposalApproveMinStepResponse {
  ok: boolean;
  proposal: TeamProposal;
  approval: {
    action_type: TeamProposalApprovedActionType;
    preview_hash: string;
    side_effect_after_conversion: string;
  };
}

export interface TeamProposalConvertResponse {
  ok: boolean;
  proposal: TeamProposal;
  task: {
    id: string;
    title: string;
    status: string | null;
  };
}

export interface TeamProposalPlanConvertResponse {
  ok: boolean;
  proposal: TeamProposal;
  plan: {
    parent_task_id: string;
    child_task_ids: string[];
    title: string;
    parent_assignee?: string | null;
    child_assignees?: Array<string | null>;
  };
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
  kind: "text" | "secret" | "select" | "boolean";
  description: string;
  placeholder: string;
  required: boolean;
  value: string | boolean;
  is_set: boolean;
  options: MemoryProviderFieldOption[];
  url: string;
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
  /** Supported dashboard auth flows for the client to choose from. In gated
   * mode always includes ``"cookie"``; includes ``"native_pkce"`` when a
   * brokerable OAuth provider is registered, signalling that the desktop can
   * use the RFC 8252 system-browser + loopback + PKCE flow (no embedded
   * webview, no session cookies). Absent / missing ``"native_pkce"`` ⇒ an
   * older gateway ⇒ the desktop falls back to the embedded-webview flow. */
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
  release_date: string;
  version: string;
}

export interface RemoteBrowserStatus {
  session: string;
  available: boolean;
  connected: boolean;
  url: string;
  title: string;
  error?: string;
}

export interface RemoteBrowserActionResponse {
  ok: boolean;
  data: Record<string, unknown>;
  error: string;
  status?: RemoteBrowserStatus;
}

export interface RemoteBrowserScreenshotResponse {
  ok: boolean;
  data: {
    image_data_url: string;
    captured_at: number;
    status: RemoteBrowserStatus;
  };
  error: string;
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
    max_tokens: number;
    /** Optional advisor output cap — round-tripped, not edited here. */
    reference_max_tokens?: number | null;
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
  max_tokens: number;
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
