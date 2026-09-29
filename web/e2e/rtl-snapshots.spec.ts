/**
 * RTL visual regression — key dashboard pages under locale=fa.
 *
 * Every page renders with:
 *  - `localStorage["hermes-locale"] = "fa"` → I18nProvider boots Persian and
 *    sets `<html dir="rtl" lang="fa">`.
 *  - `localStorage["hermes-dashboard-font"] = "vazirmatn"` → the Persian
 *    webfont override is active (same letterforms in CI and locally).
 *  - A stubbed `/api` (page.route) so no Python backend is needed; pages
 *    render their real components against deterministic empty/fake data.
 *
 * What a diff actually catches (the point of this suite):
 *  - `dir` flipping back to ltr, or logical classes (`ps/pe/start/end/ms/me`)
 *    regressing to physical (`pl/pr/left/right`) — sidebar, chips, menus and
 *    pagination jump to the wrong edge.
 *  - direction-semantic icons (chevrons/arrows) losing their `rtl:-scale-x-100`
 *    mirroring.
 *  - the Vazirmatn font stack falling back to system faces.
 *  - broken Persian copy (missing keys falling back to English mid-surface).
 *
 * One baseline per covered page (17 as of 2026-09-28, including /mcp with its
 * per-page server/catalog fixtures via MCP_API_OVERRIDES below) plus element
 * baselines for the densest mixed-content rows (files entry row, cron meta
 * strip, MCP server card).
 * Baselines: `npx playwright test --update-snapshots` (see config). On CI PRs,
 * diffs are surfaced as artifacts, not failures — same policy as the desktop
 * visual suite.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test, expect, type Page } from "@playwright/test";

/** Fixed viewport — must match the config's `use.viewport`. */
const VIEWPORT = { width: 1280, height: 800 };

/*
 * Deterministic fonts: the dashboard's `vazirmatn` override pulls its
 * @font-face from Google Fonts at runtime — a network race that flips letter
 * forms between runs (observed as ~1% pixel drift across all text rows).
 * Instead we serve the Vazirmatn woff2 files BUNDLED with the repo (added for
 * the desktop RTL support) and intercept the Google Fonts hosts, so every
 * run — local or CI — renders identical letterforms with zero network.
 */
const FONT_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../apps/desktop/src/fonts",
);
const VAZIRMATN = readFileSync(path.join(FONT_DIR, "Vazirmatn-Regular.woff2"));
const VAZIRMATN_CSS = `@font-face {
  font-family: 'Vazirmatn';
  font-style: normal;
  font-weight: 100 900;
  font-display: block;
  src: url('https://fonts.gstatic.com/vazirmatn-test.woff2') format('woff2');
}`;

/** Fixed offsets from run start; both stay inside the hour bucket of
 *  timeAgo ("N hours ago") so the rendered words are stable for hours.
 *  SessionInfo/ModelsAnalytics timestamps are Unix SECONDS. */
const HOUR = 3_600_000;
const TS = {
  yesterday: Math.floor((Date.now() - 3 * HOUR) / 1000),
  fiveDaysAgo: Math.floor((Date.now() - 8 * HOUR) / 1000),
};

export const CHAT_API_OVERRIDES: Record<string, { status: number; body: string }> = {
  "/api/sessions": stubJson({
    sessions: [
      {
        id: "sess-ingest",
        source: "cli",
        model: "stub-model",
        title: "Refactor the ingest pipeline",
        started_at: TS.yesterday - 1_800_000,
        ended_at: TS.yesterday,
        last_active: TS.yesterday,
        is_active: false,
        message_count: 14,
        tool_call_count: 3,
        input_tokens: 0,
        output_tokens: 0,
        preview: null,
      },
      {
        id: "sess-weekly",
        source: "cli",
        model: "stub-model",
        title: "گزارش هفتگی",
        started_at: TS.fiveDaysAgo - 600_000,
        ended_at: TS.fiveDaysAgo,
        last_active: TS.fiveDaysAgo,
        is_active: false,
        message_count: 7,
        tool_call_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        preview: null,
      },
    ],
    total: 2,
    limit: 20,
    offset: 0,
  }),
};

export const MODELS_API_OVERRIDES: Record<string, { status: number; body: string }> = {
  "/api/analytics/models": stubJson({
    models: [
      {
        model: "stub-model",
        provider: "stub",
        input_tokens: 48_210,
        output_tokens: 9_120,
        cache_read_tokens: 0,
        reasoning_tokens: 0,
        estimated_cost: 1.24,
        actual_cost: 0,
        sessions: 9,
        api_calls: 132,
        tool_calls: 21,
        last_used_at: TS.yesterday,
        avg_tokens_per_session: 6_367,
        capabilities: {
          supports_tools: true,
          supports_vision: false,
          supports_reasoning: true,
          context_window: 200_000,
          max_output_tokens: 8_192,
          model_family: "stub",
        },
      },
      {
        model: "stub-vision",
        provider: "stub",
        input_tokens: 12_400,
        output_tokens: 2_210,
        cache_read_tokens: 0,
        reasoning_tokens: 0,
        estimated_cost: 0.42,
        actual_cost: 0,
        sessions: 3,
        api_calls: 28,
        tool_calls: 0,
        last_used_at: TS.fiveDaysAgo,
        avg_tokens_per_session: 4_870,
        capabilities: {
          supports_tools: false,
          supports_vision: true,
          supports_reasoning: false,
          context_window: 128_000,
          max_output_tokens: 4_096,
          model_family: "stub",
        },
      },
    ],
    totals: {
      distinct_models: 2,
      total_input: 60_610,
      total_output: 11_330,
      total_cache_read: 0,
      total_reasoning: 0,
      total_estimated_cost: 1.66,
      total_actual_cost: 0,
      total_sessions: 12,
      total_api_calls: 160,
    },
    period_days: 7,
  }),
};

/**
 * Channels conditional-panel fixtures — one fixture row per conditional
 * panel pinned below. Both panels mount only when their platform id appears
 * in `/api/messaging/platforms`, so a single fixture row list drives both
 * tests (and the full-page channels shot stays the empty-state default).
 *
 * Names/URLs stay Latin: they are user data. Field copy is server-provided
 * (`prompt`/`help`/`description`), so the Telegram allowed-users test pins
 * the DATA side of the modal in Persian while the en.ts chrome around it is
 * translated through t.channels.*.
 */
export const CHANNELS_API_OVERRIDES: Record<string, { status: number; body: string }> = {
  "/api/messaging/platforms": stubJson({
    env_path: "D:\\hermes-test-home\\.env",
    gateway_start_command: "hermes gateway run",
    platforms: [
      {
        id: "whatsapp",
        name: "WhatsApp",
        description: "هرمس را از طریق پل داخلی واتس‌اپ با اتصال مبتنی بر QR به کار ببرید.",
        docs_url: "https://example.invalid/docs/whatsapp",
        enabled: false,
        configured: false,
        gateway_running: false,
        state: "not_configured",
        error_code: null,
        error_message: null,
        updated_at: null,
        home_channel: null,
        whatsapp_setup: null,
        env_vars: [],
      },
      {
        id: "telegram",
        name: "Telegram",
        description: "هرمس را از دایرکت‌ها، گروه‌ها و تاپیک‌های تلگرام اجرا کنید.",
        docs_url: "https://example.invalid/docs/telegram",
        enabled: true,
        configured: true,
        gateway_running: false,
        state: "not_configured",
        error_code: null,
        error_message: null,
        updated_at: null,
        home_channel: null,
        env_vars: [
          {
            key: "TELEGRAM_BOT_TOKEN",
            required: true,
            is_set: true,
            redacted_value: "•••••• (set — leave blank to keep)",
            description: "",
            prompt: "توکن بات",
            help: "با @BotFather یک بات بسازید، سپس توکنی که می‌دهد را جای‌گذاری کنید.",
            url: null,
            is_password: true,
            advanced: false,
          },
          {
            key: "TELEGRAM_ALLOWED_USERS",
            required: false,
            is_set: true,
            redacted_value: "8792111505",
            description: "شناسه‌های عددی جداشده با کاما از @userinfobot.",
            prompt: "شناسه‌های کاربر مجاز Telegram",
            help: "توصیه‌شده. بدون این، هرکسی می‌تواند به بات شما پیام بدهد.",
            url: null,
            is_password: false,
            advanced: false,
          },
        ],
      },
    ],
  }),
  // Telegram QR pairing flow for the allowed-users test: the start response
  // feeds QRCode.toDataURL directly (qr_payload is required or the page
  // throws), and the status route answers "ready" on the first poll with a
  // detected owner — pre-filling the chip row the baseline pins. Both
  // expires_at values sit in the future of the test's frozen clock
  // (09:00:00Z), so the mm:ss badge renders a stable "15:00".
  "/api/messaging/telegram/onboarding/start": stubJson({
    pairing_id: "pair-rtl-1",
    suggested_username: "hermes_pair_bot",
    deep_link: "https://t.me/hermes_pair_bot?start=pair-rtl-1",
    qr_payload: "https://t.me/HermesPairBot?start=pair-rtl-1",
    expires_at: "2026-09-27T09:15:00Z",
  }),
  "/api/messaging/telegram/onboarding/pair-rtl-1": stubJson({
    status: "ready",
    bot_username: "hermes_pair_bot",
    owner_user_id: "8792111505",
    expires_at: "2026-09-27T09:15:00Z",
  }),
};

export const MCP_API_OVERRIDES: Record<string, { status: number; body: string }> = {
  // McpServerListResponse — three server shapes so one page pins all the
  // chrome: http+oauth (Latin URL island, auth badge, env-var chip),
  // stdio+header (command-line island), and a disabled server (opacity-60 +
  // the «غیرفعال» badge). Names stay Latin: they are user data.
  "/api/mcp/servers": stubJson({
    servers: [
      {
        name: "github-mcp",
        transport: "http",
        url: "https://mcp.example.invalid/github",
        command: null,
        args: [],
        env: { GITHUB_TOKEN: "gho_stub_token" },
        auth: "oauth",
        enabled: true,
        tools: ["search_repos", "read_file"],
      },
      {
        name: "filesystem-bridge",
        transport: "stdio",
        url: null,
        command: "npx",
        args: ["-y", "@example/fs-bridge", "--root", "/data"],
        env: {},
        auth: "header",
        enabled: true,
        tools: null,
      },
      {
        name: "weather-legacy",
        transport: "http",
        url: "https://weather.example.invalid/mcp",
        command: null,
        args: [],
        env: {},
        auth: null,
        enabled: false,
        tools: null,
      },
    ],
  }),
  // Catalog response — one installed-but-disabled stdio entry (carries the
  // matching diagnostic + the «نصب‌شده»/«غیرفعال» badge pair) and one
  // not-installed http entry with the Persian description, install CTA,
  // endpoint line and git-bootstrap line. required_env/default_enabled are
  // read by the (closed) install modal but must be shape-complete.
  "/api/mcp/catalog": stubJson({
    entries: [
      {
        name: "git-tools",
        description: "Local git worktree tools for the agent.",
        source: "builtin",
        transport: "stdio",
        auth_type: "none",
        required_env: [],
        command: "uvx",
        args: ["git-tools-mcp"],
        url: null,
        install_url: null,
        install_ref: null,
        bootstrap: [],
        default_enabled: null,
        post_install: "",
        needs_install: false,
        installed: true,
        enabled: false,
      },
      {
        name: "notion-mcp",
        description: "خواندن و نوشتن صفحات Notion از طریق پروتکل MCP.",
        source: "https://example.invalid/catalog/notion",
        transport: "http",
        auth_type: "api_key",
        required_env: [
          { name: "NOTION_TOKEN", prompt: "Notion integration token", required: true },
        ],
        command: null,
        args: [],
        url: "https://mcp.notion.example.invalid/mcp",
        install_url: "https://example.invalid/notion-mcp.git",
        install_ref: "v2.1.0",
        bootstrap: ["npm", "ci"],
        default_enabled: ["search", "fetch"],
        post_install: "Set NOTION_TOKEN before first use.",
        needs_install: true,
        installed: false,
        enabled: false,
      },
    ],
    diagnostics: [
      { name: "git-tools", kind: "disabled", message: "Server installed but disabled." },
    ],
  }),
};

/**
 * Deterministic stub responses for every `/api/*` endpoint the covered pages
 * call on boot. `200 {}` is a safe default: the dashboard's fetchJSON accepts
 * empty objects and pages degrade to their Persian empty states.
 */
function stubJson(body: unknown): { status: number; body: string } {
  return { status: 200, body: JSON.stringify(body) };
}

const API_DEFAULT = stubJson({});

/** Per-endpoint stubs where an empty object would break a page's render. */
/** Routes covered. Keep the list to stable, content-light pages. */
const PAGES: Array<{
  path: string;
  name: string;
  /** Extra per-page stubs merged over API_STUBS (e.g. fixture rows that make
   *  locale-sensitive chips render for this surface only). */
  apiOverrides?: Record<string, { status: number; body: string }>;
  /** ISO instant to freeze the page clock at (page.clock.setFixedTime). Needed
   *  for surfaces whose rendering branches on Date.now() vs a pinned fixture
   *  timestamp — cron flips its next-run cell into an "overdue since" variant
   *  once the fixture's next_run_at drifts into the past, which would rot the
   *  baseline on a real calendar day. Freezing keeps the capture reproducible
   *  forever; the instant must predate every pinned next_run_at. */
  freezeClockAt?: string;
}> = [
  { path: "/sessions", name: "sessions" },
  // Chat + models render locale-sensitive relative-time chips (timeAgo →
  // Intl.RelativeTimeFormat, «دیروز»/«۵ روز پیش» under fa). Their default
  // stubs are empty lists, so the per-page overrides below seed fixture
  // rows — bucketed at "yesterday"/"5 days ago" so the rendered words stay
  // stable for the whole run (minute-level buckets would drift mid-capture).
  { path: "/chat", name: "chat", apiOverrides: CHAT_API_OVERRIDES },
  { path: "/models", name: "models", apiOverrides: MODELS_API_OVERRIDES },
  { path: "/skills", name: "skills" },
  { path: "/profiles", name: "profiles" },
  // NEW-SURFACE baselines. Both render meaningful rows so the diff catches
  // RTL regressions in real content, not just empty states.
  // Dates inside rows are TZ-sensitive — the config pins timezoneId UTC.
  { path: "/docs", name: "docs" },
  { path: "/channels", name: "channels" }, // localized since 6fef7145510 — the full-page shot pins the empty-state default; the conditional per-platform panels (WhatsApp QR pairing, Telegram allowed-users editor) are pinned by the dedicated element-baseline tests below.
  { path: "/config", name: "config" },
  { path: "/env", name: "env" },
  // NEW-SURFACE baselines (a378d64eaf9 round). Both render meaningful rows so
  // the diff catches RTL regressions in real content, not just empty states.
  // Dates inside rows are TZ-sensitive — the config pins timezoneId UTC.
  { path: "/files", name: "files" }, // entry rows: Latin file names, byte sizes, Jalali dates.
  { path: "/cron", name: "cron", freezeClockAt: "2026-09-27T08:00:00Z" }, // job rows: humanized schedule sentences, repeat counters, last/next timestamps — the sharpest mixed-direction text on the dashboard. Clock frozen an hour before the fixture's 09:00Z next_run_at so the overdue badge can never flip.
  // MCP (localized in f263bcc8ea9): server cards pin transport/auth badges,
  // URL + command-line Latin islands inside Persian chrome, env-var chips and
  // the disabled state; catalog cards pin the install CTA, Persian entry
  // description, bootstrap lines and the installed/disabled badge pair. The
  // `{}` defaults would crash the page (res.servers/res.entries are mapped).
  { path: "/mcp", name: "mcp", apiOverrides: MCP_API_OVERRIDES },
];

const API_STUBS: Record<string, { status: number; body: string }> = {
  // Array-shaped endpoints (consumers call .length/.filter directly):
  "/api/dashboard/plugins": stubJson([]),
  "/api/dashboard/themes": stubJson([]),
  // ProfileListResponse shape — MUST be { profiles: [...] }: CronPage does
  // setProfiles(res.profiles) and renders profiles.map, so the `{}` default
  // (or a bare array) crashes the whole page.
  "/api/profiles": stubJson({ profiles: [] }),
  "/api/analytics": stubJson([]),
  // AuxiliaryModelsResponse shape; an empty object crashes ModelSettingsPanel
  // reading aux.main.provider (aux?.main is undefined, .provider throws).
  "/api/model/auxiliary": stubJson({
    tasks: [],
    main: { provider: "stub-provider", model: "stub-model" },
  }),
  // MoaConfigResponse shape — truthy-but-empty ({}) crashes the MoA summary
  // row reading moa.reference_models.length.
  "/api/model/moa": stubJson({
    default_preset: "balanced",
    active_preset: "balanced",
    presets: {},
    reference_models: [],
    aggregator: { provider: "stub-provider", model: "stub-model" },
    reference_temperature: 0.7,
    aggregator_temperature: 0.7,
    reference_timeout: null,
    degraded_reference_policy: "silent",
    enabled: false,
  }),
  // Flattened ModelsAnalyticsResponse shape (models + totals + period_days);
  // an empty object here crashes ModelsPage reading data.totals.distinct_models.
  "/api/analytics/models": stubJson({
    models: [],
    totals: {
      distinct_models: 0,
      total_input: 0,
      total_output: 0,
      total_cache_read: 0,
      total_reasoning: 0,
      total_estimated_cost: 0,
      total_actual_cost: 0,
      total_sessions: 0,
      total_api_calls: 0,
    },
    period_days: 7,
  }),
  "/api/skills": stubJson([]),
  // SkillHubSourcesResponse shape — SkillsPage boots with
  // setSources/setFeatured/setInstalled from this response; the `{}` default
  // leaves them undefined and the resolve-vs-paint race was flipping the
  // rendered landing between runs (skills.png diffing ~2% run to run).
  "/api/skills/hub/sources": stubJson({
    sources: [],
    index_available: true,
    featured: [],
    installed: {},
  }),
  "/api/model/options": stubJson([]),
  "/api/messaging/platforms": stubJson({ platforms: [] }),
  // ManagedFilesResponse shape — an empty object crashes FilesPage reading
  // listing.entries; the fixture pins dir/file rows with Persian-safe names.
  "/api/files": stubJson({
    root: "D:\\hermes-test-home",
    path: "D:\\hermes-test-home\\memory",
    parent: "D:\\hermes-test-home",
    locked_root: null,
    can_change_path: true,
    entries: [
      {
        name: "MEMORY.md",
        path: "D:\\hermes-test-home\\memory\\MEMORY.md",
        is_directory: false,
        size: 2048,
        mtime: 1758900000,
        mime_type: "text/markdown",
      },
      {
        name: "USER.md",
        path: "D:\\hermes-test-home\\memory\\USER.md",
        is_directory: false,
        size: 1024,
        mtime: 1758800000,
        mime_type: "text/markdown",
      },
      {
        name: "اطلاعات",
        path: "D:\\hermes-test-home\\memory\\اطلاعات",
        is_directory: true,
        size: null,
        mtime: 1758700000,
        mime_type: null,
      },
    ],
  }),
  // CronJob[] — two rows (paused + active, one with an enabled-toolsets chip)
  // so the enabled/paused badges, profile chip and next-run column all render.
  "/api/cron/jobs": stubJson([
    {
      id: "job-daily-digest",
      profile: "default",
      profile_name: "default",
      hermes_home: null,
      is_default_profile: true,
      name: "گزارش روزانه",
      prompt: "Summarize today's session activity.",
      script: null,
      skills: null,
      schedule: { kind: "recurring", expr: "0 9 * * *", display: "daily at 09:00" },
      schedule_display: "daily at 09:00",
      repeat: null,
      enabled: true,
      state: "scheduled",
      deliver: null,
      model: null,
      provider: null,
      base_url: null,
      no_agent: false,
      context_from: null,
      enabled_toolsets: ["web_search"],
      workdir: null,
      last_run_at: "2026-09-26T09:00:00Z",
      next_run_at: "2026-09-27T09:00:00Z",
      scheduler_heartbeat_age_s: 42,
      last_status: "ok",
      last_error: null,
      last_delivery_error: null,
      last_fire_error: null,
    },
    {
      id: "job-inbox-pause",
      profile: "work",
      profile_name: "work",
      hermes_home: null,
      is_default_profile: false,
      name: "Inbox triage",
      prompt: "Triage the mailbox.",
      script: null,
      skills: null,
      schedule: { kind: "recurring", expr: "0 */2 * * *", display: "every 2 hours" },
      schedule_display: "every 2 hours",
      repeat: null,
      enabled: false,
      state: "paused",
      deliver: "telegram",
      model: null,
      provider: null,
      base_url: null,
      no_agent: false,
      context_from: null,
      enabled_toolsets: null,
      workdir: null,
      last_run_at: null,
      next_run_at: null,
      scheduler_heartbeat_age_s: null,
      last_status: null,
      last_error: null,
      last_delivery_error: null,
      last_fire_error: null,
    },
  ]),
  // CronDeliveryTarget[] — fetched on boot; only the (closed) create modal
  // renders them, but an empty object would set the state to undefined.
  "/api/cron/delivery-targets": stubJson({
    targets: [
      { id: "local", name: "Local", home_target_set: true, home_env_var: null },
      { id: "telegram", name: "Telegram", home_target_set: false, home_env_var: "HERMES_TELEGRAM_HOME" },
    ],
  }),
  // ToolsetInfo[] — MUST be an array: CronPage boots with
  // `[...toolsets].sort(...)`, so the `{}` default (no stub) crashes the
  // whole page with "toolsets is not iterable" (its .catch only guards
  // rejected fetches, not 200-with-an-object).
  "/api/tools/toolsets": stubJson([]),
  // Object-shaped endpoints (shape mismatches crash pages):
  // ChatWorkspacesResponse shape — ChatPage boots reading data.projects, so
  // the `{}` default crashes /chat ("projects is not iterable") and the side
  // panel never mounts.
  "/api/chat/workspaces": stubJson({
    projects: [],
    repos: [],
    default_cwd: "D:\\hermes-test-home",
    home: "D:\\hermes-test-home",
    scan_enabled: false,
  }),
  "/api/sessions": stubJson({ sessions: [], total: 0 }),
  "/api/sessions/stats":
    stubJson({ total: 0, active_store: 0, archived: 0, messages: 0, by_source: {} }),
  "/api/sessions/empty/count": stubJson({ count: 0 }),
  "/api/status": stubJson({
    version: "0.0.0-test",
    provider: "stub",
    model: "stub-model",
    gateway: { connected: false },
  }),
  "/api/model/info": stubJson({
    provider: "stub",
    model: "stub-model",
    display_name: "Stub Model",
  }),
};

/**
 * Install route interception for the given page. Runs BEFORE any app script
 * (routes apply to subsequent requests), so boot-time fetches are covered.
 * `overrides` wins over the shared API_STUBS for its exact pathname.
 */
export async function stubBackend(
  page: Page,
  overrides: Record<string, { status: number; body: string }> = {},
): Promise<void> {
  await page.route("**/api/**", (route) => {
    const url = new URL(route.request().url());
    const stub = overrides[url.pathname] ?? API_STUBS[url.pathname] ?? API_DEFAULT;
    void route.fulfill({ status: stub.status, contentType: "application/json", body: stub.body });
  });
  // Plugin manifest scripts: none, so nothing else loads.
  await page.route("**/dashboard-plugins/**", (route) =>
    void route.fulfill({ status: 200, contentType: "application/javascript", body: "" }),
  );
  // Serve Vazirmatn locally instead of Google Fonts (determinism — see above).
  await page.route("**/fonts.googleapis.com/**", (route) =>
    void route.fulfill({ status: 200, contentType: "text/css", body: VAZIRMATN_CSS }),
  );
  await page.route("**/fonts.gstatic.com/**", (route) =>
    void route.fulfill({ status: 200, contentType: "font/woff2", body: VAZIRMATN }),
  );
  // Any other external request: fail fast instead of flaking. (Local dev
  // server traffic — 127.0.0.1/localhost — passes through untouched.)
  await page.route((url) => url.protocol.startsWith("http"), (route) => {
    const host = new URL(route.request().url()).hostname;
    if (host === "127.0.0.1" || host === "localhost") return route.fallback();
    return route.abort();
  });
}

/** Seed locale + font BEFORE any app script runs (init scripts run first). */
export async function seedPersian(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.localStorage.setItem("hermes-locale", "fa");
    window.localStorage.setItem("hermes-dashboard-font", "vazirmatn");
  });
}

/** Hard assertions that the RTL contract actually holds before snapshotting. */
export async function assertRtlBoot(page: Page): Promise<void> {
  await expect(page.locator("html")).toHaveAttribute("dir", /rtl/i);
  await expect(page.locator("html")).toHaveAttribute("lang", /fa/i);
  // Force the Persian webfont to load NOW (not lazily during the screenshot)
  // and wait for it — screenshots must never race the font swap.
  await page.evaluate(async () => {
    await Promise.all([
      document.fonts.load('400 16px Vazirmatn'),
      document.fonts.load('500 16px Vazirmatn'),
      document.fonts.load('600 16px Vazirmatn'),
      document.fonts.load('700 16px Vazirmatn'),
    ]);
    await document.fonts.ready;
  });
}

/**
 * Screenshot helper: settles the page, takes a full-page snapshot, and on
 * update-mode writes a copy into test-results (mirrors the desktop helper so
 * CI artifacts include every screenshot, not just diffs).
 */
async function rtlSnapshot(page: Page, name: string): Promise<void> {
  await page.waitForLoadState("networkidle");
  // Let lazy-loaded route chunks finish painting after networkidle.
  await page.waitForTimeout(250);
  const shot = await page.screenshot({
    animations: "disabled",
    caret: "hide",
    fullPage: true,
  });
  const info = test.info();
  try {
    await expect(shot).toMatchSnapshot(`${name}.png`);
  } finally {
    mkdirSync(info.outputDir, { recursive: true });
    writeFileSync(info.outputPath(`${name}-actual.png`), shot);
  }
}

test.describe("Persian RTL visual snapshots", () => {
  for (const { path, name, apiOverrides, freezeClockAt } of PAGES) {
    test(`locale=fa ${name} renders RTL`, async ({ page }) => {
      await stubBackend(page, apiOverrides);
      await seedPersian(page);
      await page.setViewportSize(VIEWPORT);
      if (freezeClockAt) await page.clock.setFixedTime(new Date(freezeClockAt));
      await page.goto(path, { waitUntil: "domcontentloaded" });
      await assertRtlBoot(page);
      await rtlSnapshot(page, name);
    });
  }

  test("sidebar opens from the correct (right) edge in RTL", async ({ page }) => {
    // The mobile-width drawer is the sharpest direction probe: it slides in
    // from the inline-start edge, which is the RIGHT side under RTL.
    await stubBackend(page);
    await seedPersian(page);
    await page.setViewportSize({ width: 620, height: 800 });
    await page.goto("/sessions", { waitUntil: "domcontentloaded" });
    await assertRtlBoot(page);
    await page.waitForLoadState("networkidle");

    const opener = page.getByRole("button", { name: /باز کردن ناوبری|open navigation/i });
    await opener.click();
    await page.waitForTimeout(400); // drawer slide-in (reduced motion → instant)

    const drawer = page.locator("aside").first();
    const box = await drawer.boundingBox();
    expect(box).toBeTruthy();
    // Visible portion must hug the RIGHT edge of a 620px viewport.
    expect(box!.x + box!.width).toBeGreaterThan(620 - 4);
    await rtlSnapshot(page, "mobile-drawer");
  });

  test("docs page opens the Persian guide walkthrough with all five figures", async ({ page }) => {
    // The in-dashboard Persian guide mirrors guide.html's five-step first-run
    // walkthrough (kept in sync by scripts/check_guide_walkthrough_sync.py).
    // This test pins the RUNTIME side: the Docs page must actually open the
    // guide, decode all five guide-images screenshots, and render their
    // captions in the canonical ۱..۵ order — plus a dedicated element
    // baseline of the walkthrough block, so any future guide change (new
    // figure, reworded caption, layout tweak) shows up as a reviewable
    // snapshot diff instead of hiding inside the tall full-page docs shot.
    await stubBackend(page);
    await seedPersian(page);
    await page.setViewportSize(VIEWPORT);
    await page.goto("/docs", { waitUntil: "domcontentloaded" });
    await assertRtlBoot(page);
    await page.waitForLoadState("networkidle");

    // The walkthrough section opened, with exactly five figures.
    await expect(page.getByText("🚀 شروع سریع در ۵ گام")).toBeVisible();
    const figures = page.locator("figure");
    await expect(figures).toHaveCount(5);

    // Captions must be present and numbered ۱..۵ in order (the mirrored
    // guide.html numbering).
    const captions = await figures.locator("figcaption").allInnerTexts();
    expect(captions).toHaveLength(5);
    expect(captions[0]).toContain("۱ —");
    expect(captions[4]).toContain("۵ —");

    // Every screenshot must actually decode from public/guide-images/ —
    // a broken/renamed asset renders as an empty box that only this
    // assertion (not a pixel diff) catches deterministically.
    await expect
      .poll(
        () =>
          figures.locator("img").evaluateAll((imgs) =>
            imgs.map((img) => (img as HTMLImageElement).naturalWidth > 0),
          ),
        { timeout: 5_000 },
      )
      .toEqual([true, true, true, true, true]);

    // Tight element baseline of the walkthrough block itself.
    const block = page.getByText("🚀 شروع سریع در ۵ گام").locator("..");
    await block.scrollIntoViewIfNeeded();
    await page.waitForTimeout(250); // lazy images settle after scroll
    const shot = await block.screenshot({ animations: "disabled" });
    const info = test.info();
    try {
      await expect(shot).toMatchSnapshot("docs-guide-open.png");
    } finally {
      mkdirSync(info.outputDir, { recursive: true });
      writeFileSync(info.outputPath("docs-guide-open-actual.png"), shot);
    }
  });

  test("files entry row pins mixed-content chrome (Latin name, size, Jalali date)", async ({
    page,
  }) => {
    // The memory/MEMORY.md row is the densest mixed-content row on the
    // dashboard: a Latin file name, tabular byte counts and a Jalali date
    // (Persian digits) sharing one grid row. The full-page /files snapshot
    // covers the whole surface; this element baseline isolates the row so
    // a column-order or date-format regression is reviewable in isolation.
    await stubBackend(page);
    await seedPersian(page);
    await page.setViewportSize(VIEWPORT);
    await page.goto("/files", { waitUntil: "domcontentloaded" });
    await assertRtlBoot(page);
    await page.waitForLoadState("networkidle");

    // The stubbed listing actually rendered (names from the fixture).
    await expect(page.getByText("MEMORY.md")).toBeVisible();
    await expect(page.getByText("اطلاعات")).toBeVisible();

    const row = page
      .locator("div.grid.items-center")
      .filter({ hasText: "MEMORY.md" })
      .first();
    await expect(row).toBeVisible();
    const shot = await row.screenshot({ animations: "disabled" });
    const info = test.info();
    try {
      await expect(shot).toMatchSnapshot("files-entry-row.png");
    } finally {
      mkdirSync(info.outputDir, { recursive: true });
      writeFileSync(info.outputPath("files-entry-row-actual.png"), shot);
    }
  });

  test("cron job-row meta strip pins schedule sentence, repeat and timestamps", async ({
    page,
  }) => {
    // The meta strip under each job title carries the humanized schedule
    // sentence, the repeat counter and the last/next timestamps — Latin
    // digits inside Persian prose, plus TZ-sensitive Intl formatting
    // (pinned to UTC in the config). The daily-digest fixture row renders
    // the active state; the full-page /cron snapshot covers both rows and
    // the paused badge. The page clock is frozen an hour before the fixture's
    // next_run_at (mirroring freezeClockAt on the /cron PAGES entry) — without
    // that, the overdue branch flips once real wall-clock time crosses the
    // pinned timestamp and the baselines rot on the calendar.
    await stubBackend(page);
    await seedPersian(page);
    await page.setViewportSize(VIEWPORT);
    await page.clock.setFixedTime(new Date("2026-09-27T08:00:00Z"));
    await page.goto("/cron", { waitUntil: "domcontentloaded" });
    await assertRtlBoot(page);
    await page.waitForLoadState("networkidle");

    // Both fixture jobs rendered (Persian + Latin names, both states).
    await expect(page.getByText("گزارش روزانه")).toBeVisible();
    await expect(page.getByText("Inbox triage")).toBeVisible();

    const strip = page
      .locator("div.text-xs.text-muted-foreground")
      .filter({ hasText: "repeat:" })
      .first();
    await expect(strip).toBeVisible();
    const shot = await strip.screenshot({ animations: "disabled" });
    const info = test.info();
    try {
      await expect(shot).toMatchSnapshot("cron-job-meta.png");
    } finally {
      mkdirSync(info.outputDir, { recursive: true });
      writeFileSync(info.outputPath("cron-job-meta-actual.png"), shot);
    }
  });

  test("mcp server card pins mixed-direction identity chrome (Latin name + URL, Persian badges)", async ({
    page,
  }) => {
    // The github-mcp fixture card is the MCP surface's sharpest mixed-content
    // element: a Latin server name and URL (user data, LTR runs) inside
    // Persian chrome — transport badge, «احراز هویت: oauth», the env-var chip
    // «۱ متغیر محیطی» and the action buttons. The full-page /mcp snapshot
    // covers the whole surface; this element baseline isolates one card so a
    // badge-order or URL-island regression is reviewable in isolation.
    await stubBackend(page, MCP_API_OVERRIDES);
    await seedPersian(page);
    await page.setViewportSize(VIEWPORT);
    await page.goto("/mcp", { waitUntil: "domcontentloaded" });
    await assertRtlBoot(page);
    await page.waitForLoadState("networkidle");

    // The stubbed servers actually rendered (fixture names).
    await expect(page.getByText("github-mcp")).toBeVisible();
    await expect(page.getByText("filesystem-bridge")).toBeVisible();
    await expect(page.getByText("weather-legacy")).toBeVisible();

    const card = page
      .locator("div.border.bg-background-base\\/80")
      .filter({ hasText: "github-mcp" })
      .first();
    await expect(card).toBeVisible();
    const shot = await card.screenshot({ animations: "disabled" });
    const info = test.info();
    try {
      await expect(shot).toMatchSnapshot("mcp-server-card.png");
    } finally {
      mkdirSync(info.outputDir, { recursive: true });
      writeFileSync(info.outputPath("mcp-server-card-actual.png"), shot);
    }
  });

  test("whatsapp QR-pairing panel pins the mode picker and allowed-numbers field", async ({
    page,
  }) => {
    // The WhatsApp panel mounts ONLY when the fixture carries a whatsapp row,
    // so the full-page channels shot (empty-state default) never shows it.
    // This element baseline pins its idle chrome: the Persian mode picker
    // («بات» / «گفتگو با خود»), the «شماره‌های واتس‌اپ مجاز» label over a
    // Latin phone-number placeholder — the panel's sharpest mixed-direction
    // pair — and the create-with-QR button (t.channels.telegramCreateWithQr,
    // shared with the Telegram flow). The clock is frozen because the panel
    // re-renders its mm:ss expiry badge every second while a pairing is up;
    // at real wall-clock the badge would rot the baseline within a minute.
    await stubBackend(page, CHANNELS_API_OVERRIDES);
    await seedPersian(page);
    await page.setViewportSize(VIEWPORT);
    await page.clock.setFixedTime(new Date("2026-09-27T09:00:00Z"));
    await page.goto("/channels", { waitUntil: "domcontentloaded" });
    await assertRtlBoot(page);
    await page.waitForLoadState("networkidle");

    // The fixture row actually rendered (localized catalog name + intro).
    await expect(page.getByText("WhatsApp")).toBeVisible();
    await expect(
      page.getByText("هرمس را از طریق پل داخلی واتس‌اپ"),
    ).toBeVisible();

    const panel = page
      .locator("div.rounded-sm.border")
      .filter({ has: page.locator("#whatsapp-allowed-users") })
      .first();
    await expect(panel).toBeVisible();
    await expect(panel.getByText("حالت", { exact: true })).toBeVisible();
    await expect(panel.getByText("بات", { exact: true })).toBeVisible();
    await expect(panel.getByText("گفتگو با خود")).toBeVisible();
    await expect(panel.getByText("شماره‌های واتس‌اپ مجاز")).toBeVisible();

    const shot = await panel.screenshot({ animations: "disabled" });
    const info = test.info();
    try {
      await expect(shot).toMatchSnapshot("channels-whatsapp-qr-panel.png");
    } finally {
      mkdirSync(info.outputDir, { recursive: true });
      writeFileSync(info.outputPath("channels-whatsapp-qr-panel-actual.png"), shot);
    }
  });

  test("telegram allowed-users editor pins the pairing-ready state chips", async ({
    page,
  }) => {
    // The Telegram quick-setup card renders on every /channels page, but its
    // allowed-users editor appears only once a QR pairing reaches "ready" —
    // a state the full-page shot can never reach against a stubbed backend.
    // The stub serves the ready status directly: one poll cycle after the
    // create click, the panel shows the «آماده» badge, the detected-owner
    // badge («مالک شناسایی شد») and the numeric ID chip row over Persian
    // chrome with Latin user data (bot @handle, numeric IDs, mm:ss countdown
    // — frozen by the fixed clock so the badge never rots).
    await stubBackend(page, CHANNELS_API_OVERRIDES);
    await seedPersian(page);
    await page.setViewportSize(VIEWPORT);
    await page.clock.setFixedTime(new Date("2026-09-27T09:00:00Z"));
    await page.goto("/channels", { waitUntil: "domcontentloaded" });
    await assertRtlBoot(page);
    await page.waitForLoadState("networkidle");

    // Both methods of the quick-setup card rendered in Persian.
    await expect(page.getByText("راه‌اندازی سریع")).toBeVisible();
    await expect(page.getByText("پیشنهادشده")).toBeVisible();
    await expect(page.getByText("بات خودتان")).toBeVisible();

    // Start the QR flow; the stub's first status poll answers "ready". The
    // WhatsApp panel shares the same create-with-Qr string, so scope to the
    // Telegram card (its localized catalog intro) — a page-wide name query
    // is a strict-mode violation by design.
    const tgCard = page
      .locator("div.border.bg-background-base\\/80")
      .filter({ hasText: "تاپیک‌های تلگرام" })
      .first();
    await tgCard.getByRole("button", { name: "ساخت با QR" }).click();
    await expect(page.getByText("@hermes_pair_bot")).toBeVisible({
      timeout: 10_000,
    });

    const panel = page
      .locator("div.rounded-sm.border")
      .filter({ has: page.locator('img[alt="Telegram setup QR code"]') })
      .first();
    await expect(panel).toBeVisible();
    await expect(panel.getByText("کاربران مجاز")).toBeVisible();
    await expect(panel.getByText("مالک شناسایی شد")).toBeVisible();
    await expect(panel.getByText("8792111505")).toBeVisible();

    const shot = await panel.screenshot({ animations: "disabled" });
    const info = test.info();
    try {
      await expect(shot).toMatchSnapshot("channels-telegram-allowed-users.png");
    } finally {
      mkdirSync(info.outputDir, { recursive: true });
      writeFileSync(info.outputPath("channels-telegram-allowed-users-actual.png"), shot);
    }
  });
});
