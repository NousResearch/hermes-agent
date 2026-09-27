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
  { path: "/channels", name: "channels" }, // localized since 6fef7145510 — the full-page shot only covers the default view; the conditional per-platform panels (Telegram QR flow, allowed-users field) are pinned separately by the DOM assertions in the channels test below.
  { path: "/config", name: "config" },
  { path: "/env", name: "env" },
  // NEW-SURFACE baselines (a378d64eaf9 round). Both render meaningful rows so
  // the diff catches RTL regressions in real content, not just empty states.
  // Dates inside rows are TZ-sensitive — the config pins timezoneId UTC.
  { path: "/files", name: "files" }, // entry rows: Latin file names, byte sizes, Jalali dates.
  { path: "/cron", name: "cron", freezeClockAt: "2026-09-27T08:00:00Z" }, // job rows: humanized schedule sentences, repeat counters, last/next timestamps — the sharpest mixed-direction text on the dashboard. Clock frozen an hour before the fixture's 09:00Z next_run_at so the overdue badge can never flip.
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
});
