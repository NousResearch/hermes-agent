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

/** Routes covered. Keep the list to stable, content-light pages. */
const PAGES: Array<{ path: string; name: string }> = [
  { path: "/sessions", name: "sessions" },
  { path: "/models", name: "models" },
  { path: "/chat", name: "chat" },
  { path: "/skills", name: "skills" },
  { path: "/profiles", name: "profiles" },
  { path: "/docs", name: "docs" },
  { path: "/channels", name: "channels" }, // NOTE: copy is still English-hardcoded (no i18n on ChannelsPage yet) — this baseline pins that pre-localization state; the diff will visualize the Persian conversion when it lands.
  { path: "/config", name: "config" },
  { path: "/env", name: "env" },
];

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
const API_STUBS: Record<string, { status: number; body: string }> = {
  // Array-shaped endpoints (consumers call .length/.filter directly):
  "/api/dashboard/plugins": stubJson([]),
  "/api/dashboard/themes": stubJson([]),
  "/api/profiles": stubJson([]),
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
  "/api/model/options": stubJson([]),
  "/api/messaging/platforms": stubJson({ platforms: [] }),
  // Object-shaped endpoints (shape mismatches crash pages):
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
 */
export async function stubBackend(page: Page): Promise<void> {
  await page.route("**/api/**", (route) => {
    const url = new URL(route.request().url());
    const stub = API_STUBS[url.pathname] ?? API_DEFAULT;
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
  for (const { path, name } of PAGES) {
    test(`locale=fa ${name} renders RTL`, async ({ page }) => {
      await stubBackend(page);
      await seedPersian(page);
      await page.setViewportSize(VIEWPORT);
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
});
