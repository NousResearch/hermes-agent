import { defineConfig, type ReporterDescription } from "@playwright/test";

/**
 * Visual regression config for the web dashboard — RTL (locale=fa) focus.
 *
 * Mirrors apps/desktop/playwright.config.ts conventions:
 *  - baselines are generated on `main` (via --update-snapshots) and cached;
 *    PRs compare against them but DO NOT fail on visual diffs — diffs are
 *    uploaded as artifacts for human review.
 *  - toHaveScreenshot tolerates ~1% pixel drift and disables animations.
 *
 * Unlike the desktop suite, this drives a plain Chromium page against Vite's
 * dev server with the `/api` proxy pointed at a **stub backend** (see
 * e2e/rtl-snapshots.spec.ts → `stubBackend`). No Python, no Electron, no
 * gateway — the pages render their real Persian UI against empty/fake data.
 *
 * Update baselines after an intentional UI change:
 *   cd web && npx playwright test --update-snapshots
 */
const reporters: ReporterDescription[] = [["list"]];
if (process.env.CI) {
  reporters.push(["json", { outputFile: "playwright-report/results.json" }]);
}

/** Dev-server port; 5173 is the dashboard's own default. */
const PORT = Number(process.env.RTL_SNAP_PORT ?? 5173);
const BASE_URL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: "./e2e",
  // The dev server boots fast; 60s/test is generous for lazy route loads.
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  // One dev server serves the whole serial run; each test navigates fresh.
  fullyParallel: false,
  workers: 1,
  reporter: reporters,
  // Store baselines in one folder WITHOUT the platform suffix, so a Linux CI
  // baseline matches Windows local runs for this pure-web surface (fonts are
  // pinned via --force-prefers-system-colors + bundled stack below).
  snapshotPathTemplate: "{testDir}/__rtl-snapshots__/{arg}{ext}",
  use: {
    baseURL: BASE_URL,
    viewport: { width: 1280, height: 800 },
    screenshot: "off", // we take explicit snapshots only
    trace: "retain-on-failure",
    contextOptions: {
      reducedMotion: "reduce",
      deviceScaleFactor: 1,
      locale: "fa-IR",
    },
  },
  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.01,
      animations: "disabled",
      caret: "hide",
      threshold: 0.2,
    },
  },
  webServer: {
    command: `npx vite --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
