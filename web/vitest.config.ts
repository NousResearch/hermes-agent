import { defineConfig } from "vitest/config";
import babel from "@rolldown/plugin-babel";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";

/** Same component/hook-scoped compiler preset as vite.config.ts. */
function compilerPreset() {
  const preset = reactCompilerPreset();
  preset.rolldown.filter.code = /\/>|<\/|from\s*['"][^'"]*react/;
  return preset;
}
import path from "path";

export default defineConfig({
  plugins: [react(), babel({ presets: [compilerPreset()] })],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.{ts,tsx}"],
    // The first test in a file pays env init + full module transform, and page
    // suites (SessionsPage) legitimately run 3.5-5.5s on green CI runners —
    // right against vitest's 5s default, so a loaded runner tips them into a
    // timeout (main run 34600757569: 5079ms). Same headroom rationale as
    // apps/desktop/vitest.config.ts; genuinely hung tests still fail.
    testTimeout: 15_000,
    // Contention hardening for the page suites (the profile-routing test is
    // the canary: its 5 waits/4 act flushes amplify scheduler jitter).
    // - retry: a timeout under a co-tenant CPU spike is noise; re-run once
    //   before reporting. A real regression fails twice and still fails.
    // - Single-fork pool serializes files inside one process. jsdom page
    //   suites are CPU-bound (environment setup ~17s/50 files, module
    //   transform ~26s); a worker pool fights the runner's own other lanes
    //   for cores and is what pushed the first test past 5s in the first
    //   place. Serial files are slightly slower wall-clock on an idle
    //   runner and far more stable on a loaded one — the correct trade for
    //   a check lane.
    retry: 1,
    pool: "forks",
    poolOptions: { forks: { singleFork: true } },
  },
});
