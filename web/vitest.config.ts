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
    setupFiles: ["./src/test-setup.ts"],
    // The first test in a file pays env init + full module transform, and page
    // suites (SessionsPage) legitimately run 3.5-4.5s on green CI runners —
    // right against vitest's 5s default, so a loaded runner tips them into a
    // timeout (main run 34600757569: 5079ms). Same headroom rationale as
    // apps/desktop/vitest.config.ts; genuinely hung tests still fail.
    testTimeout: 15_000,
    // React 19 ships `act` only in its development build, and the jsdom suites
    // here drive components through `act()`. A shell that already exports
    // NODE_ENV=production (the one that launched the app, a packaged CI step,
    // an editor terminal) is inherited by vitest, `React.act` resolves to
    // undefined, and every `createRoot` test in this project fails with
    // "act is not a function". Pin the worker to development so the suite is
    // independent of the ambient environment; app runtime and production
    // builds are untouched.
    env: {
      NODE_ENV: "development",
    },
  },
});
