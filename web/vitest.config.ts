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
