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
    // TEMPORARY: this job moved from a 32-core Larger Runner to the
    // standard 4-core `ubuntu-latest` (GitHub Free has no Larger Runners),
    // and vitest's 5000ms default is too tight for that much less CPU per
    // test. 15s gives headroom without masking genuinely hung tests.
    testTimeout: 15_000,
  },
});
