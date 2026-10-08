import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  oxc: { jsx: { runtime: "automatic" } },
  resolve: {
    alias: {
      // Use the site's React, not the workspace runner's hoisted version.
      ...Object.fromEntries(["react", "react-dom"].map((name) => [
        name, fileURLToPath(new URL(`./node_modules/${name}`, import.meta.url)),
      ])),
      "@theme/Layout": fileURLToPath(new URL("./tests/Layout.tsx", import.meta.url)),
      ...Object.fromEntries(["Link", "router", "useBaseUrl"].map((name) => [
        `@docusaurus/${name}`,
        fileURLToPath(new URL(`./node_modules/@docusaurus/core/lib/client/exports/${name}.js`, import.meta.url)),
      ])),
    },
  },
  test: { include: ["tests/**/*.test.{ts,tsx}"], environment: "jsdom" },
});
