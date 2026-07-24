import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@nous-research/ui/ui/components/button": path.resolve(__dirname, "./src/test/noui-mock.tsx"),
      "@nous-research/ui/ui/components/badge": path.resolve(__dirname, "./src/test/noui-mock.tsx"),
      "@nous-research/ui/ui/components/list-item": path.resolve(__dirname, "./src/test/noui-list-item-mock.tsx"),
      "@nous-research/ui/ui/components/spinner": path.resolve(__dirname, "./src/test/noui-spinner-mock.tsx"),
      "@nous-research/ui/ui/components/select": path.resolve(__dirname, "./src/test/noui-select-mock.tsx"),
      "@nous-research/ui/dist/utils": path.resolve(__dirname, "./src/test/noui-utils-mock.ts"),
      "@nous-research/ui/ui/components/switch": path.resolve(__dirname, "./src/test/noui-switch-mock.tsx"),
      "@nous-research/ui/ui/components/command-block": path.resolve(__dirname, "./src/test/noui-command-block-mock.tsx"),
      "@nous-research/ui/ui/components/segmented": path.resolve(__dirname, "./src/test/noui-segmented-mock.tsx"),
      "@nous-research/ui/ui/components/stats": path.resolve(__dirname, "./src/test/noui-stats-mock.tsx"),
      "@nous-research/ui/ui/components/selection-switcher": path.resolve(__dirname, "./src/test/noui-selection-switcher-mock.tsx"),
      "@nous-research/ui/ui/components/tabs": path.resolve(__dirname, "./src/test/noui-tabs-mock.tsx"),
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
