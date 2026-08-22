import { describe, expect, it } from "vitest";
import { resolvePageTitle } from "./resolve-page-title";
import type { Translations } from "@/i18n/types";

// Minimal translations stub — only the fields resolvePageTitle touches.
const t = {
  app: {
    webUi: "Web UI",
    nav: {
      analytics: "Analytics",
      chat: "Chat",
      config: "Config",
      cron: "Cron",
      documentation: "Documentation",
      keys: "Keys",
      logs: "Logs",
      models: "Models",
      profiles: "Profiles",
      plugins: "Plugins",
      sessions: "Sessions",
      skills: "Skills",
    },
  },
} as unknown as Translations;

describe("resolvePageTitle", () => {
  it("uses i18n nav keys for translated routes", () => {
    expect(resolvePageTitle("/sessions", t, [])).toBe("Sessions");
    expect(resolvePageTitle("/env", t, [])).toBe("Keys");
  });

  it("renders initialisms and literal labels correctly", () => {
    // Regression: the naive capitalize fallback produced "Mcp".
    expect(resolvePageTitle("/mcp", t, [])).toBe("MCP");
    expect(resolvePageTitle("/system", t, [])).toBe("System");
    expect(resolvePageTitle("/channels", t, [])).toBe("Channels");
    expect(resolvePageTitle("/webhooks", t, [])).toBe("Webhooks");
    expect(resolvePageTitle("/pairing", t, [])).toBe("Pairing");
    expect(resolvePageTitle("/files", t, [])).toBe("Files");
  });

  it("prefers plugin tab labels", () => {
    expect(
      resolvePageTitle("/kanban", t, [{ path: "/kanban", label: "Kanban" }]),
    ).toBe("Kanban");
  });

  it("falls back to capitalized path segment for unknown routes", () => {
    expect(resolvePageTitle("/whatever", t, [])).toBe("Whatever");
  });

  it("treats root as sessions and trailing slashes as equivalent", () => {
    expect(resolvePageTitle("/", t, [])).toBe("Sessions");
    expect(resolvePageTitle("/mcp/", t, [])).toBe("MCP");
  });

  it("prefers a plugin's own label when it overrides the root path (#80891)", () => {
    // A plugin with tab: { path: "/example", override: "/" } is registered
    // in pluginTabs under the "/" key (the override target's path, not
    // the plugin's own) -- resolvePageTitle must consult pluginTabs
    // before special-casing "/" to the hardcoded "Sessions" title.
    expect(
      resolvePageTitle("/", t, [{ path: "/", label: "Example Plugin" }]),
    ).toBe("Example Plugin");
  });

  it("still falls back to Sessions at root when no plugin overrides it", () => {
    // Sanity: a plugin registered at a DIFFERENT path must not affect the
    // root title -- behavior for everyone else is unchanged.
    expect(
      resolvePageTitle("/", t, [{ path: "/kanban", label: "Kanban" }]),
    ).toBe("Sessions");
  });
});
