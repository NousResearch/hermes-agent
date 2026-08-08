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

  it("uses root dashboard plugin metadata before the Sessions fallback", () => {
    expect(
      resolvePageTitle("/", t, [{ path: "/", label: "Command Center" }]),
    ).toBe("Command Center");
  });

  it("treats root as sessions without a root plugin and trailing slashes as equivalent", () => {
    expect(resolvePageTitle("/", t, [])).toBe("Sessions");
    expect(resolvePageTitle("/mcp/", t, [])).toBe("MCP");
  });
});
