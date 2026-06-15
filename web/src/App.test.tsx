import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "@/App";
import { renderWithAppProviders } from "@/test/render";

vi.mock("@/lib/api", () => ({
  getManagementProfile: vi.fn(() => ""),
  setManagementProfile: vi.fn(),
  buildWsAuthParam: vi.fn().mockResolvedValue(["", ""]),
  buildWsUrl: vi.fn().mockResolvedValue("ws://localhost/api/events"),
  api: {
    getMemory: vi.fn().mockResolvedValue({
      builtin_active: true,
      provider: "",
      provider_label: "built-in only",
      directory: "/tmp/memories",
      stores: {
        user: { path: "/tmp/memories/USER.md", entry_count: 0, char_count: 0, char_limit: 1375, updated_at: null, entries: [] },
        memory: { path: "/tmp/memories/MEMORY.md", entry_count: 0, char_count: 0, char_limit: 2200, updated_at: null, entries: [] },
      },
    }),
    addMemoryEntry: vi.fn(),
    updateMemoryEntry: vi.fn(),
    removeMemoryEntry: vi.fn(),
    getAuthMe: vi.fn().mockRejectedValue(new Error("unauthorized")),
    logout: vi.fn(),
    getStatus: vi.fn().mockResolvedValue({}),
    getProfiles: vi.fn().mockResolvedValue({ profiles: [] }),
    getActiveProfile: vi.fn().mockResolvedValue({ current: "default" }),
    getSessions: vi.fn().mockResolvedValue({ sessions: [], total: 0, limit: 20, offset: 0 }),
    getSessionMessages: vi.fn(),
    getEmptySessionsCount: vi.fn().mockResolvedValue({ count: 0 }),
    deleteEmptySessions: vi.fn(),
    getSessionStats: vi.fn().mockResolvedValue({
      total_sessions: 0,
      total_messages: 0,
      total_size_bytes: 0,
      sources: [],
    }),
    deleteSession: vi.fn(),
    getLogs: vi.fn(),
    getAnalytics: vi.fn(),
    getConfig: vi.fn().mockResolvedValue({ dashboard: {} }),
    getDefaults: vi.fn(),
    getSchema: vi.fn(),
    getModelInfo: vi.fn().mockResolvedValue({ model: "gpt-test", capabilities: {} }),
    saveConfig: vi.fn(),
    getConfigRaw: vi.fn(),
    saveConfigRaw: vi.fn(),
    getEnvVars: vi.fn(),
    setEnvVar: vi.fn(),
    deleteEnvVar: vi.fn(),
    revealEnvVar: vi.fn(),
    getCronJobs: vi.fn(),
    createCronJob: vi.fn(),
    pauseCronJob: vi.fn(),
    resumeCronJob: vi.fn(),
    triggerCronJob: vi.fn(),
    deleteCronJob: vi.fn(),
    getSkills: vi.fn().mockResolvedValue([]),
    toggleSkill: vi.fn(),
    getToolsets: vi.fn().mockResolvedValue([]),
    getThemes: vi.fn().mockResolvedValue({ themes: [], active: "default" }),
    setTheme: vi.fn().mockResolvedValue({ ok: true, theme: "default" }),
    getPlugins: vi.fn().mockResolvedValue([]),
    searchSessions: vi.fn().mockResolvedValue({ results: [] }),
    getOAuthProviders: vi.fn(),
    disconnectOAuthProvider: vi.fn(),
    startOAuthLogin: vi.fn(),
    submitOAuthCode: vi.fn(),
    pollOAuthSession: vi.fn(),
    cancelOAuthSession: vi.fn(),
  },
}));

describe("App", () => {
  it("shows Memory to the right of Sessions in the header nav", () => {
    renderWithAppProviders(<App />, { routerProps: { initialEntries: ["/memory"] } });

    const links = screen.getAllByRole("link");
    const labels = links.map((link) => link.textContent?.trim()).filter(Boolean);

    expect(labels).toContain("Sessions");
    expect(labels).toContain("Memory");
    expect(labels.indexOf("Sessions")).toBeLessThan(labels.indexOf("Memory"));
  });

  it("renders the Memory page when routed to /memory", async () => {
    renderWithAppProviders(<App />, { routerProps: { initialEntries: ["/memory"] } });

    expect((await screen.findAllByRole("heading", { name: /memory/i })).length).toBeGreaterThan(0);
  });
});
