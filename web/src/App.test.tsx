import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "@/App";
import { renderWithAppProviders } from "@/test/render";

vi.mock("@/lib/api", () => ({
  api: {
    getMemory: vi.fn().mockResolvedValue({
      builtin_active: true,
      provider: "",
      provider_label: "built-in only",
      directory: "/tmp/memories",
      note: "Saved immediately. Changes apply to future sessions; current sessions keep their existing snapshot.",
      stores: {
        user: { path: "/tmp/memories/USER.md", entry_count: 0, char_count: 0, char_limit: 1375, updated_at: null, entries: [] },
        memory: { path: "/tmp/memories/MEMORY.md", entry_count: 0, char_count: 0, char_limit: 2200, updated_at: null, entries: [] },
      },
    }),
    addMemoryEntry: vi.fn(),
    updateMemoryEntry: vi.fn(),
    removeMemoryEntry: vi.fn(),
    getStatus: vi.fn().mockResolvedValue({}),
    getSessions: vi.fn().mockResolvedValue({ sessions: [], total: 0, limit: 20, offset: 0 }),
    getSessionMessages: vi.fn(),
    deleteSession: vi.fn(),
    getLogs: vi.fn(),
    getAnalytics: vi.fn(),
    getConfig: vi.fn(),
    getDefaults: vi.fn(),
    getSchema: vi.fn(),
    getModelInfo: vi.fn(),
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
  it("shows Memory in the header nav", () => {
    renderWithAppProviders(<App />);

    expect(screen.getByRole("link", { name: /memory/i })).toBeInTheDocument();
  });

  it("renders the Memory page when routed to /memory", async () => {
    renderWithAppProviders(<App />, { routerProps: { initialEntries: ["/memory"] } });

    expect((await screen.findAllByRole("heading", { name: /memory/i })).length).toBeGreaterThan(0);
  });
});
