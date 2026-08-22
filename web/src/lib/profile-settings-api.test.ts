import { beforeEach, describe, expect, it, vi } from "vitest";

const reloadMocks = vi.hoisted(() => ({
  attemptDashboardTokenReloadOnce: vi.fn(() => false),
  clearDashboardTokenReloadAttempt: vi.fn(),
}));

vi.mock("./dashboard-auth-reload", () => reloadMocks);

import { api } from "./api";

function response(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("profile settings API", () => {
  beforeEach(() => {
    vi.stubGlobal("window", { __HERMES_SESSION_TOKEN__: "test-token" });
    reloadMocks.clearDashboardTokenReloadAttempt.mockReset();
  });

  it("writes model and reasoning together without fallback data", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => response({
      ok: true,
      provider: "provider-a",
      model: "model-b",
      reasoning_effort: "high",
    }));
    vi.stubGlobal("fetch", fetchMock);

    await api.setProfileSettings("worker", "provider-a", "model-b", "high");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/profiles/worker/settings",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ provider: "provider-a", model: "model-b", effort: "high" }),
      }),
    );
  });

  it("round-trips ordered fallback entries through the dedicated endpoint", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => response({ ok: true, fallbacks: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const fallbacks = [
      {
        source_index: 1,
        source_provider: "provider-a",
        source_model: "model-a",
        source_base_url: "https://a.example/v1",
        source_api_mode: null,
        provider: "provider-b",
        model: "model-b",
        reasoning_effort: "low",
        base_url: null,
        api_mode: null,
      },
    ];

    await api.updateProfileFallbacks("worker", fallbacks);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/profiles/worker/fallbacks",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ fallbacks }),
      }),
    );
  });
});
