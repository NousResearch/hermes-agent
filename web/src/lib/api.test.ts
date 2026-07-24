import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./api";

const SESSION_HEADER = "X-Hermes-Session-Token";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function jsonFetchMock(body: unknown = { ok: true }) {
  return vi.fn<typeof fetch>(
    async () =>
      new Response(JSON.stringify(body), {
        headers: { "Content-Type": "application/json" },
        status: 200,
      }),
  );
}

describe("api.getModelOptions", () => {
  it("requests a live model refresh when asked", async () => {
    vi.stubGlobal("window", {});

    const fetchMock = jsonFetchMock({ providers: [] });
    vi.stubGlobal("fetch", fetchMock);

    await api.getModelOptions({ refresh: true });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/model/options?refresh=1&include_unconfigured=1",
      expect.objectContaining({ credentials: "include" }),
    );
  });

  it("keeps explicit profile scoping when refreshing", async () => {
    vi.stubGlobal("window", {});

    const fetchMock = jsonFetchMock({ providers: [] });
    vi.stubGlobal("fetch", fetchMock);

    await api.getModelOptions({ profile: "default", refresh: true });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/model/options?profile=default&refresh=1&include_unconfigured=1",
      expect.objectContaining({ credentials: "include" }),
    );
  });
});

describe("api Team Proposal gated launch helpers", () => {
  it("requests a read-only plan preview before launch conversion", async () => {
    vi.stubGlobal("window", {});
    const fetchMock = jsonFetchMock({ preview_hash: "hash-123", plan: { tasks: [] } });
    vi.stubGlobal("fetch", fetchMock);

    await api.getTeamProposalPlanPreview("proposal/with spaces");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/team-proposals/proposal%2Fwith%20spaces/plan-preview",
      expect.objectContaining({ credentials: "include" }),
    );
    expect(fetchMock.mock.calls[0][1]?.method).toBeUndefined();
  });

  it("requires an explicit approval payload before converting a proposal to a plan", async () => {
    vi.stubGlobal("window", {});
    const fetchMock = jsonFetchMock({ ok: true });
    vi.stubGlobal("fetch", fetchMock);

    await api.approveTeamProposalMinStep("proposal-1", {
      action_type: "plan",
      board: "mission-control",
      confirmed_preview_hash: "hash-123",
      note: "human confirmed preview",
    });
    await api.convertTeamProposalToPlan("proposal-1", "hash-123", "mission-control");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/team-proposals/proposal-1/approve-min-step",
      expect.objectContaining({ method: "POST" }),
    );
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      action_type: "plan",
      board: "mission-control",
      confirmed_preview_hash: "hash-123",
      note: "human confirmed preview",
    });
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/team-proposals/proposal-1/convert-to-plan",
      expect.objectContaining({ method: "POST" }),
    );
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual({
      board: "mission-control",
      confirmed_preview_hash: "hash-123",
    });
  });

  it("keeps chief review registry-only actions separate from conversion", async () => {
    vi.stubGlobal("window", {});
    const fetchMock = jsonFetchMock({ ok: true });
    vi.stubGlobal("fetch", fetchMock);

    await api.reviewTeamProposalAsChief("proposal-1", "shortlist", "standby only");
    await api.updateTeamProposal("proposal-1", { acceptance: "clear acceptance", title: "Refined title" });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/team-proposals/proposal-1/chief-review",
      expect.objectContaining({ method: "POST" }),
    );
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      action: "shortlist",
      note: "standby only",
    });
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/team-proposals/proposal-1",
      expect.objectContaining({ method: "PATCH" }),
    );
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual({
      acceptance: "clear acceptance",
      title: "Refined title",
    });
  });
});

describe("api OAuth helpers", () => {
  it("starts OAuth login in gated mode without requiring an injected session token", async () => {
    vi.stubGlobal("window", { __HERMES_AUTH_REQUIRED__: true });
    const fetchMock = jsonFetchMock({
      flow: "device_code",
      session_id: "oauth-session",
    });
    vi.stubGlobal("fetch", fetchMock);

    await api.startOAuthLogin("openai-codex");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/providers/oauth/openai-codex/start",
      expect.objectContaining({
        body: "{}",
        credentials: "include",
        method: "POST",
      }),
    );
    const headers = fetchMock.mock.calls[0][1]?.headers as Headers;
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(headers.has(SESSION_HEADER)).toBe(false);
  });

  it("still sends the injected session token for OAuth login in loopback mode", async () => {
    vi.stubGlobal("window", { __HERMES_SESSION_TOKEN__: "loopback-token" });
    const fetchMock = jsonFetchMock({
      flow: "device_code",
      session_id: "oauth-session",
    });
    vi.stubGlobal("fetch", fetchMock);

    await api.startOAuthLogin("openai-codex");

    const headers = fetchMock.mock.calls[0][1]?.headers as Headers;
    expect(headers.get(SESSION_HEADER)).toBe("loopback-token");
  });

  it("runs provider auth mutations in gated mode via cookie auth", async () => {
    vi.stubGlobal("window", { __HERMES_AUTH_REQUIRED__: true });
    const fetchMock = jsonFetchMock({ ok: true });
    vi.stubGlobal("fetch", fetchMock);

    await api.disconnectOAuthProvider("anthropic");
    await api.submitOAuthCode("anthropic", "oauth-session", "code-123");
    await api.cancelOAuthSession("oauth-session");
    await api.revealEnvVar("OPENAI_API_KEY");

    for (const call of fetchMock.mock.calls) {
      const init = call[1] as RequestInit;
      expect(init.credentials).toBe("include");
      expect((init.headers as Headers).has(SESSION_HEADER)).toBe(false);
    }
  });
});
