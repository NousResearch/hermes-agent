import { afterEach, describe, expect, it, vi } from "vitest";

import { api, setManagementProfile } from "./api";

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

describe("Weixin onboarding profile scoping", () => {
  afterEach(() => {
    setManagementProfile("");
  });

  it("appends ?profile= to Weixin onboarding start when a management profile is set", async () => {
    vi.stubGlobal("window", {});
    setManagementProfile("coder");

    const fetchMock = jsonFetchMock({ session_id: "s1", state: "starting", expires_at: 0 });
    vi.stubGlobal("fetch", fetchMock);

    await api.startWeixinOnboarding({});

    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("profile=coder");
    expect(url).toContain("/api/messaging/weixin/onboarding/start");
  });

  it("appends ?profile= to Weixin onboarding status when a management profile is set", async () => {
    vi.stubGlobal("window", {});
    setManagementProfile("coder");

    const fetchMock = jsonFetchMock({ session_id: "s1", state: "waiting", expires_at: 0 });
    vi.stubGlobal("fetch", fetchMock);

    await api.getWeixinOnboardingStatus("s1");

    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("profile=coder");
    expect(url).toContain("/api/messaging/weixin/onboarding/s1/status");
  });

  it("appends ?profile= to Weixin onboarding apply when a management profile is set", async () => {
    vi.stubGlobal("window", {});
    setManagementProfile("coder");

    const fetchMock = jsonFetchMock({ ok: true, platform: "weixin", account_id: "a1" });
    vi.stubGlobal("fetch", fetchMock);

    await api.applyWeixinOnboarding("s1", {});

    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("profile=coder");
    expect(url).toContain("/api/messaging/weixin/onboarding/s1/apply");
  });

  it("appends ?profile= to Weixin onboarding cancel when a management profile is set", async () => {
    vi.stubGlobal("window", {});
    setManagementProfile("coder");

    const fetchMock = jsonFetchMock({ ok: true });
    vi.stubGlobal("fetch", fetchMock);

    await api.cancelWeixinOnboarding("s1");

    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("profile=coder");
    expect(url).toContain("/api/messaging/weixin/onboarding/s1");
  });
});
