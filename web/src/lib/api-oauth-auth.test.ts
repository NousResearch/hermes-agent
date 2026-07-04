import { afterEach, describe, expect, it, vi } from "vitest";

const SESSION_HEADER = "X-Hermes-Session-Token";

function stubWindow(overrides: Record<string, unknown> = {}) {
  vi.stubGlobal("window", {
    location: {
      assign: vi.fn(),
      pathname: "/",
      reload: vi.fn(),
      search: "",
    },
    ...overrides,
  });
}

function stubJsonFetch(payload: unknown = { ok: true }) {
  const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => ({
    ok: true,
    status: 200,
    json: async () => payload,
  }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("provider OAuth API auth", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it("starts provider OAuth in gated cookie-auth mode without a session token", async () => {
    stubWindow({ __HERMES_AUTH_REQUIRED__: true });
    const fetchMock = stubJsonFetch({ session_id: "oauth-session" });

    const { api } = await import("./api");

    await expect(api.startOAuthLogin("nous")).resolves.toEqual({
      session_id: "oauth-session",
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/providers/oauth/nous/start");
    expect(init.credentials).toBe("include");
    expect(new Headers(init.headers).has(SESSION_HEADER)).toBe(false);
  });

  it("lets fetchJSON attach the legacy token in loopback mode", async () => {
    stubWindow({ __HERMES_SESSION_TOKEN__: "loopback-token" });
    const fetchMock = stubJsonFetch({ ok: true });

    const { api } = await import("./api");

    await api.disconnectOAuthProvider("nous");
    await api.submitOAuthCode("nous", "session-1", "code-1");
    await api.cancelOAuthSession("session-1");

    for (const [, init] of fetchMock.mock.calls as [string, RequestInit][]) {
      expect(new Headers(init.headers).get(SESSION_HEADER)).toBe("loopback-token");
      expect(init.credentials).toBe("include");
    }
  });
});
