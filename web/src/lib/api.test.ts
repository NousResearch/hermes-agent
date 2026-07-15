import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DASHBOARD_UNREACHABLE_CODE,
  api,
  fetchJSON,
  isDashboardUnreachableError,
} from "./api";
import {
  getDashboardReachability,
  markDashboardReachable,
  markDashboardUnreachable,
  subscribeDashboardReachability,
} from "./dashboard-reachability";

const SESSION_HEADER = "X-Hermes-Session-Token";

afterEach(() => {
  markDashboardReachable();
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

describe("fetchJSON reachability", () => {
  it("turns network failures into a structured dashboard error", async () => {
    vi.stubGlobal("window", { location: { origin: "http://localhost:3000" } });
    const cause = new TypeError("Failed to fetch");
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockRejectedValue(cause));
    const listener = vi.fn();
    const unsubscribe = subscribeDashboardReachability(listener);

    try {
      const request = fetchJSON("/api/status");
      await expect(request).rejects.toMatchObject({
        baseUrl: "http://localhost:3000",
        code: DASHBOARD_UNREACHABLE_CODE,
        name: "DashboardUnreachableError",
      });
      await expect(request).rejects.toSatisfy(isDashboardUnreachableError);
      await expect(request).rejects.toHaveProperty("cause", cause);
      expect(getDashboardReachability()).toBe("unreachable");
      expect(listener).toHaveBeenCalledTimes(1);

      await expect(fetchJSON("/api/status")).rejects.toSatisfy(
        isDashboardUnreachableError,
      );
      expect(listener).toHaveBeenCalledTimes(1);
    } finally {
      unsubscribe();
    }
  });

  it("clears the outage state when the backend returns any HTTP response", async () => {
    vi.stubGlobal("window", {});
    markDashboardUnreachable();
    const fetchMock = vi.fn<typeof fetch>(
      async () => new Response("temporarily unavailable", { status: 503 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const listener = vi.fn();
    const unsubscribe = subscribeDashboardReachability(listener);

    try {
      const request = fetchJSON("/api/status");
      await expect(request).rejects.toThrow("503: temporarily unavailable");
      await expect(request).rejects.not.toSatisfy(isDashboardUnreachableError);
      expect(getDashboardReachability()).toBe("reachable");
      expect(listener).toHaveBeenCalledTimes(1);
    } finally {
      unsubscribe();
    }
  });

  it("preserves aborted requests without changing reachability", async () => {
    vi.stubGlobal("window", {});
    const controller = new AbortController();
    controller.abort();
    const abortError = new DOMException("This operation was aborted", "AbortError");
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockRejectedValue(abortError));

    await expect(
      fetchJSON("/api/status", { signal: controller.signal }),
    ).rejects.toBe(abortError);
    expect(getDashboardReachability()).toBe("reachable");
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
