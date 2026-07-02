import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AuthUnauthorizedError,
  fetchJSON,
  getWsTicket,
  isAuthUnauthorizedError,
} from "@/lib/api";

const fetchMock = vi.fn();

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
    clone() {
      return this;
    },
    headers: new Headers({ "content-type": "application/json" }),
  } as unknown as Response;
}

describe("dashboard API auth handling", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    window.__HERMES_SESSION_TOKEN__ = "session-token";
    window.__HERMES_AUTH_REQUIRED__ = false;
    window.__HERMES_BASE_PATH__ = "";
  });

  afterEach(() => {
    fetchMock.mockReset();
    vi.unstubAllGlobals();
    delete window.__HERMES_SESSION_TOKEN__;
    delete window.__HERMES_AUTH_REQUIRED__;
    delete window.__HERMES_BASE_PATH__;
  });

  it("attaches existing auth credentials to dashboard API fetches", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true }));

    await fetchJSON<{ ok: boolean }>("/api/profiles", { method: "POST" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/profiles",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        headers: expect.any(Headers),
      }),
    );
    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("X-Hermes-Session-Token")).toBe("session-token");
  });

  it.each([401, 403])("throws typed unauthorized errors for HTTP %s", async (status) => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "denied" }, false, status));

    await expect(fetchJSON("/api/memory/content")).rejects.toMatchObject({
      status,
      message: "Authentication required. Sign in again or retry the request.",
    });
  });

  it("recognizes unauthorized errors without matching arbitrary strings", () => {
    const error = new AuthUnauthorizedError(403, "Forbidden");

    expect(isAuthUnauthorizedError(error)).toBe(true);
    expect(isAuthUnauthorizedError(new Error("403: Forbidden"))).toBe(false);
  });

  it("uses the same auth layer for WebSocket ticket requests", async () => {
    window.__HERMES_AUTH_REQUIRED__ = true;
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "forbidden" }, false, 403));

    await expect(getWsTicket()).rejects.toBeInstanceOf(AuthUnauthorizedError);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/ws-ticket",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        headers: expect.any(Headers),
      }),
    );
  });
});
