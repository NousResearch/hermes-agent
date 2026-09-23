// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchJSON } from "./api";
import { API_UNREACHABLE_MESSAGE, ApiError, errorMessage, extractDetail } from "./api-error";

vi.mock("./dashboard-auth-reload", () => ({
  attemptDashboardTokenReloadOnce: vi.fn(() => false),
  clearDashboardTokenReloadAttempt: vi.fn(),
}));

beforeEach(() => {
  Object.defineProperty(window, "__HERMES_SESSION_TOKEN__", {
    configurable: true,
    value: "tok",
    writable: true,
  });
  Object.defineProperty(window, "__HERMES_AUTH_REQUIRED__", {
    configurable: true,
    value: false,
    writable: true,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("fetchJSON error contract", () => {
  it("throws an ApiError whose message is the JSON detail, not the status or body", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(
        async () =>
          new Response(JSON.stringify({ detail: "Server 'foo' not found" }), {
            headers: { "Content-Type": "application/json" },
            status: 404,
          }),
      ),
    );

    const err = await fetchJSON("/api/mcp/servers/foo").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const apiErr = err as ApiError;
    expect(apiErr.message).toBe("Server 'foo' not found");
    expect(apiErr.message).not.toMatch(/404|detail|\{/);
    expect(apiErr.status).toBe(404);
    expect(apiErr.details).toContain("404");
    // The toast pattern `\`Error: ${e}\`` must no longer double-prefix.
    expect(`${errorMessage(apiErr)}`).not.toMatch(/^Error:/);
  });

  it("maps a detail-less status to a plain sentence with no HTTP code lead", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(async () => new Response("", { status: 500 })),
    );
    const err = (await fetchJSON("/api/status").catch((e: unknown) => e)) as ApiError;
    expect(err.message).not.toMatch(/^\d{3}/);
  });

  it("turns a network failure into the 'is hermes dashboard running' sentence", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    const err = (await fetchJSON("/api/status").catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe(API_UNREACHABLE_MESSAGE);
    expect(err.message).not.toContain("Failed to fetch");
    expect(err.status).toBe(0);
    expect(err.body).toContain("Failed to fetch");
  });
});

describe("extractDetail", () => {
  it("joins FastAPI validation errors and ignores HTML bodies", () => {
    expect(
      extractDetail(JSON.stringify({ detail: [{ msg: "field required", loc: ["body", "x"] }] })),
    ).toBe("field required");
    expect(extractDetail("<html><body>502 Bad Gateway</body></html>")).toBeNull();
    expect(extractDetail("")).toBeNull();
  });
});
