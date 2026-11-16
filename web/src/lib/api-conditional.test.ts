// @vitest-environment jsdom
/**
 * #C3 (t_406dbcb3): conditional GETs on the dashboard's session list.
 *
 * The server stamps an ETag and answers a matching If-None-Match with 304;
 * the client must (a) send the saved validator, (b) serve the cached payload
 * on 304 without ever parsing a body, and (c) adopt a fresh ETag + fresh body
 * the moment the list actually changes. Pins live on api.getSessions — the
 * one surface the reconnect refresh rides.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  api,
  fetchJSON,
  HTTP_NOT_MODIFIED,
  __resetConditionalCacheForTests,
} from "./api";

beforeEach(() => {
  __resetConditionalCacheForTests();
  Object.defineProperty(window, "__HERMES_SESSION_TOKEN__", {
    configurable: true,
    value: "test-token",
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

function sessionsBody(ids: string[]) {
  return {
    sessions: ids.map((id) => ({ id })),
    total: ids.length,
    limit: 20,
    offset: 0,
    storage: {},
  };
}

describe("fetchJSON conditional support", () => {
  it("returns the NOT_MODIFIED sentinel for a 304 and never parses a body", async () => {
    const fetchMock = vi.fn<typeof fetch>(
      async () => new Response(null, { status: 304 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchJSON("/api/sessions?limit=20", undefined, {
      ifNoneMatch: '"abc"',
    });

    expect(result).toBe(HTTP_NOT_MODIFIED);
    expect(fetchMock.mock.calls[0][1]?.headers).toBeInstanceOf(Headers);
    expect(
      (fetchMock.mock.calls[0][1]?.headers as Headers).get("if-none-match"),
    ).toBe('"abc"');
  });

  it("reports the response ETag to the caller on 200", async () => {
    const fetchMock = vi.fn<typeof fetch>(
      async () =>
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "Content-Type": "application/json", ETag: '"v1"' },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const seen: Array<string | null> = [];
    await fetchJSON("/api/sessions?limit=20", undefined, {
      onEtag: (etag) => seen.push(etag),
    });

    expect(seen).toEqual(['"v1"']);
  });
});

describe("api.getSessions conditional refresh (#C3)", () => {
  it("sends the saved ETag and serves the cached payload on 304", async () => {
    const first = sessionsBody(["a", "b"]);
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(first), {
          status: 200,
          headers: { "Content-Type": "application/json", ETag: '"e1"' },
        }),
      )
      .mockImplementationOnce(async () => new Response(null, { status: 304 }));
    vi.stubGlobal("fetch", fetchMock);

    const a = await api.getSessions(20, 0, undefined as never);
    const b = await api.getSessions(20, 0, undefined as never);

    expect(a.sessions.map((s) => s.id)).toEqual(["a", "b"]);
    expect(b).toBe(a); // the SAME cached value — zero re-parse, zero re-render churn

    // Second request carried the validator the first response stamped.
    const secondHeaders = fetchMock.mock.calls[1][1]?.headers as Headers;
    expect(secondHeaders.get("if-none-match")).toBe('"e1"');
  });

  it("adopts the fresh body and validator when the list changed", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(sessionsBody(["a"])), {
          status: 200,
          headers: { "Content-Type": "application/json", ETag: '"e1"' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(sessionsBody(["a", "b"])), {
          status: 200,
          headers: { "Content-Type": "application/json", ETag: '"e2"' },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await api.getSessions(20, 0, undefined as never);
    const changed = await api.getSessions(20, 0, undefined as never);

    expect(changed.sessions.map((s) => s.id)).toEqual(["a", "b"]);

    // And the NEW validator is what the next refresh sends.
    const third = new Response(null, { status: 304 });
    fetchMock.mockResolvedValueOnce(third);
    const cachedAgain = await api.getSessions(20, 0, undefined as never);

    expect(cachedAgain).toBe(changed);
    const thirdHeaders = fetchMock.mock.calls[2][1]?.headers as Headers;
    expect(thirdHeaders.get("if-none-match")).toBe('"e2"');
  });

  it("an error response clears nothing and never reads as not-modified", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(sessionsBody(["a"])), {
          status: 200,
          headers: { "Content-Type": "application/json", ETag: '"e1"' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "boom" }), { status: 500 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const first = await api.getSessions(20, 0, undefined as never);
    await expect(api.getSessions(20, 0, undefined as never)).rejects.toThrow();

    // The failure did not poison the cache: a later hit still revalidates.
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 304 }));
    const recovered = await api.getSessions(20, 0, undefined as never);
    expect(recovered).toBe(first);
  });
});
