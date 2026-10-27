// @vitest-environment jsdom
// Server-preference I/O: the picker → server write and the load-time read.
// `fetchJSON` is mocked so the assertions are about the request we build
// (URL, method, body) and the fail-soft behaviour — not the HTTP stack.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const fetchJSONMock = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchJSON: (...args: unknown[]) => fetchJSONMock(...args),
}));

import { fetchServerLocale, persistServerLocale } from "./locale-preference";

const LOCALE_ENDPOINT = "/api/dashboard/locale";

beforeEach(() => {
  fetchJSONMock.mockReset();
  // The guard only talks to the backend when the server-rendered bootstrap ran.
  (window as { __HERMES_SESSION_TOKEN__?: string }).__HERMES_SESSION_TOKEN__ = "test-token";
});

afterEach(() => {
  delete (window as { __HERMES_SESSION_TOKEN__?: string }).__HERMES_SESSION_TOKEN__;
  delete (window as { __HERMES_AUTH_REQUIRED__?: boolean }).__HERMES_AUTH_REQUIRED__;
});

describe("persistServerLocale (picker → server write)", () => {
  it("PUTs the chosen locale to the dashboard locale endpoint", async () => {
    fetchJSONMock.mockResolvedValue({ ok: true, locale: "zh" });

    await persistServerLocale("zh");

    expect(fetchJSONMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchJSONMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(LOCALE_ENDPOINT);
    expect(init.method).toBe("PUT");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(String(init.body))).toEqual({ locale: "zh" });
  });

  it("stays silent when the write fails (localStorage is the fallback)", async () => {
    fetchJSONMock.mockRejectedValue(new Error("offline"));
    await expect(persistServerLocale("ja")).resolves.toBeUndefined();
  });

  it("does not call the backend when the SPA was not served by it", async () => {
    delete (window as { __HERMES_SESSION_TOKEN__?: string }).__HERMES_SESSION_TOKEN__;

    await persistServerLocale("de");

    expect(fetchJSONMock).not.toHaveBeenCalled();
  });
});

describe("fetchServerLocale (load-time read)", () => {
  it("GETs the endpoint and returns the saved locale", async () => {
    fetchJSONMock.mockResolvedValue({ locale: "zh-hant" });

    await expect(fetchServerLocale()).resolves.toBe("zh-hant");
    expect(fetchJSONMock).toHaveBeenCalledWith(LOCALE_ENDPOINT);
  });

  it("ignores an unset or unsupported server value", async () => {
    fetchJSONMock.mockResolvedValue({ locale: null });
    await expect(fetchServerLocale()).resolves.toBeNull();

    fetchJSONMock.mockResolvedValue({ locale: "nl" });
    await expect(fetchServerLocale()).resolves.toBeNull();
  });

  it("returns null instead of throwing when the fetch fails", async () => {
    fetchJSONMock.mockRejectedValue(new Error("boom"));
    await expect(fetchServerLocale()).resolves.toBeNull();
  });
});
