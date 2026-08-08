import { afterEach, describe, expect, it, vi } from "vitest";

type WorkerListener = (event: {
  request: {
    method: string;
    mode: string;
    destination: string;
    url: string;
  };
  respondWith: (response: Promise<unknown>) => void;
}) => void;

const originalSelf = globalThis.self;
const originalCaches = globalThis.caches;
const originalFetch = globalThis.fetch;

async function loadFetchHandler(scope = "http://dashboard.test/hermes/") {
  const listeners = new Map<string, WorkerListener>();
  globalThis.self = {
    location: { origin: "http://dashboard.test" },
    registration: { scope },
    addEventListener(name: string, listener: WorkerListener) {
      listeners.set(name, listener);
    },
    skipWaiting() {},
    clients: { claim() {} },
  } as unknown as typeof self;
  globalThis.caches = {
    keys: async () => [],
    open: async () => ({
      match: async () => undefined,
      put() {},
    }),
  } as unknown as CacheStorage;
  globalThis.fetch = vi.fn(async () => new Response("asset")) as typeof fetch;

  vi.resetModules();
  // @ts-expect-error The production worker deliberately lives in Vite's public directory.
  await import("../public/sw.js");
  const handler = listeners.get("fetch");
  if (!handler) throw new Error("service worker did not register a fetch handler");
  return handler;
}

afterEach(() => {
  globalThis.self = originalSelf;
  globalThis.caches = originalCaches;
  globalThis.fetch = originalFetch;
});

function request(path: string, destination = "script") {
  return {
    method: "GET",
    mode: "cors",
    destination,
    url: `http://dashboard.test${path}`,
  };
}

describe("dashboard service worker cache boundary", () => {
  it("caches only declared static shell paths inside a prefixed deployment", async () => {
    const fetchHandler = await loadFetchHandler();
    const respondWith = vi.fn();

    fetchHandler({ request: request("/hermes/assets/app.js"), respondWith });

    expect(respondWith).toHaveBeenCalledOnce();
  });

  it.each([
    "/hermes/manifest.webmanifest",
    "/hermes/favicon.ico",
    "/hermes/pwa-icon-180.png",
    "/hermes/pwa-icon-192.png",
    "/hermes/pwa-icon-512.png",
    "/hermes/pwa-icon.svg",
  ])("refetches mutable PWA metadata on every request for %s", async (path) => {
    const fetchHandler = await loadFetchHandler();
    const respondWith = vi.fn();

    fetchHandler({ request: request(path, "image"), respondWith });

    expect(respondWith).not.toHaveBeenCalled();
  });

  it.each([
    ["/hermes/api/auth/me", "script"],
    ["/hermes/auth/login.js", "script"],
    ["/hermes/handoff", "script"],
    ["/hermes/chat", "script"],
    ["/hermes/chat", "document"],
  ])("does not intercept %s", async (path, destination) => {
    const fetchHandler = await loadFetchHandler();
    const respondWith = vi.fn();

    fetchHandler({ request: request(path, destination), respondWith });

    expect(respondWith).not.toHaveBeenCalled();
  });
});
