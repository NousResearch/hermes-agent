import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it, vi } from "vitest";

const WORKER_SOURCE = readFileSync(
  new URL("../../public/service-worker.js", import.meta.url),
  "utf8",
);

interface FakeRequest {
  destination: string;
  method: string;
  mode: string;
  url: string;
}

interface WorkerEvent {
  request?: FakeRequest;
  respondWith?: (value: Promise<unknown>) => void;
  waitUntil?: (value: Promise<unknown>) => void;
}

type WorkerListener = (event: WorkerEvent) => void;

function request(
  relativePath: string,
  overrides: Partial<FakeRequest> = {},
): FakeRequest {
  return {
    destination: "script",
    method: "GET",
    mode: "same-origin",
    url: `https://hermes.test/hermes/${relativePath}`,
    ...overrides,
  };
}

function createWorkerHarness(options?: {
  cacheNames?: string[];
  responseOk?: boolean;
  responseType?: string;
}) {
  const listeners = new Map<string, WorkerListener>();
  const response = {
    ok: options?.responseOk ?? true,
    type: options?.responseType ?? "basic",
    clone: vi.fn(() => ({ cached: true })),
  };
  const cache = {
    match: vi.fn(async () => undefined),
    put: vi.fn(async () => undefined),
  };
  const cacheStorage = {
    delete: vi.fn(async () => true),
    keys: vi.fn(async () => options?.cacheNames ?? []),
    open: vi.fn(async () => cache),
  };
  const fetchMock = vi.fn(async () => response);
  const skipWaiting = vi.fn(async () => undefined);
  const claim = vi.fn(async () => undefined);
  const workerSelf = {
    addEventListener: (type: string, listener: WorkerListener) => {
      listeners.set(type, listener);
    },
    clients: { claim },
    location: { origin: "https://hermes.test" },
    registration: { scope: "https://hermes.test/hermes/" },
    skipWaiting,
  };

  vm.runInNewContext(WORKER_SOURCE, {
    Promise,
    Set,
    URL,
    caches: cacheStorage,
    fetch: fetchMock,
    self: workerSelf,
  });

  async function dispatchFetch(fakeRequest: FakeRequest) {
    const respondWith = vi.fn();
    listeners.get("fetch")?.({
      request: fakeRequest,
      respondWith,
    });
    const handled = respondWith.mock.calls[0]?.[0] as
      | Promise<unknown>
      | undefined;
    if (handled) await handled;
    return respondWith;
  }

  async function dispatchLifetime(type: "activate" | "install") {
    let lifetime: Promise<unknown> | undefined;
    listeners.get(type)?.({
      waitUntil: (value) => {
        lifetime = value;
      },
    });
    if (lifetime) await lifetime;
  }

  return {
    cache,
    cacheStorage,
    claim,
    dispatchFetch,
    dispatchLifetime,
    fetchMock,
    response,
    skipWaiting,
  };
}

describe("Hermes PWA service-worker allowlist", () => {
  it.each([
    ["manifest.webmanifest", "manifest"],
    ["icons/hermes-192.png", "image"],
    ["icons/hermes-512.png", "image"],
    ["assets/index-BxxFYFhq.js", "script"],
    ["assets/index-BbJ3HYO2.css", "style"],
    ["assets/Mondwest-Regular-CWscgue7.woff2", "font"],
  ])("cache-handles the public static file %s", async (path, destination) => {
    const worker = createWorkerHarness();

    const respondWith = await worker.dispatchFetch(
      request(path, { destination }),
    );

    expect(respondWith).toHaveBeenCalledOnce();
    expect(worker.fetchMock).toHaveBeenCalledOnce();
    expect(worker.cache.put).toHaveBeenCalledOnce();
  });

  it.each([
    ["sessions", { destination: "document", mode: "navigate" }],
    ["api/status", {}],
    ["api/auth/me", {}],
    ["api/pty", {}],
    ["api/ws", {}],
    ["api/files/download", {}],
    ["dashboard-plugins/example/index.js", {}],
    ["logs", {}],
    ["files", {}],
    ["skills", {}],
    ["service-worker.js", {}],
    ["assets/index.js", {}],
  ])("does not intercept dynamic or unhashed path %s", async (path, overrides) => {
    const worker = createWorkerHarness();

    const respondWith = await worker.dispatchFetch(request(path, overrides));

    expect(respondWith).not.toHaveBeenCalled();
    expect(worker.fetchMock).not.toHaveBeenCalled();
    expect(worker.cache.put).not.toHaveBeenCalled();
  });

  it("rejects non-GET, cross-origin, and query-bearing requests", async () => {
    const worker = createWorkerHarness();

    const attempts = [
      request("assets/index-BxxFYFhq.js", { method: "POST" }),
      request("assets/index-BxxFYFhq.js?token=secret"),
      request("assets/index-BxxFYFhq.js", {
        url: "https://other.test/hermes/assets/index-BxxFYFhq.js",
      }),
    ];

    for (const attempt of attempts) {
      const respondWith = await worker.dispatchFetch(attempt);
      expect(respondWith).not.toHaveBeenCalled();
    }
    expect(worker.fetchMock).not.toHaveBeenCalled();
  });

  it("does not cache unsuccessful or non-basic responses", async () => {
    const unsuccessful = createWorkerHarness({ responseOk: false });
    await unsuccessful.dispatchFetch(request("assets/index-BxxFYFhq.js"));
    expect(unsuccessful.cache.put).not.toHaveBeenCalled();

    const crossOriginResponse = createWorkerHarness({ responseType: "cors" });
    await crossOriginResponse.dispatchFetch(
      request("assets/index-BxxFYFhq.js"),
    );
    expect(crossOriginResponse.cache.put).not.toHaveBeenCalled();
  });
});

describe("Hermes PWA service-worker lifecycle", () => {
  it("activates immediately and claims clients", async () => {
    const worker = createWorkerHarness();

    await worker.dispatchLifetime("install");
    await worker.dispatchLifetime("activate");

    expect(worker.skipWaiting).toHaveBeenCalledOnce();
    expect(worker.claim).toHaveBeenCalledOnce();
  });

  it("deletes only older Hermes PWA static caches", async () => {
    const worker = createWorkerHarness({
      cacheNames: [
        "hermes-pwa-static-v0",
        "hermes-pwa-static-v1",
        "hermes-pwa-data-v1",
        "workbox-precache-v1",
      ],
    });

    await worker.dispatchLifetime("activate");

    expect(worker.cacheStorage.delete).toHaveBeenCalledExactlyOnceWith(
      "hermes-pwa-static-v0",
    );
  });
});
