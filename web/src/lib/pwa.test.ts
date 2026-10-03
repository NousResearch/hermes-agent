import { afterEach, describe, expect, it, vi } from "vitest";
import {
  registerServiceWorker,
  serviceWorkerScope,
  serviceWorkerUrl,
} from "./pwa";

/** Minimal `navigator.serviceWorker` double recording what the module did. */
function fakeServiceWorkerContainer(
  overrides: {
    register?: (url: string, opts: { scope: string }) => Promise<unknown>;
    registrations?: { unregister: () => Promise<boolean> }[];
  } = {},
) {
  const registrations = overrides.registrations ?? [];
  return {
    register: vi.fn(
      overrides.register ?? (async () => ({ scope: "stub" })),
    ),
    getRegistrations: vi.fn(async () => registrations),
  };
}

function stubBrowser(
  serviceWorker: unknown | undefined,
  { secure = true }: { secure?: boolean } = {},
) {
  vi.stubGlobal("window", { isSecureContext: secure });
  vi.stubGlobal("navigator", serviceWorker ? { serviceWorker } : {});
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("service worker location", () => {
  it("serves the worker from the mount root so it can control every route", () => {
    // The scope must be the directory the worker script sits in — a worker
    // registered deeper than its scope cannot intercept sibling routes.
    for (const base of ["", "/hermes", "/tools/hermes"]) {
      expect(serviceWorkerUrl(base)).toBe(`${base}/sw.js`);
      expect(serviceWorkerScope(base)).toBe(`${base}/`);
      expect(serviceWorkerUrl(base).startsWith(serviceWorkerScope(base))).toBe(
        true,
      );
    }
  });

  it("keeps the worker inside the reverse-proxy prefix", () => {
    // Registering at /sw.js from a /hermes mount would claim the whole origin.
    expect(serviceWorkerUrl("/hermes")).not.toBe("/sw.js");
    expect(serviceWorkerScope("/hermes")).not.toBe("/");
  });
});

describe("registerServiceWorker", () => {
  it("is a no-op where service workers are unavailable", async () => {
    stubBrowser(undefined);
    await expect(registerServiceWorker(true)).resolves.toBeUndefined();
  });

  it("is a no-op on an insecure origin", async () => {
    // Plain-HTTP LAN binds cannot host a worker; skip rather than throw.
    const container = fakeServiceWorkerContainer();
    stubBrowser(container, { secure: false });
    await expect(registerServiceWorker(true)).resolves.toBeUndefined();
    expect(container.register).not.toHaveBeenCalled();
  });

  it("registers the worker at its scope in a production build", async () => {
    const container = fakeServiceWorkerContainer();
    stubBrowser(container);
    await registerServiceWorker(true);
    expect(container.register).toHaveBeenCalledTimes(1);
    const [url, opts] = container.register.mock.calls[0];
    expect(url).toBe(serviceWorkerUrl());
    expect(opts).toEqual({ scope: serviceWorkerScope() });
  });

  it("tears down a leftover worker in dev instead of registering one", async () => {
    // `npm run dev` shares localhost with a previous `hermes dashboard`; a
    // surviving worker would serve that build's cached assets over HMR.
    const unregister = vi.fn(async () => true);
    const container = fakeServiceWorkerContainer({
      registrations: [{ unregister }, { unregister }],
    });
    stubBrowser(container);
    await registerServiceWorker(false);
    expect(container.register).not.toHaveBeenCalled();
    expect(unregister).toHaveBeenCalledTimes(2);
  });

  it("swallows a registration failure — the dashboard works without a worker", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const container = fakeServiceWorkerContainer({
      register: async () => {
        throw new Error("SecurityError");
      },
    });
    stubBrowser(container);
    await expect(registerServiceWorker(true)).resolves.toBeUndefined();
  });
});
