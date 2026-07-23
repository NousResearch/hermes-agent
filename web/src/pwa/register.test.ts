import { describe, expect, it, vi } from "vitest";
import {
  registerHermesServiceWorker,
  resolveServiceWorkerTarget,
  type ServiceWorkerRegistrar,
} from "./register";

describe("resolveServiceWorkerTarget", () => {
  it("prefers and normalizes the runtime reverse-proxy base path", () => {
    expect(resolveServiceWorkerTarget("/hermes", "/vite-base/")).toEqual({
      scope: "/hermes/",
      scriptUrl: "/hermes/service-worker.js",
    });
  });

  it("falls back to the Vite base URL", () => {
    expect(resolveServiceWorkerTarget("", "/dashboard/")).toEqual({
      scope: "/dashboard/",
      scriptUrl: "/dashboard/service-worker.js",
    });
  });

  it("uses root scope when both base paths are empty", () => {
    expect(resolveServiceWorkerTarget("", "")).toEqual({
      scope: "/",
      scriptUrl: "/service-worker.js",
    });
  });
});

describe("registerHermesServiceWorker", () => {
  function registrar() {
    const registration = {} as ServiceWorkerRegistration;
    const register = vi.fn(async () => registration);
    return {
      registration,
      register,
      serviceWorker: { register } as ServiceWorkerRegistrar,
    };
  }

  it("does not register outside production", async () => {
    const worker = registrar();

    const result = await registerHermesServiceWorker({
      isProduction: false,
      serviceWorker: worker.serviceWorker,
    });

    expect(result).toBeUndefined();
    expect(worker.register).not.toHaveBeenCalled();
  });

  it("registers at the runtime base path without HTTP cache reuse", async () => {
    const worker = registrar();

    const result = await registerHermesServiceWorker({
      isProduction: true,
      runtimeBasePath: "/hermes/",
      viteBaseUrl: "/",
      serviceWorker: worker.serviceWorker,
    });

    expect(result).toBe(worker.registration);
    expect(worker.register).toHaveBeenCalledWith(
      "/hermes/service-worker.js",
      {
        scope: "/hermes/",
        updateViaCache: "none",
      },
    );
  });

  it("reports registration failure without logging error details", async () => {
    const warn = vi.fn();
    const serviceWorker: ServiceWorkerRegistrar = {
      register: vi.fn(async () => {
        throw new Error("potentially sensitive browser detail");
      }),
    };

    await expect(
      registerHermesServiceWorker({
        isProduction: true,
        serviceWorker,
        warn,
      }),
    ).resolves.toBeUndefined();

    expect(warn).toHaveBeenCalledExactlyOnceWith(
      "Hermes PWA service worker registration failed.",
    );
  });
});
