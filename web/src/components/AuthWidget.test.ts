import { afterEach, describe, expect, it, vi } from "vitest";

import { shouldProbeDashboardAuth } from "@/lib/dashboard-auth";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("shouldProbeDashboardAuth", () => {
  it("skips the identity endpoint in loopback mode", () => {
    vi.stubGlobal("window", { __HERMES_AUTH_REQUIRED__: false });
    expect(shouldProbeDashboardAuth()).toBe(false);
  });

  it("probes identity only when the server enables the auth gate", () => {
    vi.stubGlobal("window", { __HERMES_AUTH_REQUIRED__: true });
    expect(shouldProbeDashboardAuth()).toBe(true);
  });
});
