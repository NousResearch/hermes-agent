import { describe, expect, it } from "vitest";

import { resolveProfileStatusLoad } from "./profile-load";

describe("resolveProfileStatusLoad", () => {
  it("settles a rejected load for the selected profile without stale status", () => {
    const result = resolveProfileStatusLoad("work", {
      status: "rejected",
      reason: new Error("unavailable"),
    });

    expect(result).toEqual({
      loadedProfile: "work",
      status: null,
      error: "unavailable",
    });
  });

  it("clears the selected profile error after a successful retry", () => {
    const result = resolveProfileStatusLoad("work", {
      status: "fulfilled",
      value: { gateway_running: true },
    });

    expect(result).toEqual({
      loadedProfile: "work",
      status: { gateway_running: true },
      error: null,
    });
  });
});
