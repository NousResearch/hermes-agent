import { describe, expect, it } from "vitest";

import { getUsageAccountUnavailableMessage } from "@/lib/usage-quota";

describe("getUsageAccountUnavailableMessage", () => {
  it("returns the account-specific reason when an account is unavailable", () => {
    expect(
      getUsageAccountUnavailableMessage({
        available: false,
        windows: [],
        unavailable_reason: "Usage endpoint returned HTTP 429.",
      }),
    ).toBe("Usage endpoint returned HTTP 429.");
  });

  it("does not mark an account with usage windows as unavailable", () => {
    expect(
      getUsageAccountUnavailableMessage({
        available: true,
        windows: [{ label: "Session", used_percent: 25 }],
      }),
    ).toBeNull();
  });
});
