import { describe, expect, it } from "vitest";

import { needsRestartBanner } from "./restart-banner";

describe("needsRestartBanner", () => {
  it("is false when gateway_start_time is null (telegram_allowlist_updated_at set)", () => {
    expect(
      needsRestartBanner({ gateway_start_time: null, telegram_allowlist_updated_at: 200 }),
    ).toBe(false);
  });

  it("is false when telegram_allowlist_updated_at is null (gateway_start_time set)", () => {
    expect(
      needsRestartBanner({ gateway_start_time: 100, telegram_allowlist_updated_at: null }),
    ).toBe(false);
  });

  it("is false when both are null", () => {
    expect(
      needsRestartBanner({ gateway_start_time: null, telegram_allowlist_updated_at: null }),
    ).toBe(false);
  });

  it("is true when the allowlist changed after the gateway started", () => {
    expect(
      needsRestartBanner({ gateway_start_time: 100, telegram_allowlist_updated_at: 200 }),
    ).toBe(true);
  });

  it("is false when the allowlist last changed before the gateway started", () => {
    expect(
      needsRestartBanner({ gateway_start_time: 200, telegram_allowlist_updated_at: 100 }),
    ).toBe(false);
  });
});
