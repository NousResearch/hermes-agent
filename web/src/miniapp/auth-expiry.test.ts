import { describe, expect, it } from "vitest";
import { createAuthExpiryGate } from "./auth-expiry";

describe("createAuthExpiryGate", () => {
  it("does NOT treat a 401 as expiry before any session was established", () => {
    // The regression this pins: an unauthenticated browser visitor's (or an
    // unpaired Telegram user's) very first /api/miniapp/me call 401s, and
    // an earlier version showed them "Session expired" instead of the
    // correct "not authorized" / "not paired" screens.
    const gate = createAuthExpiryGate();
    expect(gate.isExpiry401()).toBe(false);
  });

  it("treats a 401 as expiry once a session was established", () => {
    const gate = createAuthExpiryGate();
    gate.markSessionEstablished();
    expect(gate.isExpiry401()).toBe(true);
  });

  it("stays armed across repeated checks (expiry is not consumed)", () => {
    const gate = createAuthExpiryGate();
    gate.markSessionEstablished();
    expect(gate.isExpiry401()).toBe(true);
    expect(gate.isExpiry401()).toBe(true);
  });

  it("gates are independent instances", () => {
    const armed = createAuthExpiryGate();
    const fresh = createAuthExpiryGate();
    armed.markSessionEstablished();
    expect(armed.isExpiry401()).toBe(true);
    expect(fresh.isExpiry401()).toBe(false);
  });
});
