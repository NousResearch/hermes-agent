import { describe, expect, it } from "vitest";

import {
  shouldOpenPtySocketAfterUrlBuild,
  shouldRotatePtyAttachToken,
} from "./pty-attach";

describe("shouldRotatePtyAttachToken", () => {
  it("rotates when the selected profile changes (A → B)", () => {
    expect(shouldRotatePtyAttachToken(false, true)).toBe(true);
  });

  it("reuses the token on same-profile reconnect", () => {
    expect(shouldRotatePtyAttachToken(false, false)).toBe(false);
  });

  it("rotates on forced-fresh even when the profile is unchanged", () => {
    expect(shouldRotatePtyAttachToken(true, false)).toBe(true);
  });

  it("rotates when both forced-fresh and profile change apply", () => {
    expect(shouldRotatePtyAttachToken(true, true)).toBe(true);
  });
});

describe("shouldOpenPtySocketAfterUrlBuild", () => {
  it("opens the socket when the connect effect is still active", () => {
    expect(shouldOpenPtySocketAfterUrlBuild(false)).toBe(true);
  });

  it("aborts after a deferred URL build when the effect was cleaned up", () => {
    // Profile switch / unmount during await api.buildWsUrl(...) sets
    // unmounting=true in cleanup. Opening the socket would assign a stale
    // wsRef for the previous profile scope.
    expect(shouldOpenPtySocketAfterUrlBuild(true)).toBe(false);
  });
});
