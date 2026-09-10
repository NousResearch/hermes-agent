import { describe, expect, it } from "vitest";

import type { ManagedPlatformInfo } from "./api";
import { managedChannelControls } from "./managed-channels";

const native: ManagedPlatformInfo = { kind: "native", label: "Nous Portal", url: null };
const relay: ManagedPlatformInfo = { kind: "relay", label: "Nous Portal", url: null };

describe("managedChannelControls", () => {
  it.each([
    ["older server without the field", { configured: true }],
    ["explicit null", { managed: null, configured: false }],
  ])("keeps every control for an unmanaged card (%s)", (_label, platform) => {
    expect(managedChannelControls(platform)).toEqual({
      managed: null,
      showToggle: true,
      showConfigure: true,
      showOnboarding: true,
      showTest: true,
      showNativeState: true,
    });
  });

  it.each([
    ["configured", true],
    ["not configured", false],
  ])("native: hides setup, keeps state, tests only when %s", (_label, configured) => {
    expect(managedChannelControls({ managed: native, configured })).toEqual({
      managed: native,
      showToggle: false,
      showConfigure: false,
      showOnboarding: false,
      showTest: configured,
      showNativeState: true,
    });
  });

  it("relay: hides everything native, including the state badge", () => {
    expect(managedChannelControls({ managed: relay, configured: true })).toEqual({
      managed: relay,
      showToggle: false,
      showConfigure: false,
      showOnboarding: false,
      showTest: false,
      showNativeState: false,
    });
  });
});
