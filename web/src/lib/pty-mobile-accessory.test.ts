import { describe, expect, it } from "vitest";

import {
  ACCESSORY_BAR_HEIGHT_PX,
  accessoryDockBottomPx,
  shouldShowMobileAccessory,
  terminalBottomReservePx,
} from "./pty-mobile-accessory";

describe("mobile accessory visibility", () => {
  it("shows only on a touch device, and already while the keyboard is still opening", () => {
    // Focus lands before visualViewport reports the inset; the bar must not flicker in.
    expect(shouldShowMobileAccessory(0, true, true)).toBe(true);
    expect(shouldShowMobileAccessory(320, true, false)).toBe(true);
    expect(shouldShowMobileAccessory(0, true, false)).toBe(false);
    // A desktop mouse never gets the bar, however the viewport behaves.
    expect(shouldShowMobileAccessory(320, false, true)).toBe(false);
  });
});

describe("accessory geometry", () => {
  it("reserves room under xterm and docks the bar on top of the keyboard", () => {
    expect(terminalBottomReservePx(320, true)).toBe(320 + ACCESSORY_BAR_HEIGHT_PX);
    expect(terminalBottomReservePx(320, false)).toBe(320);
    // `position: fixed` measures the layout viewport, which the keyboard overlays.
    expect(accessoryDockBottomPx(320)).toBe(320);
    expect(accessoryDockBottomPx(0)).toBe(0);
    expect(accessoryDockBottomPx(Number.NaN)).toBe(0);
  });
});
