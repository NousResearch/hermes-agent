import { describe, expect, it } from "vitest";
import {
  CHROME_INSET_MAX_PX,
  CHROME_INSET_MIN_PX,
  CHROME_INSET_STABLE_SAMPLES,
  computeKeyboardInset,
  computeViewportChromeInset,
  KEYBOARD_INSET_MIN_PX,
  shouldPinScroll,
} from "./keyboard-inset";

describe("computeKeyboardInset", () => {
  it("returns 0 when visualViewport is unavailable", () => {
    expect(computeKeyboardInset(null, 800)).toBe(0);
    expect(computeKeyboardInset(undefined, 800)).toBe(0);
  });

  it("returns 0 when no keyboard is showing (vv fills layout)", () => {
    expect(computeKeyboardInset({ height: 800, offsetTop: 0 }, 800)).toBe(0);
  });

  it("measures the obscured region below the visual viewport", () => {
    // 800px layout, keyboard eats 320px: vv.height = 480.
    expect(computeKeyboardInset({ height: 480, offsetTop: 0 }, 800)).toBe(320);
  });

  it("accounts for visual-viewport offsetTop (iOS keyboard scroll)", () => {
    // iOS nudged the visual viewport down 40px; keyboard covers the rest.
    expect(computeKeyboardInset({ height: 480, offsetTop: 40 }, 800)).toBe(
      280,
    );
  });

  it("ignores small deltas from collapsing browser chrome", () => {
    // URL bar show/hide produces deltas well under a real keyboard height.
    const delta = KEYBOARD_INSET_MIN_PX - 1;
    expect(
      computeKeyboardInset({ height: 800 - delta, offsetTop: 0 }, 800),
    ).toBe(0);
  });

  it("accepts insets at exactly the threshold", () => {
    expect(
      computeKeyboardInset(
        { height: 800 - KEYBOARD_INSET_MIN_PX, offsetTop: 0 },
        800,
      ),
    ).toBe(KEYBOARD_INSET_MIN_PX);
  });

  it("never goes negative when vv is larger than layout height", () => {
    // Rotation / zoom races can transiently report vv.height > innerHeight.
    expect(computeKeyboardInset({ height: 900, offsetTop: 0 }, 800)).toBe(0);
  });

  it("returns 0 for degenerate layout heights", () => {
    expect(computeKeyboardInset({ height: 480, offsetTop: 0 }, 0)).toBe(0);
    expect(computeKeyboardInset({ height: 480, offsetTop: 0 }, -1)).toBe(0);
    expect(computeKeyboardInset({ height: 480, offsetTop: 0 }, NaN)).toBe(0);
  });

  it("returns 0 for non-finite viewport values", () => {
    expect(computeKeyboardInset({ height: NaN, offsetTop: 0 }, 800)).toBe(0);
    expect(computeKeyboardInset({ height: 480, offsetTop: NaN }, 800)).toBe(0);
  });

  it("rounds fractional geometry to whole pixels", () => {
    // iOS reports fractional vv heights under pinch zoom.
    expect(
      computeKeyboardInset({ height: 479.5, offsetTop: 0.25 }, 800),
    ).toBe(320);
  });
});

describe("shouldPinScroll", () => {
  it("pins while a keyboard inset is active", () => {
    expect(shouldPinScroll(320)).toBe(true);
  });

  it("does not pin without a keyboard", () => {
    expect(shouldPinScroll(0)).toBe(false);
  });
});

describe("computeViewportChromeInset", () => {
  it("returns 0 when visualViewport is unavailable", () => {
    expect(
      computeViewportChromeInset(null, 800, 0, CHROME_INSET_STABLE_SAMPLES),
    ).toBe(0);
    expect(
      computeViewportChromeInset(undefined, 800, 0, CHROME_INSET_STABLE_SAMPLES),
    ).toBe(0);
  });

  it("returns 0 while the viewport fills the layout (no bar)", () => {
    expect(
      computeViewportChromeInset({ height: 800, offsetTop: 0 }, 800, 0, 5),
    ).toBe(0);
  });

  it("compensates a parked bottom URL bar in the (MIN, MAX] band", () => {
    // 800px layout, 56px bottom bar: vv.height = 744, stable across
    // samples (the stability run is maintained by the caller).
    expect(
      computeViewportChromeInset(
        { height: 744, offsetTop: 0 },
        800,
        0,
        CHROME_INSET_STABLE_SAMPLES,
      ),
    ).toBe(56);
  });

  it("waits for the value to be stable before applying it", () => {
    // First sighting of a new bar value: the stability run is still 1.
    expect(
      computeViewportChromeInset(
        { height: 744, offsetTop: 0 },
        800,
        0,
        CHROME_INSET_STABLE_SAMPLES - 1,
      ),
    ).toBe(0);
    expect(
      computeViewportChromeInset(
        { height: 744, offsetTop: 0 },
        800,
        0,
        CHROME_INSET_STABLE_SAMPLES,
      ),
    ).toBe(56);
  });

  it("rejects sub-MIN noise (measurement jitter, tiny bars)", () => {
    const delta = CHROME_INSET_MIN_PX - 1;
    expect(
      computeViewportChromeInset(
        { height: 800 - delta, offsetTop: 0 },
        800,
        0,
        CHROME_INSET_STABLE_SAMPLES,
      ),
    ).toBe(0);
  });

  it("accepts a bar of exactly MIN pixels once stable", () => {
    expect(
      computeViewportChromeInset(
        { height: 800 - CHROME_INSET_MIN_PX, offsetTop: 0 },
        800,
        0,
        CHROME_INSET_STABLE_SAMPLES,
      ),
    ).toBe(CHROME_INSET_MIN_PX);
  });

  it("cannot double-apply with the keyboard path", () => {
    // The applied inset is always max(keyboard, gatedChrome):
    //
    //   raw region ≥ 80px  → keyboard = raw (same formula), chrome = 0
    //   12 ≤ raw < 80px    → keyboard = 0, chrome = raw (once stable)
    //
    // so the full obscured region is applied exactly once in every
    // case. Check both branches:
    const keyboardUp = { height: 800 - 376, offsetTop: 0 }; // 376px:
    // keyboard with a bar still visible under it
    const kb = computeKeyboardInset(keyboardUp, 800);
    const chromeUnderKb = computeViewportChromeInset(
      keyboardUp,
      800,
      kb,
      CHROME_INSET_STABLE_SAMPLES,
    );
    expect(kb).toBe(376);
    expect(chromeUnderKb).toBe(0);
    expect(Math.max(kb, chromeUnderKb)).toBe(376); // exactly once

    const barOnly = { height: 800 - 56, offsetTop: 0 }; // 56px bar
    const kb2 = computeKeyboardInset(barOnly, 800);
    const chromeBar = computeViewportChromeInset(
      barOnly,
      800,
      kb2,
      CHROME_INSET_STABLE_SAMPLES,
    );
    expect(kb2).toBe(0);
    expect(chromeBar).toBe(56);
    expect(Math.max(kb2, chromeBar)).toBe(56); // exactly once
  });

  it("caps the chrome value strictly below the keyboard threshold", () => {
    // No matter how large the measured region, the chrome path can never
    // report a keyboard-sized value, so it can never compete with the
    // keyboard path even if the caller ignored the keyboard inset.
    expect(
      computeViewportChromeInset(
        { height: 800 - 400, offsetTop: 0 },
        800,
        0,
        CHROME_INSET_STABLE_SAMPLES,
      ),
    ).toBe(CHROME_INSET_MAX_PX);
    expect(CHROME_INSET_MAX_PX).toBeLessThan(KEYBOARD_INSET_MIN_PX);
  });

  it("accounts for visual-viewport offsetTop", () => {
    // 800px layout, visual viewport nudged down 10px with 46px obscured
    // below it (e.g. iOS keyboard-scroll + a thin bar).
    expect(
      computeViewportChromeInset(
        { height: 744, offsetTop: 10 },
        800,
        0,
        CHROME_INSET_STABLE_SAMPLES,
      ),
    ).toBe(46);
  });

  it("never goes negative or non-finite", () => {
    expect(
      computeViewportChromeInset({ height: 900, offsetTop: 0 }, 800, 0, 5),
    ).toBe(0);
    expect(
      computeViewportChromeInset(
        { height: 744, offsetTop: NaN },
        800,
        0,
        CHROME_INSET_STABLE_SAMPLES,
      ),
    ).toBe(0);
  });

  it("returns 0 for degenerate layout heights", () => {
    expect(
      computeViewportChromeInset(
        { height: 744, offsetTop: 0 },
        0,
        0,
        CHROME_INSET_STABLE_SAMPLES,
      ),
    ).toBe(0);
  });
});
