import { describe, expect, it } from "vitest";
import {
  computeKeyboardInset,
  keyboardRevealScrollDelta,
  KEYBOARD_INSET_MIN_PX,
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

describe("keyboardRevealScrollDelta", () => {
  it("accounts for iOS visual-viewport offsetTop", () => {
    expect(
      keyboardRevealScrollDelta(800, { height: 480, offsetTop: 40 }),
    ).toBe(280);
  });

  it("does not move when the composer is already on the visible bottom", () => {
    expect(
      keyboardRevealScrollDelta(480, { height: 480, offsetTop: 0 }),
    ).toBe(0);
  });
});

describe("keyboardRevealScrollDelta", () => {
  it("scrolls the page so the composer sits on the visual-viewport bottom", () => {
    expect(
      keyboardRevealScrollDelta(800, { height: 480, offsetTop: 0 }),
    ).toBe(320);
  });

  it("accounts for iOS visual-viewport offsetTop", () => {
    expect(
      keyboardRevealScrollDelta(800, { height: 480, offsetTop: 40 }),
    ).toBe(280);
  });

  it("does not move when the composer is already on the visible bottom", () => {
    expect(
      keyboardRevealScrollDelta(480, { height: 480, offsetTop: 0 }),
    ).toBe(0);
  });

  it("returns 0 for non-finite geometry", () => {
    expect(
      keyboardRevealScrollDelta(NaN, { height: 480, offsetTop: 0 }),
    ).toBe(0);
  });
});
