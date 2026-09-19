import { describe, expect, it } from "vitest";

import {
  advanceTouchAnchor,
  isTouchPan,
  touchLineTravel,
  touchScrollLines,
  wheelScrollLines,
} from "./pty-touch-scroll";

describe("one-finger transcript pan", () => {
  it("scrolls one row per finger travel and keeps a caret tap from scrolling", () => {
    const travel = touchLineTravel(20);
    expect(touchScrollLines(240, 240 - travel, 20)).toBe(1);
    expect(touchScrollLines(240, 240 + travel, 20)).toBe(-1);
    expect(advanceTouchAnchor(240, 1, travel)).toBe(240 - travel);
    // Below one row of travel nothing moves, so a tap can still place the caret.
    expect(touchScrollLines(240, 229, 20)).toBe(0);
    expect(isTouchPan(100, 108)).toBe(false);
    expect(isTouchPan(100, 120)).toBe(true);
    // Safari's synthetic flick deltas are large; one tick stays one row.
    expect(wheelScrollLines(400)).toBe(1);
    expect(wheelScrollLines(0)).toBe(0);
  });
});
