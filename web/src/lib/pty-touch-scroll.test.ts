import { describe, expect, it } from "vitest";

import {
  advanceTouchAnchor,
  isTouchPan,
  ptyWheelSequence,
  touchLineTravel,
  touchScrollLines,
  wheelScrollLines,
} from "./pty-touch-scroll";

describe("touchScrollLines", () => {
  it("moves at most one terminal line per drag sample so a stop in the middle is possible", () => {
    expect(touchScrollLines(400, 100, 20)).toBe(1);
    expect(touchScrollLines(100, 400, 20)).toBe(-1);
  });

  it("needs a clear finger travel before the first line", () => {
    expect(touchLineTravel(20)).toBe(25);
    expect(touchScrollLines(240, 220, 20)).toBe(0);
    expect(touchScrollLines(240, 240 - touchLineTravel(20), 20)).toBe(1);
  });

  it("caps line travel so a tall terminal still scrolls on a short swipe", () => {
    expect(touchLineTravel(80)).toBe(36);
    expect(touchLineTravel(20)).toBe(25);
  });

  it("ignores finger jitter smaller than one terminal row", () => {
    expect(touchScrollLines(200, 211, 20)).toBe(0);
  });
});

describe("advanceTouchAnchor", () => {
  it("moves the anchor toward the finger so reversing the swipe reverses scroll", () => {
    const travel = touchLineTravel(20);
    let y = 400;
    const up = touchScrollLines(y, 359, 20);
    expect(up).toBe(1);
    y = advanceTouchAnchor(y, up, travel);
    expect(touchScrollLines(y, 359, 20)).toBe(0);
    expect(touchScrollLines(y, y + 20, 20)).toBe(0);
    expect(touchScrollLines(y, y + travel, 20)).toBe(-1);
  });
});

describe("isTouchPan", () => {
  it("treats a short press as a tap so the caret can move", () => {
    expect(isTouchPan(100, 108)).toBe(false);
  });

  it("treats a longer drag as a pan so the transcript can scroll", () => {
    expect(isTouchPan(100, 120)).toBe(true);
  });
});

describe("ptyWheelSequence", () => {
  it("sends real SGR wheel events so Ink scrolls the transcript one step at a time", () => {
    expect(ptyWheelSequence(1)).toBe("\x1b[<64;1;1M");
    expect(ptyWheelSequence(-1)).toBe("\x1b[<65;1;1M");
    expect(ptyWheelSequence(0)).toBeNull();
  });
});

describe("wheelScrollLines", () => {
  it("maps a modest pixel wheel tick to one terminal line", () => {
    expect(wheelScrollLines(160)).toBe(1);
  });

  it("does not jump several rows on one iPhone flick tick", () => {
    expect(wheelScrollLines(400)).toBe(1);
  });
});
