import { describe, expect, it } from "vitest";

import { touchScrollLines } from "./pty-touch-scroll";

describe("touchScrollLines", () => {
  it("turns an upward finger swipe into positive terminal scroll lines", () => {
    expect(touchScrollLines(240, 180, 20)).toBe(3);
  });

  it("turns a downward finger swipe into negative terminal scroll lines", () => {
    expect(touchScrollLines(180, 240, 20)).toBe(-3);
  });

  it("allows slow drags to accumulate against a retained anchor", () => {
    expect(touchScrollLines(240, 231, 20)).toBe(0);
    expect(touchScrollLines(240, 220, 20)).toBe(1);
  });

  it("ignores finger jitter smaller than one terminal row", () => {
    expect(touchScrollLines(200, 211, 20)).toBe(0);
  });
});
