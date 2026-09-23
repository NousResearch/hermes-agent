import { describe, expect, it } from "vitest";

import {
  SHIFT_DOWN,
  SHIFT_UP,
  WHEEL_MAX_ROWS_PER_TICK,
  wheelScrollSequences,
} from "./pty-wheel-scroll";

describe("wheelScrollSequences", () => {
  it("returns empty for zero or non-finite delta", () => {
    expect(wheelScrollSequences(0)).toEqual([]);
    expect(wheelScrollSequences(Number.NaN)).toEqual([]);
    expect(wheelScrollSequences(Number.POSITIVE_INFINITY)).toEqual([]);
  });

  it("maps positive delta to Shift+Down rows", () => {
    expect(wheelScrollSequences(25)).toEqual([SHIFT_DOWN]);
    expect(wheelScrollSequences(50)).toEqual([SHIFT_DOWN]);
    expect(wheelScrollSequences(75)).toEqual([SHIFT_DOWN, SHIFT_DOWN]);
  });

  it("maps negative delta to Shift+Up rows", () => {
    expect(wheelScrollSequences(-25)).toEqual([SHIFT_UP]);
    expect(wheelScrollSequences(-50)).toEqual([SHIFT_UP]);
    expect(wheelScrollSequences(-120)).toEqual([SHIFT_UP, SHIFT_UP]);
  });

  it("caps rows per tick", () => {
    const large = wheelScrollSequences(500);
    expect(large).toHaveLength(WHEEL_MAX_ROWS_PER_TICK);
    expect(large.every((seq) => seq === SHIFT_DOWN)).toBe(true);
  });
});
