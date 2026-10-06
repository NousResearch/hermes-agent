import { describe, expect, it } from "vitest";

import { scaledTerminalFontSize } from "./terminal-font-scale";

const cases = [
  [299, 0.5, 4],
  [300, 1, 8],
  [359, 1.25, 10],
  [360, 0.5, 5],
  [419, 1, 9],
  [420, 1.25, 13],
  [519, 0.5, 5],
  [520, 1, 11],
  [719, 1.25, 14],
  [720, 0.5, 6],
  [1023, 1, 12],
  [1024, 1.25, 18],
  [1280, 0.5, 7],
  [1280, 1, 14],
  [1280, 1.25, 18],
] as const;

describe("scaledTerminalFontSize", () => {
  it.each(cases)(
    "returns %dpx for %dpx width at %d scale",
    (width, scale, expected) => {
      expect(scaledTerminalFontSize(width, scale)).toBe(expected);
    },
  );

  it("keeps a positive minimum for a sub-1 scale", () => {
    expect(scaledTerminalFontSize(299, 0.01)).toBe(1);
  });
});
