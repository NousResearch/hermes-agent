import { describe, expect, it } from "vitest";

import { needsPtyResize } from "./terminal-pty-resize";

describe("needsPtyResize", () => {
  const grid = { cols: 80, rows: 24 };

  it("resizes on a height-only change with the font unchanged", () => {
    // iOS Safari URL bar collapsing: same width (so same font), more rows.
    expect(needsPtyResize(grid, { cols: 80, rows: 31 }, false)).toBe(true);
  });

  it("resizes on a width-only change with the font unchanged", () => {
    expect(needsPtyResize(grid, { cols: 96, rows: 24 }, false)).toBe(true);
  });

  it("resizes when the font changed even if the grid did not", () => {
    expect(needsPtyResize(grid, { ...grid }, true)).toBe(true);
  });

  it("does not resize when neither the grid nor the font changed", () => {
    expect(needsPtyResize(grid, { ...grid }, false)).toBe(false);
  });
});
