import { describe, expect, it } from "vitest";

import {
  caretColumnFromOffset,
  caretOffsetInSuffix,
  composerPaintRow,
  gridCellFromPointer,
  moveNativeCaret,
} from "./pty-native-caret";

describe("native PTY caret from pointer", () => {
  it("maps a click on the acknowledged suffix to a textarea offset without guessing", () => {
    const cells = [
      { text: "h", row: 1, col: 2 },
      { text: "e", row: 1, col: 3 },
      { text: "l", row: 1, col: 4 },
      { text: "l", row: 1, col: 5 },
      { text: "o", row: 1, col: 6 },
    ];
    expect(caretOffsetInSuffix(cells, 1, 4)).toBe(2);
    expect(caretOffsetInSuffix(cells, 0, 4)).toBeNull();
  });

  it("maps viewport coordinates onto the terminal grid", () => {
    expect(gridCellFromPointer(25, 15, { left: 0, top: 0, width: 80, height: 20 }, 8, 2)).toEqual({
      col: 2,
      row: 1,
    });
  });

  it("moves the native caret with keyboard keys without leaving the suffix", () => {
    expect(moveNativeCaret("hello", 5, 5, "ArrowLeft")).toEqual({ start: 4, end: 4 });
    expect(moveNativeCaret("hello", 0, 0, "ArrowLeft")).toEqual({ start: 0, end: 0 });
    expect(moveNativeCaret("hello", 0, 0, "ArrowRight")).toEqual({ start: 1, end: 1 });
    expect(moveNativeCaret("hello", 2, 2, "Home")).toEqual({ start: 0, end: 0 });
    expect(moveNativeCaret("hello", 2, 2, "End")).toEqual({ start: 5, end: 5 });
    expect(moveNativeCaret("hello", 2, 2, "a")).toBeNull();
  });

  it("falls back to the PTY cursor when the buffer suffix is not yet matched", () => {
    expect(caretColumnFromOffset(10, 5, 5)).toBe(10);
    expect(caretColumnFromOffset(10, 5, 2)).toBe(7);
    expect(caretColumnFromOffset(2, 5, 0)).toBe(0);
  });

  it("paints the overlay on the composer row even if the PTY cursor left it", () => {
    expect(composerPaintRow(20, 0)).toBe(19);
    expect(composerPaintRow(20, 19)).toBe(19);
    expect(composerPaintRow(20, 3)).toBe(19);
  });
});
