import { describe, expect, it } from "vitest";

import {
  applyStickyModifiers,
  bufferToPlainText,
  ctrlByteFor,
  EXTRA_KEY_SEQUENCES,
} from "./pty-extra-keys";

describe("pty-extra-keys", () => {
  it("maps ctrl chords", () => {
    expect(ctrlByteFor("c")).toBe("\x03");
    expect(ctrlByteFor("A")).toBe("\x01");
    expect(ctrlByteFor("[")).toBe("\x1b");
    expect(ctrlByteFor(" ")).toBe("\x00");
    expect(ctrlByteFor("1")).toBeNull();
  });

  it("passes through when no modifier is latched", () => {
    expect(applyStickyModifiers("x", { ctrl: false, alt: false })).toEqual({
      data: "x",
      consumed: false,
    });
  });

  it("applies ctrl to a single char and arrows", () => {
    expect(applyStickyModifiers("r", { ctrl: true, alt: false }).data).toBe("\x12");
    expect(
      applyStickyModifiers(EXTRA_KEY_SEQUENCES.left, { ctrl: true, alt: false }).data,
    ).toBe("\x1b[1;5D");
  });

  it("applies alt as an ESC prefix", () => {
    expect(applyStickyModifiers("b", { ctrl: false, alt: true }).data).toBe("\x1bb");
    expect(applyStickyModifiers("\r", { ctrl: false, alt: true }).data).toBe("\x1b\r");
    expect(
      applyStickyModifiers(EXTRA_KEY_SEQUENCES.right, { ctrl: false, alt: true }).data,
    ).toBe("\x1b[1;3C");
  });

  it("flattens the buffer, joining wrapped rows and trimming blanks", () => {
    const rows = [
      { isWrapped: false, text: "hello " },
      { isWrapped: true, text: "world" },
      { isWrapped: false, text: "next" },
      { isWrapped: false, text: "" },
    ];
    const buffer = {
      length: rows.length,
      getLine: (y: number) => ({
        isWrapped: rows[y].isWrapped,
        translateToString: () => rows[y].text,
      }),
    };
    expect(bufferToPlainText(buffer)).toBe("hello world\nnext");
  });
});
