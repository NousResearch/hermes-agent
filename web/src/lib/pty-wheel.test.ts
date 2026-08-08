import { describe, expect, it } from "vitest";

import { encodePtyWheel } from "./pty-wheel";

describe("encodePtyWheel", () => {
  it("encodes wheel up and down as SGR mouse reports", () => {
    expect(encodePtyWheel(-1)).toBe("\u001b[<64;1;1M");
    expect(encodePtyWheel(120)).toBe("\u001b[<65;1;1M");
  });

  it("preserves SGR modifier bits", () => {
    expect(encodePtyWheel(-1, { shiftKey: true })).toBe("\u001b[<68;1;1M");
    expect(encodePtyWheel(1, { altKey: true, ctrlKey: true })).toBe(
      "\u001b[<89;1;1M",
    );
    expect(encodePtyWheel(-1, { metaKey: true })).toBe("\u001b[<72;1;1M");
  });

  it("ignores zero and non-finite deltas", () => {
    expect(encodePtyWheel(0)).toBeNull();
    expect(encodePtyWheel(Number.NaN)).toBeNull();
    expect(encodePtyWheel(Number.POSITIVE_INFINITY)).toBeNull();
  });
});
