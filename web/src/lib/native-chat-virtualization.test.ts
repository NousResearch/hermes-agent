import { describe, expect, it } from "vitest";

import { getVirtualRange } from "./native-chat-virtualization";

describe("native chat virtual transcript range", () => {
  it("returns an overscanned range and spacer sizes for variable row heights", () => {
    const range = getVirtualRange(5, 170, 120, [100, 200, 80, 140, 90], 120, 1);

    expect(range).toEqual({
      start: 0,
      end: 3,
      offsetTop: 0,
      bottomSpacer: 230,
      totalHeight: 610,
    });
  });

  it("uses estimated heights for unmeasured rows and clamps scroll bounds", () => {
    expect(getVirtualRange(100, -20, 240, [], 100, 2)).toMatchObject({
      start: 0,
      end: 5,
      offsetTop: 0,
      totalHeight: 10000,
    });
    expect(getVirtualRange(100, 999999, 240, [], 100, 2)).toMatchObject({
      start: 95,
      end: 100,
      offsetTop: 9500,
      bottomSpacer: 0,
    });
  });

  it("handles an empty transcript without negative spacers", () => {
    expect(getVirtualRange(0, 0, 400)).toEqual({ start: 0, end: 0, offsetTop: 0, bottomSpacer: 0, totalHeight: 0 });
  });
});
