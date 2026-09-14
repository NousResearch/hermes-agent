import { describe, expect, it } from "vitest";

import { advanceTouchScroll } from "./pty-touch-scroll";

describe("advanceTouchScroll", () => {
  it("maps finger movement to rows and carries partial rows", () => {
    const first = advanceTouchScroll({ lastY: 200, remainderPx: 0 }, 185, 10);
    expect(first).toEqual({
      lines: 1,
      state: { lastY: 185, remainderPx: 5 },
    });

    expect(advanceTouchScroll(first.state, 180, 10)).toEqual({
      lines: 1,
      state: { lastY: 180, remainderPx: 0 },
    });
  });

  it("uses terminal scroll direction in both pan directions", () => {
    expect(advanceTouchScroll({ lastY: 100, remainderPx: 0 }, 80, 10).lines).toBe(2);
    expect(advanceTouchScroll({ lastY: 100, remainderPx: 0 }, 120, 10).lines).toBe(-2);
  });
});
