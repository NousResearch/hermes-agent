import { describe, expect, it } from "vitest";

import { shouldDropPtyMouseReport } from "./pty-mouse";

describe("shouldDropPtyMouseReport", () => {
  it("forwards left-click press and release so Ink can move its caret", () => {
    expect(shouldDropPtyMouseReport("\x1b[<0;12;8M")).toBe(false);
    expect(shouldDropPtyMouseReport("\x1b[<0;12;8m")).toBe(false);
    expect(shouldDropPtyMouseReport("a")).toBe(false);
  });

  it("drops the motion and wheel reports a drag generates", () => {
    expect(shouldDropPtyMouseReport("\x1b[<32;12;8M")).toBe(true);
    expect(shouldDropPtyMouseReport("\x1b[<64;12;8M")).toBe(true);
  });
});
