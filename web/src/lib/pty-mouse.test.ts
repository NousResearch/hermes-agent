import { describe, expect, it } from "vitest";

import { shouldDropPtyMouseReport } from "./pty-mouse";

describe("PTY mouse reports", () => {
  it("forwards left-click press and release so the TUI can move its caret", () => {
    expect(shouldDropPtyMouseReport("\x1b[<0;12;8M")).toBe(false);
    expect(shouldDropPtyMouseReport("\x1b[<0;12;8m")).toBe(false);
  });

  it("drops wheel and motion reports that are not caret clicks", () => {
    expect(shouldDropPtyMouseReport("\x1b[<32;12;8M")).toBe(true);
    expect(shouldDropPtyMouseReport("\x1b[<64;12;8M")).toBe(true);
  });

  it("does not treat ordinary keystrokes as mouse reports", () => {
    expect(shouldDropPtyMouseReport("a")).toBe(false);
  });
});
