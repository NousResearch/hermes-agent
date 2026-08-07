import { describe, expect, it } from "vitest";

import { isDashboardPasteShortcut, type DashboardKeyEvent } from "./dashboard-keyboard";

function key(overrides: Partial<DashboardKeyEvent>): DashboardKeyEvent {
  return { altKey: false, ctrlKey: false, key: "v", metaKey: false, ...overrides };
}

describe("isDashboardPasteShortcut", () => {
  it("accepts Cmd+V and Ctrl+V on macOS for dashboard dictation paste chords", () => {
    expect(isDashboardPasteShortcut(key({ metaKey: true }), true)).toBe(true);
    expect(isDashboardPasteShortcut(key({ ctrlKey: true }), true)).toBe(true);
  });

  it("accepts Ctrl+V and Ctrl+Shift+V on non-mac platforms", () => {
    expect(isDashboardPasteShortcut(key({ ctrlKey: true }), false)).toBe(true);
  });

  it("does not hijack Alt/AltGr-modified chords", () => {
    expect(isDashboardPasteShortcut(key({ altKey: true, ctrlKey: true }), false)).toBe(false);
    expect(isDashboardPasteShortcut(key({ altKey: true, metaKey: true }), true)).toBe(false);
  });

  it("rejects non-v keys", () => {
    expect(isDashboardPasteShortcut(key({ ctrlKey: true, key: "c" }), false)).toBe(false);
  });
});
