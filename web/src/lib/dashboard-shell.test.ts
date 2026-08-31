import { describe, expect, it } from "vitest";

import { getDashboardSidebarMode } from "./dashboard-shell";

describe("getDashboardSidebarMode", () => {
  it("uses the compact rail for the desktop chat home surface", () => {
    expect(getDashboardSidebarMode("/chat", false, false)).toBe("chat-rail");
    expect(getDashboardSidebarMode("/chat/", false, true)).toBe("chat-rail");
  });

  it("keeps mobile navigation expanded even on chat", () => {
    expect(getDashboardSidebarMode("/chat", true, false)).toBe("expanded");
  });

  it("preserves the stored expanded/collapsed state on normal desktop routes", () => {
    expect(getDashboardSidebarMode("/sessions", false, false)).toBe("expanded");
    expect(getDashboardSidebarMode("/sessions", false, true)).toBe("collapsed");
  });

  it("lets the chat rail toggle back to the expanded shell without changing the stored preference", () => {
    expect(getDashboardSidebarMode("/chat", false, true, true)).toBe("expanded");
  });
});
