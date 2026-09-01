import { describe, expect, it } from "vitest";

import {
  getDashboardHomePath,
  getDashboardSidebarMode,
  shouldRenderDashboardPageHeader,
  shouldRenderMobileNavigationHeader,
  shouldShowMobileSessionNavigator,
} from "./dashboard-shell";

describe("getDashboardHomePath", () => {
  it("opens embedded chat at the dashboard root", () => {
    expect(getDashboardHomePath(true)).toBe("/chat");
  });

  it("falls back to sessions when embedded chat is unavailable", () => {
    expect(getDashboardHomePath(false)).toBe("/sessions");
  });
});

describe("chat shell chrome ownership", () => {
  it("lets the native chat header own the chat route", () => {
    expect(shouldRenderDashboardPageHeader("/chat")).toBe(false);
    expect(shouldRenderDashboardPageHeader("/chat/")).toBe(false);
    expect(shouldRenderDashboardPageHeader("/sessions")).toBe(true);
  });

  it("keeps the global mobile nav header on non-chat routes only", () => {
    expect(shouldRenderMobileNavigationHeader("/chat", true)).toBe(false);
    expect(shouldRenderMobileNavigationHeader("/sessions", true)).toBe(true);
    expect(shouldRenderMobileNavigationHeader("/sessions", false)).toBe(false);
  });

  it("shows the session navigator on desktop and only when opened on mobile", () => {
    expect(shouldShowMobileSessionNavigator(false, false)).toBe(true);
    expect(shouldShowMobileSessionNavigator(true, false)).toBe(false);
    expect(shouldShowMobileSessionNavigator(true, true)).toBe(true);
  });
});

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
