import { describe, expect, it } from "vitest";
import {
  DASHBOARD_LABEL,
  DASHBOARD_PATH,
  buildBuiltinNavOrder,
  hasRootDashboardPlugin,
  shouldDeferBuiltinRootRoute,
  shouldIncludePluginPageTitle,
  shouldUseExactNavMatch,
} from "./dashboard-navigation";

describe("dashboard shell navigation", () => {
  it("puts Dashboard first when a root dashboard plugin is available", () => {
    expect(
      buildBuiltinNavOrder({
        dashboard: "Dashboard",
        chat: "Chat",
        rest: ["Sessions", "Files"],
        includeDashboard: true,
        includeChat: true,
      }),
    ).toEqual(["Dashboard", "Chat", "Sessions", "Files"]);
  });

  it("preserves the existing navigation when no root dashboard exists", () => {
    expect(
      buildBuiltinNavOrder({
        dashboard: "Dashboard",
        chat: "Chat",
        rest: ["Sessions", "Files"],
        includeDashboard: false,
        includeChat: true,
      }),
    ).toEqual(["Chat", "Sessions", "Files"]);
  });

  it("detects only plugins that override the root route", () => {
    expect(hasRootDashboardPlugin([{ tab: { override: "/" } }])).toBe(true);
    expect(
      hasRootDashboardPlugin([{ tab: { hidden: true, override: "/" } }]),
    ).toBe(false);
    expect(hasRootDashboardPlugin([{ tab: { override: "/chat" } }])).toBe(false);
    expect(hasRootDashboardPlugin([{ tab: {} }])).toBe(false);
  });

  it("keeps hidden root overrides available for page titles", () => {
    expect(
      shouldIncludePluginPageTitle({ tab: { hidden: true, override: "/" } }),
    ).toBe(true);
    expect(
      shouldIncludePluginPageTitle({ tab: { hidden: true, override: "/chat" } }),
    ).toBe(false);
    expect(shouldIncludePluginPageTitle({ tab: {} })).toBe(true);
  });

  it("uses exact root matching and defers only the loading root route", () => {
    expect(DASHBOARD_PATH).toBe("/");
    expect(DASHBOARD_LABEL).toBe("Dashboard");
    expect(shouldUseExactNavMatch("/")).toBe(true);
    expect(shouldUseExactNavMatch("/sessions")).toBe(true);
    expect(shouldUseExactNavMatch("/files")).toBe(false);
    expect(shouldDeferBuiltinRootRoute("/", true)).toBe(true);
    expect(shouldDeferBuiltinRootRoute("/", false)).toBe(false);
    expect(shouldDeferBuiltinRootRoute("/sessions", true)).toBe(false);
  });
});
