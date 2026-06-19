import { describe, it, expect, vi, afterEach } from "vitest";
import { getDashboardBranding, wordmarkLines } from "./dashboard-branding";

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubWindow(branding?: Window["__HERMES_DASHBOARD_BRANDING__"]) {
  vi.stubGlobal("window", { __HERMES_DASHBOARD_BRANDING__: branding });
}

describe("getDashboardBranding", () => {
  it("returns current Hermes defaults when no branding is injected", () => {
    stubWindow();

    expect(getDashboardBranding()).toEqual({
      appName: "Hermes Agent",
      assistantName: "Hermes",
      wordmarkLines: ["Hermes", "Agent"],
      title: "Hermes Agent - Dashboard",
    });
  });

  it("uses server-injected branding overrides", () => {
    stubWindow({
      app_name: "Transformation Lab",
      assistant_name: "Debra",
      wordmark_lines: ["Transformation", "Lab"],
      title: "Transformation Lab",
    });

    expect(getDashboardBranding()).toEqual({
      appName: "Transformation Lab",
      assistantName: "Debra",
      wordmarkLines: ["Transformation", "Lab"],
      title: "Transformation Lab",
    });
  });
});

describe("wordmarkLines", () => {
  it("splits a two-word app name when no explicit wordmark is provided", () => {
    expect(wordmarkLines("Transformation Lab", undefined)).toEqual([
      "Transformation",
      "Lab",
    ]);
  });

  it("falls back to the app name as one line for longer names", () => {
    expect(wordmarkLines("My Excellent Control Center", undefined)).toEqual([
      "My Excellent Control Center",
    ]);
  });
});
