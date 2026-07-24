import { describe, expect, it } from "vitest";
import { isPluginTabActive, normalizeDashboardPath } from "./plugin-path";

describe("normalizeDashboardPath", () => {
  it("strips trailing slashes", () => {
    expect(normalizeDashboardPath("/cron/")).toBe("/cron");
    expect(normalizeDashboardPath("/")).toBe("/");
  });
});

describe("isPluginTabActive", () => {
  it("matches exact tab paths", () => {
    expect(isPluginTabActive("/livingcolor", "/livingcolor")).toBe(true);
    expect(isPluginTabActive("/sessions", "/livingcolor")).toBe(false);
  });

  it("matches sub-routes under the tab", () => {
    expect(isPluginTabActive("/livingcolor/projects/foo", "/livingcolor")).toBe(
      true,
    );
  });
});
