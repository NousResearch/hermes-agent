import { describe, expect, it } from "vitest";
import {
  isPluginTabActive,
  normalizeDashboardPath,
  pluginTabRoutePath,
} from "./plugin-path";

describe("normalizeDashboardPath", () => {
  it("strips trailing slashes", () => {
    expect(normalizeDashboardPath("/cron/")).toBe("/cron");
    expect(normalizeDashboardPath("/")).toBe("/");
  });
});

describe("pluginTabRoutePath", () => {
  it("uses tab.path when the plugin is a new route", () => {
    expect(pluginTabRoutePath({ path: "/livingcolor" })).toBe("/livingcolor");
  });

  it("uses tab.override when the plugin replaces a builtin route", () => {
    expect(
      pluginTabRoutePath({ path: "/livingcolor", override: "/cron" }),
    ).toBe("/cron");
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

  it("activates an override plugin on the replaced route, not tab.path", () => {
    const tab = { path: "/livingcolor", override: "/cron" };
    const route = pluginTabRoutePath(tab);
    expect(isPluginTabActive("/cron", route)).toBe(true);
    expect(isPluginTabActive("/cron/jobs/1", route)).toBe(true);
    expect(isPluginTabActive("/livingcolor", route)).toBe(false);
  });
});
