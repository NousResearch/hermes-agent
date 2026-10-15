import { describe, expect, it } from "vitest";
import { pluginRoutePaths } from "./pluginRoutes";

describe("pluginRoutePaths", () => {
  it("mounts a plugin at its base path and all descendant paths", () => {
    expect(pluginRoutePaths("/eng-runs")).toEqual([
      "/eng-runs",
      "/eng-runs/*",
    ]);
  });

  it("does not create a double slash for paths with a trailing slash", () => {
    expect(pluginRoutePaths("/eng-runs/")).toEqual([
      "/eng-runs/",
      "/eng-runs/*",
    ]);
  });
});
