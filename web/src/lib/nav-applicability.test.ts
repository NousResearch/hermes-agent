import { describe, expect, it } from "vitest";

import {
  MACHINE_LEVEL_NAV_PATHS,
  filterApplicableNav,
  isNavPathApplicable,
} from "./nav-applicability";

const navItems = [
  { path: "/models", label: "Models" },
  { path: "/logs", label: "Logs", machineLevel: true },
  { path: "/system", label: "System", machineLevel: true },
  { path: "/profiles", label: "Profiles" },
];

describe("nav applicability", () => {
  it("offers every entry while the switcher is on this dashboard's own profile", () => {
    // "" is the ProfileSwitcher's value for "this dashboard" — the only scope
    // where a machine-level page is actually the one the banner describes.
    expect(filterApplicableNav(navItems, "")).toEqual(navItems);
  });

  it("hides machine-level entries when another profile is managed", () => {
    const shown = filterApplicableNav(navItems, "architect").map((i) => i.path);

    // The profile-scoped pages stay: they honour ?profile=architect.
    expect(shown).toEqual(["/models", "/profiles"]);
  });

  it("keeps a machine-level page out of reach for every non-empty scope", () => {
    for (const profile of ["default", "architect", "visual"]) {
      expect(isNavPathApplicable("/system", profile)).toBe(false);
      expect(isNavPathApplicable("/models", profile)).toBe(true);
    }
  });

  it("treats the flagged path set as the contract, not the flag alone", () => {
    // Both carriers matter: the explicit flag on a nav entry and the path set
    // (plugin tabs arrive from manifests and carry no flag).
    expect(MACHINE_LEVEL_NAV_PATHS.has("/logs")).toBe(true);
    expect(isNavPathApplicable("/logs", "architect")).toBe(false);
    expect(
      isNavPathApplicable("/some/plugin/tab", "architect", true),
    ).toBe(false);
  });
});
