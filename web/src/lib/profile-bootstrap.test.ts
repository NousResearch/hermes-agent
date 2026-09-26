import { afterEach, describe, expect, it, vi } from "vitest";

import {
  dashboardServingProfile,
  initialProfileScope,
  shouldAdoptActiveProfile,
  shouldReassertProfileParam,
} from "./profile-bootstrap";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("initialProfileScope", () => {

  it("does not replace a launch profile with the sticky active profile", () => {
    expect(
      shouldAdoptActiveProfile(null, "worker_x", "default", "review"),
    ).toBe(false);
  });

  it("uses the sticky active profile without a URL or launch profile", () => {
    expect(
      shouldAdoptActiveProfile(null, "", "default", "review"),
    ).toBe(true);
  });
});

describe("dashboardServingProfile", () => {
  it("names no profile when there is no window at all", () => {
    expect(dashboardServingProfile()).toBe("");
  });

  it.each([
    ["an injected serving profile", { __HERMES_DASHBOARD_PROFILE__: "served" }, "served"],
    ["a window without one", {}, ""],
  ])("reports %s", (_label, windowStub, expected) => {
    vi.stubGlobal("window", windowStub);
    expect(dashboardServingProfile()).toBe(expected);
  });
});

describe("initialProfileScope precedence", () => {
  // URL > bootstrap > serving. The serving profile is the LAST resort: it says
  // out loud what an unnamed request already meant, so it must never override a
  // scope the URL or the bootstrap payload already named.
  it.each([
    ["the URL profile outranks bootstrap and serving", "profile=url", "boot", "served", "url"],
    ["an explicit empty URL profile still outranks both", "profile=", "boot", "served", ""],
    ["the bootstrap profile outranks the serving profile", "resume=s1", "boot", "served", "boot"],
    ["the serving profile is used when nothing else names one", "resume=s1", "", "served", "served"],
    ["no scope is invented when nothing names one", "resume=s1", "", "", ""],
  ])("%s", (_label, query, bootstrap, serving, expected) => {
    expect(
      initialProfileScope(new URLSearchParams(query), bootstrap, serving),
    ).toBe(expected);
  });

  it("defaults the serving profile to the one this backend injected", () => {
    vi.stubGlobal("window", { __HERMES_DASHBOARD_PROFILE__: "served" });
    expect(initialProfileScope(new URLSearchParams("resume=s1"), "")).toBe("served");
  });
});

describe("shouldReassertProfileParam", () => {
  // The root path renders `<Navigate to="/sessions" replace />` and nothing else.
  // Writing ?profile= into the location there replaces the URL out from under that
  // navigate: on a cold load the replace lands first, the already-mounted
  // <Navigate> never fires again, and the dashboard parks on /?profile=… with an
  // empty page until the user reloads by hand. The effect re-runs on /sessions.
  it("never rewrites the location on the redirect-only root path", () => {
    expect(shouldReassertProfileParam("/", "default", null)).toBe(false);
    expect(shouldReassertProfileParam("/", "default", "")).toBe(false);
    expect(shouldReassertProfileParam("/", "default", "other")).toBe(false);
    expect(shouldReassertProfileParam("/", "", null)).toBe(false);
  });

  it.each([
    ["a bare nav link", "/skills", "default", null, true],
    ["a nav link that dropped the param", "/config", "review", "", true],
    ["an in-app scope switch", "/profiles", "review", "default", true],
    ["an already-synced URL", "/skills", "default", "default", false],
    ["the dashboard's own profile with no param", "/skills", "", null, false],
    ["a matching empty param", "/skills", "", "", false],
  ])(
    "%s (%s?profile=%s, scope=%s)",
    (_label, pathname, profile, urlProfile, expected) => {
      expect(shouldReassertProfileParam(pathname, profile, urlProfile)).toBe(expected);
    },
  );
});
