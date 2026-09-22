import { afterEach, describe, expect, it, vi } from "vitest";

import {
  dashboardServingProfile,
  initialProfileScope,
  shouldAdoptActiveProfile,
} from "./profile-bootstrap";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("initialProfileScope", () => {
  it("inherits the dashboard bootstrap profile when the URL omits profile", () => {
    expect(initialProfileScope(new URLSearchParams("resume=session-1"), "worker_x"))
      .toBe("worker_x");
  });

  it("keeps an explicit URL profile authoritative", () => {
    expect(
      initialProfileScope(
        new URLSearchParams("resume=session-1&profile=review"),
        "worker_x",
      ),
    ).toBe("review");
  });

  it("preserves an explicit empty profile", () => {
    expect(
      initialProfileScope(new URLSearchParams("profile="), "worker_x"),
    ).toBe("");
  });

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
