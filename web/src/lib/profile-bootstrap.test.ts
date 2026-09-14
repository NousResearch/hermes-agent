import { describe, expect, it } from "vitest";

import {
  initialProfileScope,
  shouldAdoptActiveProfile,
} from "./profile-bootstrap";

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
