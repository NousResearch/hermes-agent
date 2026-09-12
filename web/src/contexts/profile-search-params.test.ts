import { describe, expect, it } from "vitest";

import { profileSelectionSearchParams } from "./profile-search-params";

describe("profileSelectionSearchParams", () => {
  it("drops a resume target when the effective profile changes", () => {
    const previous = new URLSearchParams(
      "profile=default&resume=session-from-default&panel=models",
    );

    const next = profileSelectionSearchParams(
      previous,
      "research",
      "default",
      "default",
    );

    expect(next.get("profile")).toBe("research");
    expect(next.has("resume")).toBe(false);
    expect(next.get("panel")).toBe("models");
  });

  it("keeps the resume target when the effective profile is unchanged", () => {
    const previous = new URLSearchParams(
      "profile=default&resume=session-from-default",
    );

    const next = profileSelectionSearchParams(
      previous,
      "",
      "default",
      "default",
    );

    expect(next.has("profile")).toBe(false);
    expect(next.get("resume")).toBe("session-from-default");
  });
});
