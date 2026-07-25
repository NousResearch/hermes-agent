import { describe, expect, it } from "vitest";
import {
  buildKeepSkillsPayload,
  profileSkillSelectionSummary,
} from "./profile-builder-skills";

describe("profileSkillSelectionSummary", () => {
  it("reports the full skill set even when the visible list is filtered", () => {
    const skills = [{ name: "alpha" }, { name: "beta" }, { name: "gamma" }];
    const kept = new Set(["alpha", "beta", "gamma"]);

    expect(profileSkillSelectionSummary(skills, kept)).toEqual({
      selected: 3,
      total: 3,
    });
  });

  it("ignores stale selections that are no longer in the loaded skill set", () => {
    const skills = [{ name: "alpha" }, { name: "beta" }];
    const kept = new Set(["alpha", "removed-skill"]);

    expect(profileSkillSelectionSummary(skills, kept)).toEqual({
      selected: 1,
      total: 2,
    });
  });
});

describe("buildKeepSkillsPayload", () => {
  it("preserves the full-bundle sentinel", () => {
    expect(buildKeepSkillsPayload(true, new Set(["alpha"]))).toBeUndefined();
  });

  it("emits an empty keep list after deselect all", () => {
    expect(buildKeepSkillsPayload(false, new Set())).toEqual([]);
  });
});
