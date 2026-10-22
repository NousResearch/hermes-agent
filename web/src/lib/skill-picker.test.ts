import { describe, expect, it } from "vitest";

import { filterSkills, launchSkillCommand } from "./skill-picker";

const ffmpeg = { name: "ffmpeg", description: "video encode", usage: 1 };
const hyperframes = { name: "hyperframes", description: "frames", usage: 9 };

describe("filterSkills", () => {
  it("keeps skills whose name or description contains the query (case-insensitive)", () => {
    expect(filterSkills([ffmpeg, hyperframes], "video")).toEqual([ffmpeg]);
  });

  it("sorts by usage descending, then name", () => {
    expect(filterSkills([ffmpeg, hyperframes], "")).toEqual([
      hyperframes,
      ffmpeg,
    ]);
  });

  it("treats missing usage as 0", () => {
    const noUsage = { name: "alpha", description: "first" };
    const zero = { name: "zeta", description: "last", usage: 0 };
    expect(filterSkills([zero, noUsage], "")).toEqual([noUsage, zero]);
  });
});

describe("launchSkillCommand", () => {
  it("prefixes a slash-skill command", () => {
    expect(launchSkillCommand("ffmpeg")).toBe("/ffmpeg");
  });

  it("returns null for an empty name so nothing is sent", () => {
    expect(launchSkillCommand("")).toBeNull();
    expect(launchSkillCommand("   ")).toBeNull();
  });
});
