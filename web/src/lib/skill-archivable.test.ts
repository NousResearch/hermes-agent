import { describe, expect, it } from "vitest";

import { isArchivableSkill } from "@/lib/skill-archivable";

describe("isArchivableSkill", () => {
  it("offers Delete for a local skill listed by its plain name", () => {
    expect(isArchivableSkill({ name: "dup", provenance: "agent" })).toBe(true);
  });

  it("hides Delete on a same-tier duplicate listed by its relative path", () => {
    expect(isArchivableSkill({ name: "a/dup", provenance: "agent" })).toBe(false);
  });

  it("hides Delete for skills the backend refuses to archive", () => {
    for (const provenance of ["hub", "bundled", "external", undefined]) {
      expect(isArchivableSkill({ name: "x", provenance })).toBe(false);
    }
  });
});
