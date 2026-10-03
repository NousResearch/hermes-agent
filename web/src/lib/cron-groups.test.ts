import { describe, expect, it } from "vitest";

import {
  UNCATEGORISED_GROUP,
  cronJobCategories,
  cronJobCategory,
  groupCronJobsByCategory,
  hasCategorisedJobs,
} from "./cron-groups";
import type { CronJob } from "./api";

function job(id: string, category?: string): CronJob {
  return { id, name: id, category } as unknown as CronJob;
}

describe("cronJobCategory", () => {
  it("trims a stored label and treats blanks as unset", () => {
    expect(cronJobCategory(job("a", "  Family  "))).toBe("Family");
    expect(cronJobCategory(job("a", "   "))).toBe("");
    expect(cronJobCategory(job("a"))).toBe("");
    expect(cronJobCategory({ category: 42 } as unknown as CronJob)).toBe("");
  });
});

describe("hasCategorisedJobs", () => {
  it("is true only when some job carries a label", () => {
    expect(hasCategorisedJobs([job("a"), job("b")])).toBe(false);
    expect(hasCategorisedJobs([job("a"), job("b", "  ")])).toBe(false);
    expect(hasCategorisedJobs([job("a", "Tech"), job("b")])).toBe(true);
  });
});

describe("groupCronJobsByCategory", () => {
  it("puts every job with the same label in one group, in input order", () => {
    const groups = groupCronJobsByCategory([
      job("b1", "Family"),
      job("a1", "agent"),
      job("b2", " Family "),
    ]);
    expect(groups.map((g) => g.category)).toEqual(["agent", "Family"]);
    expect(groups[1].jobs.map((j) => j.id)).toEqual(["b1", "b2"]);
  });

  it("sorts labelled groups case-insensitively and keeps the unlabelled bucket last", () => {
    const groups = groupCronJobsByCategory([
      job("x"),
      job("beta", "beta"),
      job("alpha", "Alpha"),
      job("y"),
    ]);
    expect(groups.map((g) => g.category)).toEqual(["Alpha", "beta", UNCATEGORISED_GROUP]);
    expect(groups[2].jobs.map((j) => j.id)).toEqual(["x", "y"]);
  });

  it("never mutates or reorders the caller's array", () => {
    const jobs = [job("b", "Family"), job("a"), job("c", "Family")];
    const before = jobs.map((j) => j.id);
    groupCronJobsByCategory(jobs);
    expect(jobs.map((j) => j.id)).toEqual(before);
  });

  it("is a partition: every job appears exactly once", () => {
    const jobs = [job("a", "one"), job("b"), job("c", "two"), job("d", "one")];
    const seen = groupCronJobsByCategory(jobs).flatMap((g) => g.jobs.map((j) => j.id));
    expect(seen.sort()).toEqual(["a", "b", "c", "d"]);
  });
});

describe("cronJobCategories", () => {
  it("lists each label once, trimmed and case-insensitively deduped, sorted", () => {
    const jobs = [
      job("a", "  Family  "),
      job("b", "backups"),
      job("c"),
      job("d", "family"), // same label, different spelling
      job("e", "Backups"),
    ];
    expect(cronJobCategories(jobs)).toEqual(["backups", "Family"]);
  });

  it("returns nothing when no job carries a label", () => {
    expect(cronJobCategories([job("a"), job("b", "   ")])).toEqual([]);
  });
});
