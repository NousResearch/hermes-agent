import type { CronJob } from "./api";

/** Bucket for jobs that carry no category (or only whitespace). */
export const UNCATEGORISED_GROUP = "";

export interface CronJobGroup {
  /** Trimmed label, `UNCATEGORISED_GROUP` for the fallback bucket. */
  category: string;
  jobs: CronJob[];
}

/** A job's category, trimmed; "" when unset or blank. */
export function cronJobCategory(job: Pick<CronJob, "category">): string {
  return typeof job.category === "string" ? job.category.trim() : "";
}

/** True when at least one job carries a category — the grouped view (and its
 *  toggle) is meaningless otherwise. */
export function hasCategorisedJobs(jobs: CronJob[]): boolean {
  return jobs.some((job) => cronJobCategory(job) !== "");
}

/** Every category currently in use, deduped case-insensitively and sorted.
 *  Feeds the editor's autocomplete: what you can pick is what already exists. */
export function cronJobCategories(jobs: CronJob[]): string[] {
  const seen = new Map<string, string>();
  for (const job of jobs) {
    const category = cronJobCategory(job);
    if (!category) continue;
    const key = category.toLowerCase();
    if (!seen.has(key)) seen.set(key, category);
  }
  return [...seen.values()].sort((a, b) =>
    a.localeCompare(b, undefined, { sensitivity: "base" }),
  );
}

/** Group jobs by category for the dashboard's grouped list.
 *
 *  Labels merge on an exact trimmed match ("family" and " family " are one
 *  group; "Family" stays its own label), labelled groups sort
 *  case-insensitively, the uncategorised bucket always comes last, and each
 *  group keeps the caller's order. The input array is not reordered or
 *  mutated — grouping an already-sorted list can never reshuffle it. */
export function groupCronJobsByCategory(jobs: CronJob[]): CronJobGroup[] {
  const buckets = new Map<string, CronJob[]>();
  for (const job of jobs) {
    const category = cronJobCategory(job);
    const bucket = buckets.get(category);
    if (bucket) bucket.push(job);
    else buckets.set(category, [job]);
  }
  const groups: CronJobGroup[] = [];
  for (const [category, groupJobs] of buckets) {
    groups.push({ category, jobs: groupJobs });
  }
  return groups.sort((a, b) => {
    const aUncategorised = a.category === UNCATEGORISED_GROUP;
    const bUncategorised = b.category === UNCATEGORISED_GROUP;
    if (aUncategorised !== bUncategorised) return aUncategorised ? 1 : -1;
    return a.category.localeCompare(b.category, undefined, { sensitivity: "base" });
  });
}
