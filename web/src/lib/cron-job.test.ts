import { describe, expect, it } from "vitest";

import {
  buildCronJobPayload,
  cronJobHasExecutionContent,
  cronJobFormFromJob,
  findJobByDeepLink,
  parseDeepLinkJobKey,
  splitCronList,
  type CronJobFormState,
} from "./cron-job";
import type { CronJob } from "./api";

function form(overrides: Partial<CronJobFormState> = {}): CronJobFormState {
  return {
    name: "",
    prompt: "prompt",
    schedule: "every 1h",
    deliver: "local",
    skills: [],
    provider: "",
    model: "",
    base_url: "",
    script: "",
    no_agent: false,
    context_from: "",
    enabled_toolsets: [],
    workdir: "",
    ...overrides,
  };
}

describe("splitCronList", () => {
  it("normalizes comma and newline separated cron list fields", () => {
    expect(splitCronList(" web, terminal\nfile ,, ")).toEqual([
      "web",
      "terminal",
      "file",
    ]);
  });
});

describe("buildCronJobPayload", () => {
  it("normalizes list fields and base URLs", () => {
    const payload = buildCronJobPayload(
      form({
        base_url: "https://example.invalid/v1/",
        enabled_toolsets: ["web", ""],
        context_from: "upstream-a\nupstream-b",
      }),
    );

    expect(payload).toMatchObject({
      base_url: "https://example.invalid/v1",
      context_from: ["upstream-a", "upstream-b"],
      enabled_toolsets: ["web"],
    });
  });

  it("keeps clear operations explicit for update payloads", () => {
    const payload = buildCronJobPayload(form({ schedule: "every 2h" }));

    expect(payload).toMatchObject({
      schedule: "every 2h",
      provider: null,
      model: null,
      base_url: null,
      script: null,
      no_agent: false,
      context_from: null,
      enabled_toolsets: null,
      workdir: null,
    });
  });
});

describe("cronJobHasExecutionContent", () => {
  it("treats a script as execution content for agent-backed cron jobs", () => {
    const payload = buildCronJobPayload(
      form({ prompt: "", skills: [], script: "collect-status.py" }),
    );

    expect(cronJobHasExecutionContent(payload)).toBe(true);
  });

  it("rejects payloads with no prompt, skills, or script", () => {
    const payload = buildCronJobPayload(form({ prompt: "", skills: [], script: "" }));

    expect(cronJobHasExecutionContent(payload)).toBe(false);
  });
});

describe("cronJobFormFromJob", () => {
  it("preserves schedule fallback and editable list fields", () => {
    const job: CronJob = {
      id: "abc",
      enabled: true,
      schedule_display: "every 1h",
      context_from: ["upstream-a", "upstream-b"],
      enabled_toolsets: ["web"],
    };

    expect(cronJobFormFromJob(job)).toMatchObject({
      schedule: "every 1h",
      context_from: "upstream-a\nupstream-b",
      enabled_toolsets: ["web"],
    });
  });

  it("prefers one-shot run_at over the human display string", () => {
    const job: CronJob = {
      id: "once-job",
      enabled: true,
      schedule: {
        kind: "once",
        run_at: "2026-02-03T14:00:00+08:00",
      },
      schedule_display: "once at 2026-02-03 14:00",
    };

    expect(cronJobFormFromJob(job)).toMatchObject({
      schedule: "2026-02-03T14:00:00+08:00",
    });
  });
});

describe("cron deep-link helpers", () => {
  it("parses a profile:job key", () => {
    expect(parseDeepLinkJobKey("delinat:abc123")).toEqual({
      profile: "delinat",
      id: "abc123",
    });
  });

  it("defaults a bare job id to the default profile", () => {
    expect(parseDeepLinkJobKey("abc123")).toEqual({
      profile: "default",
      id: "abc123",
    });
  });

  it("handles an empty profile segment", () => {
    expect(parseDeepLinkJobKey(":abc123")).toEqual({
      profile: "default",
      id: "abc123",
    });
  });

  it("finds a job by its bare id within a loaded list", () => {
    const jobs = [{ id: "one", enabled: true }, { id: "two", enabled: true }];
    expect(findJobByDeepLink(jobs, "delinat:two")?.id).toBe("two");
  });

  it("returns null when the job is not in the loaded list", () => {
    const jobs = [{ id: "one", enabled: true }];
    expect(findJobByDeepLink(jobs, "delinat:missing")).toBeNull();
  });
});
