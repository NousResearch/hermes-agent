import { describe, expect, it } from "vitest";

import {
  buildCronJobPayload,
  cronJobHasExecutionContent,
  cronJobFormFromJob,
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
    reasoning_effort: "",
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
      reasoning_effort: null,
      script: null,
      no_agent: false,
      context_from: null,
      enabled_toolsets: null,
      workdir: null,
    });
  });

  it("round-trips explicit reasoning overrides and clear semantics", () => {
    expect(buildCronJobPayload(form({ reasoning_effort: "ultra" }))).toMatchObject({
      reasoning_effort: "ultra",
    });
    expect(buildCronJobPayload(form({ reasoning_effort: "none" }))).toMatchObject({
      reasoning_effort: "none",
    });
    expect(buildCronJobPayload(form({ reasoning_effort: "" }))).toMatchObject({
      reasoning_effort: null,
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
      reasoning_effort: "max",
    };

    expect(cronJobFormFromJob(job)).toMatchObject({
      schedule: "every 1h",
      context_from: "upstream-a\nupstream-b",
      enabled_toolsets: ["web"],
      reasoning_effort: "max",
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

  it("does not resubmit a malformed stored override on unrelated edits", () => {
    const job: CronJob = {
      id: "legacy-job",
      enabled: true,
      reasoning_effort: "turbo",
      reasoning_effort_status: "invalid_inherits_global",
    };

    expect(cronJobFormFromJob(job).reasoning_effort).toBe("");
  });
});
