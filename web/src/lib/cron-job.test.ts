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
    script: "",
    no_agent: false,
    context_from: "",
    continuity: false,
    enabled_toolsets: [],
    workdir: "",
    reasoning_effort: "",
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

  it("stores continuity as the reserved self entry", () => {
    const payload = buildCronJobPayload(
      form({ continuity: true, context_from: "upstream-a" }),
    );

    expect(payload.context_from).toEqual(["upstream-a", "self"]);
  });

  it("continuity off strips any hand-typed self entry", () => {
    const payload = buildCronJobPayload(
      form({ continuity: false, context_from: "SELF\nupstream-a" }),
    );

    expect(payload.context_from).toEqual(["upstream-a"]);
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
      reasoning_effort: null,
    });
  });

  it("pins reasoning_effort only for known levels and clears otherwise", () => {
    expect(
      buildCronJobPayload(form({ reasoning_effort: "none" })).reasoning_effort,
    ).toBe("none");
    expect(
      buildCronJobPayload(form({ reasoning_effort: "HIGH" })).reasoning_effort,
    ).toBe("high");
    // "" = follow config; update payloads must explicitly clear the pin.
    expect(
      buildCronJobPayload(form({ reasoning_effort: "" })).reasoning_effort,
    ).toBeNull();
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
      continuity: false,
      enabled_toolsets: ["web"],
    });
  });

  it("splits the stored self entry into the continuity toggle", () => {
    const job: CronJob = {
      id: "abc",
      enabled: true,
      schedule_display: "every 1h",
      context_from: ["self", "upstream-a"],
    };

    expect(cronJobFormFromJob(job)).toMatchObject({
      context_from: "upstream-a",
      continuity: true,
    });
  });

  it("hydrates the reasoning effort pin and defaults to follow-config", () => {
    expect(
      cronJobFormFromJob({
        id: "abc",
        enabled: true,
        schedule_display: "every 1h",
        reasoning_effort: "none",
      } as CronJob).reasoning_effort,
    ).toBe("none");
    expect(
      cronJobFormFromJob({
        id: "abc",
        enabled: true,
        schedule_display: "every 1h",
      } as CronJob).reasoning_effort,
    ).toBe("");
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
