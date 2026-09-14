import { describe, expect, it } from "vitest";

import {
  buildCronJobPayload,
  cronJobHasExecutionContent,
  cronJobFormFromJob,
  cronLastResult,
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

describe("cronLastResult", () => {
  it("renders nothing for a job that never ran", () => {
    expect(cronLastResult({ last_status: null })).toBeNull();
    expect(cronLastResult({ last_status: "" })).toBeNull();
  });

  it("is green for ok with no detail", () => {
    expect(cronLastResult({ last_status: "ok", last_error: null })).toEqual({
      status: "ok",
      tone: "success",
      detail: null,
    });
  });

  it("is amber for delivery_failed and explains it from last_delivery_error", () => {
    // The agent run succeeded (last_error is null for these runs); the reason
    // lives in last_delivery_error. Must never render as green or as "unknown".
    expect(
      cronLastResult({
        last_status: "delivery_failed",
        last_error: null,
        last_delivery_error: "telegram: 502 Bad Gateway",
      }),
    ).toEqual({
      status: "delivery_failed",
      tone: "warning",
      detail: "telegram: 502 Bad Gateway",
    });
  });

  it("is red for error and any unrecognised literal", () => {
    expect(cronLastResult({ last_status: "error", last_error: "boom" })).toEqual({
      status: "error",
      tone: "destructive",
      detail: "boom",
    });
    expect(cronLastResult({ last_status: "something_new" })?.tone).toBe("destructive");
  });

  it("is amber for blocked_config (preflight refused to burn a run)", () => {
    expect(
      cronLastResult({ last_status: "blocked_config", last_error: "missing API key" }),
    ).toEqual({ status: "blocked_config", tone: "warning", detail: "missing API key" });
  });
});
