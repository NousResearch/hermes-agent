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
    reasoning_effort: null,
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
        reasoning_effort: " NONE ",
        enabled_toolsets: ["web", ""],
        context_from: "upstream-a\nupstream-b",
      }),
    );

    expect(payload).toMatchObject({
      base_url: "https://example.invalid/v1",
      context_from: ["upstream-a", "upstream-b"],
      enabled_toolsets: ["web"],
      reasoning_effort: "none",
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
  it("emits null when the reasoning selector is intentionally cleared", () => {
    expect(
      buildCronJobPayload(form({ reasoning_effort: "" })).reasoning_effort,
    ).toBeNull();
  });
  it("canonicalizes malformed legacy reasoning before unrelated edits", () => {
    const hydrated = cronJobFormFromJob({
      id: "legacy-invalid",
      enabled: true,
      reasoning_effort: " turbo ",
    });

    expect(hydrated.reasoning_effort).toBeNull();
    expect(
      buildCronJobPayload({ ...hydrated, name: "renamed" }).reasoning_effort,
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
  it("hydrates inherit/default separately from explicit none", () => {
    const inherited = cronJobFormFromJob({ id: "inherit", enabled: true });
    const none = cronJobFormFromJob({
      id: "none",
      enabled: true,
      reasoning_effort: "none",
    });
    const legacyNone = cronJobFormFromJob({
      id: "legacy-none",
      enabled: true,
      reasoning_effort: false,
    });

    expect(inherited.reasoning_effort).toBeNull();
    expect(none.reasoning_effort).toBe("none");
    expect(legacyNone.reasoning_effort).toBe("none");
  });

  it("retains configured effort for no_agent jobs", () => {
    const formState = cronJobFormFromJob({
      id: "script-only",
      enabled: true,
      no_agent: true,
      reasoning_effort: "HIGH",
    });

    expect(formState).toMatchObject({
      no_agent: true,
      reasoning_effort: "high",
    });
    expect(buildCronJobPayload(formState).reasoning_effort).toBe("high");
  });
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
