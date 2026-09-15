import { describe, expect, it } from "vitest";

import {
  buildScheduleString,
  DEFAULT_SCHEDULE_STATE,
  parseScheduleString,
} from "./schedule";

describe("parseScheduleString", () => {
  it("parses recurring interval strings", () => {
    expect(parseScheduleString("every 30m")).toMatchObject({
      mode: "interval",
      intervalValue: 30,
      intervalUnit: "minutes",
    });
    expect(parseScheduleString("every 2h")).toMatchObject({
      mode: "interval",
      intervalValue: 2,
      intervalUnit: "hours",
    });
    expect(parseScheduleString("every 1d")).toMatchObject({
      mode: "interval",
      intervalValue: 1,
      intervalUnit: "days",
    });
  });

  it("parses ISO timestamps into once mode", () => {
    expect(parseScheduleString("2026-02-03T14:00:00")).toMatchObject({
      mode: "once",
      onceAt: "2026-02-03T14:00",
    });
    expect(parseScheduleString("2026-02-03T14:00")).toMatchObject({
      mode: "once",
      onceAt: "2026-02-03T14:00",
    });
  });

  it("parses daily cron expressions", () => {
    expect(parseScheduleString("0 9 * * *")).toMatchObject({
      mode: "daily",
      timeOfDay: "09:00",
    });
  });

  it("parses weekly cron expressions", () => {
    expect(parseScheduleString("30 14 * * 1,3,5")).toMatchObject({
      mode: "weekly",
      timeOfDay: "14:30",
      weekdays: [1, 3, 5],
    });
  });

  it("normalizes cron Sunday 7 into the builder's Sunday 0", () => {
    expect(parseScheduleString("30 14 * * 1,7")).toMatchObject({
      mode: "weekly",
      timeOfDay: "14:30",
      weekdays: [1, 0],
    });
  });

  it("parses monthly cron expressions", () => {
    expect(parseScheduleString("0 9 15 * *")).toMatchObject({
      mode: "monthly",
      timeOfDay: "09:00",
      dayOfMonth: 15,
    });
  });

  it("falls back to custom for unsupported schedule strings", () => {
    expect(parseScheduleString("0 9 * * 1-5")).toMatchObject({
      mode: "custom",
      custom: "0 9 * * 1-5",
    });
    expect(parseScheduleString("@daily")).toMatchObject({
      mode: "custom",
      custom: "@daily",
    });
    expect(parseScheduleString("2026-02-03T14:00:00Z")).toMatchObject({
      mode: "custom",
      custom: "2026-02-03T14:00:00Z",
    });
    expect(parseScheduleString("2026-02-03T14:00:00+08:00")).toMatchObject({
      mode: "custom",
      custom: "2026-02-03T14:00:00+08:00",
    });
    expect(parseScheduleString("0 9 * * 1,8")).toMatchObject({
      mode: "custom",
      custom: "0 9 * * 1,8",
    });
    expect(parseScheduleString("0 9 1,15 * *")).toMatchObject({
      mode: "custom",
      custom: "0 9 1,15 * *",
    });
  });

  it("returns the default state for empty input", () => {
    expect(parseScheduleString("")).toEqual(DEFAULT_SCHEDULE_STATE);
  });
});

describe("buildScheduleString round-trip", () => {
  it("rebuilds the schedule string from parsed state", () => {
    const cases: [string, string][] = [
      ["every 30m", "every 30m"],
      ["every 2h", "every 2h"],
      ["every 1d", "every 1d"],
      ["0 9 * * *", "0 9 * * *"],
      ["30 14 * * 1,3,5", "30 14 * * 1,3,5"],
      ["30 14 * * 1,7", "30 14 * * 0,1"],
      ["0 9 15 * *", "0 9 15 * *"],
      ["0 9 1,15 * *", "0 9 1,15 * *"],
      ["2026-02-03T14:00:00", "2026-02-03T14:00:00"],
      ["2026-02-03T14:00", "2026-02-03T14:00:00"],
      ["2026-02-03T14:00:00Z", "2026-02-03T14:00:00Z"],
      ["2026-02-03T14:00:00+08:00", "2026-02-03T14:00:00+08:00"],
    ];
    for (const [input, expected] of cases) {
      const state = parseScheduleString(input);
      expect(buildScheduleString(state)).toBe(expected);
    }
  });
});
