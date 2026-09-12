import { describe, expect, it } from "vitest";

import type { CronJob } from "./api";
import { classifyOccurrences, enumerateOccurrences } from "./cron-occurrences";

/** Build a minimal CronJob for the occurrence enumerator. */
function job(
  schedule: CronJob["schedule"],
  extra: Partial<CronJob> = {},
): CronJob {
  return { id: "j1", enabled: true, schedule, ...extra };
}

/** Epoch ms for a UTC wall-clock instant (month 1-12). */
function at(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute = 0,
): number {
  return Date.UTC(year, month - 1, day, hour, minute);
}

/** Convenience: enumerate and return the firing instants (times array). */
function times(
  cronJob: CronJob,
  fromMs: number,
  toMs: number,
  timeZone: string | null = "UTC",
): number[] {
  return enumerateOccurrences(cronJob, fromMs, toMs, timeZone).times;
}

describe("enumerateOccurrences — interval", () => {
  it("anchors on next_run_at and steps by the interval", () => {
    const j = job(
      { kind: "interval", minutes: 30 },
      { next_run_at: "2026-09-10T12:00:00Z" },
    );
    const result = times(j, at(2026, 9, 10, 10, 0), at(2026, 9, 10, 14, 0));
    expect(result).toEqual([
      at(2026, 9, 10, 10, 0),
      at(2026, 9, 10, 10, 30),
      at(2026, 9, 10, 11, 0),
      at(2026, 9, 10, 11, 30),
      at(2026, 9, 10, 12, 0),
      at(2026, 9, 10, 12, 30),
      at(2026, 9, 10, 13, 0),
      at(2026, 9, 10, 13, 30),
      at(2026, 9, 10, 14, 0),
    ]);
  });
});

describe("enumerateOccurrences — once", () => {
  it("emits the single run_at instant when inside the window", () => {
    const j = job({ kind: "once", run_at: "2026-09-10T14:00:00Z" });
    expect(times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 11, 0, 0))).toEqual([
      at(2026, 9, 10, 14, 0),
    ]);
  });

  it("emits nothing when run_at is outside the window", () => {
    const j = job({ kind: "once", run_at: "2026-09-10T14:00:00Z" });
    expect(times(j, at(2026, 9, 11, 0, 0), at(2026, 9, 12, 0, 0))).toEqual([]);
  });
});

describe("enumerateOccurrences — 5-field cron", () => {
  it("enumerates a daily schedule", () => {
    const j = job({ kind: "cron", expr: "0 9 * * *" });
    expect(times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 12, 0, 0))).toEqual([
      at(2026, 9, 10, 9, 0),
      at(2026, 9, 11, 9, 0),
    ]);
  });

  it("enumerates a weekly schedule (Mon/Wed/Fri)", () => {
    // 2026-09-07 is a Monday; 09-09 Wednesday; 09-11 Friday.
    const j = job({ kind: "cron", expr: "30 14 * * 1,3,5" });
    expect(times(j, at(2026, 9, 7, 0, 0), at(2026, 9, 14, 0, 0))).toEqual([
      at(2026, 9, 7, 14, 30),
      at(2026, 9, 9, 14, 30),
      at(2026, 9, 11, 14, 30),
    ]);
  });

  it("honours day-of-month / day-of-week OR semantics", () => {
    // `0 0 1 * 1` fires on the 1st of the month OR on Mondays.
    const j = job({ kind: "cron", expr: "0 0 1 * 1" });
    expect(times(j, at(2026, 9, 1, 0, 0), at(2026, 9, 15, 0, 0))).toEqual([
      at(2026, 9, 1, 0, 0), // the 1st (a Tuesday)
      at(2026, 9, 7, 0, 0), // Monday
      at(2026, 9, 14, 0, 0), // Monday
    ]);
  });

  it("supports ranges and steps", () => {
    const j = job({ kind: "cron", expr: "*/15 9-10 * * *" });
    expect(times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 11, 0, 0))).toEqual([
      at(2026, 9, 10, 9, 0),
      at(2026, 9, 10, 9, 15),
      at(2026, 9, 10, 9, 30),
      at(2026, 9, 10, 9, 45),
      at(2026, 9, 10, 10, 0),
      at(2026, 9, 10, 10, 15),
      at(2026, 9, 10, 10, 30),
      at(2026, 9, 10, 10, 45),
    ]);
  });
});

describe("enumerateOccurrences — 6-field cron (seconds)", () => {
  it("enumerates a 6-field expression (seconds field does not shift the minute)", () => {
    // `0 9 * * * 30` = 09:00:30 daily; the timeline is minute-granular so the
    // marker lands on 09:00. The old 5-field-only matcher rejected this and
    // fell back to next/last markers only.
    const j = job({ kind: "cron", expr: "0 9 * * * 30" });
    expect(times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 12, 0, 0))).toEqual([
      at(2026, 9, 10, 9, 0),
      at(2026, 9, 11, 9, 0),
    ]);
  });

  it("enumerates a 6-field wildcard-seconds expression", () => {
    const j = job({ kind: "cron", expr: "0 9 * * * *" });
    expect(times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 12, 0, 0))).toEqual([
      at(2026, 9, 10, 9, 0),
      at(2026, 9, 11, 9, 0),
    ]);
  });

  it("rejects a malformed seconds field and yields no client-derived markers", () => {
    const j = job({ kind: "cron", expr: "0 9 * * * 61" });
    const result = enumerateOccurrences(j, at(2026, 9, 10, 0, 0), at(2026, 9, 12, 0, 0), "UTC");
    expect(result.resolved).toBe(false);
    expect(result.times).toEqual([]);
  });
});

describe("enumerateOccurrences — timezone", () => {
  it("matches cron in the configured Hermes timezone, not the browser's", () => {
    // 14:30 in Asia/Tokyo (UTC+9, no DST) = 05:30 UTC. A browser in UTC that
    // matched locally would place this at 14:30 UTC instead.
    const j = job({ kind: "cron", expr: "30 14 * * *" });
    expect(times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 11, 0, 0), "Asia/Tokyo")).toEqual([
      at(2026, 9, 10, 5, 30),
    ]);
  });

  it("produces different instants for the same expression in different zones", () => {
    const j = job({ kind: "cron", expr: "30 14 * * *" });
    const tokyo = times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 11, 0, 0), "Asia/Tokyo");
    const utc = times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 11, 0, 0), "UTC");
    expect(tokyo).toEqual([at(2026, 9, 10, 5, 30)]);
    expect(utc).toEqual([at(2026, 9, 10, 14, 30)]);
    expect(tokyo).not.toEqual(utc);
  });
});

describe("enumerateOccurrences — 30-day window", () => {
  const DAY_MS = 86_400_000;
  const from = at(2026, 10, 10, 0, 0);
  const to = from + 30 * DAY_MS;
  const TZ = "Europe/Amsterdam";

  /** The zone's local calendar date, e.g. "2026-10-10" (host-zone independent). */
  const localDate = (ms: number): string =>
    new Intl.DateTimeFormat("en-CA", { timeZone: TZ }).format(new Date(ms));

  /** The zone's local hour of an instant. */
  const localHour = (ms: number): number =>
    Number(
      new Intl.DateTimeFormat("en-US", {
        timeZone: TZ,
        hour: "2-digit",
        hourCycle: "h23",
      })
        .formatToParts(new Date(ms))
        .find((p) => p.type === "hour")?.value,
    );

  it("keeps a daily job on its wall-clock hour across a DST transition", () => {
    // The window spans the Europe/Amsterdam DST end (2026-10-25), so the
    // zone's UTC offset changes mid-window. The job must still fire at 06:00
    // local on all 30 days — no day skipped, none doubled.
    const result = times(job({ kind: "cron", expr: "0 6 * * *" }), from, to, TZ);

    expect(result).toHaveLength(30);
    expect(result.map(localHour)).toEqual(Array(30).fill(6));
    const dates = result.map(localDate);
    expect(new Set(dates).size).toBe(30);
    for (let i = 1; i < dates.length; i++) {
      expect(Date.parse(dates[i]) - Date.parse(dates[i - 1])).toBe(DAY_MS);
    }
  });

  it("caps a minute-frequency job into a dense lane instead of ~43k markers", () => {
    const lane = enumerateOccurrences(
      job({ kind: "cron", expr: "* * * * *" }),
      from,
      to,
      TZ,
    );
    expect(lane.dense).toBe(true);
    expect(lane.times.length).toBeLessThanOrEqual(302);
  });
});

describe("enumerateOccurrences — backend marker fallback", () => {
  it("folds in next_run_at / last_run_at even when the matcher can't parse the expr", () => {
    // `@daily` is not modelled by the client matcher; the authoritative
    // backend markers must still appear.
    const j = job(
      { kind: "cron", expr: "@daily" },
      {
        last_run_at: "2026-09-10T06:00:00Z",
        next_run_at: "2026-09-11T06:00:00Z",
      },
    );
    expect(times(j, at(2026, 9, 10, 0, 0), at(2026, 9, 12, 0, 0))).toEqual([
      at(2026, 9, 10, 6, 0),
      at(2026, 9, 11, 6, 0),
    ]);
  });
});

describe("classifyOccurrences", () => {
  const t1 = at(2026, 9, 10, 8, 0);
  const t2 = at(2026, 9, 10, 9, 0);
  const t3 = at(2026, 9, 10, 10, 0);

  it("marks the soonest at-or-after now as next, earlier past, later future", () => {
    const now = at(2026, 9, 10, 8, 30);
    expect(classifyOccurrences([t1, t2, t3], now)).toEqual([
      { time: t1, kind: "past" },
      { time: t2, kind: "next" },
      { time: t3, kind: "future" },
    ]);
  });

  it("marks everything past once now is beyond the last instant", () => {
    const now = at(2026, 9, 10, 11, 0);
    expect(classifyOccurrences([t1, t2, t3], now).map((o) => o.kind)).toEqual([
      "past",
      "past",
      "past",
    ]);
  });

  it("marks everything future when now precedes the first instant", () => {
    const now = at(2026, 9, 10, 7, 0);
    expect(classifyOccurrences([t1, t2, t3], now).map((o) => o.kind)).toEqual([
      "next",
      "future",
      "future",
    ]);
  });
});
