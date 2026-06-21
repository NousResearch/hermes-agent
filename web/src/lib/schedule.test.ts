import { describe, it, expect } from "vitest";
import {
  describeSchedule,
  englishOrdinal,
  type ScheduleDescribeStrings,
} from "./schedule";

/** Minimal English strings so the assertions read naturally. The
 * humanized branches interpolate {time}/{days}/{day}; the fallback
 * branches return the raw expression untouched. */
const strings: ScheduleDescribeStrings = {
  none: "No schedule",
  everyMinutes: "Every {n} minutes",
  everyHours: "Every {n} hours",
  everyDays: "Every {n} days",
  dailyAt: "Daily at {time}",
  weeklyAt: "Weekly on {days} at {time}",
  monthlyAt: "Monthly on the {day} at {time}",
  onceAt: "Once at {time}",
  weekdaysShort: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
  ordinal: englishOrdinal,
};

describe("describeSchedule cron day-of-month humanization", () => {
  it("does NOT silently drop extra days for a multi-DOM cron", () => {
    // "0 9 1,15 * *" fires on the 1st AND the 15th. The humanizer must
    // not render it as "Monthly on the 1st ...", which omits the 15th.
    const result = describeSchedule(
      { kind: "cron", expr: "0 9 1,15 * *" },
      "0 9 1,15 * *",
      strings,
    );
    // Falls back to the raw expression rather than a misleading sentence.
    expect(result).toBe("0 9 1,15 * *");
    expect(result).not.toContain("Monthly on the 1st");
  });

  it("still humanizes a single day-of-month cron (guard is not over-broad)", () => {
    const result = describeSchedule(
      { kind: "cron", expr: "0 9 15 * *" },
      "0 9 15 * *",
      strings,
    );
    expect(result).toBe("Monthly on the 15th at 09:00");
  });

  it("still humanizes a multi-day-of-WEEK cron (fix is narrow to the dom branch)", () => {
    const result = describeSchedule(
      { kind: "cron", expr: "0 9 * * 1,3" },
      "0 9 * * 1,3",
      strings,
    );
    expect(result).toBe("Weekly on Mon, Wed at 09:00");
  });

  it("falls back to raw cron for a multi-DOM expr even with no fallbackDisplay", () => {
    // describeSchedule is commonly called with fallbackDisplay=undefined.
    // The "don't render a misleading single-day sentence" outcome must not
    // depend on fallbackDisplay being present — the raw expr is recovered
    // from schedule.expr in that case.
    const result = describeSchedule(
      { kind: "cron", expr: "0 9 1,15 * *" },
      undefined,
      strings,
    );
    expect(result).toBe("0 9 1,15 * *");
    expect(result).not.toContain("Monthly on the 1st");
  });

  it("does NOT misrender a DOM cron operator as a single day", () => {
    // parseInt("1-15"|"1/2"|"1L"|"1W"|"15#2", 10) would truncate to a
    // single number and produce a wrong "Monthly on the Nth" sentence. The
    // literal-or-list guard must reject every non-digits-only DOM token and
    // bail to the raw expression instead.
    for (const expr of [
      "0 9 1-15 * *",
      "0 9 1/2 * *",
      "0 9 1L * *",
      "0 9 1W * *",
      "0 9 15#2 * *",
    ]) {
      const result = describeSchedule({ kind: "cron", expr }, expr, strings);
      expect(result).toBe(expr);
      expect(result).not.toContain("Monthly");
    }
  });
});
