/**
 * Schedule builder helpers for the cron page.
 *
 * The hermes-agent backend (cron/jobs.py::parse_schedule) accepts a
 * surprisingly broad set of string formats:
 *
 *   - Duration (one-shot):       "30m", "2h", "1d"
 *   - Interval (recurring):      "every 30m", "every 2h", "every 1d"
 *   - Cron expression (5-field): "0 9 * * *", "30 14 * * 1,3,5"
 *   - ISO timestamp (one-shot):  "2026-02-03T14:00:00"
 *
 * Power users can hand-type any of those, but for everyone else the
 * dashboard now offers a human-readable picker. This module is the
 * pure logic layer behind that picker:
 *
 *   - {@link buildScheduleString} turns the picker's structured state
 *     into one of the strings above.
 *   - {@link describeSchedule} goes the other way: takes the structured
 *     schedule shape the API returns (``CronJob.schedule``) and produces
 *     a human-readable sentence for the job list. It recognises common
 *     cron-expression shapes (daily/weekly/monthly) so users don't have
 *     to parse "30 14 * * 1,3,5" by eye.
 *
 * Kept dependency-free and locale-string-driven so it tree-shakes
 * cleanly and is testable in isolation if we ever wire up vitest here.
 */

/** Picker modes — each renders a different set of inputs in the UI but
 * all funnel through {@link buildScheduleString} to a backend-compatible
 * string. ``custom`` is the escape hatch for power users who still want
 * to type a raw cron expression. */
export type ScheduleMode =
  | "interval"
  | "daily"
  | "weekly"
  | "monthly"
  | "once"
  | "custom";

/** Unit used by interval mode. Backend parses ``m``/``h``/``d`` suffixes. */
export type IntervalUnit = "minutes" | "hours" | "days";

/** Cron weekday convention: Sunday = 0 .. Saturday = 6. Matches what
 * croniter expects on the backend (no need to remap on submit). */
export const WEEKDAY_INDEXES = [0, 1, 2, 3, 4, 5, 6] as const;
export type Weekday = (typeof WEEKDAY_INDEXES)[number];

export interface ScheduleBuilderState {
  /** Index of which "custom" radio is selected. */
  mode: ScheduleMode;

  /** Interval mode: positive integer, paired with ``intervalUnit``. */
  intervalValue: number;
  intervalUnit: IntervalUnit;

  /** Daily/weekly/monthly mode: "HH:MM" 24h format from <input type=time>. */
  timeOfDay: string;

  /** Weekly mode: 0..6, Sunday-first. Empty means "every day", which is
   * still valid — we send "*" for the day-of-week cron field. */
  weekdays: Weekday[];

  /** Monthly mode: 1..31 (no support for "last day of month" sugar — the
   * croniter ``L`` extension isn't enabled in the parse_schedule regex). */
  dayOfMonth: number;

  /** Once mode: ``YYYY-MM-DDTHH:MM`` from <input type=datetime-local>. */
  onceAt: string;

  /** Custom mode: raw user-typed cron expression. Stored separately so
   * flipping between modes doesn't erase the user's work. */
  custom: string;
}

/** Default state — "every 30 minutes" is the most-common-cron-pattern
 * starting point and avoids forcing the user to pick everything from
 * scratch. */
export const DEFAULT_SCHEDULE_STATE: ScheduleBuilderState = {
  mode: "interval",
  intervalValue: 30,
  intervalUnit: "minutes",
  timeOfDay: "09:00",
  weekdays: [1, 2, 3, 4, 5],
  dayOfMonth: 1,
  onceAt: "",
  custom: "",
};

const UNIT_SUFFIX: Record<IntervalUnit, string> = {
  minutes: "m",
  hours: "h",
  days: "d",
};

/** Build the schedule string from picker state. Returns ``""`` when the
 * state is incomplete enough that the backend would 400 — the caller
 * uses that to disable the Submit button.
 *
 * Why we lean on the broad parse_schedule grammar instead of always
 * emitting cron expressions: interval syntax ("every 30m") survives a
 * backend without ``croniter`` installed and renders more readably in
 * the job list. We only emit raw cron when the picker truly needs the
 * cron field expressiveness (specific weekdays, specific day-of-month). */
export function buildScheduleString(state: ScheduleBuilderState): string {
  switch (state.mode) {
    case "interval": {
      const n = Math.floor(state.intervalValue);
      if (!Number.isFinite(n) || n < 1) return "";
      return `every ${n}${UNIT_SUFFIX[state.intervalUnit]}`;
    }
    case "daily": {
      const parsed = parseTimeOfDay(state.timeOfDay);
      if (!parsed) return "";
      return `${parsed.minute} ${parsed.hour} * * *`;
    }
    case "weekly": {
      const parsed = parseTimeOfDay(state.timeOfDay);
      if (!parsed) return "";
      // Empty weekday selection → "*" (every day) rather than a backend
      // 400. The Daily mode is the cleaner choice for that, but if the
      // user toggles all days off in Weekly mode we still emit a valid
      // expression instead of breaking the submit.
      const days =
        state.weekdays.length === 0
          ? "*"
          : [...state.weekdays].sort((a, b) => a - b).join(",");
      return `${parsed.minute} ${parsed.hour} * * ${days}`;
    }
    case "monthly": {
      const parsed = parseTimeOfDay(state.timeOfDay);
      if (!parsed) return "";
      const dom = Math.floor(state.dayOfMonth);
      if (!Number.isFinite(dom) || dom < 1 || dom > 31) return "";
      return `${parsed.minute} ${parsed.hour} ${dom} * *`;
    }
    case "once": {
      const v = state.onceAt.trim();
      if (!v) return "";
      // <input type=datetime-local> already emits the
      // "YYYY-MM-DDTHH:MM" shape that fromisoformat() accepts directly.
      // Append ":00" so the backend's regex hits the "T" branch and
      // the seconds component lines up with isoformat() output.
      return v.length === 16 ? `${v}:00` : v;
    }
    case "custom":
      return state.custom.trim();
  }
}

function parseTimeOfDay(value: string): { hour: number; minute: number } | null {
  if (!value || !/^\d{1,2}:\d{2}$/.test(value)) return null;
  const [hh, mm] = value.split(":");
  const hour = parseInt(hh, 10);
  const minute = parseInt(mm, 10);
  if (
    !Number.isFinite(hour) ||
    !Number.isFinite(minute) ||
    hour < 0 ||
    hour > 23 ||
    minute < 0 ||
    minute > 59
  ) {
    return null;
  }
  return { hour, minute };
}

/** Translation surface the human-readable describer needs. Passing it
 * in (instead of importing ``useI18n``) keeps the helper pure and
 * testable; the CronPage threads ``t.cron.scheduleDescribe`` through. */
export interface ScheduleDescribeStrings {
  /** Display when no schedule can be resolved (e.g. legacy/blank job). */
  none: string;
  /** "Every {n} minute(s)" — caller pluralises via {n}. */
  everyMinutes: string;
  everyHours: string;
  everyDays: string;
  /** "Daily at {time}" */
  dailyAt: string;
  /** "Weekly on {days} at {time}" */
  weeklyAt: string;
  /** "Monthly on the {day} at {time}" */
  monthlyAt: string;
  /** "Once at {time}" */
  onceAt: string;
  /** Weekday short names indexed 0..6 (Sunday-first). */
  weekdaysShort: [string, string, string, string, string, string, string];
  /** Ordinal suffix builder, e.g. "1st", "22nd". For locales that
   * don't use English ordinals, just return ``String(day)``. */
  ordinal: (day: number) => string;
}

/** Schedule shape stored on a ``CronJob`` row (see api.ts). */
export interface ScheduleLike {
  kind?: string;
  expr?: string;
  minutes?: number;
  run_at?: string;
  display?: string;
}

/** Human-readable description of a stored schedule.
 *
 * Prefers a structured render over the raw ``display`` string so cron
 * expressions like ``30 14 * * 1,3,5`` show up as "Weekly on Mon, Wed,
 * Fri at 14:30" instead of the raw five-field gibberish. Falls back to
 * ``display`` / ``expr`` / ``none`` in that order if we can't make sense
 * of the schedule (e.g. exotic cron with ranges, step values, or @reboot
 * macros that we'd misrepresent if we tried to "humanize"). */
export function describeSchedule(
  schedule: ScheduleLike | undefined,
  fallbackDisplay: string | undefined,
  strings: ScheduleDescribeStrings,
): string {
  if (!schedule) return fallbackDisplay || strings.none;

  if (schedule.kind === "interval" && typeof schedule.minutes === "number") {
    return describeInterval(schedule.minutes, strings);
  }

  if (schedule.kind === "once" && schedule.run_at) {
    return strings.onceAt.replace(
      "{time}",
      formatIsoLocal(schedule.run_at, false),
    );
  }

  if (schedule.kind === "cron" && schedule.expr) {
    const cronDesc = describeCronExpression(schedule.expr, strings);
    if (cronDesc) return cronDesc;
  }

  // Try the raw expression as a last attempt — for legacy jobs stored
  // without ``kind``, the ``schedule_display`` field often *is* the cron
  // expression.
  if (fallbackDisplay) {
    const cronDesc = describeCronExpression(fallbackDisplay, strings);
    if (cronDesc) return cronDesc;
    return fallbackDisplay;
  }
  if (schedule.display) return schedule.display;
  if (schedule.expr) return schedule.expr;
  return strings.none;
}

function describeInterval(
  minutes: number,
  strings: ScheduleDescribeStrings,
): string {
  if (minutes <= 0) return strings.none;
  if (minutes % 1440 === 0) {
    return strings.everyDays.replace("{n}", String(minutes / 1440));
  }
  if (minutes % 60 === 0) {
    return strings.everyHours.replace("{n}", String(minutes / 60));
  }
  return strings.everyMinutes.replace("{n}", String(minutes));
}

/** Recognise the common, well-shaped cron patterns and return a
 * human sentence for them. Returns ``null`` when the expression has any
 * ranges, steps, or other complexity that would be misleading to
 * "humanize" — caller falls back to displaying the raw expression so
 * the user sees what's actually scheduled.
 *
 * Strictly 5-field only: the backend ``parse_schedule`` also accepts the
 * 6-field ``minute hour dom month dow year`` form, but humanising those
 * by destructuring only the first five fields would silently drop the
 * year and mislead the user (e.g. ``0 9 * * * 2099`` would read as
 * "Daily at 09:00"). 6+ field expressions intentionally fall through to
 * the raw-string fallback in {@link describeSchedule}. */
function describeCronExpression(
  expr: string,
  strings: ScheduleDescribeStrings,
): string | null {
  const parts = expr.trim().split(/\s+/);
  if (parts.length !== 5) return null;
  const [minField, hourField, domField, monField, dowField] = parts;

  const month = monField === "*";
  if (!month) return null; // we don't try to humanize per-month rules

  const isLiteralOrList = (f: string) =>
    /^\d+(,\d+)*$/.test(f) || /^\*$/.test(f);
  if (!isLiteralOrList(minField) || !isLiteralOrList(hourField)) return null;
  if (!isLiteralOrList(domField) || !isLiteralOrList(dowField)) return null;

  // Star minutes/hours would mean "every minute" / "every hour" — we'd
  // need a step-value handler ("*/15") to describe that cleanly, and
  // that path is power-user territory. Bail to raw display.
  if (minField === "*" || hourField === "*") return null;

  const minutes = minField.split(",").map((n) => parseInt(n, 10));
  const hours = hourField.split(",").map((n) => parseInt(n, 10));
  if (minutes.length !== 1 || hours.length !== 1) return null;
  if (
    !Number.isFinite(minutes[0]) ||
    !Number.isFinite(hours[0]) ||
    hours[0] < 0 ||
    hours[0] > 23 ||
    minutes[0] < 0 ||
    minutes[0] > 59
  ) {
    return null;
  }
  const time = `${pad2(hours[0])}:${pad2(minutes[0])}`;

  const domAll = domField === "*";
  const dowAll = dowField === "*";

  if (domAll && dowAll) {
    return strings.dailyAt.replace("{time}", time);
  }

  if (domAll && !dowAll) {
    const days = dowField
      .split(",")
      .map((n) => parseInt(n, 10))
      .filter((n) => Number.isFinite(n) && n >= 0 && n <= 6) as Weekday[];
    if (days.length === 0) return null;
    const labels = days
      .map((d) => strings.weekdaysShort[d])
      .filter(Boolean)
      .join(", ");
    return strings.weeklyAt
      .replace("{days}", labels)
      .replace("{time}", time);
  }

  if (!domAll && dowAll) {
    const dom = parseInt(domField, 10);
    if (!Number.isFinite(dom) || dom < 1 || dom > 31) return null;
    return strings.monthlyAt
      .replace("{day}", strings.ordinal(dom))
      .replace("{time}", time);
  }

  // Both day-of-month AND day-of-week set is unusual and cron's
  // OR-semantics for that combo are confusing — fall back to raw.
  return null;
}

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

/** Format an ISO date for inline display. Drops the seconds + TZ
 * suffix so the cron list stays compact. Falls back to the raw string
 * if Date parsing fails. */
function formatIsoLocal(iso: string, includeSeconds: boolean): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const yyyy = d.getFullYear();
  const mm = pad2(d.getMonth() + 1);
  const dd = pad2(d.getDate());
  const hh = pad2(d.getHours());
  const mi = pad2(d.getMinutes());
  if (includeSeconds) {
    return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${pad2(d.getSeconds())}`;
  }
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
}

/** Convenience: build an English ordinal suffix ("1st", "2nd", "23rd").
 * Most non-English locales should just return ``String(day)`` from
 * their ``ordinal`` override. */
export function englishOrdinal(day: number): string {
  const d = Math.floor(day);
  if (!Number.isFinite(d) || d < 1) return String(day);
  const lastTwo = d % 100;
  if (lastTwo >= 11 && lastTwo <= 13) return `${d}th`;
  switch (d % 10) {
    case 1:
      return `${d}st`;
    case 2:
      return `${d}nd`;
    case 3:
      return `${d}rd`;
    default:
      return `${d}th`;
  }
}
