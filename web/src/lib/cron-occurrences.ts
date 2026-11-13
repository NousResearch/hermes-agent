/**
 * Occurrence enumeration for the cron timeline view.
 *
 * The job list shows *when next* a cron fires (from the backend's
 * `next_run_at`), but a timeline needs *every* firing inside a visible
 * window so it can plot a lane of markers. This module is the pure logic
 * that turns a {@link CronJob}'s stored schedule into a list of run times
 * between two instants.
 *
 * It deliberately re-derives occurrences client-side instead of asking
 * the backend for a series: the schedule grammar is small and stable
 * (see `lib/schedule.ts`), the windows are short (24h–30d), and keeping it
 * local means the timeline stays responsive while panning/zooming without
 * a round-trip per frame.
 *
 * Supported schedule shapes (mirrors `cron/jobs.py::parse_schedule`):
 *   - interval  → `schedule.minutes` (every N minutes)
 *   - cron      → `schedule.expr` 5- or 6-field, with `*`, lists, ranges,
 *                 steps (a 6th field is the seconds field, matching
 *                 croniter's `min hour dom mon dow second` order)
 *   - once      → `schedule.run_at` single ISO instant
 *
 * Cron recurrence is matched in the *configured Hermes timezone* (passed in
 * as an IANA name by the caller), never the browser's own timezone — the
 * scheduler evaluates the same expressions through croniter from Hermes
 * time (`cron/jobs.py`), so a browser in another zone must not shift the
 * markers. When the timezone is unavailable, matching falls back to the
 * browser's local zone (the historical behaviour).
 *
 * The backend's authoritative `next_run_at` / `last_run_at` are always
 * folded in as well (deduped to the minute) so the timeline reflects the
 * live scheduler even for exotic expressions the local matcher can't fully
 * model (named months/weekdays, `L`, `#`, 7-field forms).
 *
 * Occurrence *enumeration* (the expensive part) is kept separate from the
 * past/next/future *classification* (which must follow a live `now`): see
 * {@link classifyOccurrences}.
 */

import type { CronJob } from "@/lib/api";

export type OccurrenceKind = "past" | "next" | "future";

export interface Occurrence {
  /** Epoch milliseconds of the firing. */
  time: number;
  kind: OccurrenceKind;
}

export interface LaneOccurrences {
  /** Epoch-millisecond firing instants, ascending, deduped to the minute. */
  times: number[];
  /**
   * True when the schedule fires so often that we capped enumeration
   * (e.g. `* * * * *`). The timeline renders a continuous band instead
   * of discrete dots in that case.
   */
  dense: boolean;
  /** False when no schedule could be resolved at all (no markers). */
  resolved: boolean;
}

/** Hard cap on markers per lane to keep rendering snappy and to detect
 * "fires every minute" style schedules. */
const MARKER_CAP = 300;

// ---------------------------------------------------------------------------
// Cron field parsing (5- and 6-field)
// ---------------------------------------------------------------------------

interface CronMatcher {
  minutes: Set<number>;
  hours: Set<number>;
  doms: Set<number>;
  months: Set<number>;
  dows: Set<number>;
  /** Present only for 6-field expressions; never consulted for matching
   * (the timeline is minute-granular and seconds cannot move a firing into a
   * different minute) — parsed solely so a valid 6-field expression isn't
   * rejected and a malformed seconds field still falls back cleanly. */
  seconds: Set<number> | null;
  domRestricted: boolean;
  dowRestricted: boolean;
}

/** Parse one cron field into the set of values it permits.
 * Supports `*`, `a`, `a,b`, `a-b`, `* /n` (step), and `a-b/n`.
 * Returns null on anything it doesn't understand so the caller can fall
 * back to the backend-provided next/last markers. */
function parseField(
  field: string,
  min: number,
  max: number,
  wrap7to0 = false,
): Set<number> | null {
  const out = new Set<number>();
  for (const part of field.split(",")) {
    const stepSplit = part.split("/");
    if (stepSplit.length > 2) return null;
    const step = stepSplit.length === 2 ? parseInt(stepSplit[1], 10) : 1;
    if (!Number.isFinite(step) || step < 1) return null;

    const range = stepSplit[0];
    let lo: number;
    let hi: number;
    if (range === "*") {
      lo = min;
      hi = max;
    } else if (range.includes("-")) {
      const [a, b] = range.split("-");
      lo = parseInt(a, 10);
      hi = parseInt(b, 10);
    } else {
      lo = parseInt(range, 10);
      hi = stepSplit.length === 2 ? max : lo;
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
    if (lo > hi) return null;
    if (lo < min || hi > max) {
      // dow allows 7 as an alias for 0 (Sunday).
      if (!(wrap7to0 && hi === 7 && lo >= min)) return null;
    }
    for (let v = lo; v <= hi; v += step) {
      out.add(wrap7to0 && v === 7 ? 0 : v);
    }
  }
  return out.size > 0 ? out : null;
}

function buildCronMatcher(expr: string): CronMatcher | null {
  const parts = expr.trim().split(/\s+/);
  if (parts.length !== 5 && parts.length !== 6) return null;
  const [minF, hourF, domF, monF, dowF] = parts;
  const secF = parts.length === 6 ? parts[5] : null;

  const minutes = parseField(minF, 0, 59);
  const hours = parseField(hourF, 0, 23);
  const doms = parseField(domF, 1, 31);
  const months = parseField(monF, 1, 12);
  const dows = parseField(dowF, 0, 7, true);
  if (!minutes || !hours || !doms || !months || !dows) return null;

  let seconds: Set<number> | null = null;
  if (secF !== null) {
    seconds = parseField(secF, 0, 59);
    if (!seconds) return null;
  }

  return {
    minutes,
    hours,
    doms,
    months,
    dows,
    seconds,
    domRestricted: domF !== "*",
    dowRestricted: dowF !== "*",
  };
}

function matchesDate(m: CronMatcher, wc: WallClock): boolean {
  if (!m.minutes.has(wc.minute)) return false;
  if (!m.hours.has(wc.hour)) return false;
  if (!m.months.has(wc.month)) return false;
  const domOk = m.doms.has(wc.dom);
  const dowOk = m.dows.has(wc.dow);
  // Cron's day-of-month / day-of-week OR-semantics: when both are
  // restricted the firing matches if *either* does; when only one is
  // restricted, only that one must match.
  if (m.domRestricted && m.dowRestricted) return domOk || dowOk;
  if (m.domRestricted) return domOk;
  if (m.dowRestricted) return dowOk;
  return true;
}

// ---------------------------------------------------------------------------
// Timezone-aware wall-clock extraction
// ---------------------------------------------------------------------------

interface WallClock {
  minute: number;
  hour: number;
  dom: number; // day of month, 1-31
  month: number; // 1-12
  dow: number; // 0=Sun .. 6=Sat (croniter convention)
  year: number;
}

/** Build a reader that extracts a `Date`'s wall-clock fields in a given
 * IANA zone (or the browser-local zone when `timeZone` is null). The reader
 * is created once per enumeration and reused across the minute loop, so the
 * `Intl` formatter is not rebuilt per minute. */
function makeWallClockReader(timeZone: string | null): (d: Date) => WallClock {
  if (!timeZone) {
    return (d) => ({
      minute: d.getMinutes(),
      hour: d.getHours(),
      dom: d.getDate(),
      month: d.getMonth() + 1,
      dow: d.getDay(),
      year: d.getFullYear(),
    });
  }
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hourCycle: "h23",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
  /** The zone's UTC offset (ms) at `t`, read off the formatter. */
  const numberPart = (
    parts: Intl.DateTimeFormatPart[],
    type: string,
  ): number => {
    const found = parts.find((p) => p.type === type);
    return found ? Number(found.value) : NaN;
  };
  const zoneOffsetMs = (t: number): number => {
    const parts = formatter.formatToParts(new Date(t));
    const asUtc = Date.UTC(
      numberPart(parts, "year"),
      numberPart(parts, "month") - 1,
      numberPart(parts, "day"),
      numberPart(parts, "hour"),
      numberPart(parts, "minute"),
    );
    return asUtc - t;
  };
  // A per-minute `formatToParts` costs ~2.7µs — over a 30-day window that is
  // a multi-second main-thread freeze per job, so the offset is derived once
  // per quarter-hour block and the rest of the block is read arithmetically
  // (shifting the instant by the offset and taking UTC fields *is* the zone's
  // wall clock). Every IANA offset is a whole number of quarter-hours and
  // every transition lands on a 15-minute boundary, so the refresh lands
  // exactly on a change.
  // ponytail: a zone that shifted mid-block would misread only that block —
  // drop BLOCK_MS to 5 minutes if such a zone ever appears.
  const BLOCK_MS = 15 * 60_000;
  let offsetMs = 0;
  let blockEnd = -Infinity;
  return (d) => {
    const t = d.getTime();
    if (t >= blockEnd) {
      offsetMs = zoneOffsetMs(t);
      blockEnd = (Math.floor(t / BLOCK_MS) + 1) * BLOCK_MS;
    }
    const local = new Date(t + offsetMs);
    return {
      minute: local.getUTCMinutes(),
      hour: local.getUTCHours(),
      dom: local.getUTCDate(),
      month: local.getUTCMonth() + 1,
      dow: local.getUTCDay(),
      year: local.getUTCFullYear(),
    };
  };
}

// ---------------------------------------------------------------------------
// Enumeration (expensive — memoize per window, not per tick)
// ---------------------------------------------------------------------------

/** Enumerate every firing of `job` in `[fromMs, toMs]`.
 *
 * Returns the raw (unclassified) firing instants. The caller classifies them
 * against a live `now` via {@link classifyOccurrences} so the expensive
 * enumeration never has to re-run each second just because `now` advanced.
 *
 * Backend `next_run_at` / `last_run_at` are always merged in (deduped to the
 * minute) so the timeline reflects the live scheduler even when the local
 * matcher can't model the expression. */
export function enumerateOccurrences(
  job: CronJob,
  fromMs: number,
  toMs: number,
  timeZone: string | null,
): LaneOccurrences {
  const times = new Set<number>();
  let dense = false;
  let resolved = false;

  const schedule = job.schedule ?? {};
  const kind = schedule.kind;
  const minutes =
    typeof schedule.minutes === "number" ? schedule.minutes : undefined;

  // ── interval ────────────────────────────────────────────────────────
  if (kind === "interval" && minutes && minutes > 0) {
    resolved = true;
    const stepMs = minutes * 60_000;
    // Anchor on a known real firing so the phase lines up with reality.
    const anchorIso = job.next_run_at ?? job.last_run_at;
    const anchor = anchorIso ? new Date(anchorIso).getTime() : null;
    if (anchor !== null && Number.isFinite(anchor)) {
      // Walk back to the first firing >= fromMs.
      let t = anchor;
      if (t > fromMs) {
        const stepsBack = Math.ceil((t - fromMs) / stepMs);
        t -= stepsBack * stepMs;
      }
      let guard = 0;
      for (; t <= toMs && guard < MARKER_CAP + 1; t += stepMs, guard++) {
        if (t >= fromMs) times.add(roundToMinute(t));
      }
      if (guard > MARKER_CAP) dense = true;
    }
  }

  // ── once ────────────────────────────────────────────────────────────
  else if (kind === "once" && schedule.run_at) {
    resolved = true;
    const t = new Date(schedule.run_at).getTime();
    if (Number.isFinite(t) && t >= fromMs && t <= toMs) {
      times.add(roundToMinute(t));
    }
  }

  // ── cron expression ─────────────────────────────────────────────────
  else if (schedule.expr) {
    const matcher = buildCronMatcher(schedule.expr);
    if (matcher) {
      resolved = true;
      const readWallClock = makeWallClockReader(timeZone);
      // Iterate over absolute minute boundaries (epoch math) rather than
      // calendar `setMinutes` steps so DST transitions can't skip or repeat
      // an hour; the wall-clock extraction handles the offset at each step.
      const startMs = Math.ceil(fromMs / 60_000) * 60_000;
      let guard = 0;
      for (let t = startMs; t <= toMs; t += 60_000) {
        if (matchesDate(matcher, readWallClock(new Date(t)))) {
          times.add(t);
          guard++;
          if (guard > MARKER_CAP) {
            dense = true;
            break;
          }
        }
      }
    }
  }

  // ── fold in authoritative backend markers ───────────────────────────
  for (const iso of [job.last_run_at, job.next_run_at]) {
    if (!iso) continue;
    const t = new Date(iso).getTime();
    if (Number.isFinite(t) && t >= fromMs && t <= toMs) {
      times.add(roundToMinute(t));
      resolved = true;
    }
  }

  return { times: [...times].sort((a, b) => a - b), dense, resolved };
}

// ---------------------------------------------------------------------------
// Classification (cheap — re-run each tick)
// ---------------------------------------------------------------------------

/** Classify already-enumerated firing instants against a live `now`.
 *
 * The single soonest firing at-or-after `now` is the "next" marker; earlier
 * instants are "past" and later ones "future". Kept separate from
 * {@link enumerateOccurrences} so a one-second `now` tick reclassifies
 * without re-enumerating. */
export function classifyOccurrences(
  times: number[],
  nowMs: number,
): Occurrence[] {
  const nextTime = times.find((t) => t >= nowMs) ?? null;
  return times.map((time) => ({
    time,
    kind: classify(time, nextTime),
  }));
}

function classify(time: number, nextTime: number | null): OccurrenceKind {
  if (nextTime !== null && time === nextTime) return "next";
  return time < (nextTime ?? Infinity) ? "past" : "future";
}

function roundToMinute(ms: number): number {
  return Math.round(ms / 60_000) * 60_000;
}
