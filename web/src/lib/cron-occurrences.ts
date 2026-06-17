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
 * (see `lib/schedule.ts`), the windows are short (24h–7d), and keeping it
 * local means the timeline stays responsive while panning/zooming without
 * a round-trip per frame.
 *
 * Supported schedule shapes (mirrors `cron/jobs.py::parse_schedule`):
 *   - interval  → `schedule.minutes` (every N minutes)
 *   - cron      → `schedule.expr` 5-field, with `*`, lists, ranges, steps
 *   - once      → `schedule.run_at` single ISO instant
 *
 * The backend's authoritative `next_run_at` / `last_run_at` are always
 * folded in as well, so the timeline matches the live scheduler even for
 * exotic expressions the client matcher can't fully model.
 *
 * All matching is done in the browser's local timezone — consistent with
 * the rest of the dashboard, which formats every instant via `Date`.
 */

import type { CronJob } from "@/lib/api";

export type OccurrenceKind = "past" | "next" | "future";

export interface Occurrence {
  /** Epoch milliseconds of the firing. */
  time: number;
  kind: OccurrenceKind;
}

export interface LaneOccurrences {
  occurrences: Occurrence[];
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
// 5-field cron expression matcher
// ---------------------------------------------------------------------------

interface CronMatcher {
  minutes: Set<number>;
  hours: Set<number>;
  doms: Set<number>;
  months: Set<number>;
  dows: Set<number>;
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
  if (parts.length !== 5) return null; // 6-field (with year) not modelled
  const [minF, hourF, domF, monF, dowF] = parts;

  const minutes = parseField(minF, 0, 59);
  const hours = parseField(hourF, 0, 23);
  const doms = parseField(domF, 1, 31);
  const months = parseField(monF, 1, 12);
  const dows = parseField(dowF, 0, 7, true);
  if (!minutes || !hours || !doms || !months || !dows) return null;

  return {
    minutes,
    hours,
    doms,
    months,
    dows,
    domRestricted: domF !== "*",
    dowRestricted: dowF !== "*",
  };
}

function matchesDate(m: CronMatcher, d: Date): boolean {
  if (!m.minutes.has(d.getMinutes())) return false;
  if (!m.hours.has(d.getHours())) return false;
  if (!m.months.has(d.getMonth() + 1)) return false;
  const domOk = m.doms.has(d.getDate());
  const dowOk = m.dows.has(d.getDay());
  // Cron's day-of-month / day-of-week OR-semantics: when both are
  // restricted the firing matches if *either* does; when only one is
  // restricted, only that one must match.
  if (m.domRestricted && m.dowRestricted) return domOk || dowOk;
  if (m.domRestricted) return domOk;
  if (m.dowRestricted) return dowOk;
  return true;
}

// ---------------------------------------------------------------------------
// Enumeration
// ---------------------------------------------------------------------------

function classify(time: number, nextTime: number | null): OccurrenceKind {
  if (nextTime !== null && time === nextTime) return "next";
  return time < (nextTime ?? Infinity) ? "past" : "future";
}

/** Enumerate every firing of `job` in `[fromMs, toMs]`.
 *
 * `nowMs` decides which marker is the single upcoming "next" firing.
 * Backend `next_run_at` / `last_run_at` are always merged in (deduped to
 * the minute) so the timeline reflects the live scheduler even when the
 * local matcher can't model the expression. */
export function enumerateOccurrences(
  job: CronJob,
  fromMs: number,
  toMs: number,
  nowMs: number,
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
    const anchor = anchorIso ? new Date(anchorIso).getTime() : nowMs;
    if (Number.isFinite(anchor)) {
      // Walk back to the first firing >= fromMs.
      let t = anchor;
      if (t > fromMs) {
        const stepsBack = Math.ceil((t - fromMs) / stepMs);
        t = t - stepsBack * stepMs;
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
      const start = new Date(Math.max(fromMs, fromMs));
      // Align to the next whole minute boundary.
      start.setSeconds(0, 0);
      if (start.getTime() < fromMs) start.setMinutes(start.getMinutes() + 1);
      let guard = 0;
      const end = toMs;
      for (
        const d = start;
        d.getTime() <= end;
        d.setMinutes(d.getMinutes() + 1)
      ) {
        if (matchesDate(matcher, d)) {
          times.add(d.getTime());
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

  // The single soonest firing at/after now is the "next" marker.
  const sorted = [...times].sort((a, b) => a - b);
  const nextTime = sorted.find((t) => t >= nowMs) ?? null;

  const occurrences: Occurrence[] = sorted.map((time) => ({
    time,
    kind: classify(time, nextTime),
  }));

  return { occurrences, dense, resolved };
}

function roundToMinute(ms: number): number {
  return Math.round(ms / 60_000) * 60_000;
}
