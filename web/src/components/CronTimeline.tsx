import { useEffect, useMemo, useRef, useState } from "react";
import { Crosshair, Pause, Pencil, Play, X, Zap } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Segmented } from "@nous-research/ui/ui/components/segmented";
import type { CronJob } from "@/lib/api";
import {
  describeSchedule,
  type ScheduleDescribeStrings,
} from "@/lib/schedule";
import {
  enumerateOccurrences,
  type Occurrence,
  type OccurrenceKind,
} from "@/lib/cron-occurrences";
import { cn } from "@/lib/utils";

/**
 * Interactive "Hermes Time" timeline for cron jobs.
 *
 * Every scheduled job becomes a horizontal lane; each predicted firing in
 * the visible window is a marker. A live, glowing "now" line sweeps across
 * the lanes in real time — the visual anchor of *le temps d'Hermes*. The
 * whole strip is touch-pannable (native horizontal scroll) and works down
 * to a phone width via a sticky job-name gutter and a tap-to-reveal detail
 * card instead of fiddly floating popovers.
 *
 * Occurrence math lives in `lib/cron-occurrences.ts`; this component is
 * purely presentation + interaction.
 */

// ---------------------------------------------------------------------------
// Localised strings (threaded from CronPage with English fallbacks so the
// feature ships without touching all 18 locale files at once).
// ---------------------------------------------------------------------------

export interface TimelineStrings {
  windows: { d1: string; d2: string; d7: string };
  now: string;
  recenter: string;
  legendPast: string;
  legendNext: string;
  legendFuture: string;
  legendPaused: string;
  empty: string;
  noOccurrences: string;
  schedule: string;
  last: string;
  next: string;
  triggerNow: string;
  pause: string;
  resume: string;
  edit: string;
  close: string;
  dense: string;
}

interface WindowDef {
  key: "d1" | "d2" | "d7";
  hours: number;
  pxPerHour: number;
  tickHours: number;
}

const WINDOWS: WindowDef[] = [
  { key: "d1", hours: 24, pxPerHour: 62, tickHours: 3 },
  { key: "d2", hours: 48, pxPerHour: 34, tickHours: 6 },
  { key: "d7", hours: 168, pxPerHour: 12, tickHours: 24 },
];

const GUTTER_W = 124;
const LANE_H = 46;
const RULER_H = 34;
/** Fraction of the window kept *behind* "now" so recent firings stay
 * visible for context. The rest is the upcoming horizon. */
const PAST_FRACTION = 0.18;

// ── small pure getters (kept local so the component stays standalone) ──

function asText(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function truncate(v: string, n: number): string {
  return v.length > n ? v.slice(0, n) + "…" : v;
}

function jobName(job: CronJob): string {
  return asText(job.name).trim();
}

function jobTitle(job: CronJob): string {
  const name = jobName(job);
  if (name) return name;
  const prompt = asText(job.prompt);
  if (prompt) return truncate(prompt, 48);
  const script = asText(job.script);
  if (script) return truncate(script, 48);
  return job.id || "Cron job";
}

function jobState(job: CronJob): string {
  return asText(job.state) || (job.enabled === false ? "disabled" : "scheduled");
}

function jobProfile(job: CronJob): string {
  return asText(job.profile) || asText(job.profile_name) || "default";
}

function jobKey(job: CronJob): string {
  return `${jobProfile(job)}:${job.id}`;
}

function formatClock(d: Date): string {
  return d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatTickLabel(d: Date, daily: boolean): string {
  if (daily) {
    return d.toLocaleDateString(undefined, { weekday: "short", day: "numeric" });
  }
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function formatFull(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}

const STATUS_TONE: Record<string, "success" | "warning" | "destructive"> = {
  enabled: "success",
  scheduled: "success",
  paused: "warning",
  error: "destructive",
  completed: "destructive",
};

interface Selection {
  job: CronJob;
  time: number;
  kind: OccurrenceKind;
}

export interface CronTimelineProps {
  jobs: CronJob[];
  scheduleDescribeStrings: ScheduleDescribeStrings;
  strings: TimelineStrings;
  onTrigger: (job: CronJob) => void;
  onPauseResume: (job: CronJob) => void;
  onEdit: (job: CronJob) => void;
}

export function CronTimeline({
  jobs,
  scheduleDescribeStrings,
  strings,
  onTrigger,
  onPauseResume,
  onEdit,
}: CronTimelineProps) {
  const [windowKey, setWindowKey] = useState<WindowDef["key"]>("d1");
  const win = WINDOWS.find((w) => w.key === windowKey) ?? WINDOWS[0];

  const spanMs = win.hours * 3_600_000;
  const contentW = win.hours * win.pxPerHour;

  // The window anchor (`fromMs`) is stable between ticks so the backdrop
  // doesn't jitter every second — only the now-line sweeps. We re-anchor
  // when the window/zoom changes, on an explicit recenter, or once "now"
  // drifts past the right edge.
  const [fromMs, setFromMs] = useState(() => Date.now() - PAST_FRACTION * spanMs);
  const [now, setNow] = useState(() => Date.now());

  const scrollRef = useRef<HTMLDivElement | null>(null);

  const recenter = (toKey?: WindowDef["key"]) => {
    const w = toKey ? WINDOWS.find((x) => x.key === toKey) ?? win : win;
    setFromMs(Date.now() - PAST_FRACTION * w.hours * 3_600_000);
  };



  // Live clock — sweeps the now-line and updates the Hermes-time readout.
  useEffect(() => {
    const id = setInterval(() => {
      const t = Date.now();
      setNow(t);
      // Auto-follow once the present moment runs off the right edge.
      setFromMs((prev) => (t > prev + spanMs ? t - PAST_FRACTION * spanMs : prev));
    }, 1000);
    return () => clearInterval(id);
  }, [spanMs]);

  const toMs = fromMs + spanMs;
  const xForTime = (t: number) => ((t - fromMs) / spanMs) * contentW;

  // Lanes — occurrences per job. Classification ("next") is anchored on
  // the window's own `now` so it doesn't flicker between seconds.
  const lanes = useMemo(
    () =>
      jobs.map((job) => ({
        job,
        ...enumerateOccurrences(job, fromMs, toMs, now),
      })),
    // `now` intentionally omitted: we don't want to recompute every second.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [jobs, fromMs, toMs],
  );

  // Auto-scroll so "now" lands ~28% from the left on first paint and on
  // re-anchor / zoom change.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const target = xForTime(Date.now()) - el.clientWidth * 0.28 + GUTTER_W;
    el.scrollTo({ left: Math.max(0, target), behavior: "smooth" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fromMs, windowKey]);

  // Ruler ticks + day separators.
  const ticks = useMemo(() => {
    const out: { x: number; label: string; major: boolean }[] = [];
    const daily = win.tickHours >= 24;
    const start = new Date(fromMs);
    start.setMinutes(0, 0, 0);
    // Advance to the first tick boundary at or after fromMs.
    while (
      start.getTime() < fromMs ||
      start.getHours() % win.tickHours !== 0
    ) {
      start.setHours(start.getHours() + 1);
    }
    for (
      const d = start;
      d.getTime() <= toMs;
      d.setHours(d.getHours() + win.tickHours)
    ) {
      const major = d.getHours() === 0;
      out.push({
        x: xForTime(d.getTime()),
        label: formatTickLabel(d, daily || major),
        major,
      });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fromMs, toMs, windowKey]);

  const [selected, setSelected] = useState<Selection | null>(null);
  // Drop a stale selection if its job disappears (deleted / filtered).
  if (selected && !jobs.some((j) => jobKey(j) === jobKey(selected.job))) {
    setSelected(null);
  }

  const totalH = RULER_H + Math.max(lanes.length, 1) * LANE_H;
  const nowX = xForTime(now);
  const nowInView = now >= fromMs && now <= toMs;

  if (jobs.length === 0) {
    return (
      <div className="border border-border bg-card/40 py-12 text-center text-sm text-muted-foreground">
        {strings.empty}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Controls + live Hermes clock */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Segmented
            value={windowKey}
            onChange={(v) => {
              const k = v as WindowDef["key"];
              setWindowKey(k);
              recenter(k);
            }}
            options={[
              { value: "d1", label: strings.windows.d1 },
              { value: "d2", label: strings.windows.d2 },
              { value: "d7", label: strings.windows.d7 },
            ]}
          />
          <Button
            ghost
            size="sm"
            className="gap-1.5 text-muted-foreground hover:text-foreground"
            onClick={() => recenter()}
            title={strings.recenter}
          >
            <Crosshair className="h-4 w-4" />
            <span className="hidden sm:inline">{strings.recenter}</span>
          </Button>
        </div>

        <div
          className="flex items-center gap-2 font-mono-ui text-sm tabular-nums"
          style={{ color: "var(--midground-base)" }}
          title={strings.now}
        >
          <span
            className="inline-block h-2 w-2 animate-pulse rounded-full"
            style={{
              background: "var(--midground-base)",
              boxShadow: "0 0 8px var(--warm-glow)",
            }}
          />
          <span className="uppercase tracking-[0.15em] text-display text-xs opacity-70">
            {strings.now}
          </span>
          <span>{formatClock(new Date(now))}</span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <LegendDot className="bg-muted-foreground/40" label={strings.legendPast} />
        <LegendDot
          className="bg-[var(--midground-base)]"
          glow
          label={strings.legendNext}
        />
        <LegendDot
          className="bg-[var(--midground-base)]/60"
          label={strings.legendFuture}
        />
        <LegendDot className="bg-warning/70" ring label={strings.legendPaused} />
      </div>

      {/* Timeline strip */}
      <div
        ref={scrollRef}
        className="relative overflow-x-auto overscroll-x-contain border border-border bg-card/30"
        style={{ touchAction: "pan-x pan-y", WebkitOverflowScrolling: "touch" }}
      >
        <div
          className="relative"
          style={{ width: GUTTER_W + contentW, height: totalH }}
        >
          {/* Day separators + hour ticks (behind lanes) */}
          {ticks.map((tick, i) => (
            <div
              key={i}
              className="absolute top-0 z-0"
              style={{
                left: GUTTER_W + tick.x,
                height: totalH,
                borderLeft: tick.major
                  ? "1px solid var(--midground-base)"
                  : "1px dashed color-mix(in srgb, var(--midground-base) 14%, transparent)",
                opacity: tick.major ? 0.28 : 1,
              }}
            >
              <span
                className={cn(
                  "absolute top-1.5 left-1.5 whitespace-nowrap font-mono-ui text-[10px]",
                  tick.major
                    ? "text-[var(--midground-base)] opacity-80"
                    : "text-muted-foreground",
                )}
              >
                {tick.label}
              </span>
            </div>
          ))}

          {/* Now line — "le temps d'Hermes" */}
          {nowInView && (
            <div
              className="pointer-events-none absolute z-10"
              style={{
                left: GUTTER_W + nowX,
                top: 0,
                height: totalH,
                width: 0,
                borderLeft: "2px solid var(--midground-base)",
                boxShadow: "0 0 12px 1px var(--warm-glow)",
              }}
            >
              <span
                className="absolute -top-0 -translate-x-1/2 whitespace-nowrap rounded-sm px-1 py-0.5 font-mono-ui text-[10px] uppercase tracking-wider"
                style={{
                  background: "var(--midground-base)",
                  color: "var(--background-base)",
                  left: 0,
                }}
              >
                {strings.now}
              </span>
            </div>
          )}

          {/* Ruler baseline */}
          <div
            className="absolute left-0 z-0 w-full border-b border-border"
            style={{ top: RULER_H }}
          />

          {/* Lanes */}
          {lanes.map((lane, i) => {
            const job = lane.job;
            const state = jobState(job);
            const paused = state === "paused" || job.enabled === false;
            const errored = state === "error" || Boolean(job.last_error);
            const key = jobKey(job);
            const isSelectedJob =
              selected != null && jobKey(selected.job) === key;
            return (
              <div
                key={key}
                className={cn(
                  "absolute left-0 w-full border-b border-border/40",
                  isSelectedJob && "bg-[var(--midground-base)]/[0.06]",
                )}
                style={{ top: RULER_H + i * LANE_H, height: LANE_H }}
              >
                {/* Sticky job-name gutter */}
                <button
                  type="button"
                  onClick={() =>
                    setSelected({
                      job,
                      time:
                        job.next_run_at
                          ? new Date(job.next_run_at).getTime()
                          : now,
                      kind: "next",
                    })
                  }
                  className="sticky left-0 z-30 flex h-full flex-col justify-center gap-0.5 border-r border-border bg-card px-2 text-left"
                  style={{ width: GUTTER_W }}
                  title={jobTitle(job)}
                >
                  <span
                    className={cn(
                      "truncate font-mono-ui text-[11px] leading-tight",
                      paused ? "text-muted-foreground" : "text-foreground",
                    )}
                  >
                    {jobTitle(job)}
                  </span>
                  <span className="flex items-center gap-1">
                    <span
                      className={cn(
                        "inline-block h-1.5 w-1.5 rounded-full",
                        errored
                          ? "bg-destructive"
                          : paused
                            ? "bg-warning"
                            : "bg-success",
                      )}
                    />
                    <span className="truncate text-[9px] uppercase tracking-wide text-muted-foreground">
                      {state}
                    </span>
                  </span>
                </button>

                {/* Markers */}
                {lane.dense ? (
                  <div
                    className="absolute top-1/2 -translate-y-1/2"
                    style={{
                      left: GUTTER_W,
                      width: contentW,
                      height: 6,
                      background:
                        "repeating-linear-gradient(90deg, var(--midground-base) 0 2px, transparent 2px 6px)",
                      opacity: paused ? 0.35 : 0.6,
                    }}
                    title={strings.dense}
                  />
                ) : lane.occurrences.length === 0 ? (
                  <span
                    className="absolute top-1/2 -translate-y-1/2 font-mono-ui text-[10px] text-muted-foreground/60"
                    style={{ left: GUTTER_W + 8 }}
                  >
                    {strings.noOccurrences}
                  </span>
                ) : (
                  lane.occurrences.map((occ, j) => (
                    <Marker
                      key={j}
                      occ={occ}
                      x={GUTTER_W + xForTime(occ.time)}
                      paused={paused}
                      errored={errored}
                      active={
                        isSelectedJob &&
                        selected != null &&
                        selected.time === occ.time
                      }
                      onSelect={() =>
                        setSelected({ job, time: occ.time, kind: occ.kind })
                      }
                    />
                  ))
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Detail card for the tapped marker / lane */}
      {selected && (
        <DetailCard
          selection={selected}
          scheduleDescribeStrings={scheduleDescribeStrings}
          strings={strings}
          onClose={() => setSelected(null)}
          onTrigger={onTrigger}
          onPauseResume={onPauseResume}
          onEdit={onEdit}
        />
      )}
    </div>
  );
}

function LegendDot({
  className,
  label,
  glow,
  ring,
}: {
  className: string;
  label: string;
  glow?: boolean;
  ring?: boolean;
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className={cn("inline-block h-2.5 w-2.5 rounded-full", className, ring && "ring-1 ring-warning")}
        style={glow ? { boxShadow: "0 0 6px var(--warm-glow)" } : undefined}
      />
      {label}
    </span>
  );
}

function Marker({
  occ,
  x,
  paused,
  errored,
  active,
  onSelect,
}: {
  occ: Occurrence;
  x: number;
  paused: boolean;
  errored: boolean;
  active: boolean;
  onSelect: () => void;
}) {
  const isNext = occ.kind === "next";
  const isPast = occ.kind === "past";

  // The hit area is intentionally wide (touch-friendly); the visible dot is
  // centered inside it.
  return (
    <button
      type="button"
      onClick={onSelect}
      title={new Date(occ.time).toLocaleString()}
      className="group absolute top-1/2 z-20 flex h-9 w-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center"
      style={{ left: x }}
      aria-label={new Date(occ.time).toLocaleString()}
    >
      <span
        className={cn(
          "block rounded-full transition-transform group-hover:scale-125",
          isNext ? "h-3.5 w-3.5 rotate-45 rounded-[2px]" : "h-2.5 w-2.5",
          active && "scale-125",
        )}
        style={{
          background: errored && isNext
            ? "var(--color-destructive, #ef4444)"
            : "var(--midground-base)",
          opacity: paused ? 0.4 : isPast ? 0.45 : isNext ? 1 : 0.65,
          boxShadow: isNext && !paused ? "0 0 10px 1px var(--warm-glow)" : undefined,
          outline: active ? "2px solid var(--midground-base)" : undefined,
          outlineOffset: 2,
        }}
      />
    </button>
  );
}

function DetailCard({
  selection,
  scheduleDescribeStrings,
  strings,
  onClose,
  onTrigger,
  onPauseResume,
  onEdit,
}: {
  selection: Selection;
  scheduleDescribeStrings: ScheduleDescribeStrings;
  strings: TimelineStrings;
  onClose: () => void;
  onTrigger: (job: CronJob) => void;
  onPauseResume: (job: CronJob) => void;
  onEdit: (job: CronJob) => void;
}) {
  const { job, time, kind } = selection;
  const state = jobState(job);
  const paused = state === "paused" || job.enabled === false;
  const scheduleText = describeSchedule(
    job.schedule,
    asText(job.schedule_display) || asText(job.schedule?.display),
    scheduleDescribeStrings,
  );
  const kindLabel =
    kind === "next"
      ? strings.legendNext
      : kind === "past"
        ? strings.legendPast
        : strings.legendFuture;

  return (
    <div className="relative border border-border bg-card p-4 shadow-lg">
      <Button
        ghost
        size="icon"
        onClick={onClose}
        className="absolute right-1.5 top-1.5 text-muted-foreground hover:text-foreground"
        aria-label={strings.close}
      >
        <X />
      </Button>

      <div className="flex flex-wrap items-center gap-2 pr-8">
        <span className="font-medium text-sm">{jobTitle(job)}</span>
        <Badge tone={STATUS_TONE[state] ?? "secondary"}>{state}</Badge>
        <Badge tone="outline">{jobProfile(job)}</Badge>
        <Badge tone="outline">{kindLabel}</Badge>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
        <Field label={strings.schedule} value={scheduleText} mono />
        <Field
          label={kindLabel}
          value={new Date(time).toLocaleString()}
          mono
        />
        <Field label={strings.last} value={formatFull(job.last_run_at)} mono />
        <Field label={strings.next} value={formatFull(job.next_run_at)} mono />
      </div>

      {job.last_error && (
        <p className="mt-2 font-mono-ui text-xs text-destructive">
          {job.last_error}
        </p>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <Button
          size="sm"
          className="gap-1.5 uppercase"
          onClick={() => onTrigger(job)}
        >
          <Zap className="h-4 w-4" />
          {strings.triggerNow}
        </Button>
        <Button
          size="sm"
          ghost
          className="gap-1.5"
          onClick={() => onPauseResume(job)}
        >
          {paused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
          {paused ? strings.resume : strings.pause}
        </Button>
        <Button size="sm" ghost className="gap-1.5" onClick={() => onEdit(job)}>
          <Pencil className="h-4 w-4" />
          {strings.edit}
        </Button>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span className={cn("truncate", mono && "font-mono-ui")} title={value}>
        {value}
      </span>
    </div>
  );
}
