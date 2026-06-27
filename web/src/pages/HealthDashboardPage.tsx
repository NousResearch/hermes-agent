import { useLayoutEffect, useMemo, type ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  BatteryCharging,
  BellRing,
  CalendarDays,
  CheckCircle2,
  Flame,
  HeartPulse,
  HeartHandshake,
  MoonStar,
  MoveRight,
  ShieldAlert,
  Sparkles,
  Thermometer,
  TimerReset,
  TrendingDown,
  TrendingUp,
  Waves,
  Weight,
  Dumbbell,
  Footprints,
  ScanSearch,
} from "lucide-react";
import {
  buildSparklineFillPath,
  buildSparklinePath,
  formatSignedDelta,
  healthDashboardSnapshot,
  type HealthAlert,
  type HealthTone,
  type TrendSeries,
} from "@/lib/health-dashboard";
import { cn } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";

const toneStyles: Record<HealthTone, string> = {
  good: "border-[#4ade80]/18 bg-[#4ade80]/10 text-[#a7f3d0]",
  warn: "border-[#fbbf24]/20 bg-[#fbbf24]/10 text-[#fde68a]",
  alert: "border-[#fb7185]/18 bg-[#fb7185]/10 text-[#fecdd3]",
  neutral: "border-white/10 bg-white/5 text-white/72",
};

const severityStyles: Record<HealthAlert["severity"], string> = {
  critical: "border-[#fb7185]/22 bg-[linear-gradient(180deg,rgba(251,113,133,0.14),rgba(251,113,133,0.04))]",
  warning: "border-[#f59e0b]/22 bg-[linear-gradient(180deg,rgba(245,158,11,0.14),rgba(245,158,11,0.04))]",
  info: "border-[#45d0ff]/20 bg-[linear-gradient(180deg,rgba(69,208,255,0.12),rgba(69,208,255,0.03))]",
  good: "border-[#4ade80]/20 bg-[linear-gradient(180deg,rgba(74,222,128,0.12),rgba(74,222,128,0.03))]",
};

const severityDotStyles: Record<HealthAlert["severity"], string> = {
  critical: "bg-[#fb7185] shadow-[0_0_0_4px_rgba(251,113,133,0.12)]",
  warning: "bg-[#f59e0b] shadow-[0_0_0_4px_rgba(245,158,11,0.12)]",
  info: "bg-[#45d0ff] shadow-[0_0_0_4px_rgba(69,208,255,0.12)]",
  good: "bg-[#4ade80] shadow-[0_0_0_4px_rgba(74,222,128,0.12)]",
};

const trendIcons: Record<TrendSeries["title"], ReactNode> = {
  "Sleep consistency": <MoonStar className="h-4 w-4" />,
  "Recovery trend": <HeartPulse className="h-4 w-4" />,
  "Load balance": <Activity className="h-4 w-4" />,
};

function SectionHeader({
  kicker,
  title,
  detail,
  action,
}: {
  kicker: string;
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-[0.28em] text-white/34">{kicker}</div>
        <h3 className="mt-1 text-[18px] font-semibold tracking-[-0.03em] text-white">{title}</h3>
        <p className="mt-1 max-w-3xl text-[13px] leading-6 text-white/58">{detail}</p>
      </div>
      {action ? <div className="shrink-0 pt-0.5">{action}</div> : null}
    </div>
  );
}

function ShellCard({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-[22px] border border-white/8 bg-[linear-gradient(180deg,rgba(15,18,25,0.96),rgba(9,11,16,0.98))] shadow-[0_20px_42px_rgba(0,0,0,0.32)]",
        className,
      )}
    >
      {children}
    </div>
  );
}

function MetricCard({
  label,
  value,
  unit,
  detail,
  tone,
}: {
  label: string;
  value: string;
  unit?: string;
  detail: string;
  tone: HealthTone;
}) {
  return (
    <ShellCard className="p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="text-[10px] uppercase tracking-[0.24em] text-white/34">{label}</div>
        <span className={cn("rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.24em]", toneStyles[tone])}>
          {tone}
        </span>
      </div>
      <div className="mt-4 flex items-end gap-2">
        <div className="text-[30px] font-semibold leading-none tracking-[-0.06em] text-white">
          {value}
        </div>
        {unit ? <div className="pb-0.5 text-[13px] text-white/50">{unit}</div> : null}
      </div>
      <div className="mt-3 text-[12px] leading-5 text-white/54">{detail}</div>
    </ShellCard>
  );
}

function ScoreCard({
  label,
  value,
  delta,
  detail,
  tone,
}: {
  label: string;
  value: number;
  delta: number;
  detail: string;
  tone: HealthTone;
}) {
  const deltaTone = delta > 0 ? "text-[#4ade80]" : delta < 0 ? "text-[#fb7185]" : "text-white/48";
  const Icon = delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : MoveRight;

  return (
    <ShellCard className="p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="text-[10px] uppercase tracking-[0.24em] text-white/34">{label}</div>
        <span className={cn("rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.24em]", toneStyles[tone])}>
          {tone}
        </span>
      </div>
      <div className="mt-4 flex items-end justify-between gap-4">
        <div className="text-[36px] font-semibold leading-none tracking-[-0.08em] text-white">{value}</div>
        <div className={cn("flex items-center gap-1 text-[12px] font-medium", deltaTone)}>
          <Icon className="h-3.5 w-3.5" />
          {formatSignedDelta(delta)}
        </div>
      </div>
      <div className="mt-3 text-[12px] leading-5 text-white/54">{detail}</div>
    </ShellCard>
  );
}

function TrendCard({ series }: { series: TrendSeries }) {
  const width = 340;
  const height = 124;
  const path = useMemo(() => buildSparklinePath(series.samples, width, height), [series, width, height]);
  const fill = useMemo(() => buildSparklineFillPath(series.samples, width, height), [series, width, height]);
  const yTicks = useMemo(() => [0, 1, 2, 3], []);

  return (
    <ShellCard className="p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
            <span className="text-white/54">{trendIcons[series.title]}</span>
            {series.window}
          </div>
          <div className="mt-2 text-[16px] font-semibold tracking-[-0.03em] text-white">{series.title}</div>
          <p className="mt-1 text-[12px] leading-5 text-white/54">{series.summary}</p>
        </div>
        <div className="text-right">
          <div className="text-[28px] font-semibold leading-none tracking-[-0.06em] text-white">{series.valueLabel}</div>
          <div className="mt-2 text-[11px] uppercase tracking-[0.24em] text-white/34">Trend view</div>
        </div>
      </div>

      <div className="mt-4 rounded-[18px] border border-white/8 bg-black/20 p-3">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-[124px] w-full overflow-visible">
          <defs>
            <linearGradient id={`spark-${series.title.replace(/\s+/g, "-").toLowerCase()}`} x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="#45d0ff" />
              <stop offset="52%" stopColor="#7cf0d8" />
              <stop offset="100%" stopColor="#fbbf24" />
            </linearGradient>
            <linearGradient id={`spark-fill-${series.title.replace(/\s+/g, "-").toLowerCase()}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(69,208,255,0.34)" />
              <stop offset="100%" stopColor="rgba(69,208,255,0)" />
            </linearGradient>
          </defs>
          {yTicks.map((tick) => (
            <line
              key={tick}
              x1="6"
              x2={width - 6}
              y1={(tick / 3) * (height - 16) + 8}
              y2={(tick / 3) * (height - 16) + 8}
              stroke="rgba(255,255,255,0.06)"
              strokeDasharray="4 6"
            />
          ))}
          {fill ? <path d={fill} fill={`url(#spark-fill-${series.title.replace(/\s+/g, "-").toLowerCase()})`} /> : null}
          {path ? <path d={path} fill="none" stroke={`url(#spark-${series.title.replace(/\s+/g, "-").toLowerCase()})`} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" /> : null}
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-[12px] leading-5 text-white/54 sm:grid-cols-3">
        <div>
          <div className="uppercase tracking-[0.18em] text-white/30">Window</div>
          <div className="mt-1 text-white/82">{series.window}</div>
        </div>
        <div>
          <div className="uppercase tracking-[0.18em] text-white/30">Detail</div>
          <div className="mt-1 text-white/82">{series.detail}</div>
        </div>
        <div>
          <div className="uppercase tracking-[0.18em] text-white/30">Signal</div>
          <div className="mt-1 text-white/82">{series.valueLabel}</div>
        </div>
      </div>
    </ShellCard>
  );
}

function AlertCard({ alert }: { alert: HealthAlert }) {
  return (
    <div className={cn("rounded-[18px] border p-4", severityStyles[alert.severity])}>
      <div className="flex items-start gap-3">
        <span className={cn("mt-1 h-3 w-3 shrink-0 rounded-full", severityDotStyles[alert.severity])} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/36">
            {alert.severity}
          </div>
          <div className="mt-1 text-[15px] font-semibold tracking-[-0.02em] text-white">{alert.title}</div>
          <p className="mt-1 text-[12px] leading-5 text-white/62">{alert.detail}</p>
          <div className="mt-3 rounded-[12px] border border-white/8 bg-black/18 px-3 py-2 text-[12px] text-white/80">
            {alert.impact}
          </div>
        </div>
      </div>
    </div>
  );
}

function DotTag({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/6 px-2.5 py-1 text-[10px] uppercase tracking-[0.24em] text-white/60">
      <span className="h-1.5 w-1.5 rounded-full bg-[#45d0ff]" />
      {children}
    </span>
  );
}

export default function HealthDashboardPage() {
  const snapshot = healthDashboardSnapshot;
  const { setTitle } = usePageHeader();

  useLayoutEffect(() => {
    setTitle("Health Command Centre");
    return () => setTitle(null);
  }, [setTitle]);

  return (
    <div className="relative min-h-0 overflow-hidden rounded-none bg-[#05070b] text-white lg:rounded-[28px]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_12%,rgba(69,208,255,0.12),transparent_24%),radial-gradient(circle_at_76%_10%,rgba(124,240,216,0.08),transparent_24%),radial-gradient(circle_at_70%_88%,rgba(251,191,36,0.08),transparent_22%),linear-gradient(180deg,#05070b_0%,#080b11_100%)]" />
      <div className="absolute inset-0 opacity-[0.15] [background-image:linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] [background-size:30px_30px]" />

      <div className="relative mx-auto max-w-[1600px] px-4 py-4 sm:px-6 sm:py-6 lg:px-8 lg:py-8">
        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.28em] text-white/34">
              <span className="rounded-full border border-[#45d0ff]/18 bg-[#45d0ff]/10 px-2.5 py-1 text-[#9ee8ff]">
                mission control
              </span>
              <span>health dashboard</span>
              <span>·</span>
              <span>updated {snapshot.updatedAt}</span>
            </div>
            <h1 className="mt-3 text-[clamp(42px,5vw,74px)] font-semibold tracking-[-0.08em] text-white">
              Recovery Control
            </h1>
            <p className="mt-3 max-w-3xl text-[15px] leading-7 text-white/62">
              {snapshot.summary}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:flex sm:flex-wrap sm:justify-end">
            <StatPill icon={<ShieldAlert className="h-3.5 w-3.5" />} label="Risk" value="Controlled" tone="warn" />
            <StatPill icon={<HeartPulse className="h-3.5 w-3.5" />} label="Recovery" value="73" tone="alert" />
            <StatPill icon={<BatteryCharging className="h-3.5 w-3.5" />} label="Readiness" value="78" tone="good" />
            <StatPill icon={<CalendarDays className="h-3.5 w-3.5" />} label="Plan" value="Low strain" tone="neutral" />
          </div>
        </div>

        <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
          {snapshot.scores.map((score) => {
            const { key, label, value, delta, detail, tone } = score;
            return <ScoreCard key={key} label={label} value={value} delta={delta} detail={detail} tone={tone} />;
          })}
        </section>

        <section className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.7fr)_minmax(340px,0.95fr)]">
          <div className="space-y-6">
            <section className="space-y-3">
              <SectionHeader
                kicker="Telemetry"
                title="Health metrics"
                detail="The operational view: compact metrics, clear deltas, and signal over noise."
                action={<DotTag>live mock layer</DotTag>}
              />
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {snapshot.metrics.map((metric) => (
                  <MetricCard key={metric.label} {...metric} />
                ))}
              </div>
            </section>

            <section className="space-y-3">
              <SectionHeader
                kicker="Trends"
                title="7 / 30 / 90 day view"
                detail="Simple trend panels with sparkline-style visuals keep the dashboard lightweight while still showing momentum."
              />
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
                {snapshot.trends.map((series) => (
                  <TrendCard key={series.title} series={series} />
                ))}
              </div>
            </section>

            <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
              <ShellCard className="p-4">
                <SectionHeader
                  kicker="Training"
                  title="Activity / load"
                  detail="Recent sessions, training load and cadence at a glance."
                  action={<DotTag>consistency streak active</DotTag>}
                />
                <div className="mt-4 space-y-3">
                  {snapshot.workouts.map((workout) => (
                    <div key={`${workout.name}-${workout.time}`} className="rounded-[18px] border border-white/8 bg-black/20 p-4">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <div className="text-[15px] font-semibold tracking-[-0.02em] text-white">{workout.name}</div>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-white/34">
                            <span>{workout.type}</span>
                            <span>·</span>
                            <span>{workout.time}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-[13px] font-medium text-white/82">{workout.duration}</div>
                          <div className="mt-1 text-[11px] uppercase tracking-[0.2em] text-white/34">{workout.load}</div>
                        </div>
                      </div>
                      <p className="mt-3 text-[12px] leading-5 text-white/58">{workout.note}</p>
                    </div>
                  ))}
                </div>

                <div className="mt-4 grid grid-cols-2 gap-3 text-[12px]">
                  <InfoTile icon={<Dumbbell className="h-4 w-4" />} label="Weekly volume" value="11.4h" detail="Sitting in the target band" tone="good" />
                  <InfoTile icon={<Flame className="h-4 w-4" />} label="PRs" value="2" detail="One lift, one pace record" tone="good" />
                  <InfoTile icon={<Footprints className="h-4 w-4" />} label="Consistency" value="6 days" detail="No missed movement days" tone="neutral" />
                  <InfoTile icon={<TimerReset className="h-4 w-4" />} label="Load shift" value="+8%" detail="Within tolerated range" tone="warn" />
                </div>
              </ShellCard>

              <ShellCard className="p-4">
                <SectionHeader
                  kicker="Recovery"
                  title="Recovery panel"
                  detail="The decision layer: sleep debt, readiness slope and strain balance determine today’s recommendation."
                  action={<DotTag>recommendation ready</DotTag>}
                />
                <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {snapshot.recovery.map((item) => (
                    <div key={item.label} className={cn("rounded-[18px] border p-4", toneStyles[item.tone])}>
                      <div className="text-[10px] uppercase tracking-[0.24em] text-white/34">{item.label}</div>
                      <div className="mt-3 text-[24px] font-semibold tracking-[-0.05em] text-white">{item.value}</div>
                      <div className="mt-2 text-[12px] leading-5 text-white/60">{item.detail}</div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 rounded-[22px] border border-[#45d0ff]/16 bg-[linear-gradient(180deg,rgba(69,208,255,0.12),rgba(69,208,255,0.03))] p-4">
                  <div className="flex items-start gap-3">
                    <div className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-[#45d0ff]/12 text-[#9ee8ff]">
                      <Sparkles className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="text-[15px] font-semibold tracking-[-0.02em] text-white">{snapshot.recommendation.title}</div>
                      <p className="mt-2 text-[13px] leading-6 text-white/62">{snapshot.recommendation.detail}</p>
                      <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/20 px-3 py-1.5 text-[11px] uppercase tracking-[0.22em] text-white/70">
                        {snapshot.recommendation.action}
                        <ArrowRight className="h-3.5 w-3.5" />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-4 space-y-3">
                  <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
                    <BellRing className="h-3.5 w-3.5 text-[#fbbf24]" />
                    alerts
                  </div>
                  <div className="space-y-3">
                    {snapshot.alerts.map((alert) => (
                      <AlertCard key={alert.title} alert={alert} />
                    ))}
                  </div>
                </div>
              </ShellCard>
            </section>

            <ShellCard className="p-4">
              <SectionHeader
                kicker="Notes"
                title="Notes / flags / annotations"
                detail="Manual observations stay visible so the dashboard captures context, not just numbers."
              />
              <div className="mt-4 grid grid-cols-1 gap-3 xl:grid-cols-3">
                {snapshot.notes.map((note) => (
                  <div key={note.title} className="rounded-[18px] border border-white/8 bg-black/18 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-[15px] font-semibold tracking-[-0.02em] text-white">{note.title}</div>
                      <span className="rounded-full border border-white/10 bg-white/6 px-2 py-1 text-[10px] uppercase tracking-[0.22em] text-white/52">
                        {note.tag}
                      </span>
                    </div>
                    <p className="mt-3 text-[12px] leading-6 text-white/62">{note.body}</p>
                    <div className="mt-4 text-[11px] uppercase tracking-[0.2em] text-white/32">{note.updatedAt}</div>
                  </div>
                ))}
              </div>
            </ShellCard>
          </div>

          <aside className="space-y-6">
            <ShellCard className="p-4">
              <SectionHeader
                kicker="Source layer"
                title="Pluggable data inputs"
                detail="Mock data is wired now; real adapters can be slotted in later without changing the dashboard layout."
              />
              <div className="mt-4 space-y-2">
                {snapshot.sources.map((source) => (
                  <div key={source.name} className="rounded-[18px] border border-white/8 bg-black/18 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-[14px] font-semibold tracking-[-0.02em] text-white">{source.name}</div>
                      <span className={cn("rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.22em]", toneStyles[source.status === "connected" ? "good" : source.status === "pending" ? "warn" : "neutral"])}>
                        {source.status}
                      </span>
                    </div>
                    <p className="mt-2 text-[12px] leading-5 text-white/58">{source.description}</p>
                    <div className="mt-3 text-[11px] uppercase tracking-[0.22em] text-white/32">{source.readiness}</div>
                  </div>
                ))}
              </div>
            </ShellCard>

            <ShellCard className="p-4">
              <SectionHeader
                kicker="Status"
                title="Readiness summary"
                detail="A compact view of the key health signals most likely to alter the day’s plan."
              />
              <div className="mt-4 space-y-3">
                <MiniLine icon={<MoonStar className="h-4 w-4" />} label="Sleep debt" value="1h 42m" detail="Clear it across 3 nights" tone="warn" />
                <MiniLine icon={<HeartPulse className="h-4 w-4" />} label="HRV trend" value="-6 ms" detail="Down but still in range" tone="alert" />
                <MiniLine icon={<Waves className="h-4 w-4" />} label="Hydration" value="2.4L" detail="Top up before afternoon" tone="warn" />
                <MiniLine icon={<Weight className="h-4 w-4" />} label="Body comp" value="15.8%" detail="Holding steady" tone="neutral" />
              </div>
            </ShellCard>

            <ShellCard className="p-4">
              <SectionHeader
                kicker="Check"
                title="Operational guardrails"
                detail="A health dashboard should be useful under pressure: clear, fast and hard to misread."
              />
              <div className="mt-4 space-y-2 text-[12px] leading-5 text-white/62">
                <Guardrail icon={<CheckCircle2 className="h-4 w-4" />} text="Near-black background with teal / cyan / green / amber / red status accents." />
                <Guardrail icon={<ScanSearch className="h-4 w-4" />} text="No hidden state — everything is visible in the page model." />
                <Guardrail icon={<Thermometer className="h-4 w-4" />} text="Lightweight inline charts keep the page fast and dependency-free." />
                <Guardrail icon={<HeartHandshake className="h-4 w-4" />} text="Real sources can be mapped in later without restructuring the UI." />
              </div>
            </ShellCard>
          </aside>
        </section>
      </div>
    </div>
  );
}

function StatPill({
  icon,
  label,
  value,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  tone: HealthTone;
}) {
  return (
    <div className={cn("min-w-[150px] rounded-[18px] border px-3 py-3 shadow-[0_16px_36px_rgba(0,0,0,0.24)]", toneStyles[tone])}>
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/40">
        {icon}
        {label}
      </div>
      <div className="mt-2 text-[18px] font-semibold tracking-[-0.04em] text-white">{value}</div>
    </div>
  );
}

function InfoTile({
  icon,
  label,
  value,
  detail,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  tone: HealthTone;
}) {
  return (
    <div className={cn("rounded-[18px] border p-4", toneStyles[tone])}>
      <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.24em] text-white/36">
        <span className="inline-flex items-center gap-2">{icon}{label}</span>
      </div>
      <div className="mt-3 text-[24px] font-semibold tracking-[-0.05em] text-white">{value}</div>
      <div className="mt-2 text-[12px] leading-5 text-white/60">{detail}</div>
    </div>
  );
}

function MiniLine({
  icon,
  label,
  value,
  detail,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  tone: HealthTone;
}) {
  return (
    <div className={cn("rounded-[18px] border p-4", toneStyles[tone])}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.24em] text-white/34">
            {icon}
            {label}
          </div>
          <div className="mt-3 text-[22px] font-semibold tracking-[-0.05em] text-white">{value}</div>
        </div>
        <ArrowRight className="mt-1 h-4 w-4 text-white/36" />
      </div>
      <div className="mt-2 text-[12px] leading-5 text-white/60">{detail}</div>
    </div>
  );
}

function Guardrail({ icon, text }: { icon: ReactNode; text: string }) {
  return (
    <div className="flex items-start gap-3 rounded-[16px] border border-white/8 bg-black/18 p-3">
      <span className="mt-0.5 text-[#45d0ff]">{icon}</span>
      <span>{text}</span>
    </div>
  );
}
