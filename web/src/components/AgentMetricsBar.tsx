/**
 * AgentMetricsBar — slim, non-intrusive metrics strip for the Chat tab.
 *
 * Renders a single row of mini-gauges above the terminal:
 *   [memory arc] [disk arc] [token bar] [clock]
 *
 * Each gauge shows two values (active profile vs. all) so the user can
 * instantly see how much resources the current agent is consuming relative
 * to the system total.  Layout is tight (~32px tall) so it never
 * obstructs the terminal content.
 */

import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api, type AgentMetricsResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useRef, useState } from "react";

const POLL_INTERVAL_MS = 30_000;

// ── helpers ────────────────────────────────────────────────────────────────

// Upper bounds for the mini arc-gauges.  Chosen to cover "normal" developer
// workloads without the bars maxing out constantly.  Adjust if your agents
// routinely use >8 GB resident memory or >50 GB disk.
const MEM_MAX = 8 * 1024;   // 8 GB in MB
const DISK_MAX = 50 * 1024; // 50 GB in MB

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function fmtMb(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(0)} MB`;
}

/** SVG arc path for a mini donut. cx,cy,radius,startPct,endPct,strokeWidth */
export function arcPath(
  cx: number,
  cy: number,
  r: number,
  startPct: number,
  endPct: number,
  stroke: string,
  sw: number,
  opacity = 1,
): string {
  const circumference = 2 * Math.PI * r;
  const start = circumference * startPct;
  const total = circumference * (endPct - startPct);
  const dasharray = `${total.toFixed(2)} ${(circumference - total).toFixed(2)}`;
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${stroke}" stroke-width="${sw}" stroke-dasharray="${dasharray}" stroke-dashoffset="${start.toFixed(2)}" opacity="${opacity}" stroke-linecap="round" transform="rotate(-90 ${cx} ${cy})" />`;
}

interface ArcGaugeProps {
  value: number; // 0-100
  max: number; // absolute max (e.g., 100 for 100%)
  label: string;
  sublabel?: string;
  color: string;
  trackColor?: string;
  size?: number;
  className?: string;
}

/** Smallest-possible arc donut.  SVG-only, no canvas. */
function ArcGauge({
  value,
  max,
  label,
  sublabel,
  color,
  trackColor = "rgba(255,255,255,0.08)",
  size = 28,
  className,
}: ArcGaugeProps) {
  const pct = Math.min(value / max, 1);
  const sw = 3;
  const r = (size - sw) / 2 - 1;
  const cx = size / 2;
  const cy = size / 2;

  return (
    <div className={cn("flex flex-col items-center gap-0.5", className)}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        aria-label={`${label}: ${value}`}
      >
        {/* track */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={trackColor}
          strokeWidth={sw}
          transform={`rotate(-90 ${cx} ${cy})`}
        />
        {/* fill — only render if pct > 0 */}
        {pct > 0.01 && (
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={sw}
            strokeDasharray={`${(2 * Math.PI * r * pct).toFixed(2)} ${(2 * Math.PI * r * (1 - pct)).toFixed(2)}`}
            strokeDashoffset="0"
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
          />
        )}
      </svg>
      <span className="text-[0.55rem] leading-none text-muted-foreground font-medium tracking-wide">
        {label}
      </span>
      {sublabel && (
        <span className="text-[0.5rem] leading-none text-muted-foreground/60">
          {sublabel}
        </span>
      )}
    </div>
  );
}

interface TokenBarProps {
  activeTokens: number; // input + output for active profile
  allTokens: number;    // input + output for all profiles
  byModel: Record<string, { input: number; output: number }>;
  className?: string;
}

/** Compact token bar showing active vs all, with model breakdown tooltip. */
function TokenBar({ activeTokens, allTokens, byModel, className }: TokenBarProps) {
  const pct = allTokens > 0 ? Math.min(activeTokens / allTokens, 1) : 0;

  const topModels = Object.entries(byModel)
    .sort(([, a], [, b]) => (b.input + b.output) - (a.input + a.output))
    .slice(0, 3);

  const total = activeTokens;
  const label = total > 0 ? fmt(total) : "0";

  return (
    <div className={cn("flex flex-col items-center gap-0.5", className)}>
      {/* Bar */}
      <div className="w-10 flex flex-col gap-0.5">
        <div className="h-1.5 rounded-full bg-white/[0.08] overflow-hidden">
          <div
            className="h-full rounded-full bg-[#4ade80] transition-all duration-500"
            style={{ width: `${pct * 100}%` }}
          />
        </div>
        <div className="text-[0.55rem] leading-none text-muted-foreground font-medium tracking-wide text-center">
          {label}
        </div>
      </div>
      {/* Model breakdown on hover via title attr */}
      {topModels.length > 0 && (
        <span
          className="text-[0.5rem] leading-none text-muted-foreground/60 max-w-[48px] truncate"
          title={topModels.map(([m, v]) => `${m}: ${fmt(v.input + v.output)}`).join(" | ")}
        >
          {topModels[0][0].split("/").pop()}
        </span>
      )}
    </div>
  );
}

interface ClockProps {
  iso: string;
  className?: string;
}

function Clock({ iso, className }: ClockProps) {
  // Parse HH:MM:SS from "2025-07-07T14:32:01"
  const time = iso.split("T")[1] ?? "";
  const [hh, mm, ss] = time.split(":");
  const [display, setDisplay] = useState(`${hh}:${mm}:${ss}`);

  useEffect(() => {
    const id = setInterval(() => {
      const now = new Date();
      setDisplay(
        `${now.getHours().toString().padStart(2, "0")}:${now
          .getMinutes()
          .toString()
          .padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`,
      );
    }, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className={cn("flex flex-col items-center gap-0.5", className)}>
      <span className="text-[0.7rem] font-mono font-medium text-foreground/80 leading-none tabular-nums">
        {display}
      </span>
      <span className="text-[0.5rem] leading-none text-muted-foreground/60">
        {iso.split("T")[0]}
      </span>
    </div>
  );
}

// ── main component ──────────────────────────────────────────────────────────

interface AgentMetricsBarProps {
  className?: string;
  /** When false, renders nothing and stops polling. Default true. */
  visible?: boolean;
}

export function AgentMetricsBar({
  className,
  visible = true,
}: AgentMetricsBarProps) {
  const [metrics, setMetrics] = useState<AgentMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const fetchMetrics = useCallback(async () => {
    if (!visible) return;
    try {
      const data = await api.getAgentMetrics();
      if (!mountedRef.current || !visible) return;
      setMetrics(data);
      setError(null);
    } catch (e) {
      if (!mountedRef.current || !visible) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (!mountedRef.current || !visible) return;
      setLoading(false);
    }
  }, [visible]);

  useEffect(() => {
    mountedRef.current = true;
    if (!visible) return;
    void fetchMetrics();
    const id = setInterval(() => void fetchMetrics(), POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [fetchMetrics, visible]);

  if (!visible) return null;

  if (loading && !metrics) {
    return (
      <div
        className={cn(
          "flex items-center justify-center gap-2 px-3 py-1.5",
          "border-b border-white/[0.06]",
          "bg-[#0d2626]/80 backdrop-blur-sm",
          className,
        )}
      >
        <Spinner className="text-[0.6rem]" />
        <span className="text-[0.6rem] text-muted-foreground">loading metrics…</span>
      </div>
    );
  }

  if (error && !metrics) {
    return (
      <div
        className={cn(
          "flex items-center px-3 py-1.5 border-b border-white/[0.06]",
          "bg-[#0d2626]/80 backdrop-blur-sm",
          className,
        )}
      >
        <span className="text-[0.6rem] text-destructive/70">{error}</span>
      </div>
    );
  }

  if (!metrics) return null;

  const activeInput = metrics.token_totals.active.input;
  const activeOutput = metrics.token_totals.active.output;
  const activeTotal = activeInput + activeOutput;
  const allTotal =
    metrics.token_totals.all.input + metrics.token_totals.all.output;

  // ArcGauge uses value/max internally — no need to pre-compute percentages here.

  return (
    <div
      className={cn(
        "flex items-center gap-4 px-4 py-1.5",
        "border-b border-white/[0.06]",
        "bg-[#0d2626]/80 backdrop-blur-sm",
        "select-none",
        className,
      )}
    >
      {/* Profile badge */}
      <div className="flex items-center gap-1.5 shrink-0">
        <div
          className="h-1.5 w-1.5 rounded-full bg-[#4ade80] animate-pulse"
          title="agent active"
        />
        <span className="text-[0.6rem] font-medium text-foreground/70 tracking-wide truncate max-w-[80px]">
          {metrics.active_profile}
        </span>
      </div>

      {/* Separator */}
      <div className="h-4 w-px bg-white/[0.06]" />

      {/* Memory */}
      <ArcGauge
        value={metrics.memory_mb}
        max={MEM_MAX}
        label={fmtMb(metrics.memory_mb)}
        sublabel="mem"
        color="#60a5fa"
        size={30}
      />

      {/* Disk (active) */}
      <ArcGauge
        value={metrics.disk_active_mb}
        max={DISK_MAX}
        label={fmtMb(metrics.disk_active_mb)}
        sublabel="disk"
        color="#a78bfa"
        size={30}
      />

      {/* Token bar */}
      <div className="flex flex-col items-center gap-0.5">
        <TokenBar
          activeTokens={activeTotal}
          allTokens={allTotal}
          byModel={metrics.by_model}
        />
        <span className="text-[0.5rem] leading-none text-muted-foreground/60">
          tokens
        </span>
      </div>

      {/* Spacer pushes clock to right */}
      <div className="flex-1" />

      {/* Clock */}
      <Clock iso={metrics.time.iso} />
    </div>
  );
}
