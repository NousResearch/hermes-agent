import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BookOpen,
  Bot,
  Boxes,
  Brain,
  CalendarClock,
  CheckCircle2,
  CircleDot,
  ClipboardCheck,
  Command,
  Cpu,
  Database,
  Fingerprint,
  Gauge,
  Globe2,
  Layers3,
  ListChecks,
  LockKeyhole,
  Radar,
  RefreshCw,
  Route,
  Search,
  ServerCog,
  ShieldCheck,
  Smartphone,
  Sparkles,
  TimerReset,
  Workflow,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type {
  MissionControlCoverageItem,
  MissionControlDomainScore,
  MissionControlSnapshot,
  MissionControlStatus,
} from "@/lib/api";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";
import { PluginSlot } from "@/plugins";

type AnyRecord = Record<string, unknown>;

function num(record: AnyRecord, key: string, fallback = 0): number {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function str(record: AnyRecord, key: string, fallback = "—"): string {
  const value = record[key];
  if (typeof value === "string" && value.trim()) return value;
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  return fallback;
}

function boolText(record: AnyRecord, key: string): string {
  return record[key] === true ? "yes" : "no";
}

function strArray(record: AnyRecord, key: string): string[] {
  const value = record[key];
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
}

function recordOf(value: unknown): AnyRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as AnyRecord) : {};
}

function objectArray(value: unknown): AnyRecord[] {
  return Array.isArray(value)
    ? value.filter((item): item is AnyRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "item";
}

function checkTone(status: unknown): "success" | "warning" | "secondary" | "outline" {
  const value = String(status ?? "unknown");
  if (["pass", "active", "ok"].includes(value)) return "success";
  if (["warn", "warning", "partial", "fail", "exceeded"].includes(value)) return "warning";
  if (["watch", "unknown", "not_applicable", "target"].includes(value)) return "secondary";
  return "outline";
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);
}

function formatDuration(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "unknown";
  const seconds = Math.abs(value);
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function signedDuration(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "unknown";
  return value < 0 ? `${formatDuration(value)} overdue` : `in ${formatDuration(value)}`;
}

function sumRecordValues(value: unknown): number {
  const record = recordOf(value);
  return Object.values(record).reduce<number>((total, item) => total + (typeof item === "number" && Number.isFinite(item) ? item : 0), 0);
}

function statusMatch(item: MissionControlCoverageItem, status: string): boolean {
  return status === "all" || item.status === status;
}

function coverageMatches(item: MissionControlCoverageItem, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [item.id, item.number, item.title, item.domain, item.part, item.summary, item.next]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
    .includes(needle);
}

function statusTone(status: MissionControlStatus): "success" | "warning" | "secondary" | "outline" {
  if (status === "active") return "success";
  if (status === "partial") return "warning";
  if (status === "watch") return "secondary";
  return "outline";
}

function statusLabel(status: MissionControlStatus): string {
  if (status === "active") return "active";
  if (status === "partial") return "partial";
  if (status === "watch") return "watch";
  if (status === "planned") return "planned";
  return String(status);
}

function ScoreRing({ value, label }: { value: number; label: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div
      className="relative grid place-items-center"
      role="progressbar"
      aria-label={`${label}: ${clamped}%`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={clamped}
      data-testid="mission-score-ring"
    >
      <div
        className="h-36 w-36 rounded-full p-[1px] shadow-[0_0_80px_rgba(255,230,203,0.16)]"
        style={{
          background: `conic-gradient(var(--midground-base) ${clamped * 3.6}deg, color-mix(in srgb, var(--midground-base) 10%, transparent) 0deg)`,
        }}
      >
        <div className="grid h-full w-full place-items-center rounded-full bg-background-base/92 backdrop-blur-xl">
          <div className="text-center">
            <div className="text-4xl font-semibold tracking-[-0.05em] text-midground">
              {clamped}
            </div>
            <div className="mt-1 text-[10px] uppercase tracking-[0.28em] text-text-tertiary">
              readiness
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: LucideIcon;
}) {
  return (
    <Card className="min-w-0 overflow-hidden border-current/15 bg-card/70 backdrop-blur-xl">
      <CardContent className="p-4 sm:p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-[11px] uppercase tracking-[0.22em] text-text-tertiary">{label}</p>
            <p className="mt-2 truncate text-2xl font-semibold tracking-[-0.03em] text-foreground sm:text-3xl">
              {value}
            </p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{detail}</p>
          </div>
          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl border border-current/15 bg-midground/10 text-midground">
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function DomainBar({ domain }: { domain: MissionControlDomainScore }) {
  return (
    <div className="min-w-0 rounded-2xl border border-current/10 bg-background-base/35 p-3" data-testid={`mission-domain-${slugify(domain.name)}`}>
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="min-w-0 truncate font-medium text-foreground">{domain.name}</span>
        <span className="font-mono text-text-tertiary">{domain.score}% · {domain.items}</span>
      </div>
      <div
        className="mt-2 h-2 overflow-hidden rounded-full bg-midground/10"
        role="progressbar"
        aria-label={`${domain.name} readiness`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={domain.score}
      >
        <div
          className="h-full rounded-full bg-midground shadow-[0_0_24px_rgba(255,230,203,0.25)]"
          style={{ width: `${Math.max(4, Math.min(100, domain.score))}%` }}
        />
      </div>
    </div>
  );
}

function CoverageCard({ item, compact = false }: { item: MissionControlCoverageItem; compact?: boolean }) {
  return (
    <div
      className={cn(
        "group min-w-0 overflow-hidden rounded-[1.35rem] border border-current/10 bg-background-base/35 p-4 transition-colors",
        "hover:border-current/20 hover:bg-midground/[0.055]",
      )}
      data-testid={`mission-coverage-${item.id}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={statusTone(item.status)}>{statusLabel(item.status)}</Badge>
            <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-text-tertiary">
              {item.number ?? item.id}
            </span>
            {item.missionControl && <Badge tone="outline">route live</Badge>}
          </div>
          <h3 className="mt-3 line-clamp-2 text-base font-semibold leading-snug tracking-[-0.02em] text-foreground">
            {item.title}
          </h3>
        </div>
        <div className="shrink-0 rounded-full border border-current/10 bg-midground/10 px-2.5 py-1 font-mono text-xs text-midground">
          {item.readiness}%
        </div>
      </div>
      {!compact && (
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{item.summary}</p>
      )}
      <div
        className="mt-4 h-1.5 overflow-hidden rounded-full bg-midground/10"
        role="progressbar"
        aria-label={`${item.title} readiness`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={item.readiness}
      >
        <div className="h-full rounded-full bg-midground/80" style={{ width: `${Math.max(4, item.readiness)}%` }} />
      </div>
      <div className="mt-4 flex flex-wrap gap-2 text-[11px] text-text-tertiary">
        <span className="rounded-full border border-current/10 px-2 py-1">{item.domain}</span>
        {item.part && <span className="rounded-full border border-current/10 px-2 py-1">{item.part}</span>}
        {item.route && <span className="rounded-full border border-current/10 px-2 py-1">{item.route}</span>}
      </div>
      <div className="mt-4 space-y-1.5 text-xs leading-relaxed text-muted-foreground">
        {item.evidence.slice(0, compact ? 1 : 2).map((line) => (
          <div key={line} className="flex gap-2">
            <CircleDot className="mt-0.5 h-3.5 w-3.5 shrink-0 text-midground/70" />
            <span className="min-w-0 break-words">{line}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ActionCard({ action }: { action: MissionControlSnapshot["actionQueue"][number] }) {
  const className = "group block min-h-[44px] min-w-0 rounded-[1.35rem] border border-current/10 bg-background-base/40 p-4 text-left transition hover:border-current/20 hover:bg-midground/[0.06] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-midground/60";
  const body = (
    <div className="flex items-start gap-3">
      <div className="grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-midground/10 font-mono text-xs text-midground">
        {action.rank}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone={action.tone === "now" ? "warning" : action.tone === "watch" ? "secondary" : "outline"}>{action.tone}</Badge>
          {action.category && <Badge tone="outline">{action.category}</Badge>}
          {action.effort && <span className="rounded-full border border-current/10 px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] text-text-tertiary">{action.effort}</span>}
          <ArrowUpRight className="h-3.5 w-3.5 text-text-tertiary transition group-hover:text-midground" />
        </div>
        <h3 className="mt-2 text-sm font-semibold text-foreground">{action.title}</h3>
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{action.reason}</p>
      </div>
    </div>
  );
  const testKey = slugify(action.title || action.route || action.category || "action");
  if (action.route.startsWith("http")) {
    return <a href={action.route} target="_blank" rel="noreferrer" className={className} data-testid={`mission-action-card-${testKey}`}>{body}</a>;
  }
  return <Link to={action.route} className={className} data-testid={`mission-action-card-${testKey}`}>{body}</Link>;
}

function Section({
  id,
  eyebrow,
  title,
  children,
  icon: Icon,
}: {
  id?: string;
  eyebrow: string;
  title: string;
  children: ReactNode;
  icon: LucideIcon;
}) {
  return (
    <section id={`mission-${id ?? slugify(eyebrow)}`} className="space-y-4" data-testid={`mission-section-${id ?? slugify(eyebrow)}`}>
      <div className="flex items-center gap-3">
        <div className="grid h-9 w-9 place-items-center rounded-2xl border border-current/10 bg-midground/10 text-midground">
          <Icon className="h-4 w-4" />
        </div>
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-[0.24em] text-text-tertiary">{eyebrow}</p>
          <h2 className="text-lg font-semibold tracking-[-0.03em] text-foreground sm:text-xl">{title}</h2>
        </div>
      </div>
      {children}
    </section>
  );
}

function RuntimePanel({ data }: { data: MissionControlSnapshot }) {
  const { runtime } = data;
  const model = runtime.model;
  const sessions = runtime.sessions;
  const gateway = runtime.gateway;
  const skills = runtime.skills;
  const cron = runtime.cron;
  const mcp = runtime.mcp;
  const safety = runtime.safety;
  const env = runtime.env;
  const voice = runtime.voice;
  const families = strArray(env, "families");
  const platforms = strArray(gateway, "configuredPlatforms");

  const analytics = recordOf(runtime.analytics);
  const analyticsTotals = recordOf(analytics.totals);
  const memory = recordOf(runtime.memory);
  const memorySqlite = recordOf(memory.sqlite);
  const semantic = recordOf(runtime.semantic);
  const production = recordOf(runtime.production);
  const cronStatusCount = sumRecordValues(cron.lastStatusCounts);
  const toolBuckets = recordOf(runtime.tools.configuredToolsetBuckets);
  const mcpStatus = recordOf(mcp.statusCounts);
  const outputLimits = recordOf(safety.toolOutputLimits);

  const rows = [
    ["Model", `${str(model, "provider")} · ${str(model, "model")}`, `Reasoning ${str(model, "reasoning")}`],
    ["Sessions", `${num(sessions, "total")} sessions · ${num(sessions, "messages")} messages`, `${num(sessions, "toolCalls")} tool calls · ${num(sessions, "apiCalls")} API calls`],
    ["Session graph", `${num(sessions, "rootSessionCount")} roots · ${num(sessions, "childSessionCount")} child`, `${num(sessions, "complexSessionCount")} complex · ${num(sessions, "rewindTotal")} rewinds`],
    ["Memory", `${num(memorySqlite, "sessions")} sessions · FTS ${boolText(memorySqlite, "ftsPresent")}`, `${num(memorySqlite, "summaries")} summaries · ${num(memorySqlite, "archivedSessions")} archived`],
    ["Semantic", `${str(semantic, "provider")}`, `configured ${boolText(semantic, "configured")} · index ${boolText(semantic, "indexConfigured")}`],
    ["Gateway", `${num(gateway, "configuredCount")} configured`, platforms.length ? platforms.join(", ") : `running: ${boolText(gateway, "running")}`],
    ["Skills", `${num(skills, "total")} installed`, `${num(skills, "agentskillsCompliant")} portable · ${num(skills, "autoCreatedCount")} auto-created`],
    ["Cron", `${num(cron, "enabled")} enabled / ${num(cron, "total")} total`, `${num(cron, "overdueCount")} overdue · ${cronStatusCount} status sample(s) · reflection ${str(cron, "reflectionFreshness", "unknown")}`],
    ["MCP", `${num(mcp, "configured")} configured`, `${num(mcpStatus, "enabled")} enabled · ${num(mcpStatus, "disabled")} disabled · names redacted`],
    ["Tools", `${num(runtime.tools, "registeredToolCount")} registered`, `${num(toolBuckets, "builtin")} builtin · ${num(toolBuckets, "custom")} custom bucket(s)`],
    ["Safety", `approvals: ${str(safety, "approvalsMode")}`, `isolated: ${boolText(safety, "terminalIsolated")} · redaction: ${boolText(safety, "redactSecrets")} · output cap: ${boolText(outputLimits, "configured")}`],
    ["Voice", `STT ${boolText(voice, "sttEnabled")}`, `TTS ${str(voice, "ttsProvider", "not configured")}`],
    ["Analytics", `${formatNumber(num(analyticsTotals, "inputTokens") + num(analyticsTotals, "outputTokens"))} tokens`, `$${formatNumber(num(analyticsTotals, "actualCostUsd"))} actual · $${formatNumber(num(analyticsTotals, "estimatedCostUsd"))} estimated`],
    ["Production", `${num(production, "score")}% score`, `${objectArray(production.blockers).length} blocker/watch item(s)`],
    ["Env", `${num(env, "configuredKeys")} configured values`, families.length ? families.join(", ") : "no provider families exposed"],
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3" data-testid="mission-runtime-grid">
      {rows.map(([label, value, detail]) => (
        <div key={label} className="min-w-0 rounded-2xl border border-current/10 bg-background-base/35 p-4">
          <p className="text-[11px] uppercase tracking-[0.22em] text-text-tertiary">{label}</p>
          <p className="mt-2 truncate text-sm font-semibold text-foreground">{value}</p>
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">{detail}</p>
        </div>
      ))}
    </div>
  );
}

function InsightTile({
  label,
  value,
  detail,
  icon: Icon,
  tone = "neutral",
}: {
  label: string;
  value: string;
  detail: string;
  icon: LucideIcon;
  tone?: "neutral" | "success" | "warning";
}) {
  const toneClass = tone === "success" ? "text-emerald-300 bg-emerald-400/10" : tone === "warning" ? "text-amber-300 bg-amber-400/10" : "text-midground bg-midground/10";
  return (
    <div className="min-w-0 rounded-[1.4rem] border border-current/10 bg-background-base/35 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[10px] uppercase tracking-[0.24em] text-text-tertiary">{label}</p>
          <p className="mt-2 truncate text-xl font-semibold tracking-[-0.04em] text-foreground sm:text-2xl">{value}</p>
        </div>
        <div className={cn("grid h-10 w-10 shrink-0 place-items-center rounded-2xl", toneClass)}>
          <Icon className="h-5 w-5" aria-hidden="true" />
        </div>
      </div>
      <p className="mt-3 line-clamp-2 text-xs leading-relaxed text-muted-foreground">{detail}</p>
    </div>
  );
}

function MissionCockpitPanel({ data }: { data: MissionControlSnapshot }) {
  const { runtime } = data;
  const sessions = runtime.sessions;
  const cron = runtime.cron;
  const mcp = runtime.mcp;
  const tools = runtime.tools;
  const gateway = runtime.gateway;
  const safety = runtime.safety;
  const production = recordOf(runtime.production);
  const roleCounts = recordOf(sessions.roleCounts);
  const toolBuckets = recordOf(tools.configuredToolsetBuckets);
  const mcpTransports = recordOf(mcp.transportCounts);
  const topAction = data.actionQueue[0];
  const readiness = data.coverage.summary.readiness;
  const totalMessages = num(sessions, "messages");
  const toolCalls = num(sessions, "toolCalls");
  const automationRatio = totalMessages ? Math.round((toolCalls / totalMessages) * 100) : 0;
  const blockers = objectArray(production.blockers).length;
  const nextRun = signedDuration(cron.nextRunDueInSeconds);

  return (
    <section id="mission-cockpit" className="grid gap-4 xl:grid-cols-[minmax(0,1.18fr)_minmax(0,0.82fr)]" aria-label="Mission cockpit" data-testid="mission-cockpit">
      <Card className="relative isolate overflow-hidden border-current/10 bg-card/70 shadow-[0_30px_120px_rgba(0,0,0,0.28)] backdrop-blur-2xl">
        <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_12%_18%,rgba(255,230,203,0.18),transparent_32%),radial-gradient(circle_at_92%_10%,rgba(59,130,246,0.12),transparent_28%)]" />
        <CardContent className="p-5 sm:p-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={blockers ? "warning" : "success"}>{blockers ? `${blockers} watch` : "clear to operate"}</Badge>
                <Badge tone="outline">privacy boundary on</Badge>
                <Badge tone="secondary">generated {formatDuration(Date.now() / 1000 - Date.parse(data.runtime.generatedAt) / 1000)} ago</Badge>
              </div>
              <h2 className="mt-4 max-w-3xl text-3xl font-semibold tracking-[-0.06em] text-foreground sm:text-4xl">
                Command cockpit: action, trust and runtime pulse in one sweep.
              </h2>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-muted-foreground">
                The dashboard now favors decisions over data dumps: weakest blueprint areas, live automation health, privacy posture, and the next operator move are visible before the long source map.
              </p>
            </div>
            <div className="min-w-[12rem] rounded-[1.4rem] border border-current/10 bg-background-base/40 p-4 text-left">
              <p className="text-[10px] uppercase tracking-[0.24em] text-text-tertiary">readiness vector</p>
              <div className="mt-3 flex items-end gap-2">
                <span className="text-5xl font-semibold tracking-[-0.08em] text-midground">{readiness}</span>
                <span className="pb-2 text-sm text-muted-foreground">/100</span>
              </div>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-midground/10" role="progressbar" aria-label="Overall blueprint readiness" aria-valuemin={0} aria-valuemax={100} aria-valuenow={readiness}>
                <div className="h-full rounded-full bg-midground shadow-[0_0_30px_rgba(255,230,203,0.35)]" style={{ width: `${Math.max(4, readiness)}%` }} />
              </div>
            </div>
          </div>

          {topAction && (
            <div className="mt-6 rounded-[1.5rem] border border-current/10 bg-background-base/35 p-4" data-testid="mission-cockpit-top-action">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <p className="text-[10px] uppercase tracking-[0.24em] text-text-tertiary">next best action</p>
                  <h3 className="mt-2 text-base font-semibold tracking-[-0.03em] text-foreground">{topAction.title}</h3>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{topAction.reason}</p>
                </div>
                <Link to={topAction.route} className="inline-flex min-h-[44px] shrink-0 items-center justify-center rounded-2xl border border-current/15 px-4 text-sm font-medium text-foreground transition hover:bg-midground/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-midground/60">
                  Open route <ArrowUpRight className="ml-2 h-4 w-4" aria-hidden="true" />
                </Link>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
        <InsightTile label="automation density" value={`${automationRatio}%`} detail={`${formatNumber(toolCalls)} tool calls across ${formatNumber(totalMessages)} messages; content remains hidden.`} icon={Cpu} />
        <InsightTile label="session topology" value={`${num(sessions, "rootSessionCount")} / ${num(sessions, "childSessionCount")}`} detail={`${num(sessions, "complexSessionCount")} complex sessions · avg ${formatNumber(num(sessions, "avgApiCalls"))} API calls`} icon={Workflow} />
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:col-span-2 2xl:grid-cols-4">
        <InsightTile label="gateway pulse" value={str(gateway, "state")} detail={`${num(gateway, "configuredCount")} platform family · names redacted: ${boolText(gateway, "platformNamesRedacted")}`} icon={Command} tone={gateway.running ? "success" : "warning"} />
        <InsightTile label="scheduler" value={nextRun} detail={`${num(cron, "enabled")} enabled · ${num(cron, "overdueCount")} overdue · ${num(cron, "failedJobs")} failed`} icon={TimerReset} tone={num(cron, "overdueCount") || num(cron, "failedJobs") ? "warning" : "success"} />
        <InsightTile label="tool surface" value={`${num(tools, "registeredToolCount")}`} detail={`${num(toolBuckets, "builtin")} builtin · ${num(toolBuckets, "mcp")} MCP · ${num(toolBuckets, "custom")} custom bucket(s)`} icon={Boxes} />
        <InsightTile label="MCP transports" value={`${num(mcp, "configured")}`} detail={`${num(mcpTransports, "stdio")} stdio · ${num(mcpTransports, "http")} http · ${num(mcpTransports, "sse")} sse`} icon={ServerCog} />
      </div>

      <div className="grid gap-3 xl:col-span-2 md:grid-cols-2">
        <div className="rounded-[1.5rem] border border-current/10 bg-background-base/35 p-4" data-testid="mission-role-mix">
          <p className="text-[10px] uppercase tracking-[0.24em] text-text-tertiary">conversation role mix</p>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {Object.entries(roleCounts).map(([role, value]) => (
              <div key={role} className="rounded-2xl border border-current/10 bg-card/40 p-3">
                <p className="text-xs font-medium text-foreground">{role}</p>
                <p className="mt-1 font-mono text-sm text-text-tertiary">{formatNumber(typeof value === "number" ? value : 0)}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-[1.5rem] border border-current/10 bg-background-base/35 p-4" data-testid="mission-privacy-posture">
          <p className="text-[10px] uppercase tracking-[0.24em] text-text-tertiary">privacy posture</p>
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            <Badge tone={safety.redactSecrets ? "success" : "warning"}>secrets: {boolText(safety, "redactSecrets")}</Badge>
            <Badge tone={tools.configuredToolsetNamesRedacted ? "success" : "warning"}>toolsets redacted</Badge>
            <Badge tone={mcp.serverNamesRedacted ? "success" : "warning"}>MCP names redacted</Badge>
            <Badge tone="outline">session content omitted</Badge>
          </div>
          <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
            Mission Control serializes aggregate counts and safe families only. Prompts, commands, platform IDs, file paths, custom labels, and server names stay outside the UI boundary.
          </p>
        </div>
      </div>
    </section>
  );
}

function SectionNav() {
  const sections = [
    ["cockpit", "Cockpit"],
    ["production", "Production"],
    ["operator-queue", "Actions"],
    ["live-runtime", "Runtime"],
    ["source-coverage", "Coverage"],
    ["data-flow", "Data flow"],
    ["privacy-boundary", "Privacy"],
  ];
  return (
    <nav aria-label="Mission Control sections" className="sticky top-2 z-20 -mx-1 overflow-x-auto rounded-2xl border border-current/10 bg-background-base/80 p-1 shadow-[0_20px_80px_rgba(0,0,0,0.22)] backdrop-blur-2xl" data-testid="mission-section-nav">
      <div className="flex min-w-max gap-1">
        {sections.map(([id, label]) => (
          <a key={id} href={`#mission-${id}`} className="inline-flex min-h-[44px] items-center justify-center rounded-xl px-3 text-xs font-medium text-muted-foreground transition hover:bg-midground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-midground/60 sm:px-4">
            {label}
          </a>
        ))}
      </div>
    </nav>
  );
}

function StatusList({ items, testId }: { items: AnyRecord[]; testId: string }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3" data-testid={testId}>
      {items.map((item, index) => {
        const title = str(item, "title", str(item, "label", str(item, "term", str(item, "symptom", `Item ${index + 1}`))));
        const status = item.status ?? "info";
        const evidence = objectArray(item.evidence).length ? objectArray(item.evidence).map((e) => String(e)) : strArray(item, "evidence");
        const stableId = str(item, "id", str(item, "term", str(item, "label", title)));
        return (
          <div key={`${stableId}-${index}`} className="min-w-0 rounded-[1.35rem] border border-current/10 bg-background-base/35 p-4" data-testid={`${testId}-${slugify(stableId)}`}>
            <div className="flex flex-wrap items-center gap-2">
              {typeof item.status !== "undefined" && <Badge tone={checkTone(status)}>{String(status)}</Badge>}
              {typeof item.route !== "undefined" && <Badge tone="outline">{str(item, "route")}</Badge>}
            </div>
            <h3 className="mt-3 line-clamp-2 break-words text-sm font-semibold text-foreground [overflow-wrap:anywhere]">{title}</h3>
            <p className="mt-1 line-clamp-3 break-words text-xs leading-relaxed text-muted-foreground [overflow-wrap:anywhere]">
              {str(item, "summary", str(item, "detail", str(item, "definition", str(item, "fix", "—"))))}
            </p>
            {evidence.length > 0 && (
              <div className="mt-3 space-y-1 text-[11px] leading-relaxed text-text-tertiary">
                {evidence.slice(0, 2).map((line) => <p key={line}>• {line}</p>)}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ProductionPanel({ data }: { data: MissionControlSnapshot }) {
  const production = recordOf(data.runtime.production);
  const signals = objectArray(production.signals);
  const blockers = objectArray(production.blockers);
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,0.82fr)_minmax(0,1.18fr)]" data-testid="mission-production-readiness">
      <Card className="overflow-hidden bg-card/65 backdrop-blur-xl">
        <CardContent className="p-5">
          <p className="text-[11px] uppercase tracking-[0.24em] text-text-tertiary">production score</p>
          <div className="mt-4 flex items-end gap-2">
            <span className="text-5xl font-semibold tracking-[-0.08em] text-midground">{num(production, "score")}</span>
            <span className="pb-2 text-sm text-muted-foreground">/100</span>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
            Auth, approvals, redaction, gateway reachability, cron, MCP, quality hooks, and hosting posture are evaluated separately so static support is not confused with verified runtime state.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Badge tone={blockers.length ? "warning" : "success"}>{blockers.length} blocker/watch item(s)</Badge>
            <Badge tone="outline">privacy minimized</Badge>
          </div>
        </CardContent>
      </Card>
      <div className="grid gap-3 sm:grid-cols-2">
        {signals.map((signal, index) => (
          <div key={`${str(signal, "id", String(index))}`} className="rounded-[1.2rem] border border-current/10 bg-background-base/35 p-4">
            <Badge tone={checkTone(signal.status)}>{String(signal.status ?? "unknown")}</Badge>
            <h3 className="mt-3 text-sm font-semibold text-foreground">{str(signal, "label")}</h3>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{str(signal, "detail")}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function DataFlowPanel({ data }: { data: MissionControlSnapshot }) {
  const rows = objectArray(data.runtime.dataFlow);
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5" data-testid="mission-data-flow">
      {rows.map((row) => (
        <div key={str(row, "id", str(row, "label"))} className="min-w-0 rounded-[1.2rem] border border-current/10 bg-background-base/35 p-4">
          <Badge tone={row.configured ? "success" : "secondary"}>{row.configured ? "configured" : "not configured"}</Badge>
          <h3 className="mt-3 text-sm font-semibold text-foreground">{str(row, "label")}</h3>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{str(row, "dataSent")}</p>
          <p className="mt-2 text-[11px] leading-relaxed text-text-tertiary">{str(row, "retention")}</p>
        </div>
      ))}
    </div>
  );
}

function SourceExtrasPanel({ data }: { data: MissionControlSnapshot }) {
  return (
    <div className="grid gap-6 xl:grid-cols-2">
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-foreground">Architecture pieces</h3>
        <StatusList items={data.blueprint.architecturePieces ?? []} testId="mission-architecture" />
      </div>
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-foreground">Prerequisites</h3>
        <StatusList items={data.blueprint.prerequisites ?? []} testId="mission-prerequisites" />
      </div>
      <div className="space-y-4 xl:col-span-2">
        <h3 className="text-sm font-semibold text-foreground">What to build next</h3>
        <StatusList items={data.blueprint.nextTools ?? []} testId="mission-next-tools" />
      </div>
      <div className="space-y-4 xl:col-span-2">
        <h3 className="text-sm font-semibold text-foreground">Troubleshooting playbook</h3>
        <StatusList items={data.blueprint.troubleshooting ?? []} testId="mission-troubleshooting" />
      </div>
    </div>
  );
}

function DeviceProofPanel({ data }: { data: MissionControlSnapshot }) {
  const smokeTargets = objectArray(data.deviceProof.smokeTargets);
  return (
    <Card className="overflow-hidden bg-card/65 backdrop-blur-xl" data-testid="mission-device-proof">
      <CardHeader>
        <CardTitle className="text-base">Responsive proof markers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-2 sm:grid-cols-2">
          {data.deviceProof.breakpoints.map((bp) => (
            <div key={bp} className="rounded-2xl border border-current/10 bg-background-base/35 px-3 py-2 text-sm text-foreground" data-testid={`mission-device-breakpoint-${slugify(bp)}`}>
              <CheckCircle2 className="mr-2 inline h-4 w-4 text-midground" />
              {bp}
            </div>
          ))}
        </div>
        {smokeTargets.length > 0 && (
          <div className="grid gap-2 sm:grid-cols-2">
            {smokeTargets.map((target) => (
              <div key={str(target, "id", str(target, "label"))} className="rounded-2xl border border-current/10 bg-background-base/20 px-3 py-2 text-xs text-muted-foreground">
                <Badge tone={checkTone(target.status)}>{str(target, "status")}</Badge>
                <span className="ml-2">{str(target, "label")}</span>
              </div>
            ))}
          </div>
        )}
        <div className="space-y-2 text-xs leading-relaxed text-muted-foreground">
          {data.deviceProof.principles.map((line) => (
            <p key={line}>• {line}</p>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function MissionControlPage() {
  const [data, setData] = useState<MissionControlSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [coverageQuery, setCoverageQuery] = useState("");
  const [coverageStatus, setCoverageStatus] = useState("all");
  const [showAllSteps, setShowAllSteps] = useState(false);
  const { setAfterTitle, setEnd } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getMissionControlBlueprint()
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useLayoutEffect(() => {
    setAfterTitle(
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="outline">Claude Agent blueprint</Badge>
        {data && <Badge tone="success">{data.coverage.summary.readiness}% ready</Badge>}
      </div>,
    );
    setEnd(
      <Button ghost size="icon" className="min-h-[44px] min-w-[44px]" onClick={load} disabled={loading} aria-label="Refresh Mission Control" data-testid="mission-refresh-button">
        {loading ? <Spinner /> : <RefreshCw />}
      </Button>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [data, load, loading, setAfterTitle, setEnd]);

  useEffect(() => {
    void Promise.resolve().then(load);
  }, [load]);

  useEffect(() => {
    setShowAllSteps(false);
  }, [coverageQuery, coverageStatus]);

  const topSteps = useMemo(
    () => [...(data?.coverage.steps ?? [])].sort((a, b) => a.readiness - b.readiness).slice(0, 6),
    [data],
  );
  const hermesFeatures = useMemo(
    () => (data?.coverage.features ?? []).filter((f) => f.id.startsWith("H")),
    [data],
  );
  const openClawFeatures = useMemo(
    () => (data?.coverage.features ?? []).filter((f) => f.id.startsWith("O")),
    [data],
  );
  const filteredSteps = useMemo(
    () => (data?.coverage.steps ?? []).filter((step) => statusMatch(step, coverageStatus) && coverageMatches(step, coverageQuery)),
    [coverageQuery, coverageStatus, data],
  );
  const visibleSteps = showAllSteps ? filteredSteps : filteredSteps.slice(0, 12);

  if (loading && !data) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3" aria-busy="true" aria-live="polite" data-testid="mission-loading">
        <Spinner className="text-3xl text-primary" />
        <p className="text-sm text-muted-foreground">Loading Mission Control…</p>
      </div>
    );
  }

  if (error && !data) {
    return (
      <Card role="alert" data-testid="mission-error">
        <CardContent className="py-12 text-center">
          <p className="text-sm text-destructive">{error}</p>
          <Button className="mt-4" onClick={load} data-testid="mission-retry-button">Retry</Button>
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  const sessions = data.runtime.sessions;
  const skills = data.runtime.skills;
  const cron = data.runtime.cron;
  const mcp = data.runtime.mcp;
  const summary = data.coverage.summary;

  return (
    <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-6 pb-[max(2rem,env(safe-area-inset-bottom))]" data-testid="mission-control-page">
      <PluginSlot name="mission-control:top" />

      <section
        className="relative isolate min-w-0 overflow-hidden rounded-[2rem] border border-current/10 bg-card/70 p-5 shadow-[0_24px_100px_rgba(0,0,0,0.32)] backdrop-blur-2xl sm:p-7 lg:p-8"
        data-testid="mission-hero"
      >
        <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_20%_20%,rgba(255,230,203,0.16),transparent_34%),radial-gradient(circle_at_80%_10%,rgba(52,211,153,0.12),transparent_30%),linear-gradient(135deg,rgba(255,255,255,0.045),transparent)]" />
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="outline">source: {data.source.title}</Badge>
              <Badge tone="secondary">{data.blueprint.stepCount} tracked steps</Badge>
              <Badge tone="secondary">{data.blueprint.hermesFeatureCount + data.blueprint.openclawFeatureCount} feature-picker items</Badge>
            </div>
            <h2 className="mt-5 max-w-4xl text-4xl font-semibold tracking-[-0.07em] text-foreground sm:text-5xl lg:text-6xl">
              Mission Control for the full agent blueprint.
            </h2>
            <p className="mt-5 max-w-3xl text-sm leading-7 text-muted-foreground sm:text-base">
              This is not a static mirror. Hermes maps every step, Hermes feature, and OpenClaw feature from the guide to live local runtime evidence — without exposing raw chat content, commands, logs, secrets, or absolute local paths.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <a href={data.source.url} target="_blank" rel="noreferrer" className="inline-flex min-h-[44px] items-center rounded-xl border border-current/15 px-4 text-sm font-medium text-foreground transition hover:bg-midground/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-midground/60">
                Source guide <ArrowUpRight className="ml-2 h-4 w-4" />
              </a>
              <Link to="/system" className="inline-flex min-h-[44px] items-center rounded-xl px-4 text-sm font-medium text-muted-foreground transition hover:bg-midground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-midground/60">
                System evidence
              </Link>
            </div>
          </div>
          <ScoreRing value={summary.readiness} label="Mission readiness" />
        </div>
      </section>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-5" data-testid="mission-metrics">
        <MetricCard label="Coverage" value={`${summary.total}`} detail={`${summary.counts.active ?? 0} active · ${summary.counts.partial ?? 0} partial`} icon={Gauge} />
        <MetricCard label="Sessions" value={formatNumber(num(sessions, "total"))} detail={`${formatNumber(num(sessions, "messages"))} messages counted`} icon={Database} />
        <MetricCard label="Skills" value={formatNumber(num(skills, "total"))} detail="portable procedural memory" icon={Brain} />
        <MetricCard label="Cron" value={formatNumber(num(cron, "enabled"))} detail={`${formatNumber(num(cron, "total"))} scheduled job(s)`} icon={CalendarClock} />
        <MetricCard label="MCP" value={formatNumber(num(mcp, "configured"))} detail="configured external tool servers" icon={Workflow} />
      </div>

      <SectionNav />
      <MissionCockpitPanel data={data} />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
        <Section id="production" eyebrow="production" title="Production readiness, honestly separated" icon={ServerCog}>
          <ProductionPanel data={data} />
        </Section>

        <Section id="operator-queue" eyebrow="operator queue" title="Smart next actions" icon={Radar}>
          <div className="grid gap-3 sm:grid-cols-2" data-testid="mission-action-queue">
            {data.actionQueue.map((action) => <ActionCard key={`${action.rank}-${action.title}`} action={action} />)}
          </div>
        </Section>

        <Section id="readiness-heatmap" eyebrow="readiness heatmap" title="Weakest domains first" icon={Activity}>
          <div className="grid gap-3">
            {data.coverage.weakestDomains.map((domain) => <DomainBar key={domain.name} domain={domain} />)}
          </div>
        </Section>
      </div>

      <Section id="live-runtime" eyebrow="live runtime" title="Smart things Mission Control can honestly show" icon={Bot}>
        <RuntimePanel data={data} />
      </Section>

      <div className="grid gap-6 xl:grid-cols-2">
        <Section id="preflight" eyebrow="pre-flight" title="Before you ship checklist" icon={ClipboardCheck}>
          <StatusList items={objectArray(data.runtime.preflight)} testId="mission-preflight" />
        </Section>
        <Section id="customization" eyebrow="customization" title="Operator-specific setup" icon={ListChecks}>
          <StatusList items={objectArray(data.runtime.customization)} testId="mission-customization" />
        </Section>
      </div>

      <Section id="attention" eyebrow="attention" title="Lowest readiness guide steps" icon={Globe2}>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {topSteps.map((step) => <CoverageCard key={step.id} item={step} compact />)}
        </div>
      </Section>

      <Section id="source-coverage" eyebrow="source coverage" title="All guide steps, mapped to routes and evidence" icon={Layers3}>
        <div className="space-y-4" data-testid="mission-blueprint-steps">
          <div className="rounded-[1.5rem] border border-current/10 bg-background-base/35 p-3 sm:p-4" data-testid="mission-coverage-controls">
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto_auto] lg:items-center">
              <label className="relative block min-w-0">
                <span className="sr-only">Search guide steps</span>
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-tertiary" aria-hidden="true" />
                <input
                  value={coverageQuery}
                  onChange={(event) => setCoverageQuery(event.target.value)}
                  placeholder="Search steps, domains, next actions…"
                  className="min-h-[44px] w-full rounded-2xl border border-current/10 bg-card/70 pl-10 pr-3 text-sm text-foreground outline-none transition placeholder:text-text-tertiary focus:border-midground/50 focus:ring-2 focus:ring-midground/20"
                  data-testid="mission-coverage-search"
                />
              </label>
              <label className="grid gap-1 text-[10px] uppercase tracking-[0.22em] text-text-tertiary">
                Status
                <select
                  value={coverageStatus}
                  onChange={(event) => setCoverageStatus(event.target.value)}
                  className="min-h-[44px] rounded-2xl border border-current/10 bg-card/70 px-3 text-sm normal-case tracking-normal text-foreground outline-none focus:border-midground/50 focus:ring-2 focus:ring-midground/20"
                  data-testid="mission-coverage-status-filter"
                >
                  <option value="all">All</option>
                  <option value="active">Active</option>
                  <option value="partial">Partial</option>
                  <option value="watch">Watch</option>
                  <option value="planned">Planned</option>
                </select>
              </label>
              <div className="flex min-h-[44px] items-center rounded-2xl border border-current/10 bg-card/50 px-3 text-xs text-muted-foreground">
                <Smartphone className="mr-2 h-4 w-4 text-midground" aria-hidden="true" />
                {visibleSteps.length}/{filteredSteps.length} visible · {data.coverage.steps.length} total
              </div>
            </div>
            <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-text-tertiary">
              <span className="inline-flex min-h-8 items-center rounded-full border border-current/10 px-3">evidence collapses by default</span>
              <span className="inline-flex min-h-8 items-center rounded-full border border-current/10 px-3">search is local only</span>
              <span className="inline-flex min-h-8 items-center rounded-full border border-current/10 px-3">raw prompts never rendered</span>
            </div>
          </div>

          {filteredSteps.length === 0 ? (
            <div className="rounded-[1.5rem] border border-current/10 bg-background-base/35 p-8 text-center" data-testid="mission-coverage-empty">
              <AlertTriangle className="mx-auto h-8 w-8 text-text-tertiary" aria-hidden="true" />
              <h3 className="mt-3 text-sm font-semibold text-foreground">No guide steps match that filter.</h3>
              <p className="mt-1 text-xs text-muted-foreground">Clear the search or switch status back to all.</p>
            </div>
          ) : (
            <>
              <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3" data-testid="mission-step-grid">
                {visibleSteps.map((step) => <CoverageCard key={step.id} item={step} compact />)}
              </div>
              {filteredSteps.length > visibleSteps.length && (
                <div className="flex justify-center">
                  <button
                    type="button"
                    onClick={() => setShowAllSteps(true)}
                    className="inline-flex min-h-[44px] items-center justify-center rounded-2xl border border-current/15 px-4 text-sm font-medium text-foreground transition hover:bg-midground/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-midground/60"
                    data-testid="mission-show-all-steps"
                  >
                    Show all {filteredSteps.length} mapped steps
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </Section>

      <div className="grid gap-6 xl:grid-cols-2">
        <Section id="hermes-picker" eyebrow="Hermes picker" title="H1–H11 feature status" icon={Sparkles}>
          <div className="grid gap-3" data-testid="mission-hermes-features">
            {hermesFeatures.map((feature) => <CoverageCard key={feature.id} item={feature} compact />)}
          </div>
        </Section>
        <Section id="openclaw-picker" eyebrow="OpenClaw picker" title="O1–O10 production feature status" icon={Zap}>
          <div className="grid gap-3" data-testid="mission-openclaw-features">
            {openClawFeatures.map((feature) => <CoverageCard key={feature.id} item={feature} compact />)}
          </div>
        </Section>
      </div>

      <Section id="source-extras" eyebrow="source extras" title="Architecture, prerequisites, next tools and troubleshooting" icon={BookOpen}>
        <SourceExtrasPanel data={data} />
      </Section>

      <Section id="data-flow" eyebrow="data flow" title="Where data goes — safely summarized" icon={Route}>
        <DataFlowPanel data={data} />
      </Section>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Section id="privacy-boundary" eyebrow="privacy boundary" title="Useful without leaking the operator" icon={LockKeyhole}>
          <div className="grid gap-3 sm:grid-cols-2">
            {data.privacy.map((item) => (
              <div key={item.label} className="rounded-[1.35rem] border border-current/10 bg-background-base/35 p-4">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-midground" />
                  <Badge tone="outline">{item.policy}</Badge>
                </div>
                <h3 className="mt-3 text-sm font-semibold text-foreground">{item.label}</h3>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{item.detail}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section id="device-proof" eyebrow="device proof" title="Desktop, tablet, mobile — no cockpit collapse" icon={Fingerprint}>
          <DeviceProofPanel data={data} />
        </Section>
      </div>

      <p className="text-center text-xs text-text-tertiary">
        Generated {data.runtime.generatedAt}. Source checked {data.source.lastChecked}. {data.source.note}
      </p>
      <PluginSlot name="mission-control:bottom" />
    </div>
  );
}
