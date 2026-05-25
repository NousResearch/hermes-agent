import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  CalendarClock,
  Clock,
  ExternalLink,
  FileText,
  RefreshCw,
  Search,
  ShieldCheck,
  Timer,
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { H2, Typography } from "@/components/NouiTypography";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { CronJob, SessionInfo } from "@/lib/api";
import { cn } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";

type LedgerRow = {
  id: string;
  kind: "session" | "cron";
  title: string;
  source: string;
  status: string;
  project: string;
  started?: number | string | null;
  ended?: number | string | null;
  detail: string;
  link?: string;
  error?: string | null;
};

function formatTime(value?: string | number | null): string {
  if (!value) return "—";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString();
}

function sessionProject(session: SessionInfo): string {
  const text = `${session.title || ""} ${session.preview || ""} ${session.source || ""}`.toLowerCase();
  if (text.includes("family")) return "Family Hub";
  if (text.includes("tool") || text.includes("tally") || text.includes("vendorproof")) return "Tool & Tally / VendorProof";
  if (text.includes("signal") || text.includes("video") || text.includes("impossible")) return "Video";
  if (text.includes("waha")) return "Waha";
  if (text.includes("crucible")) return "The Crucible";
  if (text.includes("research")) return "Research Ops";
  if (text.includes("cron_")) return "Automation";
  return "Hermes Ops";
}

function cronProject(job: CronJob): string {
  const text = `${job.name || ""} ${job.prompt || ""} ${job.script || ""} ${job.deliver || ""}`.toLowerCase();
  if (text.includes("family")) return "Family Hub";
  if (text.includes("tool") || text.includes("tally") || text.includes("vendorproof")) return "Tool & Tally / VendorProof";
  if (text.includes("signal") || text.includes("video") || text.includes("impossible")) return "Video";
  if (text.includes("waha")) return "Waha";
  if (text.includes("storage")) return "Hermes Ops";
  if (text.includes("discord") || text.includes("thread")) return "Hermes Ops";
  return job.profile || job.profile_name || "default";
}

function jobTitle(job: CronJob): string {
  if (job.name) return job.name.slice(0, 100);
  if (job.script) return `Script: ${job.script}`.slice(0, 100);
  const raw = (job.prompt || "").trim();
  if (raw.startsWith("[IMPORTANT:")) return `Cron job ${job.id}`;
  const prompt = raw.replace(/^\[IMPORTANT:[\s\S]*?\]\s*/i, "").trim();
  if (prompt) return prompt.slice(0, 100);
  return `Cron job ${job.id}`;
}

function jobState(job: CronJob): string {
  if (job.last_error) return "failed";
  return job.state || (job.enabled === false ? "paused" : "scheduled");
}

function statusTone(status: string): string {
  const value = status.toLowerCase();
  if (value.includes("active") || value.includes("running") || value.includes("scheduled")) return "border-emerald-400/35 text-emerald-200";
  if (value.includes("fail") || value.includes("error")) return "border-red-400/40 text-red-200";
  if (value.includes("paused") || value.includes("ended")) return "border-slate-400/40 text-slate-200";
  return "border-cyan-400/30 text-cyan-200";
}

function cleanTitle(value: string | null | undefined, fallback: string): string {
  const raw = (value || "").trim();
  if (raw.startsWith("[IMPORTANT:")) return fallback;
  const text = raw.replace(/^\[IMPORTANT:[\s\S]*?\]\s*/i, "").trim();
  return (text || fallback).slice(0, 100);
}

function buildLedger(sessions: SessionInfo[], jobs: CronJob[]): LedgerRow[] {
  const sessionRows = sessions.map((session) => ({
    id: session.id,
    kind: "session" as const,
    title: cleanTitle(session.title || session.preview, session.id),
    source: session.source || "session",
    status: session.is_active ? "active" : session.ended_at ? "ended" : "recent",
    project: sessionProject(session),
    started: session.started_at,
    ended: session.ended_at || session.last_active,
    detail: `${session.message_count} messages · ${session.tool_call_count} tool calls · ${session.input_tokens + session.output_tokens} tokens`,
    link: `/sessions?session=${encodeURIComponent(session.id)}`,
  }));

  const jobRows = jobs.map((job) => ({
    id: job.id,
    kind: "cron" as const,
    title: jobTitle(job),
    source: job.profile || job.profile_name || "cron",
    status: jobState(job),
    project: cronProject(job),
    started: job.last_run_at,
    ended: job.next_run_at,
    detail: `${job.schedule_display || job.schedule?.display || job.schedule?.expr || "manual/unknown schedule"} · deliver ${job.deliver || "origin/local"}`,
    link: "/cron",
    error: job.last_error,
  }));

  return [...jobRows, ...sessionRows].sort((a, b) => {
    const at = a.started ? new Date(typeof a.started === "number" ? a.started * 1000 : a.started).getTime() : 0;
    const bt = b.started ? new Date(typeof b.started === "number" ? b.started * 1000 : b.started).getTime() : 0;
    return bt - at;
  });
}

export default function OpsRunsPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [jobs, setJobs] = useState<CronJob[]>([]);
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<"all" | "session" | "cron" | "problem">("all");
  const [error, setError] = useState<string | null>(null);
  const { setEnd } = usePageHeader();

  const load = useCallback(() => {
    setError(null);
    Promise.allSettled([api.getSessions(40), api.getCronJobs("all")]).then(([sessionsResult, jobsResult]) => {
      if (sessionsResult.status === "fulfilled") setSessions(sessionsResult.value.sessions);
      if (jobsResult.status === "fulfilled") setJobs(jobsResult.value);
      if (sessionsResult.status === "rejected" || jobsResult.status === "rejected") {
        setError("Some run ledger sources could not refresh.");
      }
    });
  }, []);

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 30_000);
    return () => window.clearInterval(timer);
  }, [load]);

  useEffect(() => {
    setEnd(
      <Button ghost onClick={load} className="gap-2">
        <RefreshCw className="h-4 w-4" /> Refresh
      </Button>,
    );
    return () => setEnd(null);
  }, [load, setEnd]);

  const ledger = useMemo(() => buildLedger(sessions, jobs), [sessions, jobs]);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return ledger.filter((row) => {
      if (kind === "session" && row.kind !== "session") return false;
      if (kind === "cron" && row.kind !== "cron") return false;
      if (kind === "problem" && !row.error && !row.status.toLowerCase().includes("fail") && !row.status.toLowerCase().includes("error")) return false;
      if (!q) return true;
      return `${row.title} ${row.project} ${row.source} ${row.status} ${row.detail}`.toLowerCase().includes(q);
    });
  }, [kind, ledger, query]);

  const activeCount = ledger.filter((row) => row.status.toLowerCase().includes("active") || row.status.toLowerCase().includes("running") || row.status.toLowerCase().includes("scheduled")).length;
  const problemCount = ledger.filter((row) => row.error || row.status.toLowerCase().includes("fail") || row.status.toLowerCase().includes("error")).length;

  return (
    <main className="h-full overflow-auto px-4 py-5 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <section className="rounded-3xl border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(20,184,166,0.18),transparent_35%),rgba(255,255,255,0.035)] p-5 shadow-2xl shadow-black/30 lg:p-7">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-midground/30 bg-midground/10 px-3 py-1 text-xs uppercase tracking-[0.22em] text-midground">
                <Activity className="h-3.5 w-3.5" /> Jenny run ledger
              </div>
              <H2 className="text-3xl font-bold tracking-tight lg:text-5xl">Runs, sessions, and automation history</H2>
              <Typography className="mt-3 max-w-3xl text-sm leading-6 text-text-secondary lg:text-base">
                Read-only Phase 2 view. It combines recent Hermes sessions and cron jobs so Travis can see what ran, where it came from, what project it likely belongs to, and where to inspect details.
              </Typography>
            </div>
            <Badge tone="outline" className="w-fit border-cyan-400/30 text-cyan-200">read-only</Badge>
          </div>
        </section>

        {error && (
          <div className="flex items-center gap-2 rounded-xl border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
            <AlertTriangle className="h-4 w-4" /> {error}
          </div>
        )}

        <section className="grid gap-3 md:grid-cols-4">
          <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-4"><div className="text-xs uppercase tracking-wide text-text-secondary">Ledger rows</div><div className="mt-3 text-3xl font-semibold text-text-primary">{ledger.length}</div></CardContent></Card>
          <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-4"><div className="text-xs uppercase tracking-wide text-text-secondary">Sessions</div><div className="mt-3 text-3xl font-semibold text-text-primary">{sessions.length}</div></CardContent></Card>
          <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-4"><div className="text-xs uppercase tracking-wide text-text-secondary">Active/scheduled</div><div className="mt-3 text-3xl font-semibold text-text-primary">{activeCount}</div></CardContent></Card>
          <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-4"><div className="text-xs uppercase tracking-wide text-text-secondary">Problems</div><div className="mt-3 text-3xl font-semibold text-text-primary">{problemCount}</div></CardContent></Card>
        </section>

        <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="relative max-w-xl flex-1">
              <Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-text-secondary" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Filter by project, source, status, title..."
                className="w-full rounded-xl border border-white/10 bg-black/30 py-2 pl-9 pr-3 text-sm text-text-primary outline-none focus:border-midground/60"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              {(["all", "session", "cron", "problem"] as const).map((value) => (
                <Button key={value} size="sm" ghost={kind !== value} onClick={() => setKind(value)}>
                  {value === "all" ? "All" : value === "session" ? "Sessions" : value === "cron" ? "Cron" : "Problems"}
                </Button>
              ))}
            </div>
          </div>
        </section>

        <section className="space-y-3">
          {filtered.length ? filtered.map((row) => (
            <Card key={`${row.kind}:${row.id}`} className="border-white/10 bg-white/[0.03]">
              <CardContent className="p-4">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      {row.kind === "session" ? <Bot className="h-4 w-4 text-midground" /> : <CalendarClock className="h-4 w-4 text-midground" />}
                      <Badge tone="outline" className={cn("uppercase", statusTone(row.status))}>{row.status}</Badge>
                      <Badge tone="outline" className="border-white/20 text-text-secondary">{row.project}</Badge>
                      <span className="text-xs text-text-secondary">{row.source}</span>
                    </div>
                    <div className="mt-2 truncate text-base font-semibold text-text-primary">{row.title}</div>
                    <div className="mt-1 text-sm text-text-secondary">{row.detail}</div>
                    {row.error && <div className="mt-2 rounded-lg border border-red-400/25 bg-red-500/10 p-2 text-xs text-red-100">{row.error}</div>}
                  </div>
                  <div className="grid min-w-56 gap-2 text-xs text-text-secondary">
                    <div className="flex items-center gap-2"><Clock className="h-3.5 w-3.5" /> Started/last: {formatTime(row.started)}</div>
                    <div className="flex items-center gap-2"><Timer className="h-3.5 w-3.5" /> Ended/next: {formatTime(row.ended)}</div>
                    <div className="flex items-center gap-2"><FileText className="h-3.5 w-3.5" /> {row.id}</div>
                    {row.link && (
                      <Link to={row.link} className="inline-flex w-fit items-center gap-2 text-midground hover:underline">
                        <ExternalLink className="h-3.5 w-3.5" /> Inspect
                      </Link>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          )) : (
            <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-6 text-sm text-text-secondary">No runs match the current filter.</CardContent></Card>
          )}
        </section>

        <section className="rounded-2xl border border-white/10 bg-black/25 p-4 text-sm text-text-secondary">
          <div className="mb-2 flex items-center gap-2 font-semibold text-text-primary"><ShieldCheck className="h-4 w-4" /> Boundary</div>
          This page is observational. It does not trigger jobs, restart services, change credentials, publish, send outreach, or delete data. Those remain approval-gated.
        </section>
      </div>
    </main>
  );
}
