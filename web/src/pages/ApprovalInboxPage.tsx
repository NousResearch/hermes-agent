import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ClipboardCheck,
  Clock,
  ExternalLink,
  FileWarning,
  LockKeyhole,
  PauseCircle,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { H2, Typography } from "@/components/NouiTypography";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { CronJob, StatusResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";

type ApprovalItem = {
  id: string;
  title: string;
  project: string;
  risk: string;
  status: "standing-gate" | "needs-design" | "ready-to-draft";
  requestedBy: string;
  target: string;
  preview: string;
  safeNext: string;
  blockedAction: string;
  sourcePath?: string;
};

const STANDING_APPROVALS: ApprovalItem[] = [
  {
    id: "live-gateway-restart",
    title: "Live gateway/service restart or reset",
    project: "Hermes Ops",
    risk: "Live-service",
    status: "standing-gate",
    requestedBy: "Jenny safety policy",
    target: "WhatsApp / Discord / API gateway",
    preview: "Restarting can interrupt Travis's active chat surfaces and WhatsApp session continuity.",
    safeNext: "Create an approval request with reason, affected service, expected downtime, and rollback/health proof before restarting.",
    blockedAction: "Do not restart/reset the live gateway from dashboard buttons or unattended jobs.",
    sourcePath: "/home/jenny/ai-ops-brain/ai-ops/PROJECT_STATUS.md",
  },
  {
    id: "public-payment-customer-action",
    title: "Public, payment, outreach, or customer action",
    project: "Tool & Tally / Video / Apps",
    risk: "Money/customer",
    status: "standing-gate",
    requestedBy: "Jenny safety policy",
    target: "Public sites, social accounts, payment systems, customers/prospects",
    preview: "Publishing, charging, sending, or contacting can create real-world consequences.",
    safeNext: "Queue a request with exact target, copy/asset preview, account/channel, and explicit Travis decision needed.",
    blockedAction: "Do not publish, buy, send outreach, deliver reports, or enable live payments from the dashboard.",
    sourcePath: "/home/jenny/ai-ops-brain/PROJECT_COMMAND_CENTER.md",
  },
  {
    id: "credential-auth-change",
    title: "Credential, OAuth, token, or account change",
    project: "Hermes Ops",
    risk: "Credential/auth",
    status: "standing-gate",
    requestedBy: "Jenny safety policy",
    target: "API keys, OAuth tokens, platform account settings, .env/auth.json",
    preview: "Credential changes can break Jenny, publishing, storage, email, or external platform access.",
    safeNext: "Prepare a redacted approval request with which credential/account changes and how rollback will be verified.",
    blockedAction: "Do not reveal, rotate, delete, or replace credentials without current approval.",
    sourcePath: "/home/jenny/.hermes/.env",
  },
  {
    id: "destructive-data-change",
    title: "Destructive cleanup, migration, or reorganization",
    project: "All projects",
    risk: "Destructive",
    status: "standing-gate",
    requestedBy: "Jenny safety policy",
    target: "Files, project folders, OneDrive outside AI, repos, databases, archives",
    preview: "Deletes/moves can lose project state or visible Travis files if scope is wrong.",
    safeNext: "Inventory exact paths, produce manifest, classify safe vs approval-needed, and request approval before deleting outside approved areas.",
    blockedAction: "Do not delete/migrate/reorganize outside OneDrive AI or explicitly approved scope.",
    sourcePath: "/home/jenny/ai-ops-brain/AGENTS.md",
  },
  {
    id: "approval-engine-backend",
    title: "Persistent approval engine backend",
    project: "Hermes Ops",
    risk: "Security boundary",
    status: "needs-design",
    requestedBy: "Jenny Ops Center roadmap",
    target: "Approval state files/API routes/future execution hooks",
    preview: "A backend that stores approvals can become an authority for future actions. If designed poorly, it can accidentally authorize risky work.",
    safeNext: "Design schema, storage path, audit log, expiration, risk labels, and no-execution rules before adding writable endpoints.",
    blockedAction: "Do not add approve/reject write endpoints or execution hooks until the schema and safety policy are reviewed.",
    sourcePath: "/home/jenny/ai-ops-brain/ai-ops/ops-center/approval-inbox-schema-and-policy-2026-05-26.md",
  },
];

function riskTone(risk: string): string {
  const value = risk.toLowerCase();
  if (value.includes("live")) return "border-red-400/40 text-red-200";
  if (value.includes("money") || value.includes("customer")) return "border-amber-400/40 text-amber-200";
  if (value.includes("credential") || value.includes("security")) return "border-violet-400/40 text-violet-200";
  if (value.includes("destructive")) return "border-orange-400/40 text-orange-200";
  return "border-cyan-400/30 text-cyan-200";
}

function statusTone(status: ApprovalItem["status"]): string {
  if (status === "needs-design") return "border-violet-400/40 text-violet-200";
  if (status === "ready-to-draft") return "border-cyan-400/30 text-cyan-200";
  return "border-amber-400/40 text-amber-200";
}

function statusLabel(status: ApprovalItem["status"]): string {
  if (status === "needs-design") return "Needs design";
  if (status === "ready-to-draft") return "Ready to draft";
  return "Standing gate";
}

function problemJobs(jobs: CronJob[]): CronJob[] {
  return jobs.filter((job) => Boolean(job.last_error || (job.state || "").toLowerCase().includes("error") || (job.state || "").toLowerCase().includes("fail")));
}

export default function ApprovalInboxPage() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [jobs, setJobs] = useState<CronJob[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { setEnd } = usePageHeader();

  const load = useCallback(() => {
    setError(null);
    Promise.allSettled([api.getStatus(), api.getCronJobs("all")]).then(([statusResult, jobsResult]) => {
      if (statusResult.status === "fulfilled") setStatus(statusResult.value);
      if (jobsResult.status === "fulfilled") setJobs(jobsResult.value);
      if (statusResult.status === "rejected" || jobsResult.status === "rejected") {
        setError("Some approval-context sources could not refresh.");
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

  const problems = useMemo(() => problemJobs(jobs), [jobs]);
  const visibleItems = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return STANDING_APPROVALS;
    return STANDING_APPROVALS.filter((item) => `${item.title} ${item.project} ${item.risk} ${item.target} ${item.preview}`.toLowerCase().includes(q));
  }, [query]);

  return (
    <main className="h-full overflow-auto px-4 py-5 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <section className="rounded-3xl border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(245,158,11,0.18),transparent_35%),rgba(255,255,255,0.035)] p-5 shadow-2xl shadow-black/30 lg:p-7">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-amber-400/30 bg-amber-500/10 px-3 py-1 text-xs uppercase tracking-[0.22em] text-amber-200">
                <ClipboardCheck className="h-3.5 w-3.5" /> Jenny approval inbox
              </div>
              <H2 className="text-3xl font-bold tracking-tight lg:text-5xl">Decisions before side effects</H2>
              <Typography className="mt-3 max-w-3xl text-sm leading-6 text-text-secondary lg:text-base">
                Read-only gate view. This shows which classes of action require Travis approval and what an approval request must contain. It does not approve, reject, execute, or write approval state yet.
              </Typography>
            </div>
            <Badge tone="outline" className="w-fit border-amber-400/40 text-amber-200">read-only / no execution</Badge>
          </div>
        </section>

        {error && (
          <div className="flex items-center gap-2 rounded-xl border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
            <AlertTriangle className="h-4 w-4" /> {error}
          </div>
        )}

        <section className="grid gap-3 md:grid-cols-4">
          <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-4"><div className="text-xs uppercase tracking-wide text-text-secondary">Standing gates</div><div className="mt-3 text-3xl font-semibold text-text-primary">{STANDING_APPROVALS.length}</div></CardContent></Card>
          <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-4"><div className="text-xs uppercase tracking-wide text-text-secondary">Problem runs</div><div className="mt-3 text-3xl font-semibold text-text-primary">{problems.length}</div></CardContent></Card>
          <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-4"><div className="text-xs uppercase tracking-wide text-text-secondary">Gateway</div><div className={cn("mt-3 text-xl font-semibold", status?.gateway_running ? "text-emerald-300" : "text-red-300")}>{status?.gateway_running ? "Running" : "Unknown"}</div></CardContent></Card>
          <Card className="border-white/10 bg-white/[0.03]"><CardContent className="p-4"><div className="text-xs uppercase tracking-wide text-text-secondary">Execution hooks</div><div className="mt-3 text-xl font-semibold text-amber-200">Blocked</div></CardContent></Card>
        </section>

        <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="relative max-w-xl flex-1">
              <Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-text-secondary" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Filter approvals by project, risk, target..."
                className="w-full rounded-xl border border-white/10 bg-black/30 py-2 pl-9 pr-3 text-sm text-text-primary outline-none focus:border-midground/60"
              />
            </div>
            <Link to="/ops-runs" className="inline-flex">
              <Button ghost className="gap-2"><Clock className="h-4 w-4" /> Review run ledger</Button>
            </Link>
          </div>
        </section>

        <section className="grid gap-4 xl:grid-cols-[1fr_0.75fr]">
          <div className="space-y-3">
            {visibleItems.map((item) => (
              <Card key={item.id} className="border-white/10 bg-white/[0.03]">
                <CardContent className="space-y-4 p-5">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone="outline" className={riskTone(item.risk)}>{item.risk}</Badge>
                        <Badge tone="outline" className={statusTone(item.status)}>{statusLabel(item.status)}</Badge>
                        <span className="text-xs text-text-secondary">{item.project}</span>
                      </div>
                      <div className="mt-2 text-lg font-semibold text-text-primary">{item.title}</div>
                      <div className="mt-1 text-sm text-text-secondary">Target: {item.target}</div>
                    </div>
                    <div className="text-xs text-text-secondary">Requested by: {item.requestedBy}</div>
                  </div>

                  <div className="grid gap-3 lg:grid-cols-3">
                    <div className="rounded-xl border border-white/10 bg-black/25 p-3">
                      <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-text-secondary"><FileWarning className="h-3.5 w-3.5" /> Preview / risk</div>
                      <div className="text-sm leading-6 text-text-primary">{item.preview}</div>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-black/25 p-3">
                      <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-text-secondary"><CheckCircle2 className="h-3.5 w-3.5" /> Safe next</div>
                      <div className="text-sm leading-6 text-text-primary">{item.safeNext}</div>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-black/25 p-3">
                      <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-text-secondary"><LockKeyhole className="h-3.5 w-3.5" /> Blocked action</div>
                      <div className="text-sm leading-6 text-text-primary">{item.blockedAction}</div>
                    </div>
                  </div>

                  {item.sourcePath && (
                    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-white/10 bg-black/25 p-3 text-xs text-text-secondary">
                      <ExternalLink className="h-3.5 w-3.5" /> Source/control path: <span className="break-all text-text-primary">{item.sourcePath}</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>

          <aside className="space-y-4">
            <Card className="border-white/10 bg-white/[0.03]">
              <CardContent className="space-y-3 p-5">
                <div className="flex items-center gap-2 text-midground"><ShieldCheck className="h-5 w-5" /><H2 className="text-xl">Approval request minimum</H2></div>
                <ul className="space-y-2 text-sm leading-6 text-text-secondary">
                  <li>• Exact proposed action and target.</li>
                  <li>• Project/profile and risk label.</li>
                  <li>• Preview, diff, or affected path/channel/account.</li>
                  <li>• Why Jenny recommends it.</li>
                  <li>• Rollback or verification plan.</li>
                  <li>• Expiration/scope so approval cannot be reused loosely.</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-amber-400/25 bg-amber-500/10">
              <CardContent className="space-y-3 p-5">
                <div className="flex items-center gap-2 text-amber-200"><ShieldAlert className="h-5 w-5" /><H2 className="text-xl">Current boundary</H2></div>
                <Typography className="text-sm leading-6 text-amber-50/90">
                  This is not an execution system. The approve/reject backend, audit log, and any future action hooks are the next gate and require a deliberate schema/security pass first.
                </Typography>
              </CardContent>
            </Card>

            <Card className="border-white/10 bg-white/[0.03]">
              <CardContent className="space-y-3 p-5">
                <div className="flex items-center gap-2 text-midground"><PauseCircle className="h-5 w-5" /><H2 className="text-xl">Why this matters</H2></div>
                <Typography className="text-sm leading-6 text-text-secondary">
                  Mission Control should help Travis decide quickly without turning dashboard buttons into hidden autonomy. The approval inbox makes the boundary visible before we add anything writable.
                </Typography>
              </CardContent>
            </Card>
          </aside>
        </section>
      </div>
    </main>
  );
}
