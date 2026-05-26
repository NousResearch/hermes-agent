import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bot,
  CalendarDays,
  CheckCircle2,
  Clipboard,
  Clock,
  Database,
  ExternalLink,
  FileText,
  FolderKanban,
  Gauge,
  Globe2,
  Inbox,
  Link as LinkIcon,
  ListChecks,
  MessageSquare,
  Play,
  Rocket,
  ShieldCheck,
  Sparkles,
  Target,
  Terminal,
  Zap,
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { H2, Typography } from "@/components/NouiTypography";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { CronJob, OpsApprovalSummary, OpsSocialPlatformHistory, OpsSocialPlatformHistoryEvent, OpsSocialPlatformStatus, OpsSocialPlatformStatusItem, OpsSocialPlatformStatusUpdate, SessionInfo, StatusResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";

const DISCORD_GUILD = "1497927076567715921";

function discordThreadUrl(threadId: string): string {
  return `https://discord.com/channels/${DISCORD_GUILD}/${threadId}`;
}

const PROJECTS = [
  {
    name: "Hermes Ops",
    short: "Gateway, memory, workers, dashboard",
    statusPath: "/home/jenny/ai-ops-brain/ai-ops/PROJECT_STATUS.md",
    thread: "1507671041118437416",
    profile: "default",
    tone: "from-cyan-500/20 to-blue-500/10",
    next: "Keep Jenny reliable; no gateway restarts unless approved.",
    phase: "Coordinator/gateway/ops-brain stabilization",
    waitingOn: "Travis approval for any gateway restart, credential change, or new recurring cron.",
    health: "Active",
  },
  {
    name: "Main Jenny",
    short: "Coordinator, routing, project OS",
    statusPath: "/home/jenny/ai-ops-brain/PROJECT_COMMAND_CENTER.md",
    thread: "1508336998149128205",
    profile: "default",
    tone: "from-teal-500/20 to-cyan-500/10",
    next: "Use as coordinator only; route deep work into the correct project/profile.",
    phase: "Routing and project OS",
    waitingOn: "Use the right hard-walled profile before deep work.",
    health: "Active",
  },
  {
    name: "Research Ops",
    short: "Deep research, tool scouting, source packets",
    statusPath: "/home/jenny/ai-ops-brain/ai-ops/research-ops/PROJECT_STATUS.md",
    thread: "1508395290929926185",
    profile: "default",
    tone: "from-lime-500/20 to-green-500/10",
    next: "Keep research source-ledgered and convert durable findings into project notes.",
    phase: "Decision-support packets",
    waitingOn: "None; research remains source-ledgered and draft-only unless approved.",
    health: "Active",
  },
  {
    name: "Family Hub",
    short: "Private beta + APK readiness",
    statusPath: "/home/jenny/ai-ops-brain/family-hub/PROJECT_STATUS.md",
    thread: "1507598956752928820",
    profile: "family-hub",
    tone: "from-emerald-500/20 to-teal-500/10",
    next: "Verify repo/build state on Main Laptop before claiming APK readiness.",
    phase: "Private beta/APK readiness",
    waitingOn: "Explicit approval for release/store/public actions; Main Laptop proof for APK claims.",
    health: "High priority",
  },
  {
    name: "Family Hub Public App",
    short: "Separate freemium public app concept",
    statusPath: "/home/jenny/ai-ops-brain/family-hub-public-app/PROJECT_STATUS.md",
    thread: "1507081077196460185",
    profile: "family-hub",
    tone: "from-green-500/20 to-emerald-500/10",
    next: "Keep separate from the private family app; no app-store/public action without approval.",
    phase: "Public app concept/foundation",
    waitingOn: "Decision gates before backend, app-store, billing, or public launch.",
    health: "Gated",
  },
  {
    name: "Tool & Tally",
    short: "Launch, outreach, fulfillment gates",
    statusPath: "/home/jenny/ai-ops-brain/business/no-call-lead-engine-business-os/PROJECT_STATUS.md",
    thread: "1507337638603132988",
    profile: "no-call-estimateready",
    tone: "from-amber-500/20 to-orange-500/10",
    next: "Keep launch/payment/customer delivery gated unless specifically approved.",
    phase: "Pre-launch hardening/outreach prep",
    waitingOn: "Approval for launch, payment, intake, outreach, or customer delivery.",
    health: "Gated",
  },
  {
    name: "VendorProof",
    short: "Agentic managed-service proof package",
    statusPath: "/home/jenny/ai-ops-brain/business/vendorproof-agentic-build/PROJECT_STATUS.md",
    thread: "1507424793300045946",
    profile: "default",
    tone: "from-yellow-500/20 to-stone-500/10",
    next: "Private fictional/sample-data proof only; no launch, payments, outreach, or real customer data.",
    phase: "Private proof package",
    waitingOn: "Travis review before pilot, real data, outreach, or payment path.",
    health: "Review gate",
  },
  {
    name: "Video Channel",
    short: "Broad social/video production lane",
    statusPath: "/home/jenny/ai-ops-brain/social-video/signal-room/PROJECT_STATUS.md",
    thread: "1508469222290751669",
    profile: "money-signal-video",
    tone: "from-violet-500/20 to-fuchsia-500/10",
    next: "Preserve approved Lane A cinematic-plate quality and avoid old Signal Room regressions.",
    phase: "Production-quality video pipeline",
    waitingOn: "Review before publishing; protect Lane A quality baseline.",
    health: "Active",
  },
  {
    name: "Impossible Footage",
    short: "Cinematic reconstruction channel concept",
    statusPath: "/home/jenny/ai-ops-brain/social-video/impossible-footage/PROJECT_STATUS.md",
    thread: "1508453418493149224",
    profile: "money-signal-video",
    tone: "from-indigo-500/20 to-slate-500/10",
    next: "Make abstract facts instantly vivid as impossible POVs; pre-render hook/first-frame gate stays strict.",
    phase: "Channel strategy/proof lock",
    waitingOn: "Title/hook/first-frame gate before renders or public production.",
    health: "Gated",
  },
  {
    name: "Waha Inspection",
    short: "Owner-side engineering work",
    statusPath: "/home/jenny/ai-ops-brain/waha/PROJECT_STATUS.md",
    thread: "1501221381302390835",
    profile: "wahainspection",
    tone: "from-sky-500/20 to-slate-500/10",
    next: "Route deep engineering details to Waha profile/files.",
    phase: "Owner-side engineering support",
    waitingOn: "Keep details inside Waha profile/files; no cross-project leakage.",
    health: "Active",
  },
  {
    name: "Personal Life",
    short: "Reminders, travel, family admin",
    statusPath: "/home/jenny/ai-ops-brain/personal/PROJECT_STATUS.md",
    thread: "1505346970208960572",
    profile: "personal-life",
    tone: "from-rose-500/20 to-pink-500/10",
    next: "Keep personal logistics separate from business/coding/Waha.",
    phase: "Personal logistics/reminders",
    waitingOn: "Use personal-life profile for deep personal context.",
    health: "Active",
  },
  {
    name: "The Crucible",
    short: "Creative project pointer / archive",
    statusPath: "/home/jenny/ai-ops-brain/creative/the-crucible/PROJECT_STATUS.md",
    thread: "1501500319350259744",
    profile: "the-crucible",
    tone: "from-red-500/20 to-stone-500/10",
    next: "Do not restore large archive or run heavy media unless requested.",
    phase: "Archived pointer / creative continuity",
    waitingOn: "Explicit request before restoring archive or heavy media work.",
    health: "Dormant",
  },
];
const GOAL_TEMPLATES = [
  {
    name: "Daily project command review",
    project: "Hermes Ops",
    prompt:
      "Read /home/jenny/ai-ops-brain/PROJECT_COMMAND_CENTER.md and the relevant PROJECT_STATUS.md files. Give Travis a concise Done / Blocked / Next summary. Do not start public, payment, outreach, destructive, or gateway-restart actions.",
  },
  {
    name: "Family Hub APK readiness check",
    project: "Family Hub",
    prompt:
      "Use the family-hub profile/context. Inspect the Family Hub repo on Main Laptop, verify current branch/build/versionCode/test state, and update /home/jenny/ai-ops-brain/family-hub/PROJECT_STATUS.md. Do not use EAS or publish anything unless Travis explicitly approves.",
  },
  {
    name: "Tool & Tally safe next packet",
    project: "Tool & Tally",
    prompt:
      "Read Tool & Tally PROJECT_STATUS.md and choose one staging-only safe improvement or QA packet. Keep launch, payment, intake, outreach, and customer delivery blocked. Save evidence paths and update status.",
  },
  {
    name: "Signal Room production gate",
    project: "Signal Room / Video",
    prompt:
      "Read the Signal Room PROJECT_STATUS.md and production locks. Review the next video idea or artifact against Lane A baseline: cinematic text-free plates, B-roll progression, typography safe zones, TTS realism, and banned theme-language checks.",
  },
];

const OPERATING_LANES = [
  {
    label: "Capture",
    detail: "Discord / WhatsApp / browser notes land here first.",
  },
  {
    label: "Clarify",
    detail: "Convert chat into a bounded goal, risk gate, or project status update.",
  },
  {
    label: "Execute",
    detail: "Use the right profile, repo, worker, or AI Ops Brain folder.",
  },
  {
    label: "Record",
    detail: "Update PROJECT_STATUS.md so thread rollover does not lose state.",
  },
];

const DASHBOARD_METRICS = [
  { label: "Project homes", value: PROJECTS.length.toString(), detail: "Status files linked" },
  { label: "Goal templates", value: GOAL_TEMPLATES.length.toString(), detail: "Safe launch prompts" },
  { label: "Hard walls", value: "7", detail: "Profiles / domains separated" },
  { label: "Source of truth", value: "1", detail: "AI Ops Brain command center" },
];

const VAULT_LINKS = [
  "/home/jenny/ai-ops-brain/PROJECT_COMMAND_CENTER.md",
  "/home/jenny/ai-ops-brain/AGENTS.md",
  "/home/jenny/ai-ops-brain/ai-ops/project-systems/discord-to-project-os-continuity-runbook-2026-05-25.md",
];

const DECISION_QUEUE = [
  {
    label: "Live gateway/service restart",
    detail: "WhatsApp/Discord/API gateway restarts stay gated unless Travis approves the exact restart or service is already down.",
    risk: "Live-service",
  },
  {
    label: "Public/payment/customer action",
    detail: "Publishing, outreach, customer delivery, payment links, purchases, and real data intake need current approval.",
    risk: "Money/customer",
  },
  {
    label: "Credential or destructive change",
    detail: "Auth changes and deletes/migrations outside approved OneDrive AI scope require explicit approval.",
    risk: "Destructive",
  },
];

const OPERATIONS_FOLDERS = [
  { label: "00 Human", tone: "border-red-400/40 text-red-200", detail: "Travis decisions" },
  { label: "Copilot", tone: "border-orange-400/40 text-orange-200", detail: "Jenny coordination" },
  { label: "Docs", tone: "border-yellow-400/40 text-yellow-100", detail: "AI Ops Brain" },
  { label: "Machine", tone: "border-emerald-400/40 text-emerald-200", detail: "VPS + workers" },
  { label: "Sessions", tone: "border-cyan-400/40 text-cyan-200", detail: "Recent runs" },
  { label: "System", tone: "border-blue-400/40 text-blue-200", detail: "Gateway health" },
];

const DAILY_DRIVERS = [
  "Inbox triage",
  "Approval review",
  "Gateway health",
  "Project routing",
  "Record durable state",
];

const OPS_BOUNDARY_STEPS = [
  { label: "Prepare", detail: "Draft command or proposal packet" },
  { label: "Review", detail: "Travis approval stays visible" },
  { label: "Probe", detail: "Read-only status only; config disabled" },
  { label: "Stop", detail: "Restarts/sends/deletes/cron/payment gated" },
];

const SOCIAL_PLATFORM_STATUS: OpsSocialPlatformStatusItem[] = [
  {
    platform: "YouTube",
    published: "Needs sync",
    scheduled: "Needs sync",
    issues_private: "Queue reset / private check needed",
    readiness: "Canonical upload engine, but live counts require a read-only sync.",
    source: "Default dashboard status; no local sync file found.",
    status: "needs_sync",
  },
  {
    platform: "Facebook",
    published: "Needs sync",
    scheduled: "Needs sync",
    issues_private: "Legacy old-style queue blocked",
    readiness: "Native Reels path exists; use only approved current-quality packages.",
    source: "Default dashboard status; no local sync file found.",
    status: "needs_sync",
  },
  {
    platform: "Instagram",
    published: "Needs sync",
    scheduled: "0 known scheduler",
    issues_private: "Immediate publish only / API readiness check",
    readiness: "Do not call scheduling ready until a real scheduler and token check exist.",
    source: "Default dashboard status; no local sync file found.",
    status: "needs_sync",
  },
  {
    platform: "TikTok",
    published: "0",
    scheduled: "0",
    issues_private: "Onboarding/API not ready",
    readiness: "Format support is not posting readiness; OAuth/app review remains gated.",
    source: "Default dashboard status; no local sync file found.",
    status: "blocked",
  },
];

const readablePanel = "min-w-0 rounded-xl border border-[#284848] bg-[#061f1f]/85 p-4 shadow-sm shadow-black/20";
const cockpitCard = "border-[#264545] bg-[#071717]/90 shadow-[0_0_0_1px_rgba(47,214,161,0.04),0_18px_60px_rgba(0,0,0,0.28)]";
const cockpitPanel = "rounded-xl border border-[#284848] bg-black/30 p-3";
const readableTitle = "min-w-0 break-words text-base font-semibold leading-6 text-text-primary";
const readableBody = "mt-1 min-w-0 break-words text-sm leading-6 text-text-secondary";
const readableSectionHeading = "text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90";
const readableBadge = "max-w-full shrink-0 whitespace-normal text-right leading-5 sm:max-w-[10rem]";
const SOCIAL_STATUS_OPTIONS = [
  { value: "ok", label: "OK" },
  { value: "needs_review", label: "Needs review" },
  { value: "blocked", label: "Blocked" },
  { value: "not_connected", label: "Not connected" },
  { value: "needs_sync", label: "Needs sync" },
];
const SOCIAL_STATUS_STALE_DAYS = 7;

function isProblemJob(job: CronJob): boolean {
  const state = getJobState(job).toLowerCase();
  return Boolean(job.last_error || state.includes("error") || state.includes("fail"));
}

function projectHealthTone(health?: string): string {
  const value = (health || "").toLowerCase();
  if (value.includes("high")) return "border-emerald-400/40 text-emerald-200";
  if (value.includes("gate") || value.includes("review")) return "border-amber-400/40 text-amber-200";
  if (value.includes("dormant")) return "border-slate-400/40 text-slate-200";
  return "border-cyan-400/30 text-cyan-200";
}

function socialPlatformTone(item: OpsSocialPlatformStatusItem): string {
  const platform = item.platform.toLowerCase();
  const status = item.status.toLowerCase();
  if (status.includes("blocked") || status.includes("not_connected")) return "border-slate-400/30 bg-slate-500/10 text-slate-100";
  if (status.includes("needs_review")) return "border-amber-400/35 bg-amber-500/10 text-amber-100";
  if (platform.includes("youtube")) return "border-red-400/30 bg-red-500/10 text-red-100";
  if (platform.includes("facebook")) return "border-blue-400/30 bg-blue-500/10 text-blue-100";
  if (platform.includes("instagram")) return "border-pink-400/30 bg-pink-500/10 text-pink-100";
  if (platform.includes("tiktok")) return "border-slate-400/30 bg-slate-500/10 text-slate-100";
  return "border-cyan-400/30 bg-cyan-500/10 text-cyan-100";
}

function normalizeSocialStatusLabel(value?: string | null): string {
  const status = String(value || "needs_sync").trim().toLowerCase();
  return SOCIAL_STATUS_OPTIONS.find((option) => option.value === status)?.label || status.replace(/_/g, " ") || "Needs sync";
}

function getSocialStaleness(item: OpsSocialPlatformStatusItem): { label: string; stale: boolean } {
  const raw = item.last_checked_at;
  if (!raw) return { label: "Never manually checked", stale: true };
  const timestamp = new Date(raw).getTime();
  if (!Number.isFinite(timestamp)) return { label: "Last checked timestamp invalid", stale: true };
  const ageMs = Date.now() - timestamp;
  if (ageMs < 0) return { label: "Checked just now", stale: false };
  const ageDays = Math.floor(ageMs / 86_400_000);
  if (ageDays === 0) return { label: "Checked today", stale: false };
  if (ageDays === 1) return { label: "Checked 1 day ago", stale: false };
  return { label: `Checked ${ageDays} days ago`, stale: ageDays >= SOCIAL_STATUS_STALE_DAYS };
}

function formatHistoryTime(value?: string | null): string {
  if (!value) return "Unknown time";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function describeHistoryEvent(event: OpsSocialPlatformHistoryEvent): string {
  const counts = event.status_counts || {};
  const parts = SOCIAL_STATUS_OPTIONS
    .map((option) => [option.label, counts[option.value] || 0] as const)
    .filter(([, count]) => count > 0)
    .map(([label, count]) => `${count} ${label.toLowerCase()}`);
  return parts.length ? parts.join(" · ") : `${event.platform_count || 0} platform rows`;
}

function getJobTitle(job: CronJob): string {
  return (job.name || job.prompt || job.script || job.id || "Cron job").slice(0, 80);
}

function getJobState(job: CronJob): string {
  return job.state || (job.enabled === false ? "paused" : "scheduled");
}

function getSchedule(job: CronJob): string {
  return job.schedule_display || job.schedule?.display || job.schedule?.expr || "—";
}

function jobSearchText(job: CronJob): string {
  return `${job.id || ""} ${job.name || ""} ${job.prompt || ""} ${job.script || ""} ${job.deliver || ""} ${job.workdir || ""}`.toLowerCase();
}

function isActiveJob(job: CronJob): boolean {
  return job.enabled !== false && getJobState(job) !== "paused";
}

function isApprovedStandingGuard(job: CronJob): boolean {
  const text = jobSearchText(job);
  return (
    text.includes("backup") ||
    text.includes("storage hygiene guard") ||
    text.includes("memory capacity guard") ||
    text.includes("discord thread lifecycle") ||
    text.includes("redacted export")
  );
}

function isProjectAutomation(job: CronJob): boolean {
  const text = jobSearchText(job);
  return (
    text.includes("tool") ||
    text.includes("tally") ||
    text.includes("signal") ||
    text.includes("instagram") ||
    text.includes("family") ||
    text.includes("vendorproof") ||
    text.includes("outreach") ||
    text.includes("queue") ||
    text.includes("email") ||
    text.includes("launch-gate")
  );
}

function needsAutomationReview(job: CronJob): boolean {
  if (!isActiveJob(job)) return false;
  if (isProblemJob(job)) return true;
  const text = jobSearchText(job);
  const schedule = getSchedule(job).toLowerCase();
  const deliver = (job.deliver || "").toLowerCase();
  return (
    deliver === "origin" ||
    Boolean(job.last_delivery_error) ||
    schedule.includes("every 1m") ||
    schedule.includes("every 5m") ||
    text.includes("outreach") ||
    text.includes("queue runner") ||
    text.includes("email watcher") ||
    text.includes("launch-gate") ||
    (!isApprovedStandingGuard(job) && !isProjectAutomation(job))
  );
}

function describeAutomationReason(job: CronJob): string {
  if (!isActiveJob(job)) return "Paused or retired — not currently running.";
  if (isProblemJob(job)) return "Problem state or recent error reported.";
  const schedule = getSchedule(job).toLowerCase();
  const deliver = (job.deliver || "").toLowerCase();
  if (deliver === "origin") return "Delivers to the current thread; review whether it should stay visible.";
  if (schedule.includes("every 1m") || schedule.includes("every 5m")) return "High-frequency automation; confirm it is still intentionally active.";
  if (isApprovedStandingGuard(job)) return "Protective/backup-style standing guard candidate.";
  if (isProjectAutomation(job)) return "Project automation/watchdog; keep only if the project gate still approves it.";
  return "Active automation without a clear standing approval category.";
}

function buildPauseProposalCommand(jobs: CronJob[]): string {
  const lines = jobs.map((job) => `- ${job.id} (${job.profile || "default"}): ${getJobTitle(job)} — ${getSchedule(job)} — ${describeAutomationReason(job)}`);
  return `/goal Draft a pause-review proposal for these enabled Hermes cron/automation jobs. Do not pause, remove, trigger, or edit any job yet. Summarize why each job may need Travis review and ask for exact approval before mutation.\n${lines.join("\n")}`;
}

function formatTime(value?: string | number | null): string {
  if (!value) return "—";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString();
}

function platformSummary(status: StatusResponse | null): string {
  if (!status?.gateway_platforms) return "No platform details";
  const names = Object.entries(status.gateway_platforms)
    .filter(([, value]) => value?.state && value.state !== "disabled")
    .map(([name, value]) => `${name}: ${value.state}`);
  return names.length ? names.join(" · ") : "No active platforms reported";
}

function WorkspaceRail({ activeJobs, recentSessions }: { activeJobs: CronJob[]; recentSessions: SessionInfo[] }) {
  return (
    <aside className="font-readable-ui hidden xl:block">
      <div className="sticky top-5 flex max-h-[calc(100vh-2.5rem)] flex-col gap-4">
        <Card className={cn(cockpitCard, "overflow-hidden")}>
          <CardContent className="space-y-4 p-4">
            <div className="flex items-center gap-2 text-emerald-200">
              <Database className="h-4 w-4" />
              <div className="text-sm font-semibold uppercase tracking-[0.12em]">Operations vault</div>
            </div>
            <div className="space-y-2">
              {OPERATIONS_FOLDERS.map((folder) => (
                <div key={folder.label} className={cn("rounded-lg border bg-black/25 px-3 py-2", folder.tone)}>
                  <div className="text-sm font-semibold">{folder.label}</div>
                  <div className="mt-0.5 text-xs text-text-secondary">{folder.detail}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className={cn(cockpitCard, "min-h-0 flex-1 overflow-hidden")}>
          <CardContent className="flex h-full min-h-0 flex-col gap-3 p-4">
            <div className="flex items-center gap-2 text-emerald-200">
              <Bot className="h-4 w-4" />
              <div className="text-sm font-semibold uppercase tracking-[0.12em]">Recent trace</div>
            </div>
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
              {recentSessions.slice(0, 4).map((session) => (
                <Link
                  key={session.id}
                  to={`/sessions?session=${encodeURIComponent(session.id)}`}
                  className="block rounded-lg border border-[#284848] bg-black/25 p-3 transition hover:border-emerald-400/40"
                >
                  <div className="line-clamp-3 text-sm font-medium leading-5 text-text-primary">{session.title || session.preview || session.id}</div>
                  <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[0.72rem] leading-4 text-text-secondary">
                    <span className="rounded-full border border-[#284848] bg-black/25 px-2 py-0.5">{session.source || "session"}</span>
                    <span>{formatTime(session.last_active)}</span>
                  </div>
                </Link>
              ))}
              {!recentSessions.length && <div className="rounded-lg border border-[#284848] bg-black/25 p-3 text-sm text-text-secondary">No sessions loaded.</div>}
            </div>
          </CardContent>
        </Card>

        <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 p-3 text-sm text-emerald-100">
          {activeJobs.length} active automation entr{activeJobs.length === 1 ? "y" : "ies"}; risky actions still require chat approval.
        </div>
      </div>
    </aside>
  );
}

function CommandDeck({ status, activeJobs, approvalSummary }: { status: StatusResponse | null; activeJobs: CronJob[]; approvalSummary: OpsApprovalSummary | null }) {
  return (
    <aside className="font-readable-ui hidden 2xl:block">
      <div className="sticky top-5 flex max-h-[calc(100vh-2.5rem)] flex-col gap-4">
        <Card className={cockpitCard}>
          <CardContent className="space-y-4 p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-emerald-200">
                <CalendarDays className="h-4 w-4" />
                <div className="text-sm font-semibold uppercase tracking-[0.12em]">Command stack</div>
              </div>
              <Badge tone="outline" className="border-emerald-400/30 text-emerald-200">Now</Badge>
            </div>

            <div className="space-y-2">
              <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-3">
                <div className="text-xs uppercase tracking-[0.1em] text-emerald-100/80">Current focus</div>
                <div className="mt-2 text-lg font-semibold leading-6 text-text-primary">Make Ops Center a practical operator dashboard</div>
                <div className="mt-2 text-sm leading-6 text-text-secondary">Dashboard can review approvals and show gated fixed actions; live execution remains disabled unless Travis approves exact config scope.</div>
              </div>
              <Link to="/approvals" className="block rounded-xl border border-amber-400/25 bg-amber-500/10 p-3 transition hover:border-amber-300/50">
                <div className="flex items-center gap-2 text-amber-100">
                  <Inbox className="h-4 w-4" />
                  <span className="text-sm font-semibold">Approval inbox</span>
                </div>
                <div className="mt-2 text-2xl font-semibold text-text-primary">{approvalSummary?.pending_count ?? DECISION_QUEUE.length}</div>
                <div className="text-sm text-text-secondary">pending / standing gates</div>
              </Link>
            </div>
          </CardContent>
        </Card>

        <Card className={cockpitCard}>
          <CardContent className="space-y-3 p-4">
            <div className="flex items-center gap-2 text-emerald-200">
              <Zap className="h-4 w-4" />
              <div className="text-sm font-semibold uppercase tracking-[0.12em]">Daily drivers</div>
            </div>
            <div className="space-y-2">
              {DAILY_DRIVERS.map((driver, idx) => (
                <div key={driver} className="flex items-center gap-3 rounded-full border border-[#284848] bg-black/30 px-3 py-2 text-sm text-text-primary">
                  <span className={cn("h-2.5 w-2.5 rounded-full", idx < 2 ? "bg-emerald-400" : "bg-black ring-1 ring-[#56716f]")} />
                  {driver}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className={cn(cockpitCard, "min-h-0 flex-1")}>
          <CardContent className="flex h-full min-h-0 flex-col gap-3 p-4">
            <div className="flex items-center gap-2 text-emerald-200">
              <ShieldCheck className="h-4 w-4" />
              <div className="text-sm font-semibold uppercase tracking-[0.12em]">Live guardrail</div>
            </div>
            <div className={cockpitPanel}>
              <div className="text-xs uppercase tracking-[0.1em] text-text-secondary">Gateway</div>
              <div className={cn("mt-1 text-base font-semibold", status?.gateway_running ? "text-emerald-300" : "text-red-300")}>
                {status?.gateway_running ? "Running" : "Unknown"}
              </div>
            </div>
            <div className={cockpitPanel}>
              <div className="text-xs uppercase tracking-[0.1em] text-text-secondary">Active runs</div>
              <div className="mt-1 text-base font-semibold text-text-primary">{activeJobs.length}</div>
            </div>
          </CardContent>
        </Card>
      </div>
    </aside>
  );
}

function TodayView({ status, activeJobs, jobs, approvalSummary }: { status: StatusResponse | null; activeJobs: CronJob[]; jobs: CronJob[]; approvalSummary: OpsApprovalSummary | null }) {
  const problemJobs = jobs.filter(isProblemJob);
  const nextProjects = PROJECTS.slice(0, 6);
  const platformCount = status?.gateway_platforms ? Object.keys(status.gateway_platforms).length : 0;

  return (
    <section className="font-readable-ui grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(360px,0.8fr)]">
      <Card className={cockpitCard}>
        <CardContent className="space-y-4 p-5">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="flex items-center gap-2 text-midground">
                <ListChecks className="h-5 w-5" />
                <H2 className="text-xl">Today view</H2>
              </div>
              <Typography className="mt-1 text-sm leading-6 text-text-secondary">
                Operator snapshot: what needs Travis, what is active, what failed, and the safest next moves.
              </Typography>
            </div>
            <Badge tone="outline" className="shrink-0 whitespace-nowrap border-cyan-400/30 text-cyan-200">
              RO phase 1
            </Badge>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-amber-400/30 bg-amber-500/15 p-4">
              <div className="text-sm font-semibold uppercase tracking-[0.03em] text-amber-100">Needs Travis</div>
              <div className="mt-2 text-3xl font-semibold text-text-primary">{approvalSummary?.pending_count ?? DECISION_QUEUE.length}</div>
              <div className="mt-1 text-sm leading-6 text-text-secondary">pending approval records + standing gates</div>
            </div>
            <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/15 p-4">
              <div className="text-sm font-semibold uppercase tracking-[0.03em] text-emerald-100">Active runs</div>
              <div className="mt-2 text-3xl font-semibold text-text-primary">{activeJobs.length}</div>
              <div className="mt-1 text-sm leading-6 text-text-secondary">enabled cron/automation entries</div>
            </div>
            <div className="rounded-xl border border-red-400/30 bg-red-500/15 p-4">
              <div className="text-sm font-semibold uppercase tracking-[0.03em] text-red-100">Problem runs</div>
              <div className="mt-2 text-3xl font-semibold text-text-primary">{problemJobs.length}</div>
              <div className="mt-1 text-sm leading-6 text-text-secondary">jobs with error/failure state</div>
            </div>
          </div>

          <div
            className="grid gap-3"
            style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 22rem), 1fr))" }}
          >
            <div className="space-y-2">
              <div className={readableSectionHeading}>Waiting on approval / do not auto-run</div>
              {approvalSummary?.pending?.length ? (
                approvalSummary.pending.slice(0, 4).map((item) => (
                  <Link key={item.id} to="/approvals" className="block rounded-xl border border-amber-400/25 bg-amber-500/15 p-4 transition hover:border-amber-300/50">
                    <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div className="min-w-0">
                        <div className={readableTitle}>{item.title}</div>
                        <div className={readableBody}>{item.target}</div>
                      </div>
                      <Badge tone="outline" className={cn(readableBadge, "border-amber-400/30 text-amber-200")}>{item.risk_label}</Badge>
                    </div>
                  </Link>
                ))
              ) : DECISION_QUEUE.map((item) => (
                <div key={item.label} className={readablePanel}>
                  <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <div className={readableTitle}>{item.label}</div>
                      <div className={readableBody}>{item.detail}</div>
                    </div>
                    <Badge tone="outline" className={cn(readableBadge, "border-amber-400/30 text-amber-200")}>{item.risk}</Badge>
                  </div>
                </div>
              ))}
            </div>

            <div className="space-y-2">
              <div className={readableSectionHeading}>Next safe project moves</div>
              {nextProjects.map((project) => (
                <div key={project.name} className={readablePanel}>
                  <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <div className={readableTitle}>{project.name}</div>
                      <div className={readableBody}>{project.phase}</div>
                    </div>
                    <Badge tone="outline" className={cn(readableBadge, projectHealthTone(project.health))}>{project.health}</Badge>
                  </div>
                  <div className="mt-3 text-sm leading-6 text-text-secondary">{project.next}</div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className={cockpitCard}>
        <CardContent className="space-y-4 p-5">
          <div className="flex items-center gap-2 text-midground">
            <ShieldCheck className="h-5 w-5" />
            <H2 className="text-xl">Ops health snapshot</H2>
          </div>
          <div className="grid gap-3 text-sm">
            <div className={readablePanel}>
              <div className={readableSectionHeading}>Gateway</div>
              <div className={cn("mt-1 text-base font-semibold leading-6", status?.gateway_running ? "text-emerald-300" : "text-red-300")}>
                {status?.gateway_running ? "Running" : "Unknown / needs check"}
              </div>
            </div>
            <div className={readablePanel}>
              <div className={readableSectionHeading}>Platforms visible</div>
              <div className="mt-1 text-base leading-6 text-text-primary">{platformCount || "—"} · {platformSummary(status)}</div>
            </div>
            <div className={readablePanel}>
              <div className={readableSectionHeading}>Project homes</div>
              <div className="mt-1 text-base leading-6 text-text-primary">{PROJECTS.length} linked to AI Ops Brain source files</div>
            </div>
            <div className={readablePanel}>
              <div className={readableSectionHeading}>Phase 1 boundary</div>
              <div className="mt-1 text-base leading-6 text-text-primary">Approval-first dashboard wiring — no cron changes, no service restart, no public side effects. The only execute route is the disabled read-only status probe.</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
}

function AutomationPosturePanel({ jobs }: { jobs: CronJob[] }) {
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const active = jobs.filter(isActiveJob);
  const needsReview = active.filter(needsAutomationReview);
  const approvedStanding = active.filter((job) => isApprovedStandingGuard(job) && !needsAutomationReview(job));
  const projectAutomations = active.filter((job) => isProjectAutomation(job) && !needsAutomationReview(job) && !isApprovedStandingGuard(job));
  const paused = jobs.filter((job) => !isActiveJob(job));

  const groups = [
    {
      key: "approved",
      title: "Approved standing guards",
      description: "Protective or backup-style jobs that look intentionally durable. Still visible, not auto-mutated.",
      tone: "border-emerald-400/30 bg-emerald-500/10 text-emerald-100",
      jobs: approvedStanding,
    },
    {
      key: "project",
      title: "Project watchdogs / business automations",
      description: "Project-specific jobs that may be valid only while their project gate is active.",
      tone: "border-cyan-400/30 bg-cyan-500/10 text-cyan-100",
      jobs: projectAutomations,
    },
    {
      key: "review",
      title: "Needs review",
      description: "High-frequency, origin-delivering, error-prone, or unclear active jobs. Dashboard can draft a proposal, not pause them.",
      tone: "border-amber-400/35 bg-amber-500/12 text-amber-100",
      jobs: needsReview,
    },
    {
      key: "paused",
      title: "Paused / retired",
      description: "Not currently active; shown so old jobs do not blend into running automation.",
      tone: "border-slate-400/25 bg-slate-500/10 text-slate-100",
      jobs: paused,
    },
  ];

  const copyPauseProposal = async () => {
    try {
      await navigator.clipboard.writeText(buildPauseProposalCommand(needsReview));
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
  };

  return (
    <Card className={cockpitCard}>
      <CardContent className="space-y-5 p-5">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-midground">
              <Activity className="h-5 w-5" />
              <H2 className="text-xl">Automation posture</H2>
            </div>
            <Typography className="mt-1 max-w-3xl text-sm leading-6 text-text-secondary">
              Read-only grouping for enabled cron/watchdog jobs. This answers what is running and why before any pause/remove/trigger action is proposed.
            </Typography>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge tone="outline" className="border-emerald-400/30 text-emerald-200">{active.length} active</Badge>
            <Badge tone="outline" className="border-amber-400/35 text-amber-200">{needsReview.length} review</Badge>
            <Link to="/ops-runs" className="inline-flex">
              <Button ghost size="sm" className="gap-2">Run ledger <ArrowRight className="h-4 w-4" /></Button>
            </Link>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-4">
          {groups.map((group) => (
            <div key={group.key} className={cn("rounded-xl border p-4", group.tone)}>
              <div className="text-sm font-semibold uppercase tracking-[0.08em]">{group.title}</div>
              <div className="mt-2 text-3xl font-semibold text-text-primary">{group.jobs.length}</div>
              <div className="mt-2 text-sm leading-6 text-text-secondary">{group.description}</div>
            </div>
          ))}
        </div>

        <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          {groups.map((group) => (
            <div key={`${group.key}-list`} className="min-w-0 space-y-2">
              <div className={readableSectionHeading}>{group.title}</div>
              <div className="space-y-2">
                {group.jobs.length ? group.jobs.slice(0, 5).map((job) => (
                  <div key={`${group.key}:${job.profile || "default"}:${job.id}`} className={readablePanel}>
                    <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div className="min-w-0">
                        <div className={readableTitle}>{getJobTitle(job)}</div>
                        <div className={readableBody}>{describeAutomationReason(job)}</div>
                        <div className="mt-2 flex flex-wrap gap-2 text-xs leading-5 text-text-secondary">
                          <span className="rounded-full border border-[#284848] bg-black/25 px-2 py-0.5">{job.profile || "default"}</span>
                          <span className="rounded-full border border-[#284848] bg-black/25 px-2 py-0.5">{getSchedule(job)}</span>
                          <span className="rounded-full border border-[#284848] bg-black/25 px-2 py-0.5">next {formatTime(job.next_run_at)}</span>
                        </div>
                      </div>
                      <Badge tone="outline" className={cn(readableBadge, isActiveJob(job) ? "border-emerald-400/30 text-emerald-200" : "border-slate-400/30 text-slate-200")}>
                        {getJobState(job)}
                      </Badge>
                    </div>
                  </div>
                )) : (
                  <div className="rounded-xl border border-[#284848] bg-black/30 p-4 text-sm leading-6 text-text-secondary">No jobs in this group.</div>
                )}
                {group.jobs.length > 5 && (
                  <div className="rounded-xl border border-[#284848] bg-black/30 p-3 text-sm text-text-secondary">
                    +{group.jobs.length - 5} more visible in Ops Runs.
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-amber-400/25 bg-amber-500/10 p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="text-base font-semibold text-text-primary">Draft pause-review proposal</div>
              <div className="mt-1 text-sm leading-6 text-text-secondary">
                Copies a chat command for Jenny review. It does not pause, remove, trigger, edit cron, or restart anything.
              </div>
            </div>
            <Button onClick={copyPauseProposal} disabled={!needsReview.length} className="gap-2">
              <Clipboard className="h-4 w-4" />
              {copyState === "copied" ? "Copied proposal" : copyState === "failed" ? "Copy failed" : "Copy pause proposal"}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ManualSocialSnapshotForm({ platforms, onSaved }: { platforms: OpsSocialPlatformStatusItem[]; onSaved: (status: OpsSocialPlatformStatus) => void }) {
  const [rows, setRows] = useState<OpsSocialPlatformStatusItem[]>(() => platforms.map((item) => ({ ...item })));
  const [source, setSource] = useState("manual-dashboard-snapshot");
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "failed">("idle");
  const [message, setMessage] = useState<string | null>(null);

  const updateRow = (idx: number, key: keyof OpsSocialPlatformStatusItem, value: string) => {
    setRows((current) => current.map((item, itemIdx) => (itemIdx === idx ? { ...item, [key]: value } : item)));
    setSaveState("idle");
    setMessage(null);
  };

  const saveSnapshot = async () => {
    setSaveState("saving");
    setMessage(null);
    const payload: OpsSocialPlatformStatusUpdate = {
      source: source.trim() || "manual-dashboard-snapshot",
      platforms: rows.map((row) => ({
        platform: row.platform,
        published: row.published,
        scheduled: row.scheduled,
        issues_private: row.issues_private,
        readiness: row.readiness,
        source: row.source || "Manual dashboard entry",
        status: row.status || "needs_sync",
        last_checked_at: new Date().toISOString(),
      })),
    };
    try {
      const saved = await api.updateOpsSocialPlatformStatus(payload);
      onSaved(saved);
      setSaveState("saved");
      setMessage("Saved local snapshot. No platform APIs, tokens, cron, upload, schedule, delete, or privacy action ran.");
    } catch (error) {
      setSaveState("failed");
      setMessage(error instanceof Error ? error.message : "Save failed");
    }
  };

  return (
    <div className="space-y-4">
      <div className={cockpitPanel}>
        <div className="font-semibold text-text-primary">Manual local snapshot writer</div>
        <div className="mt-1 text-sm leading-6 text-text-secondary">
          Writes only `$HERMES_HOME/state/ops-center/social-platform-status.json`. It does not call YouTube, Meta, TikTok, or any platform API.
        </div>
      </div>
      <label className="block space-y-2">
        <span className="text-sm font-medium text-text-secondary">Snapshot source</span>
        <input
          value={source}
          onChange={(event) => setSource(event.target.value)}
          className="w-full rounded-xl border border-[#284848] bg-black/45 p-3 text-sm text-text-primary outline-none focus:border-emerald-400/60"
        />
      </label>
      <div className="space-y-3">
        {rows.map((row, idx) => (
          <div key={row.platform} className="rounded-xl border border-[#284848] bg-black/25 p-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-base font-semibold text-text-primary">{row.platform}</div>
              <Badge tone="outline" className="border-cyan-400/30 text-cyan-200">
                {normalizeSocialStatusLabel(row.status)}
              </Badge>
            </div>
            <div className="mt-2 text-xs leading-5 text-text-secondary">
              Last checked: {getSocialStaleness(row).label}
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <label className="space-y-1 text-sm text-text-secondary">
                <span>Published</span>
                <input value={row.published} onChange={(event) => updateRow(idx, "published", event.target.value)} className="w-full rounded-lg border border-[#284848] bg-black/45 p-2 text-text-primary outline-none focus:border-emerald-400/60" />
              </label>
              <label className="space-y-1 text-sm text-text-secondary">
                <span>Scheduled</span>
                <input value={row.scheduled} onChange={(event) => updateRow(idx, "scheduled", event.target.value)} className="w-full rounded-lg border border-[#284848] bg-black/45 p-2 text-text-primary outline-none focus:border-emerald-400/60" />
              </label>
              <label className="space-y-1 text-sm text-text-secondary sm:col-span-2">
                <span>Status</span>
                <select value={row.status || "needs_sync"} onChange={(event) => updateRow(idx, "status", event.target.value)} className="w-full rounded-lg border border-[#284848] bg-black/45 p-2 text-text-primary outline-none focus:border-emerald-400/60">
                  {SOCIAL_STATUS_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value} className="bg-[#061616] text-text-primary">
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1 text-sm text-text-secondary sm:col-span-2">
                <span>Issues / private</span>
                <input value={row.issues_private} onChange={(event) => updateRow(idx, "issues_private", event.target.value)} className="w-full rounded-lg border border-[#284848] bg-black/45 p-2 text-text-primary outline-none focus:border-emerald-400/60" />
              </label>
              <label className="space-y-1 text-sm text-text-secondary sm:col-span-2">
                <span>Readiness note</span>
                <textarea value={row.readiness} onChange={(event) => updateRow(idx, "readiness", event.target.value)} className="min-h-20 w-full rounded-lg border border-[#284848] bg-black/45 p-2 text-text-primary outline-none focus:border-emerald-400/60" />
              </label>
            </div>
          </div>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={saveSnapshot} disabled={saveState === "saving"} className="gap-2">
          <Database className="h-4 w-4" />
          {saveState === "saving" ? "Saving" : saveState === "saved" ? "Saved local snapshot" : "Save local snapshot"}
        </Button>
        {message && (
          <div className={cn("text-sm leading-6", saveState === "failed" ? "text-red-200" : "text-emerald-200")}>{message}</div>
        )}
      </div>
    </div>
  );
}

function SocialSnapshotHistoryPanel({ history }: { history: OpsSocialPlatformHistory | null }) {
  const events = history?.events || [];
  return (
    <div className={cockpitPanel}>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="font-semibold text-text-primary">Manual snapshot history</div>
        <Badge tone="outline" className="border-cyan-400/30 text-cyan-200">local JSONL audit</Badge>
      </div>
      <div className="mt-1 text-sm leading-6 text-text-secondary">
        Shows recent manual saves from the local history file. It is not a platform sync, scheduler, or posting log.
      </div>
      {history?.warning && <div className="mt-3 rounded-lg border border-amber-400/30 bg-amber-500/10 p-2 text-sm text-amber-100">{history.warning}</div>}
      <div className="mt-3 space-y-2">
        {events.length ? events.slice(0, 5).map((event, idx) => (
          <div key={`${event.timestamp || "event"}-${idx}`} className="rounded-lg border border-white/10 bg-black/25 p-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <div className="text-sm font-semibold text-text-primary">{formatHistoryTime(event.timestamp)}</div>
                <div className="mt-1 text-xs leading-5 text-text-secondary">{event.source || "manual-dashboard-snapshot"}</div>
              </div>
              <Badge tone="outline" className="border-white/20 text-text-secondary">{event.platform_count || 0} rows</Badge>
            </div>
            <div className="mt-2 text-sm leading-5 text-text-secondary">{describeHistoryEvent(event)}</div>
          </div>
        )) : (
          <div className="rounded-lg border border-white/10 bg-black/25 p-3 text-sm leading-6 text-text-secondary">
            No manual snapshot history yet. The next local save will append an audit record.
          </div>
        )}
      </div>
    </div>
  );
}

function GoalLauncher() {
  const [selected, setSelected] = useState(GOAL_TEMPLATES[0]);
  const [customGoal, setCustomGoal] = useState(GOAL_TEMPLATES[0].prompt);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");

  const command = useMemo(() => `/goal ${customGoal.trim()}`, [customGoal]);

  const selectTemplate = (idx: number) => {
    const item = GOAL_TEMPLATES[idx];
    setSelected(item);
    setCustomGoal(item.prompt);
    setCopyState("idle");
  };

  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
  };

  return (
    <Card className={cockpitCard}>
      <CardContent className="space-y-4 p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="flex items-center gap-2 text-midground">
              <Rocket className="h-5 w-5" />
              <H2 className="text-xl">Goal launcher</H2>
            </div>
            <Typography className="mt-1 text-sm text-text-secondary">
              Prepares a safe Hermes <code>/goal</code> command. It copies the command and opens Chat; it does not silently start unattended work.
            </Typography>
          </div>
          <Badge tone="outline" className="border-amber-400/40 text-amber-200">
            gated by chat send
          </Badge>
        </div>

        <div className="grid gap-2 md:grid-cols-2">
          {GOAL_TEMPLATES.map((item, idx) => (
            <button
              key={item.name}
              type="button"
              onClick={() => selectTemplate(idx)}
              className={cn(
                "rounded-xl border p-3 text-left transition",
                selected.name === item.name
                  ? "border-midground/60 bg-midground/10"
                  : "border-[#284848] bg-black/30 hover:border-emerald-400/40",
              )}
            >
              <div className="text-sm font-semibold text-text-primary">{item.name}</div>
              <div className="mt-1 text-xs text-text-secondary">{item.project}</div>
            </button>
          ))}
        </div>

        <label className="block space-y-2">
          <span className="text-sm font-medium text-text-secondary">Goal prompt</span>
          <textarea
            value={customGoal}
            onChange={(event) => setCustomGoal(event.target.value)}
            className="min-h-32 w-full rounded-xl border border-[#284848] bg-black/45 p-3 text-sm text-text-primary outline-none focus:border-emerald-400/60"
          />
        </label>

        <div className={cockpitPanel}>
          <div className="mb-1 text-xs uppercase tracking-wide text-text-secondary">Command to send in Chat</div>
          <code className="break-words text-sm text-midground">{command}</code>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button onClick={copyCommand} className="gap-2">
            <Clipboard className="h-4 w-4" />
            {copyState === "copied" ? "Copied" : copyState === "failed" ? "Copy failed" : "Copy /goal command"}
          </Button>
          <Link to="/chat" className="inline-flex">
            <Button ghost className="gap-2">
              <Terminal className="h-4 w-4" />
              Open Chat
            </Button>
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}

export default function MissionControlPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [jobs, setJobs] = useState<CronJob[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [approvalSummary, setApprovalSummary] = useState<OpsApprovalSummary | null>(null);
  const [socialStatus, setSocialStatus] = useState<OpsSocialPlatformStatus | null>(null);
  const [socialHistory, setSocialHistory] = useState<OpsSocialPlatformHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { setEnd } = usePageHeader();

  const load = useCallback(() => {
    setError(null);
    Promise.allSettled([
      api.getStatus(),
      api.getCronJobs("all"),
      api.getSessions(8),
      api.getOpsApprovalSummary(),
      api.getOpsSocialPlatformStatus(),
      api.getOpsSocialPlatformStatusHistory(8),
    ]).then(([statusResult, jobsResult, sessionsResult, approvalSummaryResult, socialStatusResult, socialHistoryResult]) => {
      if (statusResult.status === "fulfilled") setStatus(statusResult.value);
      if (jobsResult.status === "fulfilled") setJobs(jobsResult.value);
      if (sessionsResult.status === "fulfilled") setSessions(sessionsResult.value.sessions);
      if (approvalSummaryResult.status === "fulfilled") setApprovalSummary(approvalSummaryResult.value);
      if (socialStatusResult.status === "fulfilled") setSocialStatus(socialStatusResult.value);
      if (socialHistoryResult.status === "fulfilled") setSocialHistory(socialHistoryResult.value);
      const failures = [statusResult, jobsResult, sessionsResult, approvalSummaryResult, socialStatusResult, socialHistoryResult].filter((r) => r.status === "rejected");
      setError(failures.length ? "Some live status panels could not refresh." : null);
    });
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(load, 0);
    const timer = window.setInterval(load, 30_000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [load]);

  useEffect(() => {
    setEnd(
      <Button ghost onClick={load} className="gap-2">
        <Activity className="h-4 w-4" />
        Refresh
      </Button>,
    );
    return () => setEnd(null);
  }, [load, setEnd]);

  const activeJobs = useMemo(
    () => jobs.filter(isActiveJob),
    [jobs],
  );
  const recentSessions = useMemo(() => sessions.slice(0, 5), [sessions]);
  const socialPlatforms = socialStatus?.platforms?.length ? socialStatus.platforms : SOCIAL_PLATFORM_STATUS;
  const handleSocialSnapshotSaved = useCallback((saved: OpsSocialPlatformStatus) => {
    setSocialStatus(saved);
    api.getOpsSocialPlatformStatusHistory(8).then(setSocialHistory).catch(() => undefined);
  }, []);

  return (
    <main className="h-full overflow-auto bg-[radial-gradient(circle_at_50%_-10%,rgba(16,185,129,0.18),transparent_34%),linear-gradient(180deg,#031111_0%,#061616_55%,#030808_100%)] px-4 py-5 lg:px-6">
      <div className="mx-auto grid max-w-[2200px] gap-5 xl:grid-cols-[220px_minmax(0,1fr)] 2xl:grid-cols-[220px_minmax(0,1fr)_300px]">
        <WorkspaceRail activeJobs={activeJobs} recentSessions={recentSessions} />

        <div className="flex min-w-0 flex-col gap-5">
        <section className="font-readable-ui overflow-hidden rounded-3xl border border-emerald-400/20 bg-[radial-gradient(circle_at_top_right,rgba(47,214,161,0.16),transparent_30%),linear-gradient(135deg,rgba(6,31,31,0.98),rgba(3,12,12,0.96))] p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.04),0_24px_80px_rgba(0,0,0,0.38)] lg:p-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="max-w-3xl">
              <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-xs uppercase tracking-[0.2em] text-emerald-200 shadow-[0_0_18px_rgba(47,214,161,0.12)]">
                <Sparkles className="h-3.5 w-3.5" /> Jenny Command Center
              </div>
              <H2 className="text-3xl font-semibold leading-none tracking-[-0.035em] text-emerald-300 lg:text-5xl">
                Jenny Ops Center
              </H2>
              <Typography className="mt-2 max-w-2xl text-sm leading-6 text-text-secondary lg:text-base">
                Operator console for approvals, live status, project routing, traces, and durable AI Ops Brain records. No hidden execution.
              </Typography>
            </div>
            <div className="grid w-full gap-2 text-sm sm:min-w-72 lg:w-auto">
              <div className="flex items-center justify-between rounded-xl border border-[#284848] bg-black/30 px-3 py-2">
                <span className="text-text-secondary">Gateway</span>
                <span className={cn("font-semibold", status?.gateway_running ? "text-emerald-300" : "text-red-300")}>
                  {status?.gateway_running ? "Running" : "Unknown"}
                </span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-[#284848] bg-black/30 px-3 py-2">
                <span className="text-text-secondary">Active sessions</span>
                <span className="font-semibold text-text-primary">{status?.active_sessions ?? "—"}</span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-[#284848] bg-black/30 px-3 py-2">
                <span className="text-text-secondary">Cron jobs enabled</span>
                <span className="font-semibold text-text-primary">{activeJobs.length}</span>
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
            {OPS_BOUNDARY_STEPS.map((step, idx) => (
              <div key={step.label} className="rounded-xl border border-[#284848] bg-black/25 p-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-text-primary">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full border border-emerald-400/30 bg-emerald-500/10 text-xs text-emerald-200">
                    {idx + 1}
                  </span>
                  {step.label}
                </div>
                <div className="mt-1 text-xs leading-5 text-text-secondary">{step.detail}</div>
              </div>
            ))}
          </div>
        </section>

        {error && (
          <div className="flex items-center gap-2 rounded-xl border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
            <AlertTriangle className="h-4 w-4" /> {error}
          </div>
        )}

        <TodayView status={status} activeJobs={activeJobs} jobs={jobs} approvalSummary={approvalSummary} />
        <AutomationPosturePanel jobs={jobs} />
        </div>

        <CommandDeck status={status} activeJobs={activeJobs} approvalSummary={approvalSummary} />
      </div>

      <div className="font-readable-ui mx-auto mt-5 flex max-w-[1900px] min-w-0 flex-col gap-5 2xl:max-w-[2050px]">
        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {DASHBOARD_METRICS.map((metric) => (
            <Card key={metric.label} className={cockpitCard}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs uppercase tracking-wide text-text-secondary">{metric.label}</div>
                  <Gauge className="h-4 w-4 text-midground" />
                </div>
                <div className="mt-3 text-3xl font-semibold text-text-primary">{metric.value}</div>
                <div className="mt-1 text-xs text-text-secondary">{metric.detail}</div>
              </CardContent>
            </Card>
          ))}
        </section>

        <section className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <Card className={cockpitCard}>
            <CardContent className="space-y-4 p-5">
              <div className="flex items-center gap-2 text-midground">
                <Target className="h-5 w-5" />
                <H2 className="text-xl">Operating lanes</H2>
              </div>
              <div
                className="grid gap-3"
                style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 12rem), 1fr))" }}
              >
                {OPERATING_LANES.map((lane, idx) => (
                  <div key={lane.label} className={cockpitPanel}>
                    <div className="mb-2 flex items-start gap-2 text-sm font-semibold leading-5 text-text-primary">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full border border-midground/40 bg-midground/10 text-xs text-midground">
                        {idx + 1}
                      </span>
                      {lane.label}
                    </div>
                    <div className="text-xs leading-5 text-text-secondary">{lane.detail}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className={cockpitCard}>
            <CardContent className="space-y-4 p-5">
              <div className="flex items-center gap-2 text-midground">
                <FileText className="h-5 w-5" />
                <H2 className="text-xl">Vault control files</H2>
              </div>
              <div className="space-y-2">
                {VAULT_LINKS.map((path) => (
                  <div key={path} className="rounded-xl border border-[#284848] bg-black/30 p-3 text-xs text-text-secondary">
                    <div className="break-all">{path}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <GoalLauncher />

          <Card className={cockpitCard}>
            <CardContent className="space-y-4 p-5">
              <div className="flex items-center gap-2 text-midground">
                <ShieldCheck className="h-5 w-5" />
                <H2 className="text-xl">Live status</H2>
              </div>
              <div className="grid gap-3 text-sm">
                <div className={cockpitPanel}>
                  <div className="text-xs uppercase tracking-wide text-text-secondary">Hermes</div>
                  <div className="mt-1 text-text-primary">v{status?.version ?? "—"} · {status?.release_date ?? "—"}</div>
                </div>
                <div className={cockpitPanel}>
                  <div className="text-xs uppercase tracking-wide text-text-secondary">Platforms</div>
                  <div className="mt-1 text-text-primary">{platformSummary(status)}</div>
                </div>
                <div className={cockpitPanel}>
                  <div className="text-xs uppercase tracking-wide text-text-secondary">AI Ops Brain</div>
                  <div className="mt-1 break-all text-text-primary">/home/jenny/ai-ops-brain/PROJECT_COMMAND_CENTER.md</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <Card className={cockpitCard}>
            <CardContent className="space-y-4 p-5">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                <div className="flex items-center gap-2 text-midground">
                  <Globe2 className="h-5 w-5" />
                  <H2 className="text-xl">Social platform status</H2>
                </div>
                <Badge tone="outline" className="border-amber-400/40 text-amber-200">
                  read-only / counts need sync
                </Badge>
              </div>
              <Typography className="text-sm leading-6 text-text-secondary">
                Counts come from a local read-only dashboard status snapshot. Until a sync writes that snapshot, unknown counts stay labeled instead of guessed.
              </Typography>
              {socialStatus?.warning && (
                <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-3 text-sm leading-6 text-amber-100">
                  {socialStatus.warning}
                </div>
              )}
              <div
                className="grid gap-3"
                style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 14rem), 1fr))" }}
              >
                {socialPlatforms.map((item) => {
                  const checked = getSocialStaleness(item);
                  return (
                    <div key={item.platform} className={cn("rounded-xl border p-4", socialPlatformTone(item))}>
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div className="text-base font-semibold text-text-primary">{item.platform}</div>
                        <Badge tone="outline" className={cn("border-white/20 text-text-primary", checked.stale && "border-amber-400/40 text-amber-200")}>
                          {normalizeSocialStatusLabel(item.status)}
                        </Badge>
                      </div>
                      <div className="mt-2 flex items-start gap-2 rounded-lg border border-white/10 bg-black/20 p-2 text-xs leading-5 text-text-secondary">
                        {checked.stale ? <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300" /> : <Clock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-300" />}
                        <span className={checked.stale ? "text-amber-100" : "text-text-secondary"}>{checked.label}</span>
                      </div>
                      <div className="mt-3 grid gap-2 text-sm">
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-text-secondary">Published</span>
                          <span className="font-semibold text-text-primary">{item.published}</span>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-text-secondary">Scheduled</span>
                          <span className="font-semibold text-text-primary">{item.scheduled}</span>
                        </div>
                        <div className="border-t border-white/10 pt-2">
                          <div className="text-xs uppercase tracking-[0.08em] text-text-secondary">Issues / private</div>
                          <div className="mt-1 text-sm leading-5 text-text-primary">{item.issues_private}</div>
                        </div>
                      </div>
                      <div className="mt-3 text-xs leading-5 text-text-secondary">{item.readiness}</div>
                      <div className="mt-2 text-[0.7rem] leading-4 text-text-secondary/80">Source: {item.source}</div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          <Card className={cockpitCard}>
            <CardContent className="space-y-4 p-5">
              <div className="flex items-center gap-2 text-midground">
                <FileText className="h-5 w-5" />
                <H2 className="text-xl">Manual status snapshot</H2>
              </div>
              <ManualSocialSnapshotForm key={socialStatus?.updated_at || "default-social-status"} platforms={socialPlatforms} onSaved={handleSocialSnapshotSaved} />
              <SocialSnapshotHistoryPanel history={socialHistory} />
              <div className={cockpitPanel}>
                <div className="font-semibold text-text-primary">Gate line</div>
                <div className="mt-1 text-sm leading-6 text-text-secondary">
                  Live API counting for YouTube/Meta/TikTok may use existing tokens, but new token work, posting, scheduling, deletion, privacy changes, or recurring sync jobs need explicit approval.
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="font-readable-ui">
          <div className="mb-4 flex items-center gap-2 text-midground">
            <FolderKanban className="h-6 w-6" />
            <H2 className="text-2xl">Project buttons</H2>
          </div>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {PROJECTS.map((project) => (
              <Card key={project.name} className={cn(cockpitCard, "overflow-hidden")}>
                <CardContent className={cn("space-y-5 bg-gradient-to-br p-5", project.tone)}>
                  <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <div className="text-2xl font-semibold leading-8 text-text-primary">{project.name}</div>
                      <div className="mt-1 text-base leading-6 text-text-secondary">{project.short}</div>
                    </div>
                    <Badge tone="outline" className={cn(readableBadge, "border-white/20 text-text-secondary")}>
                      {project.profile}
                    </Badge>
                  </div>

                  <div className="grid gap-3 text-base leading-7 text-text-secondary">
                    <div className={cockpitPanel}>
                      <div className="mb-2 flex min-w-0 flex-col gap-2 text-base font-semibold leading-6 text-text-primary sm:flex-row sm:items-start sm:justify-between">
                        <span className="flex min-w-0 items-center gap-1"><CheckCircle2 className="h-3.5 w-3.5 shrink-0" /> Project health</span>
                        <Badge tone="outline" className={cn(readableBadge, projectHealthTone(project.health))}>{project.health}</Badge>
                      </div>
                      <div>{project.phase}</div>
                    </div>
                    <div className={cockpitPanel}>
                      <div className="mb-2 text-base font-semibold leading-6 text-text-primary">Next safe posture</div>
                      {project.next}
                    </div>
                    <div className={cockpitPanel}>
                      <div className="mb-2 text-base font-semibold leading-6 text-text-primary">Waiting / gate</div>
                      {project.waitingOn}
                    </div>
                  </div>

                  <div className="space-y-2 text-sm leading-6">
                    <div className="flex items-start gap-2 rounded-lg bg-black/20 p-2">
                      <LinkIcon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-text-secondary" />
                      <span className="break-all text-text-secondary">{project.statusPath}</span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <a href={discordThreadUrl(project.thread)} target="_blank" rel="noreferrer" className="inline-flex">
                      <Button size="sm" className="gap-2">
                        <MessageSquare className="h-4 w-4" /> Discord
                      </Button>
                    </a>
                    <Link to={`/sessions?search=${encodeURIComponent(project.name)}`} className="inline-flex">
                      <Button size="sm" ghost className="gap-2">
                        <Clock className="h-4 w-4" /> Sessions
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Card className={cockpitCard}>
            <CardContent className="space-y-4 p-5">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-midground">
                  <Play className="h-5 w-5" />
                  <H2 className="text-xl">Active runs</H2>
                </div>
                <Link to="/ops-runs" className="inline-flex">
                  <Button ghost size="sm" className="gap-2">Run ledger <ArrowRight className="h-4 w-4" /></Button>
                </Link>
              </div>
              <div className="space-y-2">
                {activeJobs.length ? activeJobs.slice(0, 6).map((job) => (
                  <div key={`${job.profile ?? "default"}:${job.id}`} className={cockpitPanel}>
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="font-medium text-text-primary">{getJobTitle(job)}</div>
                        <div className="mt-1 text-xs text-text-secondary">{getSchedule(job)} · next {formatTime(job.next_run_at)}</div>
                      </div>
                      <Badge tone="outline" className="border-emerald-400/30 text-emerald-200">
                        {getJobState(job)}
                      </Badge>
                    </div>
                  </div>
                )) : (
                  <div className="rounded-xl border border-[#284848] bg-black/30 p-4 text-sm text-text-secondary">
                    No enabled cron runs reported. That matches the “cron off unless requested” operating preference.
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className={cockpitCard}>
            <CardContent className="space-y-4 p-5">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-midground">
                  <Bot className="h-5 w-5" />
                  <H2 className="text-xl">Recent sessions</H2>
                </div>
                <Link to="/sessions" className="inline-flex">
                  <Button ghost size="sm" className="gap-2">Sessions <ArrowRight className="h-4 w-4" /></Button>
                </Link>
              </div>
              <div className="space-y-2">
                {recentSessions.length ? recentSessions.map((session) => (
                  <Link
                    key={session.id}
                    to={`/sessions?session=${encodeURIComponent(session.id)}`}
                    className="block rounded-xl border border-[#284848] bg-black/30 p-3 transition hover:border-emerald-400/40"
                  >
                    <div className="font-medium text-text-primary">{session.title || session.preview || session.id}</div>
                    <div className="mt-1 text-xs text-text-secondary">
                      {session.source || "session"} · {formatTime(session.last_active)}
                    </div>
                  </Link>
                )) : (
                  <div className="rounded-xl border border-[#284848] bg-black/30 p-4 text-sm text-text-secondary">
                    No recent sessions loaded.
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="rounded-2xl border border-[#284848] bg-black/30 p-4 text-sm text-text-secondary">
          <div className="mb-2 flex items-center gap-2 font-semibold text-text-primary">
            <ExternalLink className="h-4 w-4" /> Useful links
          </div>
          <div className="grid gap-2 md:grid-cols-3">
            <a className="hover:text-midground" href="https://jenny-hostinger-vps.taila00f3c.ts.net:9121" target="_blank" rel="noreferrer">Hermes dashboard</a>
            <a className="hover:text-midground" href="https://jenny-hostinger-vps.taila00f3c.ts.net:8080" target="_blank" rel="noreferrer">Open WebUI</a>
            <a className="hover:text-midground" href={discordThreadUrl("1507671041118437416")} target="_blank" rel="noreferrer">Hermes Ops Discord</a>
          </div>
        </section>
      </div>
    </main>
  );
}
