import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Loader2,
  MessageCircle,
  Radar,
  Satellite,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  api,
  type CockpitAgentActivityItem,
  type CockpitAgentActivityResponse,
  type CockpitLane,
  type CockpitAutomationHealthCard,
  type CockpitAutomationHealthResponse,
  type CockpitDailyOpsRadarResponse,
  type CockpitLaneMessagesResponse,
  type CockpitMessage,
  type CockpitN8nCheck,
  type CockpitN8nChecksResponse,
  type CockpitResolvedLane,
  type CockpitSendIntentResponse,
  type CockpitSelfWorkHandoffResponse,
  type CockpitSignal,
  type CockpitSignalsResponse,
  type CockpitTranscriptWindow,
  type StatusResponse,
} from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { PluginSlot } from "@/plugins";
import ChatPage from "@/pages/ChatPage";

const LANE_LIMIT = 60;
const LANE_MESSAGE_LIMIT = 16;
const REFRESH_MS = 7000;
const MAX_MESSAGE_CHARS = 520;

type CockpitSectionKey = "overview" | "local-chat" | "automation-health" | "tools";

const N8N_CHECK_NAMES = [
  "Morning Briefing",
  "Workflow Health Daily Report",
  "Auto-Remediation Monitor",
  "Immich Nightly Sync Monitor",
  "Obsidian Inbox Processor",
  "Alexa Bring Sync",
  "n8n Nightly Workflow Backup",
] as const;

const COCKPIT_SECTIONS: Array<{
  key: CockpitSectionKey;
  label: string;
  shortLabel: string;
  description: string;
}> = [
  { key: "overview", label: "Command Brief", shortLabel: "Brief", description: "Attention, changes, active work, health, and safe gated action" },
  { key: "local-chat", label: "Local Chat", shortLabel: "Chat", description: "Primary PTY-backed local Hermes command surface" },
  { key: "automation-health", label: "Automation Health", shortLabel: "Health", description: "Plain-language automation health from local read-only sources" },
  { key: "tools", label: "Actions", shortLabel: "Actions", description: "Confirmed actions, upgrade review, archive, and diagnostics" },
];

const sectionIcon: Record<CockpitSectionKey, typeof Activity> = {
  overview: Activity,
  "local-chat": MessageCircle,
  "automation-health": CheckCircle2,
  tools: ShieldCheck,
};

function formatClock(value: unknown): string {
  if (value == null || value === "") return "—";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function formatClockLabel(value: unknown, fallback = "Not available"): string {
  const label = formatClock(value);
  return label === "—" ? fallback : label;
}

function formatFreshness(value: Date | null): string {
  if (!value) return "Last refresh pending";
  return `Last refreshed ${formatClock(value.toISOString())}`;
}

function laneName(lane: CockpitLane): string {
  return lane.alias?.alias || lane.title || lane.platform || "Cockpit lane";
}

function laneStatus(lane: CockpitLane): string {
  return lane.status || "observing";
}

function roleLabel(value: unknown): string {
  const role = String(value || "").toLowerCase();
  const labels: Record<string, string> = { biff: "Biff", forge: "Forge", vex: "Vex", quill: "Quill", ranger: "Ranger" };
  return labels[role] || safeText(value) || "Agent";
}

function safeText(value: unknown): string {
  return String(value ?? "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "[media omitted]")
    .replace(/\[[^\]]*\]\((?:file|data|blob):[^)]*\)/gi, "[link omitted]")
    .replace(/\b(?:sk|pk|ghp|gho|xox[abprs])-[-_A-Za-z0-9]{12,}\b/g, "[redacted key]")
    .replace(/\b(api[_-]?key|token|secret|password|authorization)\b\s*[:=]\s*\S+/gi, "$1: [redacted]")
    .replace(/\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/gi, "[redacted id]")
    .replace(/\b(?:session|lane|thread|chat|user|message|msg)[_-]?id\b\s*[:=]\s*[-_:.A-Za-z0-9]{6,}/gi, "$1: [redacted]")
    .replace(/\b[0-9a-f]{24,}\b/gi, "[redacted id]")
    .replace(/\s+/g, " ")
    .trim();
}

function boundedCopy(value: unknown, maxChars = MAX_MESSAGE_CHARS): string {
  const text = safeText(value);
  if (!text) return "No display-safe text in this recent record.";
  return text.length > maxChars ? `${text.slice(0, maxChars).trimEnd()}…` : text;
}

function messageRoleLabel(message: CockpitMessage): string {
  const role = String(message.role || "message").toLowerCase();
  if (role === "assistant") return "Biff";
  if (role === "user") return "Marco";
  if (role === "tool" || role === "system") return "Hidden system detail";
  return safeText(role) || "Message";
}

function messageBody(message: CockpitMessage): string {
  const role = String(message.role || "").toLowerCase();
  if (role === "tool" || role === "system") {
    return "Internal execution details are hidden in Cockpit observer mode.";
  }
  return boundedCopy(message.content);
}

function messageTimestamp(message: CockpitMessage): string {
  return formatClock(message.created_at ?? message.timestamp);
}

function statusTone(status: string): "success" | "warning" | "secondary" {
  const normalized = status.toLowerCase();
  if (["active", "running", "connected", "ready", "done", "success"].some((token) => normalized.includes(token))) {
    return "success";
  }
  if (["error", "failed", "stale", "offline"].some((token) => normalized.includes(token))) {
    return "warning";
  }
  return "secondary";
}

function categorySignals(signals: CockpitSignalsResponse | null, key: string): CockpitSignal[] {
  return signals?.categories?.[key]?.signals ?? [];
}

function signalFreshness(signal: CockpitSignal): string {
  return signal.freshness?.bucket || signal.source?.freshness || "operational";
}

function uniqueSignals(signals: CockpitSignal[], limit: number): CockpitSignal[] {
  const seen = new Set<string>();
  const deduped: CockpitSignal[] = [];
  for (const signal of signals) {
    const key = signal.lane_id || signal.id || `${signal.title}-${signal.recency_label}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(signal);
    if (deduped.length >= limit) break;
  }
  return deduped;
}

function signalBriefLine(signal: CockpitSignal): string {
  const reason = safeText(signal.reason || signal.recency_label || signalFreshness(signal));
  if (!reason) return "Recent work item from the bounded command window.";
  return reason;
}

function BriefSignalSummary({ signal, onSelect }: { signal: CockpitSignal; onSelect?: () => void }) {
  const urgent = signal.category === "stuck_failed" || signal.category === "needs_marco";
  const body = (
    <>
      <div className="flex min-w-0 items-start gap-3">
        <span className={cn("mt-1 h-2.5 w-2.5 shrink-0 rounded-full", urgent ? "bg-[var(--cockpit-warning)]" : "bg-[var(--cockpit-active)]")} />
        <div className="min-w-0">
          <p className="break-words text-sm font-semibold text-[var(--cockpit-text)]">{safeText(signal.title)}</p>
          <p className="mt-1 break-words text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">{signalBriefLine(signal)}</p>
          <p className="mt-2 text-[11px] text-[var(--cockpit-muted)]">{safeText(signal.recency_label || signalFreshness(signal))}</p>
        </div>
      </div>
    </>
  );
  if (!onSelect) {
    return <article className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4">{body}</article>;
  }
  return (
    <button type="button" onClick={onSelect} className="w-full rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-left transition hover:border-[color-mix(in_srgb,var(--cockpit-active)_45%,transparent)]">
      {body}
    </button>
  );
}

function SignalSummary({ signal, onSelect }: { signal: CockpitSignal; onSelect?: () => void }) {
  const tone = signal.category === "stuck_failed" || signal.category === "needs_marco" ? "warning" : "success";
  const body = (
    <>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate font-semibold text-[var(--cockpit-text)]">{signal.title}</p>
          <p className="mt-1 text-xs text-[var(--cockpit-muted)]">{signal.reason}</p>
        </div>
        <Badge tone={tone} className="shrink-0 text-[10px] uppercase tracking-[0.18em]">
          {Math.round(signal.confidence * 100)}%
        </Badge>
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-[var(--cockpit-muted)]">
        <span>{signal.recency_label}</span>
        <span>·</span>
        <span>{signal.provenance}</span>
        <span>·</span>
        <span>{signalFreshness(signal)}</span>
        {signal.source?.platform && <><span>·</span><span>{signal.source.platform}</span></>}
      </div>
    </>
  );
  if (!onSelect) {
    return <article className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4">{body}</article>;
  }
  return (
    <button type="button" onClick={onSelect} className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-left transition hover:border-[color-mix(in_srgb,var(--cockpit-active)_45%,transparent)]">
      {body}
    </button>
  );
}

function LaneSignal({ lane, selected }: { lane: CockpitLane; selected: boolean }) {
  const status = laneStatus(lane);
  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-3xl border p-4 transition-all duration-200",
        selected
          ? "border-[color-mix(in_srgb,var(--cockpit-active)_55%,transparent)] bg-[var(--cockpit-active-soft)] shadow-[0_0_28px_color-mix(in_srgb,var(--cockpit-active)_18%,transparent)]"
          : "border-[var(--cockpit-border)] bg-[var(--cockpit-card)] hover:border-[color-mix(in_srgb,var(--cockpit-active)_35%,transparent)] hover:bg-[color-mix(in_srgb,var(--cockpit-active)_7%,transparent)]",
      )}
    >
      <div className="absolute inset-x-4 top-0 h-px bg-gradient-to-r from-transparent via-[color-mix(in_srgb,var(--cockpit-active)_45%,transparent)] to-transparent" />
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3 shrink-0">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[color-mix(in_srgb,var(--cockpit-active)_45%,transparent)] opacity-75" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-[var(--cockpit-active)]" />
            </span>
            <h3 className="truncate text-sm font-semibold text-[var(--cockpit-text)]">{laneName(lane)}</h3>
          </div>
          <p className="mt-1 truncate text-xs text-[var(--cockpit-muted)]">
            {lane.agent_role ? roleLabel(lane.agent_role) : lane.platform || "unknown platform"} · {lane.issue_identifier || lane.chat_type || "lane"}
          </p>
        </div>
        <Badge tone={statusTone(status)} className="shrink-0 text-[10px] uppercase tracking-[0.18em]">
          {status}
        </Badge>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-2xl bg-[var(--cockpit-shell-raised)] p-3">
          <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Updated</p>
          <p className="mt-1 font-mono-ui text-[var(--cockpit-text)]">{formatClockLabel(lane.updated_at)}</p>
        </div>
        <div className="rounded-2xl bg-[var(--cockpit-shell-raised)] p-3">
          <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Role / Issue</p>
          <p className="mt-1 truncate font-mono-ui text-[var(--cockpit-text)]">{lane.agent_role ? `${roleLabel(lane.agent_role)}${lane.issue_identifier ? ` · ${lane.issue_identifier}` : ""}` : lane.alias?.display_only ? "display" : "safe"}</p>
        </div>
      </div>
      {lane.alias?.ambiguous && (
        <div className="mt-3 flex items-center gap-2 rounded-2xl border border-[color-mix(in_srgb,var(--cockpit-warning)_30%,transparent)] bg-[var(--cockpit-warning-soft)] px-3 py-2 text-xs text-[color-mix(in_srgb,var(--cockpit-warning)_86%,var(--cockpit-text)_14%)]">
          <AlertTriangle className="h-3.5 w-3.5" />
          Ambiguous display alias
        </div>
      )}
    </div>
  );
}

function RiskySendComposer({
  enabled,
  allowedLanes,
}: {
  enabled: boolean;
  allowedLanes: CockpitResolvedLane[];
}) {
  const [laneAlias, setLaneAlias] = useState("%discord/hermes");
  const [messageText, setMessageText] = useState("");
  const [idempotencyKey, setIdempotencyKey] = useState(() => `cockpit-${Date.now().toString(36)}`);
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState<CockpitSendIntentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (allowedLanes[0]?.lane_alias) setLaneAlias(allowedLanes[0].lane_alias);
  }, [allowedLanes]);

  const disabled = !enabled || sending || !messageText.trim() || !idempotencyKey.trim();
  const selectedLane = allowedLanes.find((lane) => lane.lane_alias === laneAlias) ?? allowedLanes[0] ?? null;

  const submit = () => {
    setSending(true);
    setError(null);
    setResult(null);
    api.sendCockpitIntent({ lane_alias: laneAlias, idempotency_key: idempotencyKey, message_text: messageText })
      .then((response) => {
        setResult(response);
        if (response.ok) {
          setMessageText("");
          setIdempotencyKey(`cockpit-${Date.now().toString(36)}`);
        }
      })
      .catch((err) => setError(String(err)))
      .finally(() => setSending(false));
  };

  return (
    <SurfaceCard eyebrow="Risky external send" title="Discord #hermes composer">
      <div className="grid gap-4">
        <div className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_30%,transparent)] bg-[var(--cockpit-warning-soft)] p-4 text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-warning)_78%,var(--cockpit-text)_22%)]">
          <div className="flex items-center gap-2 font-semibold"><AlertTriangle className="h-4 w-4" /> Production external-send surface</div>
          <p className="mt-2 text-xs text-[color-mix(in_srgb,var(--cockpit-warning)_74%,var(--cockpit-text)_26%)]">
            Sends leave the dashboard and post to the resolved platform lane. Lane resolution is explicit and ambiguity fails closed. Attachments and voice are disabled in this pass.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="grid gap-2 text-xs uppercase tracking-[0.18em] text-[var(--cockpit-muted)]">
            Resolved lane
            <select
              value={laneAlias}
              onChange={(event) => setLaneAlias(event.target.value)}
              disabled={!enabled || allowedLanes.length === 0}
              className="rounded-2xl border border-[var(--cockpit-border)] bg-[var(--cockpit-shell)] px-3 py-2 text-sm normal-case text-[var(--cockpit-text)] outline-none focus:border-[color-mix(in_srgb,var(--cockpit-active)_60%,transparent)]"
            >
              {allowedLanes.length === 0 ? <option value="%discord/hermes">No unambiguous lane</option> : allowedLanes.map((lane) => (
                <option key={lane.lane_alias} value={lane.lane_alias}>{lane.lane_label}</option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-xs uppercase tracking-[0.18em] text-[var(--cockpit-muted)]">
            Idempotency key
            <input
              value={idempotencyKey}
              onChange={(event) => setIdempotencyKey(event.target.value)}
              disabled={!enabled}
              className="rounded-2xl border border-[var(--cockpit-border)] bg-[var(--cockpit-shell)] px-3 py-2 font-mono-ui text-sm normal-case text-[var(--cockpit-text)] outline-none focus:border-[color-mix(in_srgb,var(--cockpit-active)_60%,transparent)]"
            />
          </label>
        </div>

        <textarea
          value={messageText}
          onChange={(event) => setMessageText(event.target.value)}
          disabled={!enabled}
          placeholder={enabled ? "Type a short, intentional Discord #hermes message…" : "Risky send is disabled by feature config or kill switch."}
          className="min-h-[120px] rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-shell)] p-4 text-sm leading-6 text-[var(--cockpit-text)] outline-none placeholder:text-[var(--cockpit-muted-soft)] focus:border-[color-mix(in_srgb,var(--cockpit-active)_60%,transparent)] disabled:opacity-60"
        />

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.18em]">
            <Badge tone={enabled ? "warning" : "secondary"}>External send {enabled ? "gated" : "disabled"}</Badge>
            <Badge tone="secondary">Attachments disabled</Badge>
            <Badge tone="secondary">Voice disabled</Badge>
            {selectedLane && <Badge tone="secondary">{safeText(selectedLane.platform)}</Badge>}
          </div>
          <button
            type="button"
            disabled={disabled}
            onClick={submit}
            className="inline-flex items-center gap-2 rounded-2xl border border-[color-mix(in_srgb,var(--cockpit-warning)_45%,transparent)] bg-[color-mix(in_srgb,var(--cockpit-warning)_16%,transparent)] px-4 py-2 text-sm font-semibold text-[color-mix(in_srgb,var(--cockpit-warning)_78%,var(--cockpit-text)_22%)] transition hover:bg-[color-mix(in_srgb,var(--cockpit-warning)_24%,transparent)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            Send externally
          </button>
        </div>

        {error && <p className="rounded-2xl border border-[color-mix(in_srgb,var(--cockpit-risk)_30%,transparent)] bg-[var(--cockpit-risk-soft)] p-3 text-xs text-[color-mix(in_srgb,var(--cockpit-risk)_82%,var(--cockpit-text)_18%)]">{boundedCopy(error, 220)}</p>}
        {result && (
          <div className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-shell-raised)] p-4 text-xs leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">
            <p className="font-semibold text-[var(--cockpit-text)]">Result: {safeText(result.status)}{result.idempotent_replay ? " · idempotent replay" : ""}</p>
            {result.record && <p>Preview: {safeText(result.record.content_preview_redacted)}</p>}
            {result.record?.dispatch_result && <p>Dispatch: {result.record.dispatch_result.ok ? "ok" : safeText(result.record.dispatch_result.error_code)}</p>}
            {result.audit?.slice(-2).map((event) => (
              <p key={`${event.event_type}-${event.updated_at}`}>Audit: {safeText(event.event_type)} · {safeText(event.status)}</p>
            ))}
          </div>
        )}
      </div>
    </SurfaceCard>
  );
}

function LaneDetailPanel({
  lane,
  response,
  transcriptWindow,
  loading,
  error,
}: {
  lane: CockpitLane | null;
  response: CockpitLaneMessagesResponse | null;
  transcriptWindow: CockpitTranscriptWindow | null;
  loading: boolean;
  error: string | null;
}) {
  const messages = response?.messages ?? [];
  const window = response?.window ?? transcriptWindow;
  const bounded = response?.bounded ?? window?.bounded ?? true;
  const windowLimit = response?.window_limit ?? window?.window_limit;
  const visibleCount = messages.length;
  const totalCount = response?.total ?? visibleCount;

  return (
    <SurfaceCard eyebrow="Selected lane" title={lane ? laneName(lane) : "No lane selected"}>
      {!lane ? (
        <div className="rounded-3xl border border-dashed border-[var(--cockpit-border)] p-8 text-center text-sm text-[var(--cockpit-muted)]">
          <Satellite className="mx-auto mb-3 h-8 w-8" />
          Select a visible lane to inspect its bounded recent messages.
        </div>
      ) : (
        <div className="grid gap-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-3xl bg-[var(--cockpit-shell-raised)] p-3">
              <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Platform</p>
              <p className="mt-1 truncate text-sm font-semibold text-[var(--cockpit-text)]">{safeText(lane.platform || "Unknown")}</p>
            </div>
            <div className="rounded-3xl bg-[var(--cockpit-shell-raised)] p-3">
              <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Kind</p>
              <p className="mt-1 truncate text-sm font-semibold text-[var(--cockpit-text)]">{safeText(lane.chat_type || "Lane")}</p>
            </div>
            <div className="rounded-3xl bg-[var(--cockpit-shell-raised)] p-3">
              <p className="cockpit-micro-label text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Updated</p>
              <p className="mt-1 font-mono-ui text-sm text-[var(--cockpit-text)]">{formatClockLabel(lane.updated_at)}</p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <Badge tone={statusTone(laneStatus(lane))} className="text-[10px] uppercase tracking-[0.18em]">
              {laneStatus(lane)}
            </Badge>
            <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">
              {bounded ? "Bounded recent window" : "Recent window"}
              {windowLimit ? ` · ${windowLimit}` : ""}
            </Badge>
            <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">
              {visibleCount} shown{totalCount > visibleCount ? ` of ${totalCount}` : ""}
            </Badge>
            {lane.agent_role && <Badge tone="success" className="text-[10px] uppercase tracking-[0.18em]">{roleLabel(lane.agent_role)}</Badge>}
            {lane.issue_identifier && <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">{safeText(lane.issue_identifier)}</Badge>}
            {lane.delegate_kind && <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">{safeText(lane.delegate_kind)}</Badge>}
            {lane.alias?.display_only && (
              <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">Display alias</Badge>
            )}
            {lane.alias?.ambiguous && (
              <Badge tone="warning" className="text-[10px] uppercase tracking-[0.18em]">Ambiguous alias</Badge>
            )}
          </div>

          <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_20%,transparent)] bg-[var(--cockpit-active-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-active)_85%,var(--cockpit-text)_15%)]">
            Showing display-safe recent messages only. Older history is not loaded here, and internal execution details, raw identifiers, keys, and media payloads are redacted or hidden.
          </p>

          <Card className="flex max-h-[560px] min-h-[300px] overflow-hidden rounded-[1.75rem] border-[var(--cockpit-border)] bg-[var(--cockpit-panel)]">
            <CardContent className="flex flex-1 flex-col gap-3 overflow-auto p-4">
              {loading && (
                <div className="flex flex-1 items-center justify-center text-sm text-[var(--cockpit-muted)]">
                  <Spinner className="mr-2 text-[var(--cockpit-active)]" /> Loading selected lane messages…
                </div>
              )}
              {!loading && error && (
                <div className="flex flex-1 flex-col items-center justify-center rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-risk)_30%,transparent)] bg-[var(--cockpit-risk-soft)] p-6 text-center text-sm text-[color-mix(in_srgb,var(--cockpit-risk)_82%,var(--cockpit-text)_18%)]">
                  <AlertTriangle className="mb-3 h-8 w-8" />
                  Could not load selected lane messages.
                  <span className="mt-2 text-xs text-[color-mix(in_srgb,var(--cockpit-risk)_72%,var(--cockpit-text)_28%)]">{boundedCopy(error, 180)}</span>
                </div>
              )}
              {!loading && !error && messages.length === 0 && (
                <div className="flex flex-1 flex-col items-center justify-center rounded-3xl border border-dashed border-[var(--cockpit-border)] p-8 text-center text-sm text-[var(--cockpit-muted)]">
                  <MessageCircle className="mb-3 h-9 w-9" />
                  No display-safe recent messages are visible for this lane.
                </div>
              )}
              {!loading && !error && messages.map((message, index) => (
                <article key={`${message.created_at ?? message.timestamp ?? index}-${index}`} className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-card)] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <Badge tone="secondary" className="shrink-0 text-[10px] uppercase tracking-[0.18em]">
                      {messageRoleLabel(message)}
                    </Badge>
                    <span className="cockpit-meta font-mono-ui text-[11px] text-[var(--cockpit-muted)]">{messageTimestamp(message)}</span>
                  </div>
                  <p className="cockpit-mono-copy mt-3 whitespace-pre-wrap break-words text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_88%,transparent)]">{messageBody(message)}</p>
                  {(message.status || message.source) && (
                    <p className="mt-3 text-[11px] uppercase tracking-[0.18em] text-[var(--cockpit-muted)]">
                      {[message.status, message.source].map(safeText).filter(Boolean).join(" · ")}
                    </p>
                  )}
                </article>
              ))}
            </CardContent>
          </Card>
        </div>
      )}
    </SurfaceCard>
  );
}

function DailyOpsRadarPanel({ response, loading, error }: { response: CockpitDailyOpsRadarResponse | null; loading: boolean; error: string | null }) {
  const summary = response?.summary;
  const prepare = response?.upgrade.prepare_review;
  const recommendation = response?.upgrade.recommendation;
  const upgradeBrief = response?.upgrade.brief || summary?.upgrade_brief;
  const categories = summary?.categories ?? [];
  const topCommits = summary?.top_commits ?? [];

  return (
    <SurfaceCard eyebrow="Hermes Daily Ops Radar" title="Upstream upgrade review gate">
      <div className="grid gap-4">
        <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_20%,transparent)] bg-[var(--cockpit-active-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-active)_85%,var(--cockpit-text)_15%)]">
          Read-only projection from cron job a82830911bcd / hermes_daily_ops_radar.py. This panel does not run git pull, merge, restart services, mutate production, or send external messages. High-volume or risky radar answers: Prepare review first — Do not upgrade blindly.
        </p>
        {upgradeBrief && (
          <div className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_28%,transparent)] bg-[var(--cockpit-panel)] p-4 shadow-[0_18px_60px_rgba(0,0,0,0.20)]" data-testid="cockpit-upgrade-brief">
            <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--cockpit-muted)]">Upgrade brief</p>
            <p className="mt-2 break-words text-lg font-semibold leading-7 text-[var(--cockpit-text)]">Major improvements</p>
            <p className="mt-2 break-words text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_84%,transparent)]">{boundedCopy(upgradeBrief.headline, 420)}</p>
            {upgradeBrief.groups.length > 0 && (
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {upgradeBrief.groups.map((group) => (
                  <article key={group.label} className="min-w-0 rounded-2xl bg-[var(--cockpit-shell-raised)] p-3">
                    <p className="break-words text-xs font-semibold text-[var(--cockpit-text)]">{safeText(group.label)}</p>
                    <p className="mt-1 break-words text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-text)_78%,transparent)]">{boundedCopy(group.summary, 260)}</p>
                  </article>
                ))}
              </div>
            )}
            <p className="mt-3 break-words text-xs leading-5 text-[var(--cockpit-muted)]">{boundedCopy(upgradeBrief.why_this_matters, 260)}</p>
          </div>
        )}
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-3xl bg-[var(--cockpit-shell-raised)] p-4">
            <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Behind</p>
            <p className="mt-2 text-3xl font-semibold text-[var(--cockpit-text)]">{summary?.behind_count ?? "—"}</p>
          </div>
          <div className="rounded-3xl bg-[var(--cockpit-shell-raised)] p-4">
            <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Relevant</p>
            <p className="mt-2 text-3xl font-semibold text-[var(--cockpit-text)]">{summary?.relevant_change_count ?? "—"}</p>
          </div>
          <div className="rounded-3xl bg-[var(--cockpit-shell-raised)] p-4">
            <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Status</p>
            <p className="mt-2 text-xl font-semibold text-[var(--cockpit-text)]">{safeText(response?.job.status || "pending")}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.18em]">
          <Badge tone="secondary">Read-only {String(response?.read_only ?? true)}</Badge>
          <Badge tone="secondary">actions_enabled {String(response?.actions_enabled ?? false)}</Badge>
          <Badge tone="secondary">last {safeText(summary?.last_run || response?.job.last_run_at || "not available")}</Badge>
          <Badge tone="secondary">next {safeText(response?.job.next_run_at || "not available")}</Badge>
        </div>
        {loading && !response && <div className="flex min-h-48 items-center justify-center text-sm text-[var(--cockpit-muted)]"><Spinner className="mr-2 text-[var(--cockpit-active)]" /> Loading daily ops radar…</div>}
        {!loading && error && <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-risk)_30%,transparent)] bg-[var(--cockpit-risk-soft)] p-4 text-sm text-[color-mix(in_srgb,var(--cockpit-risk)_82%,var(--cockpit-text)_18%)]">Could not load Daily Ops Radar: {boundedCopy(error, 180)}</p>}
        {!loading && !error && !response && <p className="rounded-3xl border border-dashed border-[var(--cockpit-border)] p-6 text-center text-sm text-[var(--cockpit-muted)]">No Daily Ops Radar cron output is available yet.</p>}

        <div className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_28%,transparent)] bg-[var(--cockpit-panel)] p-4 shadow-[0_18px_60px_rgba(0,0,0,0.24)]" data-testid="cockpit-upgrade-recommendation">
          <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--cockpit-muted)]">Should we upgrade?</p>
          <div className="mt-3 grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start">
            <div className="min-w-0">
              <p className="break-words text-2xl font-semibold leading-tight text-[var(--cockpit-text)]">{safeText(recommendation?.label || "Wait")}</p>
              <p className="mt-2 break-words text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_82%,transparent)]">{boundedCopy(recommendation?.rationale || "Wait; radar is not trustworthy enough for an upgrade call.", 220)}</p>
            </div>
            <Badge tone={recommendation?.risk_level === "low" ? "success" : recommendation?.risk_level === "unknown" ? "warning" : "warning"} className="w-fit text-[10px] uppercase tracking-[0.18em]">Risk {safeText(recommendation?.risk_level || "unknown")}</Badge>
          </div>
          <div className="mt-3 grid gap-2 text-xs leading-5 text-[var(--cockpit-muted)] sm:grid-cols-2">
            <p className="min-w-0 break-words"><span className="font-semibold text-[var(--cockpit-text)]">Freshness:</span> {safeText(recommendation?.freshness || "unknown")} · {boundedCopy(recommendation?.freshness_detail || "No trustworthy radar artifact is available.", 160)}</p>
            <p className="min-w-0 break-words"><span className="font-semibold text-[var(--cockpit-text)]">Last radar basis:</span> {safeText(recommendation?.basis || summary?.last_run || response?.job.last_run_at || "not available")}</p>
          </div>
          {recommendation?.signals && recommendation.signals.length > 0 && (
            <p className="mt-3 break-words text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-text)_72%,transparent)]">Basis signals: {recommendation.signals.map((signal) => boundedCopy(signal, 120)).join(" · ")}</p>
          )}
        </div>

        <div className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <p className="text-sm font-semibold text-[var(--cockpit-text)]">{safeText(prepare?.label || "Prepare upgrade review")}</p>
              <p className="mt-1 text-xs leading-5 text-[var(--cockpit-muted)]">{boundedCopy(prepare?.description || "Disabled until a safe read-only preflight endpoint exists.", 260)}</p>
            </div>
            <button type="button" disabled className="shrink-0 rounded-2xl border border-[var(--cockpit-border)] bg-[var(--cockpit-shell-raised)] px-4 py-3 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--cockpit-muted)] opacity-80">
              {prepare?.enabled ? "Open preflight" : "Review gate disabled"}
            </button>
          </div>
          <p className="mt-3 text-[10px] uppercase tracking-[0.18em] text-[var(--cockpit-muted)]">{safeText(prepare?.status || "disabled_no_safe_preflight_endpoint")} · method {safeText(prepare?.method || "GET")} · mutates {String(prepare?.mutates ?? false)}</p>
        </div>

        {categories.length > 0 && (
          <div className="grid gap-3 md:grid-cols-2">
            {categories.map((category) => (
              <article key={category.label} className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4">
                <p className="font-semibold text-[var(--cockpit-text)]">{safeText(category.label)}</p>
                <p className="mt-2 line-clamp-4 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">{boundedCopy(category.summary, 260)}</p>
              </article>
            ))}
          </div>
        )}

        {topCommits.length > 0 && (
          <div className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4">
            <p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Top commits to review</p>
            <div className="grid gap-2">
              {topCommits.map((commit) => (
                <div key={`${commit.sha}-${commit.title}`} className="rounded-2xl bg-[var(--cockpit-shell-raised)] p-3 text-xs leading-5">
                  <span className="font-mono-ui text-[var(--cockpit-active)]">{safeText(commit.sha || "commit")}</span>
                  <span className="text-[color-mix(in_srgb,var(--cockpit-text)_80%,transparent)]"> · {boundedCopy(commit.title, 220)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {summary?.compare_command && (
          <div className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-shell-raised)] p-4">
            <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">Compare command (not executed)</p>
            <code className="mt-2 block max-w-full whitespace-pre-wrap break-words font-mono-ui text-xs text-[color-mix(in_srgb,var(--cockpit-text)_82%,transparent)]">{safeText(summary.compare_command)}</code>
          </div>
        )}
      </div>
    </SurfaceCard>
  );
}

function N8nChecksPanel({ response, loading, error }: { response: CockpitN8nChecksResponse | null; loading: boolean; error: string | null }) {
  const checks = response?.checks ?? [];
  const expectedNames = N8N_CHECK_NAMES.filter((name) => !checks.some((check) => check.name === name));
  const flags = response ?? { read_only: true, actions_enabled: false, external_delivery_enabled: false };

  return (
    <SurfaceCard eyebrow="Daily n8n checks" title="Read-only n8n daily check outputs">
      <div className="grid gap-4">
        <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_20%,transparent)] bg-[var(--cockpit-active-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-active)_85%,var(--cockpit-text)_15%)]">
          No workflow triggers, Discord sends, retries, repairs, voice, attachments, or routing are available here. This first slice renders display-safe status metadata only.
        </p>
        <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.18em]">
          <Badge tone={flags.read_only ? "secondary" : "warning"}>Read-only {String(flags.read_only)}</Badge>
          <Badge tone={flags.actions_enabled ? "warning" : "secondary"}>actions_enabled {String(flags.actions_enabled)}</Badge>
          <Badge tone={flags.external_delivery_enabled ? "warning" : "secondary"}>external_delivery_enabled {String(flags.external_delivery_enabled)}</Badge>
          <Badge tone="secondary">source {safeText(response?.source || "fixture_bif_525_inventory")}</Badge>
          <Badge tone={response?.live ? "success" : "warning"}>{response?.live ? "live n8n" : "fallback/stale"}</Badge>
          {response?.stale && <Badge tone="warning">stale fixture fallback</Badge>}
        </div>
        {response?.live_error && (
          <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_30%,transparent)] bg-[var(--cockpit-warning-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-warning)_78%,var(--cockpit-text)_22%)]">
            Live n8n read unavailable; showing fixture fallback. {boundedCopy(response.live_error, 160)}
          </p>
        )}

        {loading && checks.length === 0 && <div className="flex min-h-48 items-center justify-center text-sm text-[var(--cockpit-muted)]"><Spinner className="mr-2 text-[var(--cockpit-active)]" /> Loading daily n8n checks…</div>}
        {!loading && error && <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-risk)_30%,transparent)] bg-[var(--cockpit-risk-soft)] p-4 text-sm text-[color-mix(in_srgb,var(--cockpit-risk)_82%,var(--cockpit-text)_18%)]">Could not load n8n checks: {boundedCopy(error, 180)}</p>}
        {!loading && !error && checks.length === 0 && <p className="rounded-3xl border border-dashed border-[var(--cockpit-border)] p-6 text-center text-sm text-[var(--cockpit-muted)]">No daily n8n check fixture rows are available.</p>}

        {expectedNames.length > 0 && (
          <div className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_30%,transparent)] bg-[var(--cockpit-warning-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-warning)_78%,var(--cockpit-text)_22%)]">
            Missing inventory rows: {expectedNames.join(", ")}
          </div>
        )}

        {checks.length > 0 && (
          <div className="max-w-full overflow-x-auto rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)]" data-testid="cockpit-n8n-checks-table">
            <div className="min-w-[960px]">
              <div className="grid grid-cols-7 gap-3 border-b border-[var(--cockpit-border)] px-4 py-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--cockpit-muted)]">
                <span>Check</span><span>Status</span><span>Last run</span><span>Completed</span><span>Next</span><span>Output summary</span><span>Action needed</span>
              </div>
              {checks.map((check: CockpitN8nCheck) => (
                <article key={check.id} className="grid grid-cols-7 gap-3 border-b border-[color-mix(in_srgb,var(--cockpit-text)_7%,transparent)] px-4 py-4 text-xs last:border-b-0">
                  <div className="min-w-0">
                    <p className="truncate font-semibold text-[var(--cockpit-text)]">{safeText(check.name)}</p>
                    <p className="mt-1 line-clamp-3 leading-5 text-[var(--cockpit-muted)]">{boundedCopy(check.summary, 160)}</p>
                    <p className="mt-2 text-[10px] uppercase tracking-[0.18em] text-[var(--cockpit-muted)]">{safeText(check.live_source || "fixture_fallback")}</p>
                  </div>
                  <div><Badge tone={statusTone(check.execution_status || check.status)} className="text-[10px] uppercase tracking-[0.18em]">{safeText(check.execution_status || check.status)}</Badge></div>
                  <p className="break-words font-mono-ui text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">{safeText(check.last_started || check.last_run)}</p>
                  <p className="break-words font-mono-ui text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">{safeText(check.last_completed || "—")}</p>
                  <p className="break-words font-mono-ui text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">{safeText(check.next_schedule)}</p>
                  <div className="min-w-0">
                    <p className="line-clamp-4 break-words leading-5 text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">{boundedCopy(check.output_summary || check.summary, 220)}</p>
                    <p className="mt-2 text-[10px] uppercase tracking-[0.18em] text-[var(--cockpit-muted)]">auth {safeText(check.auth)} · {safeText(check.delivery)}</p>
                  </div>
                  <p className="break-words leading-5 text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">{safeText(check.action_needed)}</p>
                </article>
              ))}
            </div>
          </div>
        )}
      </div>
    </SurfaceCard>
  );
}

function automationBucketTone(bucket: string): "success" | "warning" | "secondary" {
  if (bucket === "healthy") return "success";
  if (bucket === "attention" || bucket === "stale_or_failing") return "warning";
  return "secondary";
}

function automationBucketLabel(bucket: string): string {
  if (bucket === "attention") return "Attention / needs Marco";
  if (bucket === "healthy") return "Healthy";
  if (bucket === "stale_or_failing") return "Stale or failing";
  return safeText(bucket || "Observed");
}

function AutomationHealthPanel({ response, loading, error, onInvestigate }: { response: CockpitAutomationHealthResponse | null; loading: boolean; error: string | null; onInvestigate: (card: CockpitAutomationHealthCard) => void }) {
  const cards = response?.cards ?? [];
  const counts = response?.summary;
  const buckets: Array<{ key: "attention" | "healthy" | "stale_or_failing"; label: string; value: number }> = [
    { key: "attention", label: "Attention / needs Marco", value: counts?.attention ?? 0 },
    { key: "healthy", label: "Healthy", value: counts?.healthy ?? 0 },
    { key: "stale_or_failing", label: "Stale or failing", value: counts?.stale_or_failing ?? 0 },
  ];

  return (
    <SurfaceCard eyebrow="Automation Health Cockpit" title="Automation health without the telemetry wall">
      <div className="grid gap-4">
        <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_20%,transparent)] bg-[var(--cockpit-active-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-active)_85%,var(--cockpit-text)_15%)]">
          Plain-language summaries from safe local sources only: n8n daily checks, Daily Ops Radar metadata, cron/job status, gateway runtime, and dashboard local status. This section is observer-only: it cannot launch automations, change services, perform version-control work, mutate production, contact outside channels, use voice, add attachments, or change routing.
        </p>

        <div className="grid gap-3 sm:grid-cols-3" aria-label="Automation health reading model">
          {buckets.map((bucket) => (
            <div key={bucket.key} className="min-w-0 rounded-3xl bg-[var(--cockpit-shell-raised)] p-4">
              <p className="cockpit-micro-label break-words text-[10px] uppercase tracking-[0.18em] text-[var(--cockpit-muted)]">{bucket.label}</p>
              <p className="mt-2 text-3xl font-semibold text-[var(--cockpit-text)]">{bucket.value}</p>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.18em]">
          <Badge tone="secondary">Read-only {String(response?.read_only ?? true)}</Badge>
          <Badge tone="secondary">actions_enabled {String(response?.actions_enabled ?? false)}</Badge>
          <Badge tone="secondary">mutation_enabled {String(response?.mutation_enabled ?? false)}</Badge>
          <Badge tone="secondary">Last checked/source</Badge>
        </div>

        {loading && !response && <div className="flex min-h-40 items-center justify-center text-sm text-[var(--cockpit-muted)]"><Spinner className="mr-2 text-[var(--cockpit-active)]" /> Loading automation health…</div>}
        {!loading && error && <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-risk)_30%,transparent)] bg-[var(--cockpit-risk-soft)] p-4 text-sm text-[color-mix(in_srgb,var(--cockpit-risk)_82%,var(--cockpit-text)_18%)]">Could not load automation health: {boundedCopy(error, 180)}</p>}
        {!loading && !error && !response && <p className="rounded-3xl border border-dashed border-[var(--cockpit-border)] p-6 text-center text-sm text-[var(--cockpit-muted)]">No automation health summary is available yet.</p>}

        {response?.summary?.headline && (
          <p className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_80%,transparent)]">{boundedCopy(response.summary.headline, 220)}</p>
        )}

        <div className="grid gap-3" data-testid="cockpit-automation-health-cards">
          {cards.map((card: CockpitAutomationHealthCard) => (
            <article key={`${card.title}-${card.source}`} className="min-w-0 rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4">
              <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="min-w-0">
                  <p className="break-words text-sm font-semibold text-[var(--cockpit-text)]">{safeText(card.title)}</p>
                  <p className="mt-2 break-words text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_78%,transparent)]">{boundedCopy(card.summary, 260)}</p>
                </div>
                <div className="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">
                  <Badge tone={automationBucketTone(card.bucket)} className="text-[10px] uppercase tracking-[0.16em]">{automationBucketLabel(card.bucket)}</Badge>
                  {card.bucket !== "healthy" && (
                    <button
                      type="button"
                      onClick={() => onInvestigate(card)}
                      className="rounded-full border border-[color-mix(in_srgb,var(--cockpit-warning)_42%,transparent)] bg-[var(--cockpit-warning-soft)] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-[color-mix(in_srgb,var(--cockpit-warning)_86%,var(--cockpit-text)_14%)] transition hover:border-[color-mix(in_srgb,var(--cockpit-warning)_60%,transparent)]"
                    >
                      Ask Biff to investigate
                    </button>
                  )}
                </div>
              </div>
              {card.details && card.details.length > 0 && (
                <ul className="mt-3 grid gap-1 text-xs leading-5 text-[var(--cockpit-muted)]">
                  {card.details.map((detail) => <li key={detail} className="break-words">{boundedCopy(detail, 160)}</li>)}
                </ul>
              )}
              <p className="cockpit-meta mt-3 break-words text-[11px] text-[var(--cockpit-muted)]">Last checked/source: {formatClockLabel(card.last_checked, safeText(card.last_checked) || "Not available")} · {safeText(card.source)}</p>
            </article>
          ))}
        </div>
      </div>
    </SurfaceCard>
  );
}

function AgentActivityPanel({ response, onSelect }: { response: CockpitAgentActivityResponse | null; onSelect: (laneId: string) => void }) {
  const items = (response?.items ?? []).slice(0, 8);
  return (
    <SurfaceCard eyebrow="Agent Activity" title="Biff / Forge / Vex / Quill / Ranger">
      <div className="grid gap-3" data-testid="cockpit-agent-activity">
        <p className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-3 text-xs leading-5 text-[var(--cockpit-muted)]">
          Read-only visibility for named agents. No interrupt, restart, send, deploy, workflow mutation, raw channel IDs, credentials, or full transcripts are exposed here.
        </p>
        {items.length ? items.map((item: CockpitAgentActivityItem) => (
          <button
            key={`agent-${item.id}`}
            type="button"
            onClick={() => onSelect(item.lane_id)}
            className="min-w-0 rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-left transition hover:border-[color-mix(in_srgb,var(--cockpit-active)_42%,transparent)]"
          >
            <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="success" className="text-[10px] uppercase tracking-[0.18em]">{roleLabel(item.agent_role)}</Badge>
                  {item.issue_identifier && <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">{safeText(item.issue_identifier)}</Badge>}
                  <Badge tone={statusTone(item.status)} className="text-[10px] uppercase tracking-[0.18em]">{safeText(item.status)}</Badge>
                  {item.delegate_kind && <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">{safeText(item.delegate_kind)}</Badge>}
                </div>
                <p className="mt-3 break-words text-sm font-semibold text-[var(--cockpit-text)]">{safeText(item.goal || item.title)}</p>
                <p className="mt-1 break-words text-xs leading-5 text-[var(--cockpit-muted)]">{boundedCopy(item.latest_evidence, 220)}</p>
                <p className="mt-2 text-[11px] text-[var(--cockpit-muted)]">Updated {formatClockLabel(item.updated_at)}{item.completed_at ? ` · completed ${formatClockLabel(item.completed_at)}` : ""}</p>
              </div>
              <span className="shrink-0 text-[11px] text-[var(--cockpit-muted)]">{safeText(item.recency_label)}</span>
            </div>
          </button>
        )) : (
          <p className="rounded-3xl border border-dashed border-[var(--cockpit-border)] p-5 text-center text-sm text-[var(--cockpit-muted)]">{response?.empty_state || "No named-agent activity is visible yet."}</p>
        )}
      </div>
    </SurfaceCard>
  );
}

function SelfWorkHandoffPanel({ response }: { response: CockpitSelfWorkHandoffResponse | null }) {
  const handoff = response?.handoff;
  if (!handoff) return null;
  const checklistLines = safeText(handoff.rendered_checklist || "").split(" ").length ? String(handoff.rendered_checklist || "").split("\n").filter(Boolean) : [];
  return (
    <SurfaceCard eyebrow="Resume brief" title="Where Biff was before restart">
      <article className="grid gap-3 rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_28%,transparent)] bg-[var(--cockpit-active-soft)] p-4" data-testid="cockpit-self-work-handoff">
        <div className="flex flex-wrap items-center gap-2">
          {handoff.issue_identifier && <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">{safeText(handoff.issue_identifier)}</Badge>}
          {handoff.current_phase && <Badge tone="success" className="text-[10px] uppercase tracking-[0.18em]">{safeText(handoff.current_phase)}</Badge>}
          <Badge tone="secondary" className="text-[10px] uppercase tracking-[0.18em]">read-only</Badge>
        </div>
        {handoff.goal && <p className="break-words text-sm font-semibold text-[var(--cockpit-text)]">{boundedCopy(handoff.goal, 180)}</p>}
        <div className="grid gap-2 text-xs leading-5 text-[var(--cockpit-muted)]">
          {handoff.last_action && <p><span className="font-semibold text-[var(--cockpit-text)]">Last action:</span> {boundedCopy(handoff.last_action, 240)}</p>}
          {handoff.next_safe_step && <p><span className="font-semibold text-[var(--cockpit-text)]">Next safe step:</span> {boundedCopy(handoff.next_safe_step, 240)}</p>}
          {handoff.pending_verification?.length ? <p><span className="font-semibold text-[var(--cockpit-text)]">Pending:</span> {handoff.pending_verification.map((item) => boundedCopy(item, 90)).join(" · ")}</p> : null}
          {handoff.known_failures?.length ? <p><span className="font-semibold text-[var(--cockpit-text)]">Known failures:</span> {handoff.known_failures.map((item) => boundedCopy(item, 90)).join(" · ")}</p> : null}
        </div>
        {checklistLines.length > 0 && <pre className="max-w-full overflow-x-auto whitespace-pre-wrap rounded-2xl bg-[var(--cockpit-shell)] p-3 text-[11px] leading-5 text-[var(--cockpit-text)]">{checklistLines.join("\n")}</pre>}
      </article>
    </SurfaceCard>
  );
}

function SurfaceCard({ title, eyebrow, children }: { title: string; eyebrow: string; children: ReactNode }) {
  return (
    <section className="max-w-full overflow-hidden rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-card-raised)] p-3 shadow-[0_20px_70px_var(--cockpit-shadow)] sm:rounded-[2rem] sm:p-5">
      <p className="cockpit-micro-label text-[9px] font-semibold uppercase tracking-[0.18em] text-[color-mix(in_srgb,var(--cockpit-active)_78%,transparent)] sm:text-[10px] sm:tracking-[0.22em]">{eyebrow}</p>
      <h2 className="cockpit-title mt-2 break-words text-lg font-semibold tracking-tight text-[var(--cockpit-text)] sm:text-xl">{title}</h2>
      <div className="mt-4 min-w-0">{children}</div>
    </section>
  );
}

export default function CockpitPage({ standalone = false }: { standalone?: boolean }) {
  const [lanes, setLanes] = useState<CockpitLane[]>([]);
  const [signals, setSignals] = useState<CockpitSignalsResponse | null>(null);
  const [agentActivity, setAgentActivity] = useState<CockpitAgentActivityResponse | null>(null);
  const [selectedLaneId, setSelectedLaneId] = useState<string | null>(null);
  const [capabilityReadOnly, setCapabilityReadOnly] = useState<boolean | null>(null);
  const [inputEnabled, setInputEnabled] = useState<boolean | null>(null);
  const [externalSendEnabled, setExternalSendEnabled] = useState(false);
  const [riskyAllowedLanes, setRiskyAllowedLanes] = useState<CockpitResolvedLane[]>([]);
  const [transcriptWindow, setTranscriptWindow] = useState<CockpitTranscriptWindow | null>(null);
  const [selectedLaneMessages, setSelectedLaneMessages] = useState<CockpitLaneMessagesResponse | null>(null);
  const [automationHealth, setAutomationHealth] = useState<CockpitAutomationHealthResponse | null>(null);
  const [n8nChecks, setN8nChecks] = useState<CockpitN8nChecksResponse | null>(null);
  const [dailyOpsRadar, setDailyOpsRadar] = useState<CockpitDailyOpsRadarResponse | null>(null);
  const [selfWorkHandoff, setSelfWorkHandoff] = useState<CockpitSelfWorkHandoffResponse | null>(null);
  const [dashboardStatus, setDashboardStatus] = useState<StatusResponse | null>(null);
  const [automationHealthLoading, setAutomationHealthLoading] = useState(false);
  const [n8nChecksLoading, setN8nChecksLoading] = useState(false);
  const [automationHealthError, setAutomationHealthError] = useState<string | null>(null);
  const [n8nChecksError, setN8nChecksError] = useState<string | null>(null);
  const [dailyOpsRadarError, setDailyOpsRadarError] = useState<string | null>(null);
  const [selectedLaneLoading, setSelectedLaneLoading] = useState(false);
  const [selectedLaneError, setSelectedLaneError] = useState<string | null>(null);
  const [healthActionPrompt, setHealthActionPrompt] = useState<string | null>(null);
  const [healthActionCopied, setHealthActionCopied] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [activeSection, setActiveSection] = useState<CockpitSectionKey>("local-chat");

  const selectedLane = useMemo(
    () => lanes.find((lane) => lane.lane_id === selectedLaneId) ?? lanes[0] ?? null,
    [lanes, selectedLaneId],
  );

  const refreshOverview = useCallback(() => {
    setError(null);
    setAutomationHealthLoading(true);
    setN8nChecksLoading(true);
    setAutomationHealthError(null);
    setN8nChecksError(null);
    setDailyOpsRadarError(null);
    return Promise.all([
      api.getCockpitCapabilities(),
      api.getCockpitLanes({ limit: LANE_LIMIT, offset: 0 }),
      api.getCockpitSignals({ limit: LANE_LIMIT, message_limit: 20 }),
      api.getCockpitAgentActivity({ limit: LANE_LIMIT, message_limit: 8 }),
      api.getCockpitAutomationHealth(),
      api.getCockpitN8nChecks(),
      api.getCockpitDailyOpsRadar(),
      api.getCockpitSelfWorkHandoff(),
      api.getStatus(),
    ])
      .then(([capabilities, laneResponse, signalResponse, agentActivityResponse, automationResponse, n8nResponse, radarResponse, handoffResponse, statusResponse]) => {
        setCapabilityReadOnly(Boolean(capabilities.read_only));
        setInputEnabled(Boolean(capabilities.input_enabled));
        setExternalSendEnabled(Boolean(capabilities.external_send_enabled));
        setRiskyAllowedLanes(capabilities.risky_send?.allowed_lanes ?? []);
        setTranscriptWindow(capabilities.transcript_window ?? null);
        setLanes(laneResponse.lanes);
        setSignals(signalResponse);
        setAgentActivity(agentActivityResponse);
        setAutomationHealth(automationResponse);
        setN8nChecks(n8nResponse);
        setDailyOpsRadar(radarResponse);
        setSelfWorkHandoff(handoffResponse);
        setDashboardStatus(statusResponse);
        const firstSignalLane = categorySignals(signalResponse, "now")[0]?.lane_id;
        setSelectedLaneId((current) => {
          if (current && laneResponse.lanes.some((lane) => lane.lane_id === current)) return current;
          return firstSignalLane ?? laneResponse.lanes[0]?.lane_id ?? null;
        });
        setLastRefresh(new Date());
      })
      .catch((err) => {
        setError(String(err));
        setAutomationHealthError(String(err));
        setN8nChecksError(String(err));
        setDailyOpsRadarError(String(err));
      })
      .finally(() => {
        setLoading(false);
        setAutomationHealthLoading(false);
        setN8nChecksLoading(false);
      });
  }, []);

  useEffect(() => {
    refreshOverview();
    const timer = window.setInterval(refreshOverview, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refreshOverview]);

  useEffect(() => {
    const laneId = selectedLane?.lane_id;
    if (!laneId) {
      setSelectedLaneMessages(null);
      setSelectedLaneError(null);
      setSelectedLaneLoading(false);
      return;
    }

    let cancelled = false;
    setSelectedLaneLoading(true);
    setSelectedLaneError(null);
    api.getCockpitLaneMessages(laneId, { limit: LANE_MESSAGE_LIMIT, offset: 0 })
      .then((response) => {
        if (!cancelled) setSelectedLaneMessages(response);
      })
      .catch((err) => {
        if (!cancelled) {
          setSelectedLaneMessages(null);
          setSelectedLaneError(String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setSelectedLaneLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedLane?.lane_id]);

  const nowSignals = categorySignals(signals, "now");
  const attentionSignals = uniqueSignals([
    ...categorySignals(signals, "needs_marco"),
    ...categorySignals(signals, "stuck_failed"),
    ...categorySignals(signals, "waiting"),
  ], 5);
  const recentContextSignals = uniqueSignals([
    ...categorySignals(signals, "recent_context"),
    ...categorySignals(signals, "archive"),
  ], 6);
  const recentChangeSignals = uniqueSignals([
    ...categorySignals(signals, "recently_completed"),
    ...categorySignals(signals, "recent_context"),
    ...categorySignals(signals, "archive"),
  ], 4);
  const activeWorkSignals = uniqueSignals([
    ...categorySignals(signals, "active_role_work"),
    ...nowSignals,
  ], 4);
  const activeCount = activeWorkSignals.length;
  const automationAttentionCount = automationHealth?.summary?.attention ?? 0;
  const totalAttentionCount = attentionSignals.length + automationAttentionCount;
  const riskySendEnabled = externalSendEnabled && riskyAllowedLanes.length > 0;
  const activeBiffMode = dashboardStatus?.biff_operating_mode;
  const biffModeName = activeBiffMode?.name ?? "normal";
  const biffModeLabel = activeBiffMode?.label ?? "Normal";
  const healthyChecks = (n8nChecks?.checks ?? []).filter((check) => statusTone(check.execution_status || check.status) === "success").length;
  const safeActionLabel = riskySendEnabled ? "External Discord #hermes send is available only behind the explicit Actions gate." : "No external action is armed. Review-only panels are safe to read.";

  const handleInvestigateHealthCard = useCallback((card: CockpitAutomationHealthCard) => {
    const prompt = [
      `[Cockpit Health action] Investigate and fix: ${safeText(card.title)}`,
      `Status: ${automationBucketLabel(card.bucket)}`,
      `Summary: ${boundedCopy(card.summary, 320)}`,
      card.details?.length ? `Details: ${card.details.map((detail) => boundedCopy(detail, 120)).join("; ")}` : "Details: none shown",
      `Source: ${safeText(card.source)} · last checked ${safeText(card.last_checked)}`,
      "Please investigate the root cause, make only safe/low-risk fixes directly, and ask before destructive, external-send, credential/access, restart, deploy, git merge, or broad automation changes.",
    ].join("\n");
    setHealthActionPrompt(prompt);
    setHealthActionCopied(false);
    void navigator.clipboard?.writeText(prompt).then(() => setHealthActionCopied(true)).catch(() => setHealthActionCopied(false));
    setActiveSection("local-chat");
  }, []);

  const copyHealthActionPrompt = useCallback(() => {
    if (!healthActionPrompt) return;
    void navigator.clipboard?.writeText(healthActionPrompt).then(() => setHealthActionCopied(true)).catch(() => setHealthActionCopied(false));
  }, [healthActionPrompt]);

  return (
    <div
      className={cn(
        "cockpit-typography relative isolate min-h-[calc(100vh-9rem)] overflow-x-hidden overflow-y-auto text-[var(--cockpit-text)] normal-case",
        standalone && "min-h-dvh",
      )}
      style={{
        background: "var(--cockpit-shell)",
        paddingTop: standalone ? "max(16px, env(safe-area-inset-top))" : undefined,
        paddingBottom: standalone ? "max(18px, env(safe-area-inset-bottom))" : undefined,
      }}
    >
      <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_18%_0%,rgba(121,216,196,0.13),transparent_30%),radial-gradient(circle_at_88%_12%,rgba(141,134,201,0.14),transparent_32%),linear-gradient(180deg,var(--cockpit-shell)_0%,#0a0806_56%,#030202_100%)]" />
      <div className="pointer-events-none absolute inset-x-6 top-0 -z-10 h-px bg-gradient-to-r from-transparent via-[color-mix(in_srgb,var(--cockpit-active)_50%,transparent)] to-transparent" />
      <PluginSlot name="cockpit:top" />

      <div className={cn("mx-auto flex w-full max-w-6xl flex-col gap-4 px-3 sm:gap-5 sm:px-6", standalone ? "pb-5" : "py-2")}>
        <header className="flex max-w-full flex-col items-start gap-3 rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-card-raised)] px-3 py-3 shadow-[0_20px_70px_var(--cockpit-shadow)] sm:flex-row sm:items-center sm:justify-between sm:gap-4 sm:rounded-[2rem] sm:px-4" data-testid="cockpit-iphone-safe-header">
          <div className="min-w-0 max-w-full">
            <p className="cockpit-micro-label flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[0.22em] text-[color-mix(in_srgb,var(--cockpit-active)_82%,transparent)] sm:text-[10px] sm:tracking-[0.28em]"><Activity className="h-3.5 w-3.5" /> Biff Cockpit</p>
            <h1 className="cockpit-title mt-1 max-w-full truncate text-xl font-semibold tracking-tight text-[var(--cockpit-text)] sm:text-2xl">Now command surface</h1>
          </div>
          <div className="grid w-full min-w-0 grid-cols-2 gap-2 sm:flex sm:w-auto sm:flex-wrap sm:items-center sm:justify-end">
            <Badge tone="success" className="min-w-0 justify-center gap-1 truncate text-[9px] uppercase tracking-[0.12em] sm:text-[10px] sm:tracking-[0.18em]">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
              Live
            </Badge>
            <Badge tone={inputEnabled ? "success" : "secondary"} className="min-w-0 justify-center gap-1 truncate text-[9px] uppercase tracking-[0.12em] sm:text-[10px] sm:tracking-[0.18em]">
              <MessageCircle className="h-3 w-3" />
              <span className="sm:hidden">Chat {inputEnabled ? "on" : "wait"}</span><span className="hidden sm:inline">Local chat {inputEnabled ? "enabled" : "pending"}</span>
            </Badge>
            <Badge tone="secondary" className="min-w-0 justify-center gap-1 truncate text-[9px] uppercase tracking-[0.12em] sm:text-[10px] sm:tracking-[0.18em]">
              <ShieldCheck className="h-3 w-3" />
              <span className="sm:hidden">Send {riskySendEnabled ? "gated" : "off"}</span><span className="hidden sm:inline">{riskySendEnabled ? "External send gated" : "External actions off"}</span>
            </Badge>
            <Badge tone="secondary" className="min-w-0 justify-center gap-1 truncate text-[9px] uppercase tracking-[0.12em] sm:text-[10px] sm:tracking-[0.18em]">
              <span className="sm:hidden">Win{transcriptWindow?.window_limit ? ` · ${transcriptWindow.window_limit}` : ""}</span><span className="hidden sm:inline">Recent window{transcriptWindow?.window_limit ? ` · ${transcriptWindow.window_limit}` : ""}</span>
            </Badge>
            <Badge tone={biffModeName === "normal" ? "secondary" : "warning"} className="min-w-0 justify-center gap-1 truncate text-[9px] uppercase tracking-[0.12em] sm:text-[10px] sm:tracking-[0.18em]" data-testid="cockpit-biff-mode-badge">
              <span className="sm:hidden">Mode {biffModeLabel}</span><span className="hidden sm:inline">Biff mode · {biffModeLabel}</span>
            </Badge>
          </div>
        </header>

        {error && (
          <div className="flex items-center gap-3 rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-risk)_35%,transparent)] bg-[var(--cockpit-risk-soft)] p-4 text-sm text-[color-mix(in_srgb,var(--cockpit-risk)_82%,var(--cockpit-text)_18%)]">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        )}

        <div className="grid gap-5 lg:grid-cols-[220px_minmax(0,1fr)]">
          <nav aria-label="Cockpit sections" className="hidden lg:block">
            <div className="sticky top-4 grid gap-2 rounded-[2rem] border border-[var(--cockpit-border)] bg-[var(--cockpit-card-raised)] p-3 shadow-[0_20px_70px_var(--cockpit-shadow)]" data-testid="cockpit-desktop-section-rail">
              {COCKPIT_SECTIONS.map((section) => {
                const Icon = sectionIcon[section.key];
                const active = activeSection === section.key;
                return (
                  <button
                    key={section.key}
                    type="button"
                    data-section-target={section.key}
                    aria-current={active ? "page" : undefined}
                    onClick={() => setActiveSection(section.key)}
                    className={cn(
                      "rounded-2xl border px-3 py-3 text-left transition",
                      active ? "border-[color-mix(in_srgb,var(--cockpit-active)_55%,transparent)] bg-[var(--cockpit-active-soft)] text-[var(--cockpit-text)]" : "border-[var(--cockpit-border)] bg-[color-mix(in_srgb,var(--cockpit-card)_82%,transparent)] text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)] hover:border-[color-mix(in_srgb,var(--cockpit-active)_35%,transparent)]",
                    )}
                  >
                    <span className="flex items-center gap-2 text-sm font-semibold"><Icon className="h-4 w-4" />{section.shortLabel}</span>
                  </button>
                );
              })}
            </div>
          </nav>

          <div className="min-w-0">
            <nav className="mb-4 max-w-full lg:hidden" data-testid="cockpit-compact-section-nav" aria-label="Cockpit sections">
              <div className="mb-2 flex items-center justify-between gap-2 text-[10px] uppercase tracking-[0.16em] text-[var(--cockpit-muted)]">
                <span>Sections</span>
                <span className="normal-case tracking-normal text-[var(--cockpit-muted)]">{COCKPIT_SECTIONS.find((section) => section.key === activeSection)?.label}</span>
              </div>
              <div className="grid grid-cols-4 gap-1.5 rounded-3xl border border-[var(--cockpit-border)] bg-[color-mix(in_srgb,var(--cockpit-shell-raised)_88%,transparent)] p-1.5" data-testid="cockpit-mobile-section-buttons">
                {COCKPIT_SECTIONS.map((section) => {
                  const Icon = sectionIcon[section.key];
                  const active = activeSection === section.key;
                  return (
                    <button
                      key={section.key}
                      type="button"
                      aria-label={section.label}
                      aria-current={active ? "page" : undefined}
                      onClick={() => setActiveSection(section.key)}
                      className={cn(
                        "flex min-w-0 flex-col items-center justify-center gap-1 rounded-2xl px-1 py-2 text-[10px] leading-none transition",
                        active ? "bg-[color-mix(in_srgb,var(--cockpit-active)_18%,transparent)] text-[var(--cockpit-text)] shadow-[0_0_18px_color-mix(in_srgb,var(--cockpit-active)_16%,transparent)]" : "text-[var(--cockpit-muted)] hover:bg-[color-mix(in_srgb,var(--cockpit-card-raised)_80%,var(--cockpit-active)_7%)] hover:text-[var(--cockpit-text)]",
                      )}
                    >
                      <Icon className="h-3.5 w-3.5" />
                      <span className="max-w-full truncate">{section.shortLabel}</span>
                    </button>
                  );
                })}
              </div>
            </nav>

            <main className="min-w-0" data-active-section={activeSection}>
              {activeSection === "overview" && (
                <section data-testid="cockpit-section-overview" className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                  <SurfaceCard eyebrow="Command brief" title="What needs Marco attention now">
                    <div className="grid gap-4">
                      <div className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_20%,transparent)] bg-[var(--cockpit-active-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-active)_85%,var(--cockpit-text)_15%)]">
                        Read top to bottom: Attention, Recent changes, Active work, System health, then Safe gated action. {formatFreshness(lastRefresh)}.
                      </div>
                      {dashboardStatus?.session_quota_recommendation && (
                        <div className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_34%,transparent)] bg-[var(--cockpit-warning-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-warning)_88%,var(--cockpit-text)_12%)]" data-testid="cockpit-session-quota-recommendation">
                          <span className="font-semibold text-[var(--cockpit-text)]">Session quota:</span> {safeText(dashboardStatus.session_quota_recommendation.text)}
                        </div>
                      )}
                      {activeBiffMode && biffModeName !== "normal" && (
                        <div className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_34%,transparent)] bg-[var(--cockpit-warning-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-warning)_88%,var(--cockpit-text)_12%)]" data-testid="cockpit-biff-mode-card">
                          <span className="font-semibold text-[var(--cockpit-text)]">Biff mode:</span> {safeText(biffModeLabel)} — {safeText(activeBiffMode.description)}
                        </div>
                      )}
                      <div className="grid gap-3" aria-label="Attention">
                        <div className="flex items-center justify-between gap-3">
                          <h3 className="text-sm font-semibold text-[var(--cockpit-text)]">Attention</h3>
                          <Badge tone={totalAttentionCount ? "warning" : "success"} className="text-[10px] uppercase tracking-[0.18em]">{totalAttentionCount ? `${totalAttentionCount} item${totalAttentionCount === 1 ? "" : "s"}` : "clear"}</Badge>
                        </div>
                        {attentionSignals.length ? attentionSignals.map((signal) => <BriefSignalSummary key={`attention-${signal.id}`} signal={signal} onSelect={() => setSelectedLaneId(signal.lane_id)} />) : automationAttentionCount ? <button type="button" onClick={() => setActiveSection("automation-health")} className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_36%,transparent)] bg-[var(--cockpit-warning-soft)] p-4 text-left text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-warning)_86%,var(--cockpit-text)_14%)] transition hover:border-[color-mix(in_srgb,var(--cockpit-warning)_54%,transparent)]"><span className="block font-semibold text-[var(--cockpit-text)]">Automation Health needs review</span><span className="mt-1 block">{automationHealth?.summary?.headline || `${automationAttentionCount} automation area${automationAttentionCount === 1 ? "" : "s"} need Marco review.`}</span></button> : <p className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">Nothing is asking for Marco right now. No waiting, stuck, failed, or automation-health review item appears in the bounded command window.</p>}
                      </div>
                      <div className="grid gap-3" aria-label="Recent changes">
                        <h3 className="text-sm font-semibold text-[var(--cockpit-text)]">Recent changes — What changed recently</h3>
                        {recentChangeSignals.length ? recentChangeSignals.map((signal) => <BriefSignalSummary key={`recent-${signal.category}-${signal.id}`} signal={signal} onSelect={() => setSelectedLaneId(signal.lane_id)} />) : <p className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">No meaningful change is visible in the recent command window yet.</p>}
                      </div>
                    </div>
                  </SurfaceCard>

                  <div className="grid gap-5">
                    <SurfaceCard eyebrow="Brief status" title="What is running and healthy">
                      <div className="grid gap-4">
                        <div className="grid gap-3 sm:grid-cols-2">
                          <div className="rounded-3xl bg-[var(--cockpit-shell-raised)] p-4"><p className="text-[10px] uppercase tracking-[0.22em] text-[var(--cockpit-muted)]">Active work</p><p className="mt-2 text-3xl font-semibold text-[var(--cockpit-text)]">{activeCount}</p><p className="mt-1 text-xs text-[var(--cockpit-muted)]">running or recently active lanes</p></div>
                          <div className="rounded-3xl bg-[var(--cockpit-shell-raised)] p-4"><p className="text-[10px] uppercase tracking-[0.22em] text-[var(--cockpit-muted)]">System health</p><p className="mt-2 text-3xl font-semibold text-[var(--cockpit-text)]">{loading ? "Sync" : "Live"}</p><p className="mt-1 text-xs text-[var(--cockpit-muted)]">dashboard reads only; local chat {inputEnabled ? "ready" : "pending"}</p></div>
                        </div>
                        <div className="grid gap-3" aria-label="Active work">
                          <h3 className="text-sm font-semibold text-[var(--cockpit-text)]">Active work</h3>
                          {activeWorkSignals.length ? activeWorkSignals.map((signal) => <BriefSignalSummary key={`active-${signal.category}-${signal.id}`} signal={signal} onSelect={() => setSelectedLaneId(signal.lane_id)} />) : <p className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">No active work is visible in the bounded command window.</p>}
                        </div>
                        <div className="grid gap-2 rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4" aria-label="System health">
                          <div className="flex flex-wrap items-center gap-2">
                            {loading ? <Badge tone="secondary" className="gap-1 text-[10px] uppercase tracking-[0.18em]"><Loader2 className="h-3 w-3 animate-spin" /> Syncing</Badge> : <Badge tone="success" className="gap-1 text-[10px] uppercase tracking-[0.18em]"><CheckCircle2 className="h-3 w-3" /> Live</Badge>}
                            <Badge tone={capabilityReadOnly ? "secondary" : "success"} className="text-[10px] uppercase tracking-[0.18em]">Dashboard {capabilityReadOnly ? "read-only" : "input-capable"}</Badge>
                            <Badge tone={riskySendEnabled ? "warning" : "secondary"} className="text-[10px] uppercase tracking-[0.18em]">External send {riskySendEnabled ? "gated" : "off"}</Badge>
                          </div>
                          <p className="text-xs leading-5 text-[var(--cockpit-muted)]">{lanes.length} visible lanes. {healthyChecks} daily checks currently report healthy. Last refresh {lastRefresh ? formatClock(lastRefresh.toISOString()) : "pending"}.</p>
                        </div>
                      </div>
                    </SurfaceCard>

                    <SelfWorkHandoffPanel response={selfWorkHandoff} />

                    <AgentActivityPanel response={agentActivity} onSelect={setSelectedLaneId} />

                    <SurfaceCard eyebrow="Safe gated action" title="Safe gated action">
                      <div className="grid gap-3">
                        <p className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-text)_78%,transparent)]">{safeActionLabel}</p>
                        <button type="button" onClick={() => setActiveSection("tools")} className="rounded-2xl border border-[color-mix(in_srgb,var(--cockpit-active)_42%,transparent)] bg-[color-mix(in_srgb,var(--cockpit-active)_12%,transparent)] px-4 py-3 text-left text-sm font-semibold text-[var(--cockpit-text)] transition hover:bg-[color-mix(in_srgb,var(--cockpit-active)_18%,transparent)]">
                          {riskySendEnabled ? "Open Actions / explicit send gate" : "Open Actions / read-only upgrade review"}
                        </button>
                      </div>
                    </SurfaceCard>
                  </div>
                </section>
              )}

              <section
                data-testid="cockpit-section-local-chat"
                data-cockpit-section-active={activeSection === "local-chat" ? "true" : "false"}
                hidden={activeSection !== "local-chat"}
                aria-hidden={activeSection !== "local-chat"}
                className="grid gap-5"
              >
                <SurfaceCard eyebrow="Primary local Hermes chat" title="Local Chat — PTY command surface">
                  <div className="max-w-full overflow-hidden rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_32%,transparent)] bg-[color-mix(in_srgb,var(--cockpit-shell-raised)_88%,transparent)] p-3 shadow-[0_0_36px_color-mix(in_srgb,var(--cockpit-active)_12%,transparent)] sm:rounded-[1.75rem] sm:p-4" data-testid="cockpit-iphone-local-chat-shell">
                    <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-center">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[var(--cockpit-active-soft)] text-[var(--cockpit-active)] sm:h-11 sm:w-11"><MessageCircle className="h-5 w-5" /></div>
                      <div className="min-w-0"><p className="break-words font-semibold text-[var(--cockpit-text)]">Dominant local PTY-backed Hermes chat</p><p className="mt-1 break-words text-xs leading-5 text-[var(--cockpit-muted)] sm:text-sm">Messages stay in the local dashboard chat route. External delivery, new routing targets, attachments, and audio/voice paths are unavailable here.</p></div>
                    </div>
                    {healthActionPrompt && (
                      <div className="mt-4 grid gap-3 rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_36%,transparent)] bg-[var(--cockpit-warning-soft)] p-4" data-testid="cockpit-health-action-prompt">
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                          <div>
                            <p className="text-sm font-semibold text-[var(--cockpit-text)]">Health action ready</p>
                            <p className="mt-1 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-warning)_82%,var(--cockpit-text)_18%)]">Paste this into Local Chat to have Biff investigate. It does not run automatically.</p>
                          </div>
                          <button type="button" onClick={copyHealthActionPrompt} className="rounded-2xl border border-[color-mix(in_srgb,var(--cockpit-warning)_45%,transparent)] bg-[var(--cockpit-shell-raised)] px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--cockpit-text)]">
                            {healthActionCopied ? "Copied" : "Copy prompt"}
                          </button>
                        </div>
                        <pre className="max-h-36 overflow-auto whitespace-pre-wrap rounded-2xl bg-[var(--cockpit-shell)] p-3 text-[11px] leading-5 text-[var(--cockpit-muted)]">{healthActionPrompt}</pre>
                      </div>
                    )}
                    <div className="mt-4 flex h-[min(68dvh,760px)] min-h-[420px] max-w-full flex-col overflow-hidden rounded-2xl border border-[var(--cockpit-border)] bg-[var(--cockpit-shell)] p-1.5 normal-case sm:min-h-[560px] sm:p-2">
                      <ChatPage isActive={activeSection === "local-chat"} sessionQuotaRecommendation={dashboardStatus?.session_quota_recommendation ?? null} />
                    </div>
                  </div>
                </SurfaceCard>
              </section>

              {activeSection === "automation-health" && (
                <section data-testid="cockpit-section-automation-health" className="grid gap-5">
                  <AutomationHealthPanel response={automationHealth} loading={automationHealthLoading} error={automationHealthError} onInvestigate={handleInvestigateHealthCard} />
                  <details className="group rounded-[2rem] border border-[var(--cockpit-border)] bg-[var(--cockpit-card)] p-4 shadow-[0_20px_70px_rgba(0,0,0,0.24)]" data-testid="cockpit-health-n8n-details">
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-semibold text-[var(--cockpit-text)]">
                      <span className="flex items-center gap-2"><Radar className="h-4 w-4 text-[var(--cockpit-active)]" /> Daily n8n checks</span>
                      <span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">{healthyChecks}/{n8nChecks?.checks?.length ?? 0} healthy<ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" aria-hidden="true" /></span>
                    </summary>
                    <div className="mt-4">
                      <N8nChecksPanel response={n8nChecks} loading={n8nChecksLoading} error={n8nChecksError} />
                    </div>
                  </details>
                </section>
              )}

              {activeSection === "tools" && (
                <section data-testid="cockpit-section-tools" className="grid gap-5">
                  <SurfaceCard eyebrow="Actions" title="Actions that need confirmation">
                    <div className="grid gap-3">
                      <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_30%,transparent)] bg-[var(--cockpit-warning-soft)] p-4 text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-warning)_78%,var(--cockpit-text)_22%)]">
                        External sends, upgrade reviews, and debug details live here so the Brief stays clean. Nothing here runs automatically; anything that can affect the outside world requires an explicit confirmation click.
                      </p>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <Badge tone={inputEnabled ? "success" : "secondary"} className="justify-center py-2 text-[10px] uppercase tracking-[0.18em]">Local chat {inputEnabled ? "enabled" : "pending"}</Badge>
                        <Badge tone={riskySendEnabled ? "warning" : "secondary"} className="justify-center py-2 text-[10px] uppercase tracking-[0.18em]"><Send className="h-3 w-3" /> External Discord #hermes {riskySendEnabled ? "gated" : "disabled"}</Badge>
                        <Badge tone={capabilityReadOnly ? "secondary" : "success"} className="justify-center py-2 text-[10px] uppercase tracking-[0.18em]">Observer {capabilityReadOnly ? "read-only" : "input-capable"}</Badge>
                        <Badge tone="secondary" className="justify-center py-2 text-[10px] uppercase tracking-[0.18em]">No voice · no attachments · no new targets</Badge>
                      </div>
                      <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-active)_20%,transparent)] bg-[var(--cockpit-active-soft)] p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-active)_85%,var(--cockpit-text)_15%)]">{formatFreshness(lastRefresh)}. Local Chat handles PWA visibility/pageshow/pagehide reconnect without requiring manual refresh as normal recovery.</p>
                    </div>
                  </SurfaceCard>

                  <details open className="group rounded-[2rem] border border-[var(--cockpit-border)] bg-[var(--cockpit-card)] p-4 shadow-[0_20px_70px_rgba(0,0,0,0.32)]" data-testid="cockpit-tools-external-send-details">
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-semibold text-[var(--cockpit-text)]"><span className="flex items-center gap-2"><Send className="h-4 w-4 text-[var(--cockpit-warning)]" /> External Send / Safe Action</span><span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">{riskySendEnabled ? "gated" : "disabled"}<ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" aria-hidden="true" /></span></summary>
                    <div className="mt-4 grid gap-4">
                      <p className="rounded-3xl border border-[color-mix(in_srgb,var(--cockpit-warning)_30%,transparent)] bg-[var(--cockpit-warning-soft)] p-4 text-sm leading-6 text-[color-mix(in_srgb,var(--cockpit-warning)_78%,var(--cockpit-text)_22%)]">
                        Secondary surface: explicit external Discord #hermes send only. This preserves BIF-516 gates; no voice, attachments, new routing targets, or broad send behavior were added.
                      </p>
                      <RiskySendComposer enabled={riskySendEnabled} allowedLanes={riskyAllowedLanes} />
                    </div>
                  </details>

                  <details className="group rounded-[2rem] border border-[var(--cockpit-border)] bg-[var(--cockpit-card)] p-4 shadow-[0_20px_70px_rgba(0,0,0,0.24)]" data-testid="cockpit-tools-upgrade-radar-details">
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-semibold text-[var(--cockpit-text)]"><span className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-[var(--cockpit-active)]" /> Upgrade Radar / review gate</span><span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">read-only<ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" aria-hidden="true" /></span></summary>
                    <div className="mt-4">
                      <DailyOpsRadarPanel response={dailyOpsRadar} loading={loading} error={dailyOpsRadarError} />
                    </div>
                  </details>

                  <details className="group rounded-[2rem] border border-[var(--cockpit-border)] bg-[var(--cockpit-card)] p-4 shadow-[0_20px_70px_rgba(0,0,0,0.24)]" data-testid="cockpit-tools-archive-context-details">
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-semibold text-[var(--cockpit-text)]"><span className="flex items-center gap-2"><ChevronDown className="h-4 w-4 text-[var(--cockpit-active)]" /> Archive/Context</span><span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">{recentContextSignals.length} signals<ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" aria-hidden="true" /></span></summary>
                    <div className="mt-4 grid gap-3">
                      <p className="rounded-3xl border border-amber-200/15 bg-amber-400/10 p-3 text-xs leading-5 text-[color-mix(in_srgb,var(--cockpit-warning)_78%,var(--cockpit-text)_22%)]/85">These records are not presented as current operations. They come from recent_context/archive buckets and stay separated from Local Chat.</p>
                      {recentContextSignals.length ? recentContextSignals.map((signal) => <SignalSummary key={`${signal.category}-${signal.id}`} signal={signal} onSelect={() => setSelectedLaneId(signal.lane_id)} />) : <p className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-4 text-sm text-[color-mix(in_srgb,var(--cockpit-text)_76%,transparent)]">No stale SessionDB context is visible in the bounded recent cockpit window.</p>}
                    </div>
                  </details>

                  <details className="group rounded-[2rem] border border-[var(--cockpit-border)] bg-[var(--cockpit-card)] p-4 shadow-[0_20px_70px_rgba(0,0,0,0.24)]" data-testid="cockpit-tools-settings-health-details">
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-semibold text-[var(--cockpit-text)]"><span className="flex items-center gap-2"><Settings className="h-4 w-4 text-[var(--cockpit-active)]" /> Diagnostics</span><span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-[var(--cockpit-muted)]">support details<ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" aria-hidden="true" /></span></summary>
                    <div className="mt-4 grid gap-4">
                      <p className="rounded-3xl border border-[var(--cockpit-border)] bg-[var(--cockpit-panel)] p-3 text-xs leading-5 text-[var(--cockpit-muted)]">For debugging Cockpit/Biff when something looks wrong. You normally do not need this for daily reading; it only shows bounded lane context, recent-window metadata, and display-safe runtime details.</p>
                      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.9fr)]">
                        <LaneDetailPanel lane={selectedLane} response={selectedLaneMessages} transcriptWindow={transcriptWindow} loading={selectedLaneLoading} error={selectedLaneError} />
                        <div className="grid max-h-[620px] gap-3 overflow-auto pr-1">
                          {loading && lanes.length === 0 && <div className="flex h-48 items-center justify-center text-[var(--cockpit-muted)]"><Spinner className="mr-2 text-[var(--cockpit-active)]" /> Loading cockpit lanes…</div>}
                          {!loading && lanes.length === 0 && <div className="rounded-3xl border border-dashed border-[var(--cockpit-border)] p-8 text-center text-sm text-[var(--cockpit-muted)]"><Satellite className="mx-auto mb-3 h-8 w-8" />No display-safe lanes are currently visible in the recent bounded window.</div>}
                          {lanes.map((lane) => <div key={lane.lane_id} role="tab" aria-selected={selectedLane?.lane_id === lane.lane_id} tabIndex={0} onClick={() => setSelectedLaneId(lane.lane_id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelectedLaneId(lane.lane_id); }} className="cursor-pointer rounded-3xl outline-none focus-visible:ring-2 focus-visible:ring-[color-mix(in_srgb,var(--cockpit-active)_50%,transparent)]"><LaneSignal lane={lane} selected={selectedLane?.lane_id === lane.lane_id} /></div>)}
                        </div>
                      </div>
                    </div>
                  </details>
                </section>
              )}
            </main>
          </div>
        </div>
      </div>
    </div>
  );
}
