import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  Building2,
  Clock,
  Database,
  Eye,
  Filter,
  Lock,
  MapPinned,
  RefreshCw,
  Route,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type OfficeDataSource, type OfficeSourceStatus, type OfficeState } from "@/lib/api";
import { buildOfficeAttentionItems, groupByText, numberField, textField, visibleRows } from "./officeView";

const FOCUS_OPTIONS = ["overview", "work", "automation", "routing"] as const;
const LIST_LIMIT = 6;
const EVENT_LIMIT = 12;
type FocusOption = (typeof FOCUS_OPTIONS)[number];

type InspectorSelection = {
  kind: string;
  title: string;
  fields: Array<[string, string]>;
};

function fmt(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return new Date(value * 1000).toLocaleString();
  if (typeof value !== "string") return String(value);
  if (/^\d{4}-\d{2}-\d{2}T/.test(value)) return new Date(value).toLocaleString();
  return value;
}

const SOURCE_TONE: Record<OfficeSourceStatus, string> = {
  ok: "border-emerald-400/40 text-emerald-300",
  partial: "border-yellow-400/40 text-yellow-300",
  missing: "border-sky-400/40 text-sky-300",
  unavailable: "border-zinc-400/40 text-zinc-300",
  error: "border-red-400/40 text-red-300",
};

const SOURCE_LABEL: Record<OfficeSourceStatus, string> = {
  ok: "ready",
  partial: "partial",
  missing: "not connected",
  unavailable: "unavailable",
  error: "error",
};

function StatusPill({ status }: { status: OfficeSourceStatus | string }) {
  const tone = SOURCE_TONE[status as OfficeSourceStatus] ?? "border-zinc-400/40 text-zinc-300";
  const label = SOURCE_LABEL[status as OfficeSourceStatus] ?? status;
  return (
    <span className={`whitespace-nowrap border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] ${tone}`}>
      {label}
    </span>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-midground/60">{children}</div>;
}

function SourceCard({ source, onInspect }: { source: OfficeDataSource; onInspect: () => void }) {
  return (
    <div className="border border-current/15 bg-black/20 p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-foreground">{source.id}</span>
        <StatusPill status={source.status} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-midground/75">
        <div>
          <div className="text-midground/45">items</div>
          <div className="text-foreground">{source.item_count ?? "—"}</div>
        </div>
        <div>
          <div className="text-midground/45">warnings</div>
          <div className={source.warning_count ? "text-yellow-300" : "text-foreground"}>{source.warning_count ?? 0}</div>
        </div>
      </div>
      {source.error_summary ? (
        <div className="mt-3 border border-red-400/30 bg-red-950/20 p-2 text-xs text-red-300/90">{source.error_summary}</div>
      ) : null}
      <button type="button" onClick={onInspect} className="mt-3 flex items-center gap-1 text-xs uppercase tracking-[0.16em] text-midground/70 hover:text-foreground">
        <Eye className="h-3 w-3" /> inspect
      </button>
    </div>
  );
}

function EmptyLine({ label, hint }: { label: string; hint?: string }) {
  return (
    <div className="border border-dashed border-current/15 bg-black/10 p-4 text-sm text-midground/65">
      <div>No {label} in the redacted OfficeState DTO.</div>
      {hint ? <div className="mt-2 text-xs leading-5 text-midground/50">{hint}</div> : null}
    </div>
  );
}

function StatCard({ label, value, detail, tone = "text-foreground" }: { label: string; value: unknown; detail: string; tone?: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-midground/70">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className={`text-3xl font-semibold ${tone}`}>{String(value ?? 0)}</div>
        <div className="mt-2 text-xs text-midground/60">{detail}</div>
      </CardContent>
    </Card>
  );
}

function EntityRow({
  title,
  meta,
  badge,
  warning,
  onInspect,
}: {
  title: string;
  meta: string;
  badge?: string;
  warning?: string | null;
  onInspect?: () => void;
}) {
  return (
    <div className="border border-current/15 bg-black/15 p-3 text-sm">
      <div className="flex items-start justify-between gap-3">
        <span className="font-semibold text-foreground">{title}</span>
        {badge ? <span className="shrink-0 text-xs text-midground/70">{badge}</span> : null}
      </div>
      <div className="mt-1 text-xs text-midground/70">{meta}</div>
      {warning ? <div className="mt-2 border border-red-400/30 bg-red-950/20 p-2 text-xs text-red-300">{warning}</div> : null}
      {onInspect ? (
        <button type="button" onClick={onInspect} className="mt-3 flex items-center gap-1 text-xs uppercase tracking-[0.16em] text-midground/70 hover:text-foreground">
          <Eye className="h-3 w-3" /> inspect
        </button>
      ) : null}
    </div>
  );
}

function MiniList({
  title,
  icon,
  children,
  meta,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  meta?: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          {icon}
          {title}
        </CardTitle>
        {meta ? <div className="text-xs text-midground/55">{meta}</div> : null}
      </CardHeader>
      <CardContent className="space-y-3">{children}</CardContent>
    </Card>
  );
}

function GroupBlock({ title, count, children }: { title: string; count: number; children: React.ReactNode }) {
  return (
    <div className="border border-current/10 bg-black/10 p-3">
      <div className="mb-3 flex items-center justify-between gap-3 text-xs uppercase tracking-[0.16em] text-midground/65">
        <span>{title}</span>
        <span>{count}</span>
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function LimitedRows<T>({
  rows,
  limit = LIST_LIMIT,
  label,
  children,
}: {
  rows: T[];
  limit?: number;
  label: string;
  children: (row: T) => React.ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  const visible = visibleRows(rows, limit, expanded);
  const hidden = Math.max(rows.length - visible.length, 0);
  return (
    <>
      {visible.map((row) => children(row))}
      {rows.length > limit ? (
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="w-full border border-dashed border-current/20 bg-black/10 px-3 py-2 text-left text-xs uppercase tracking-[0.16em] text-midground/65 hover:text-foreground"
        >
          {expanded ? `show fewer ${label}` : `show ${hidden} more ${label}`}
        </button>
      ) : null}
    </>
  );
}

function InspectorPanel({ selection }: { selection: InspectorSelection | null }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Eye className="h-4 w-4" /> Safe inspector
        </CardTitle>
      </CardHeader>
      <CardContent>
        {selection ? (
          <div className="space-y-3 text-sm">
            <div>
              <SectionLabel>{selection.kind}</SectionLabel>
              <div className="mt-1 font-semibold text-foreground">{selection.title}</div>
            </div>
            <div className="grid gap-2">
              {selection.fields.map(([label, value]) => (
                <div key={label} className="grid grid-cols-[8rem_1fr] gap-3 border border-current/10 bg-black/15 p-2 text-xs">
                  <span className="text-midground/50">{label}</span>
                  <span className="break-words text-midground/85">{value}</span>
                </div>
              ))}
            </div>
            <div className="border border-emerald-400/20 bg-emerald-950/10 p-3 text-xs leading-5 text-emerald-200/80">
              Inspector shows DTO metadata only. Raw prompts, transcripts, task bodies, cron scripts, logs, auth, and secrets remain omitted.
            </div>
          </div>
        ) : (
          <div className="border border-dashed border-current/15 bg-black/10 p-4 text-sm text-midground/65">
            Select Inspect on a source, room, session, work item, automation, topic, or event to see safe metadata here.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function OfficePage() {
  const [state, setState] = useState<OfficeState | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focus, setFocus] = useState<FocusOption>("overview");
  const [selection, setSelection] = useState<InspectorSelection | null>(null);

  const load = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const next = await api.getOfficeState();
      setState(next);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .getOfficeState()
      .then((next) => {
        if (!cancelled) setState(next);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const inspectRecord = useCallback((kind: string, title: string, fields: Array<[string, string]>) => {
    setSelection({ kind, title, fields });
  }, []);

  const needsAttention = useMemo(() => (state ? buildOfficeAttentionItems(state) : []), [state]);

  const sourceCounts = useMemo(() => {
    if (!state) return { ok: 0, partial: 0, missing: 0, unavailable: 0, error: 0 };
    return state.data_sources.reduce<Record<OfficeSourceStatus, number>>(
      (acc, source) => {
        acc[source.status] += 1;
        return acc;
      },
      { ok: 0, partial: 0, missing: 0, unavailable: 0, error: 0 },
    );
  }, [state]);

  const workGroups = useMemo(() => (state ? groupByText(state.work_items, "status", "unknown") : {}), [state]);
  const automationGroups = useMemo(() => (state ? groupByText(state.automations, "state", "unknown") : {}), [state]);

  const showOverview = focus === "overview";
  const showWork = focus === "overview" || focus === "work";
  const showAutomation = focus === "overview" || focus === "automation";
  const showRouting = focus === "overview" || focus === "routing";

  if (loading) {
    return (
      <div className="flex min-h-[420px] flex-col items-center justify-center gap-4 border border-current/15 bg-black/10 py-24">
        <Spinner className="text-2xl text-primary" />
        <div className="text-sm uppercase tracking-[0.2em] text-midground/70">Loading redacted OfficeState</div>
      </div>
    );
  }

  if (error || !state) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base text-red-300">
            <AlertTriangle className="h-4 w-4" /> Office unavailable
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-midground/80">
          <p>{error ?? "No state returned"}</p>
          <p className="text-xs text-midground/55">The page could not read the protected OfficeState DTO. This fallback still avoids exposing raw logs or secrets.</p>
          <Button onClick={load} className="gap-2 uppercase">
            <RefreshCw className="h-4 w-4" /> Try again
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-6 normal-case">
      <div className="border border-current/20 bg-gradient-to-br from-black/35 to-black/10 p-5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="max-w-3xl">
            <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.22em] text-emerald-300">
              <ShieldCheck className="h-4 w-4" /> Read-only MVP · localhost first
            </div>
            <h1 className="mt-3 text-3xl font-semibold uppercase tracking-wide text-foreground md:text-4xl">Hermes AI Office</h1>
            <p className="mt-3 text-sm leading-6 text-midground/80">
              A redacted operational map of this Mac-local Hermes runtime. It shows state, health, and provenance gaps without exposing raw sessions, prompts, logs, or secrets.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {FOCUS_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setFocus(option)}
                  className={`border px-3 py-1 text-xs uppercase tracking-[0.16em] ${focus === option ? "border-emerald-400/50 text-emerald-300" : "border-current/20 text-midground/70 hover:text-foreground"}`}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
          <div className="min-w-64 border border-current/15 bg-black/20 p-3 text-xs text-midground/70">
            <div className="flex items-center gap-2 text-foreground">
              <Lock className="h-4 w-4 text-emerald-300" /> Safe mode
            </div>
            <div className="mt-2 grid gap-1">
              <div>Generated: {fmt(state.generated_at)}</div>
              <div>Display: {state.display_mode}</div>
              <div>Remote: {state.capabilities.remote_mode}</div>
              <div>Mutations: {state.capabilities.mutations_enabled ? "enabled" : "absent"}</div>
            </div>
            <Button onClick={load} className="mt-4 w-full gap-2 uppercase" disabled={refreshing}>
              <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} /> Refresh
            </Button>
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <StatCard label="Active work" value={state.summary.active_work_count ?? 0} detail="Open work items surfaced by approved adapters" />
        <StatCard label="Needs attention" value={needsAttention.length} detail="Blocked work, source warnings, or failed automations" tone={needsAttention.length > 0 ? "text-yellow-300" : "text-emerald-300"} />
        <StatCard label="Automations" value={state.summary.automation_count ?? state.automations.length} detail="Cron-style jobs represented as read-only machines" />
        <StatCard label="Redactions" value={state.redactions.redacted_field_count} detail={`Policy v${state.redactions.policy_version}; raw sensitive fields omitted`} />
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Activity className="h-4 w-4" /> Source health
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4 flex flex-wrap gap-2 text-xs">
              <span className="border border-emerald-400/30 px-2 py-1 text-emerald-300">ready {sourceCounts.ok}</span>
              <span className="border border-yellow-400/30 px-2 py-1 text-yellow-300">partial {sourceCounts.partial}</span>
              <span className="border border-sky-400/30 px-2 py-1 text-sky-300">not connected {sourceCounts.missing}</span>
              <span className="border border-red-400/30 px-2 py-1 text-red-300">error {sourceCounts.error}</span>
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {state.data_sources.map((source) => (
                <SourceCard
                  key={source.id}
                  source={source}
                  onInspect={() => inspectRecord("data source", source.id, [
                    ["status", source.status],
                    ["checked", fmt(source.checked_at)],
                    ["items", String(source.item_count ?? "—")],
                    ["warnings", String(source.warning_count ?? 0)],
                    ["error", source.error_summary ?? "—"],
                  ])}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <AlertTriangle className="h-4 w-4" /> Attention rail
            </CardTitle>
          </CardHeader>
          <CardContent>
            {needsAttention.length === 0 ? (
              <div className="border border-emerald-400/25 bg-emerald-950/10 p-4 text-sm text-emerald-300">
                No blocked work, failed automations, or source warnings in the redacted DTO.
              </div>
            ) : (
              <div className="space-y-2">
                <LimitedRows rows={needsAttention} limit={LIST_LIMIT} label="attention items">
                  {(item) => (
                    <div key={item.id} className="border border-yellow-300/30 bg-yellow-950/10 p-3 text-sm text-yellow-200">
                      <span className="font-semibold">{item.label}</span>
                      <span className="ml-2 text-xs text-yellow-100/70">{item.detail}</span>
                    </div>
                  )}
                </LimitedRows>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_24rem]">
        <div className="flex flex-col gap-6">
          {showWork ? (
            <div className="grid gap-6 xl:grid-cols-2">
              <MiniList title="Rooms / workstreams" icon={<Building2 className="h-4 w-4" />} meta="Grouping surfaces only; rooms are not a source of truth.">
                {state.rooms.length === 0 ? (
                  <EmptyLine label="rooms" hint="No Kanban/topic/system room projection is available yet. Missing rooms are not treated as empty work." />
                ) : (
                  <LimitedRows rows={state.rooms} label="rooms">
                    {(room) => (
                      <EntityRow
                        key={String(room.id)}
                        title={textField(room, "display_name")}
                        badge={textField(room, "kind")}
                        meta={`source ${textField(room, "source")} · id ${String(room.id)}`}
                        onInspect={() => inspectRecord("room", textField(room, "display_name"), [
                          ["id", String(room.id)],
                          ["kind", textField(room, "kind")],
                          ["source", textField(room, "source")],
                        ])}
                      />
                    )}
                  </LimitedRows>
                )}
              </MiniList>

              <MiniList title="Sessions / agents" icon={<Bot className="h-4 w-4" />} meta="Session titles/previews stay redacted unless explicitly allowed later.">
                {state.agents.length === 0 ? (
                  <EmptyLine label="session metadata" hint="The sessions adapter did not surface safe metadata for this snapshot." />
                ) : (
                  <LimitedRows rows={state.agents} label="sessions">
                    {(agent) => (
                      <EntityRow
                        key={String(agent.id)}
                        title={textField(agent, "source_platform")}
                        badge={textField(agent, "status")}
                        meta={`messages ${numberField(agent, "message_count") ?? 0} · title ${textField(agent, "title_policy")}`}
                        onInspect={() => inspectRecord("session / agent", textField(agent, "source_platform"), [
                          ["id", String(agent.id)],
                          ["status", textField(agent, "status")],
                          ["messages", String(numberField(agent, "message_count") ?? 0)],
                          ["title policy", textField(agent, "title_policy")],
                        ])}
                      />
                    )}
                  </LimitedRows>
                )}
              </MiniList>
            </div>
          ) : null}

          {showWork ? (
            <MiniList title="Work items" icon={<MapPinned className="h-4 w-4" />} meta="Grouped by safe status; body/result/comments/logs stay omitted.">
              {state.work_items.length === 0 ? (
                <EmptyLine label="work items" hint="No approved adapter reported task cards. This does not imply every external board is empty." />
              ) : Object.entries(workGroups).map(([status, items]) => (
                <GroupBlock key={status} title={status} count={items.length}>
                  <LimitedRows rows={items} label="work items">
                    {(item) => (
                      <EntityRow
                        key={String(item.id)}
                        title={textField(item, "title")}
                        badge={textField(item, "status")}
                        meta={`assignee ${textField(item, "assignee")} · priority ${numberField(item, "priority") ?? 0}`}
                        onInspect={() => inspectRecord("work item", textField(item, "title"), [
                          ["id", String(item.id)],
                          ["status", textField(item, "status")],
                          ["assignee", textField(item, "assignee")],
                          ["priority", String(numberField(item, "priority") ?? 0)],
                        ])}
                      />
                    )}
                  </LimitedRows>
                </GroupBlock>
              ))}
            </MiniList>
          ) : null}

          {showAutomation ? (
            <MiniList title="Automations" icon={<Clock className="h-4 w-4" />} meta="Grouped by job state; no trigger/pause/resume/delete controls are present.">
              {state.automations.length === 0 ? (
                <EmptyLine label="automations" hint="No cron-style jobs were surfaced by the read-only adapter in this snapshot." />
              ) : Object.entries(automationGroups).map(([jobState, jobs]) => (
                <GroupBlock key={jobState} title={jobState} count={jobs.length}>
                  <LimitedRows rows={jobs} label="automations">
                    {(job) => (
                      <EntityRow
                        key={String(job.id)}
                        title={textField(job, "name")}
                        badge={textField(job, "state")}
                        meta={`last ${fmt(job.last_status)} · next ${fmt(job.next_run_at)}`}
                        warning={typeof job.last_error_summary === "string" ? job.last_error_summary : null}
                        onInspect={() => inspectRecord("automation", textField(job, "name"), [
                          ["id", String(job.id)],
                          ["state", textField(job, "state")],
                          ["last status", fmt(job.last_status)],
                          ["next run", fmt(job.next_run_at)],
                          ["delivery", textField(job, "delivery_policy")],
                        ])}
                      />
                    )}
                  </LimitedRows>
                </GroupBlock>
              ))}
            </MiniList>
          ) : null}

          {showRouting ? (
            <div className="grid gap-6 xl:grid-cols-2">
              <MiniList title="Topic routing" icon={<Route className="h-4 w-4" />} meta="Read-only routing projection; unknown provenance remains explicit.">
                {state.topics.length === 0 ? (
                  <EmptyLine label="topic routing records" hint="No approved topic registry/projection is connected. This is a known source gap, not a UI failure." />
                ) : (
                  <LimitedRows rows={state.topics} label="topics">
                    {(topic) => (
                      <EntityRow
                        key={String(topic.id)}
                        title={textField(topic, "display_name")}
                        badge={textField(topic, "platform")}
                        meta={`purpose ${textField(topic, "purpose")} · confidence ${textField(topic, "confidence")}`}
                        onInspect={() => inspectRecord("topic", textField(topic, "display_name"), [
                          ["id", String(topic.id)],
                          ["platform", textField(topic, "platform")],
                          ["purpose", textField(topic, "purpose")],
                          ["confidence", textField(topic, "confidence")],
                          ["source", textField(topic, "source")],
                        ])}
                      />
                    )}
                  </LimitedRows>
                )}
              </MiniList>

              <MiniList title="Provenance / redaction" icon={<Database className="h-4 w-4" />} meta="Shows counts and policies, not raw sensitive payloads.">
                <div className="grid gap-3 text-sm">
                  <div className="border border-current/15 bg-black/15 p-3">
                    <SectionLabel>Provenance records</SectionLabel>
                    <div className="mt-2 text-2xl text-foreground">{state.provenance.length}</div>
                    <div className="mt-1 text-xs text-midground/65">Unknown or missing provenance stays explicit; it is not inferred from sensitive text.</div>
                  </div>
                  <div className="border border-current/15 bg-black/15 p-3">
                    <SectionLabel>Omitted sections</SectionLabel>
                    <div className="mt-2 text-xs text-midground/75">
                      {state.redactions.omitted_sections.length === 0 ? "—" : state.redactions.omitted_sections.join(" · ")}
                    </div>
                  </div>
                  {state.redactions.warnings.length > 0 ? (
                    <div className="border border-yellow-300/30 bg-yellow-950/10 p-3 text-xs text-yellow-200">
                      {state.redactions.warnings.join(" · ")}
                    </div>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => inspectRecord("redaction report", `policy v${state.redactions.policy_version}`, [
                      ["redacted fields", String(state.redactions.redacted_field_count)],
                      ["omitted sections", state.redactions.omitted_sections.length === 0 ? "—" : state.redactions.omitted_sections.join(" · ")],
                      ["warnings", state.redactions.warnings.length === 0 ? "—" : state.redactions.warnings.join(" · ")],
                    ])}
                    className="flex items-center gap-1 text-xs uppercase tracking-[0.16em] text-midground/70 hover:text-foreground"
                  >
                    <Eye className="h-3 w-3" /> inspect redaction report
                  </button>
                </div>
              </MiniList>
            </div>
          ) : null}

          {showOverview ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Recent safe events</CardTitle>
              </CardHeader>
              <CardContent>
                {state.events.length === 0 ? (
                  <EmptyLine label="events" hint="No safe chronology was produced for this snapshot. Raw logs and transcripts remain hidden by design." />
                ) : (
                  <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                    <LimitedRows rows={state.events} limit={EVENT_LIMIT} label="events">
                      {(event) => (
                        <button
                          type="button"
                          key={String(event.id)}
                          onClick={() => inspectRecord("event", textField(event, "kind"), [
                            ["id", String(event.id)],
                            ["source", textField(event, "source")],
                            ["created", fmt(event.created_at)],
                          ])}
                          className="border border-current/15 bg-black/15 p-2 text-left text-xs hover:border-current/30"
                        >
                          <div className="font-semibold text-foreground">{textField(event, "kind")}</div>
                          <div className="mt-1 text-midground/70">{textField(event, "source")} · {fmt(event.created_at)}</div>
                        </button>
                      )}
                    </LimitedRows>
                  </div>
                )}
              </CardContent>
            </Card>
          ) : null}
        </div>

        <div className="xl:sticky xl:top-4 xl:self-start">
          <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-midground/55">
            <Filter className="h-3 w-3" /> Focus: {focus}
          </div>
          <InspectorPanel selection={selection} />
        </div>
      </div>
    </div>
  );
}
