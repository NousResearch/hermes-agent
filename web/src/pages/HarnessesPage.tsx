import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleDot,
  Clock3,
  Cpu,
  ExternalLink,
  FileText,
  GitBranch,
  Radio,
  RefreshCw,
  Search,
  Server,
  Users,
  XCircle,
} from "lucide-react";
import { api } from "@/lib/api";
import type { ActionStatusResponse, HarnessGatewayAction, HarnessInfo, HarnessesResponse } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";

const REFRESH_INTERVAL_MS = 10_000;

type Filter = "all" | "running" | "attention" | "stopped";

function formatDate(value: string | null, empty = "No activity") {
  if (!value) return empty;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return empty;
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function gatewayTone(harness: HarnessInfo): "success" | "warning" | "outline" {
  if (harness.gateway.state === "running" && !harness.gateway.stale) return "success";
  if (harness.attention || harness.gateway.state === "starting") return "warning";
  return "outline";
}

function gatewayLabel(harness: HarnessInfo) {
  if (harness.gateway.stale) return "Stale startup state";
  switch (harness.gateway.state) {
    case "running": return "Running";
    case "starting": return "Starting";
    case "startup_failed": return "Start failed";
    case "stopped": return "Stopped";
    default: return "Unknown";
  }
}

function GatewayIcon({ harness, className }: { harness: HarnessInfo; className?: string }) {
  if (harness.gateway.state === "running" && !harness.gateway.stale) {
    return <CheckCircle2 className={cn("text-success", className)} />;
  }
  if (harness.attention) return <AlertTriangle className={cn("text-warning", className)} />;
  if (harness.gateway.state === "stopped") return <XCircle className={cn("text-text-tertiary", className)} />;
  return <CircleDot className={cn("text-warning", className)} />;
}

function Metric({ icon: Icon, label, value, tone = "text-midground" }: {
  icon: typeof Users;
  label: string;
  value: string | number;
  tone?: string;
}) {
  return (
    <div className="flex min-w-0 items-center gap-3 border border-border/60 bg-card/35 px-4 py-3">
      <Icon className="h-4 w-4 shrink-0 text-text-tertiary" />
      <div className="min-w-0">
        <div className={cn("font-mondwest text-xl tracking-wide", tone)}>{value}</div>
        <div className="truncate text-xs uppercase tracking-[0.12em] text-text-tertiary">{label}</div>
      </div>
    </div>
  );
}

function PlatformPill({ name, state }: { name: string; state: string }) {
  const connected = state === "connected";
  return (
    <span className={cn(
      "inline-flex items-center gap-1.5 border px-2 py-1 text-[11px] uppercase tracking-[0.08em]",
      connected ? "border-success/30 bg-success/10 text-success" : "border-border bg-muted/40 text-text-tertiary",
    )}>
      <span className={cn("h-1.5 w-1.5 rounded-full", connected ? "bg-success" : "bg-text-tertiary")} />
      {name}
    </span>
  );
}

function HarnessCard({ harness, selected, onSelect }: {
  harness: HarnessInfo;
  selected: boolean;
  onSelect: () => void;
}) {
  const { connected_platforms: connectedPlatforms } = harness;
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "group w-full text-left transition-all focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
        selected ? "translate-y-[-1px]" : "hover:translate-y-[-1px]",
      )}
      aria-pressed={selected}
    >
      <Card className={cn(
        "h-full overflow-hidden border transition-colors",
        selected ? "border-midground/70 bg-midground/[0.07]" : "border-border/80 hover:border-midground/35",
      )}>
        <div className={cn("h-1 w-full", harness.gateway.state === "running" && !harness.gateway.stale ? "bg-success" : harness.attention ? "bg-warning" : "bg-border")} />
        <CardHeader className="gap-3 pb-3">
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center border border-border bg-background/50 font-mondwest text-sm uppercase text-midground">
                {harness.name.slice(0, 2)}
              </div>
              <div className="min-w-0">
                <CardTitle className="truncate text-lg normal-case tracking-normal">{harness.name}</CardTitle>
                {harness.is_default && <div className="mt-1 text-[10px] uppercase tracking-[0.16em] text-text-tertiary">Primary profile</div>}
              </div>
            </div>
            <Badge tone={gatewayTone(harness)} className="shrink-0 gap-1.5 text-[10px] uppercase tracking-[0.08em]">
              <GatewayIcon harness={harness} className="h-3 w-3" />
              {gatewayLabel(harness)}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pt-0">
          <div className="flex min-w-0 items-center gap-2 text-xs text-text-secondary">
            <Cpu className="h-3.5 w-3.5 shrink-0 text-text-tertiary" />
            <span className="truncate">{harness.model ?? "Model not configured"}</span>
            {harness.provider && <span className="shrink-0 text-text-tertiary">· {harness.provider}</span>}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {harness.gateway.platforms.length > 0 ? harness.gateway.platforms.map((platform) => (
              <PlatformPill key={platform.name} name={platform.name} state={platform.state} />
            )) : <span className="text-xs text-text-tertiary">No platform heartbeat</span>}
          </div>
          <div className="grid grid-cols-3 gap-2 border-t border-border/60 pt-3 text-xs">
            <div><div className="font-mondwest text-base text-midground">{harness.sessions.active}</div><div className="text-text-tertiary">active</div></div>
            <div><div className="font-mondwest text-base text-midground">{connectedPlatforms}</div><div className="text-text-tertiary">connected</div></div>
            <div><div className="font-mondwest text-base text-midground">{harness.skill_count}</div><div className="text-text-tertiary">skills</div></div>
          </div>
        </CardContent>
      </Card>
    </button>
  );
}

function HarnessDetails({
  harness,
  actionBusy,
  onAction,
}: {
  harness: HarnessInfo;
  actionBusy: boolean;
  onAction: (action: HarnessGatewayAction) => void;
}) {
  return (
    <Card className="h-fit border-midground/30 bg-midground/[0.04]">
      <CardHeader className="border-b border-border/60 pb-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-text-tertiary"><Server className="h-3.5 w-3.5" /> Harness detail</div>
            <CardTitle className="text-2xl normal-case tracking-normal">{harness.name}</CardTitle>
          </div>
          <GatewayIcon harness={harness} className="h-5 w-5" />
        </div>
      </CardHeader>
      <CardContent className="space-y-5 pt-5">
        <div className="grid grid-cols-2 gap-3">
          <div className="border border-border/60 bg-background/30 p-3"><div className="text-[10px] uppercase tracking-[0.12em] text-text-tertiary">Gateway</div><div className="mt-1 text-sm text-midground">{gatewayLabel(harness)}</div></div>
          <div className="border border-border/60 bg-background/30 p-3"><div className="text-[10px] uppercase tracking-[0.12em] text-text-tertiary">PID</div><div className="mt-1 font-mono text-sm text-midground">{harness.gateway.pid ?? "—"}</div></div>
          <div className="border border-border/60 bg-background/30 p-3"><div className="text-[10px] uppercase tracking-[0.12em] text-text-tertiary">Model</div><div className="mt-1 truncate text-sm text-midground">{harness.model ?? "Not configured"}</div></div>
          <div className="border border-border/60 bg-background/30 p-3"><div className="text-[10px] uppercase tracking-[0.12em] text-text-tertiary">Provider</div><div className="mt-1 truncate text-sm text-midground">{harness.provider ?? "Not configured"}</div></div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-text-tertiary"><Radio className="h-3.5 w-3.5" /> Platforms</div>
          <div className="space-y-2">
            {harness.gateway.platforms.length > 0 ? harness.gateway.platforms.map((platform) => (
              <div key={platform.name} className="flex items-center justify-between border border-border/60 bg-background/20 px-3 py-2.5">
                <div className="flex items-center gap-2"><PlatformPill name={platform.name} state={platform.state} />{platform.needs_attention && <span className="text-xs text-warning">Needs attention</span>}</div>
                <span className="text-xs text-text-tertiary">{platform.state}</span>
              </div>
            )) : <div className="border border-dashed border-border p-3 text-sm text-text-tertiary">No platform runtime data reported.</div>}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-text-tertiary"><Activity className="h-3.5 w-3.5" /> Activity</div>
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between gap-3"><span className="text-text-tertiary">Active sessions</span><span className="font-mondwest text-midground">{harness.sessions.active}</span></div>
            <div className="flex items-center justify-between gap-3"><span className="text-text-tertiary">Recent sessions</span><span className="font-mondwest text-midground">{harness.sessions.total}</span></div>
            <div className="flex items-center justify-between gap-3"><span className="text-text-tertiary">Last activity</span><span className="text-right text-midground">{formatDate(harness.sessions.last_activity_at)}</span></div>
            <div className="flex items-center justify-between gap-3"><span className="text-text-tertiary">Heartbeat</span><span className="text-right text-midground">{formatDate(harness.gateway.updated_at, "No heartbeat")}</span></div>
          </div>
        </div>

        {(harness.gateway.has_exit_reason || harness.gateway.stale) && (
          <div className="border border-warning/30 bg-warning/10 p-3 text-sm text-warning">
            {harness.gateway.stale ? "The gateway startup state is older than five minutes." : "The gateway reported an exit or startup failure."}
          </div>
        )}

        <div className="flex flex-wrap gap-2 border-t border-border/60 pt-4">
          <div className="flex w-full flex-wrap gap-2">
            {!harness.gateway.running && (
              <Button outlined size="sm" onClick={() => onAction("start")} disabled={actionBusy} className="gap-2 text-xs uppercase tracking-[0.08em]">
                <Radio className="h-3.5 w-3.5" /> Start
              </Button>
            )}
            {harness.gateway.running && (
              <Button outlined size="sm" onClick={() => onAction("stop")} disabled={actionBusy} className="gap-2 text-xs uppercase tracking-[0.08em]">
                <XCircle className="h-3.5 w-3.5" /> Stop
              </Button>
            )}
            <Button outlined size="sm" onClick={() => onAction("restart")} disabled={actionBusy} className="gap-2 text-xs uppercase tracking-[0.08em]">
              <RefreshCw className={cn("h-3.5 w-3.5", actionBusy && "animate-spin")} /> Restart
            </Button>
          </div>
          <Link to="/profiles" className="inline-flex items-center gap-2 border border-border px-3 py-2 text-xs uppercase tracking-[0.1em] text-text-secondary transition-colors hover:border-midground/50 hover:text-midground"><FileText className="h-3.5 w-3.5" /> Profile settings</Link>
          <Link to="/logs" className="inline-flex items-center gap-2 border border-border px-3 py-2 text-xs uppercase tracking-[0.1em] text-text-secondary transition-colors hover:border-midground/50 hover:text-midground"><ExternalLink className="h-3.5 w-3.5" /> Logs</Link>
        </div>
      </CardContent>
    </Card>
  );
}

export default function HarnessesPage() {
  const [data, setData] = useState<HarnessesResponse | null>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const requestId = useRef(0);
  const [activeAction, setActiveAction] = useState<{ name: string; profile: string; action: HarnessGatewayAction; invocationId: string } | null>(null);
  const [actionStatus, setActionStatus] = useState<ActionStatusResponse | null>(null);
  const { showToast } = useToast();
  const { t } = useI18n();
  const { setEnd } = usePageHeader();

  const load = useCallback(async (manual = false) => {
    const currentRequest = ++requestId.current;
    if (manual) setRefreshing(true);
    setErrorMessage(null);
    try {
      const next = await api.getHarnesses();
      if (currentRequest !== requestId.current) return;
      setData(next);
      setSelectedName((current) => current && next.harnesses.some((item) => item.name === current) ? current : next.harnesses[0]?.name ?? null);
    } catch {
      if (currentRequest !== requestId.current) return;
      setErrorMessage(t.status.error);
      showToast(t.status.error, "error");
    } finally {
      if (currentRequest === requestId.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [showToast, t.status.error]);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void load(), 0);
    const interval = window.setInterval(() => void load(), REFRESH_INTERVAL_MS);
    return () => {
      window.clearTimeout(initialLoad);
      window.clearInterval(interval);
    };
  }, [load]);

  useEffect(() => {
    if (!activeAction) return;
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const status = await api.getActionStatus(activeAction.name, 200, activeAction.invocationId);
        if (cancelled) return;
        setActionStatus(status);
        if (!status.running) {
          setActiveAction(null);
          if (status.exit_code === 0) {
            showToast("Gateway action completed and status was refreshed.", "success");
          } else {
            showToast("Gateway action failed. Review the gateway logs.", "error");
          }
          void load();
          return;
        }
      } catch {
        if (cancelled) return;
      }
      if (!cancelled) timer = window.setTimeout(() => void poll(), 1200);
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeAction, load, showToast]);

  useEffect(() => {
    setEnd(
      <Button outlined size="sm" onClick={() => void load(true)} disabled={refreshing} className="gap-2">
        <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
        Refresh
      </Button>,
    );
    return () => setEnd(null);
  }, [load, refreshing, setEnd]);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return (data?.harnesses ?? []).filter((harness) => {
      const matchesQuery = !normalized || [harness.name, harness.model, harness.provider, ...harness.gateway.platforms.map((platform) => platform.name)].some((value) => value?.toLowerCase().includes(normalized));
      const matchesFilter = filter === "all" || (filter === "running" && harness.gateway.running) || (filter === "attention" && harness.attention) || (filter === "stopped" && !harness.gateway.running);
      return matchesQuery && matchesFilter;
    });
  }, [data, filter, query]);

  const selected = filtered.find((harness) => harness.name === selectedName) ?? filtered[0] ?? null;

  const runAction = useCallback(async (action: HarnessGatewayAction) => {
    if (!selected || activeAction) return;
    const verb = action.charAt(0).toUpperCase() + action.slice(1);
    if (!window.confirm(`${verb} the ${selected.name} gateway? This changes its live process state.`)) return;
    try {
      const response = await api.runHarnessGatewayAction(selected.name, action);
      setActionStatus(null);
      setActiveAction({
        name: response.name,
        profile: response.profile,
        action: response.action,
        invocationId: response.invocation_id,
      });
    } catch {
      showToast("Unable to start the gateway action.", "error");
    }
  }, [activeAction, selected, showToast]);

  if (loading && !data) {
    return <div className="flex min-h-[50vh] items-center justify-center"><Spinner /></div>;
  }

  if (errorMessage && !data) {
    return (
      <div className="mx-auto flex min-h-[50vh] w-full max-w-xl flex-col items-center justify-center gap-4 p-8 text-center">
        <AlertTriangle className="h-8 w-8 text-warning" />
        <div className="text-lg text-midground">Unable to load harness status</div>
        <div className="text-sm text-text-tertiary">{errorMessage}. Check the dashboard connection and try again.</div>
        <Button outlined size="sm" onClick={() => void load(true)} disabled={refreshing} className="gap-2">
          <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
          Retry
        </Button>
      </div>
    );
  }

  const summary = data?.summary ?? { total: 0, running: 0, connected_platforms: 0, attention: 0 };
  return (
    <div className="mx-auto w-full max-w-[1600px] space-y-6 p-5 pb-10 md:p-8">
      <section className="relative overflow-hidden border border-border/80 bg-card/35 p-6 md:p-8">
        <div className="pointer-events-none absolute -right-16 -top-24 h-64 w-64 rounded-full bg-midground/10 blur-3xl" />
        <div className="relative flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div className="max-w-2xl">
            <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-success"><span className="h-2 w-2 animate-pulse rounded-full bg-success" /> Live control room</div>
            <h1 className="font-mondwest text-3xl uppercase tracking-[0.08em] text-midground md:text-4xl">Harnesses</h1>
            <p className="mt-3 max-w-xl text-sm leading-6 text-text-secondary">A live view of every Hermes profile, gateway heartbeat, platform connection, and active session on this host.</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-text-tertiary"><Clock3 className="h-3.5 w-3.5" />Auto-refreshes every 10 seconds</div>
        </div>
      </section>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric icon={GitBranch} label="Configured" value={summary.total} />
        <Metric icon={Server} label="Running gateways" value={summary.running} tone="text-success" />
        <Metric icon={Radio} label="Connected platforms" value={summary.connected_platforms} tone="text-success" />
        <Metric icon={AlertTriangle} label="Needs attention" value={summary.attention} tone={summary.attention > 0 ? "text-warning" : "text-success"} />
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="relative w-full md:max-w-sm"><Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-tertiary" /><Input aria-label="Search harnesses, models, and platforms" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search harnesses, models, platforms..." className="pl-9" /></div>
        <div className="flex flex-wrap gap-1 border border-border/70 bg-card/30 p-1">
          {(["all", "running", "attention", "stopped"] as Filter[]).map((item) => (
            <button key={item} type="button" onClick={() => setFilter(item)} className={cn("px-3 py-1.5 text-xs uppercase tracking-[0.1em] transition-colors", filter === item ? "bg-midground text-background" : "text-text-tertiary hover:text-midground")} aria-pressed={filter === item}>{item}</button>
          ))}
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
        <section className="min-w-0">
          {filtered.length === 0 ? (
            <div className="flex min-h-64 flex-col items-center justify-center border border-dashed border-border p-8 text-center"><Search className="mb-3 h-6 w-6 text-text-tertiary" /><div className="text-midground">No harnesses match this view</div><div className="mt-1 text-sm text-text-tertiary">Try clearing the search or selecting All.</div></div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">{filtered.map((harness) => <HarnessCard key={harness.name} harness={harness} selected={selected?.name === harness.name} onSelect={() => setSelectedName(harness.name)} />)}</div>
          )}
        </section>
        <aside className="min-w-0">{selected ? <HarnessDetails harness={selected} actionBusy={activeAction?.profile === selected.name && (actionStatus?.running ?? true)} onAction={runAction} /> : <div className="border border-dashed border-border p-8 text-center text-sm text-text-tertiary">Select a harness to inspect its live details.</div>}</aside>
      </div>
    </div>
  );
}
