import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Bot, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { DelegationEvent, DelegationTraceSummary } from "@/lib/api";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { usePageHeader } from "@/contexts/usePageHeader";
import { timeAgo } from "@/lib/utils";

const PERIODS = [7, 30, 90] as const;
const STATUSES = ["", "running", "completed", "failed", "interrupted", "backgrounded"];

function statusTone(status: string): "success" | "warning" | "secondary" | "destructive" {
  if (status === "completed") return "success";
  if (status === "failed" || status === "interrupted") return "destructive";
  if (status === "running" || status === "backgrounded") return "warning";
  return "secondary";
}

function EventRow({ event }: { event: DelegationEvent }) {
  const fallbackActivations = event.fallback_activations || [];
  const continuation = event.fallback_continuation || {};
  const continuationRisk = typeof continuation.risk === "string" ? continuation.risk : "";
  return (
    <div className="border border-border bg-muted/20 px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="secondary">{event.type}</Badge>
        <span className="font-mono text-muted-foreground">{event.agent_id || "unknown"}</span>
        {event.status ? <Badge tone={statusTone(event.status)}>{event.status}</Badge> : null}
        {fallbackActivations.length > 0 ? <Badge tone="warning">fallback x{fallbackActivations.length}</Badge> : null}
        <span className="ml-auto text-muted-foreground">{timeAgo(event.timestamp)}</span>
      </div>
      {fallbackActivations.length > 0 ? (
        <div className="mt-2 space-y-1 border-l-2 border-warning/50 pl-2 text-muted-foreground">
          {fallbackActivations.map((activation, idx) => (
            <div key={idx} className="font-mono text-[11px]">
              {String(activation.from_model || "?")} → {String(activation.to_model || "?")}
              {" · "}
              {String(activation.reason || "unknown")}
            </div>
          ))}
          {continuationRisk ? (
            <div className="text-[11px]">continuation: {continuationRisk}</div>
          ) : null}
        </div>
      ) : null}
      {event.goal_preview || event.reason || event.error ? (
        <div className="mt-2 whitespace-pre-wrap text-muted-foreground">
          {event.goal_preview || event.reason || event.error}
        </div>
      ) : null}
      <div className="mt-1 font-mono text-[10px] text-muted-foreground/70">
        {event.subagent_id || event.event_id}
      </div>
    </div>
  );
}

export default function DelegationsPage() {
  const { setTitle } = usePageHeader();
  const [days, setDays] = useState<number>(30);
  const [agentId, setAgentId] = useState("");
  const [status, setStatus] = useState("");
  const [traces, setTraces] = useState<DelegationTraceSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [events, setEvents] = useState<DelegationEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await api.getDelegations({ days, agent_id: agentId || undefined, status: status || undefined });
      setTraces(resp.delegations);
      if (selected && !resp.delegations.some((t) => t.task_id === selected)) {
        setSelected(null);
        setEvents([]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [agentId, days, selected, status]);

  const loadTrace = useCallback(async (taskId: string) => {
    setSelected(taskId);
    setDetailLoading(true);
    setError(null);
    try {
      const resp = await api.getDelegationTrace(taskId);
      setEvents(resp.events);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    setTitle("Delegations");
    return () => setTitle(null);
  }, [setTitle]);

  useEffect(() => {
    load();
  }, [load]);

  const selectedTrace = useMemo(
    () => traces.find((trace) => trace.task_id === selected),
    [selected, traces],
  );

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="flex flex-wrap items-center gap-2 py-4">
          {PERIODS.map((p) => (
            <Button key={p} size="sm" outlined={days !== p} onClick={() => setDays(p)}>
              {p}d
            </Button>
          ))}
          <Input
            className="h-8 w-48"
            placeholder="agent_id"
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
          />
          <select
            className="h-8 border border-border bg-background px-2 text-xs"
            value={status}
            onChange={(e) => setStatus(e.target.value)}
          >
            {STATUSES.map((s) => (
              <option key={s || "all"} value={s}>{s || "all statuses"}</option>
            ))}
          </select>
          <Button size="sm" outlined onClick={load} disabled={loading} prefix={loading ? <Spinner /> : <RefreshCw />}>
            Refresh
          </Button>
        </CardContent>
      </Card>

      {error ? (
        <Card className="border-destructive/40">
          <CardContent className="flex items-center gap-2 py-4 text-sm text-destructive">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.2fr)]">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Delegation Traces</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {loading && traces.length === 0 ? <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner /> Loading...</div> : null}
            {!loading && traces.length === 0 ? <div className="text-sm text-muted-foreground">No subagent events found.</div> : null}
            {traces.map((trace) => (
              <button
                key={trace.task_id}
                type="button"
                className={`w-full border px-3 py-2 text-left text-sm transition-colors ${selected === trace.task_id ? "border-primary bg-primary/5" : "border-border hover:bg-muted/40"}`}
                onClick={() => loadTrace(trace.task_id)}
              >
                <div className="flex items-center gap-2">
                  <Bot className="h-4 w-4 text-muted-foreground" />
                  <span className="truncate font-mono text-xs">{trace.task_id}</span>
                  <Badge tone={statusTone(trace.status)} className="ml-auto">{trace.status}</Badge>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {trace.event_count} events · {trace.first_at ? timeAgo(trace.first_at) : "unknown"}
                  {(trace.fallback_activation_count || 0) > 0 ? (
                    <> · fallback x{trace.fallback_activation_count}</>
                  ) : null}
                  {trace.fallback_continuation_risk ? (
                    <> · {trace.fallback_continuation_risk}</>
                  ) : null}
                </div>
              </button>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Trace Detail</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {!selectedTrace ? <div className="text-sm text-muted-foreground">Select a trace to inspect the chain.</div> : null}
            {detailLoading ? <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner /> Loading trace...</div> : null}
            {selectedTrace && !detailLoading ? (
              <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
                <Badge tone={statusTone(selectedTrace.status)}>{selectedTrace.status}</Badge>
                <span className="font-mono text-muted-foreground">{selectedTrace.task_id}</span>
              </div>
            ) : null}
            {events.map((event) => <EventRow key={event.event_id} event={event} />)}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
