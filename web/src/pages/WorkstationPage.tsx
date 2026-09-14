import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, CircleDot, RefreshCw, ScrollText } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  api,
  type WorkstationEvent,
  type WorkstationEventsResponse,
  type WorkstationResource,
  type WorkstationResourcesResponse,
} from "@/lib/api";

const EMPTY: WorkstationResourcesResponse = {
  available: false,
  schema_version: 1,
  runtime: null,
  generated_at: null,
  resources: [],
  error: null,
};

const EMPTY_EVENTS: WorkstationEventsResponse = {
  available: false,
  schema_version: 1,
  runtime: null,
  generated_at: null,
  task_id: null,
  events: [],
  error: null,
};

function stateText(resource: WorkstationResource, key: string): string {
  const value = resource.state[key];
  return value == null ? "—" : String(value);
}

function resourceTitle(resource: WorkstationResource): string {
  if (resource.resource_type === "browser") return "Electron browser";
  if (resource.resource_type === "browser_task") return resource.task_id ?? resource.resource_id;
  return `Journal · ${resource.task_id ?? resource.resource_id}`;
}

function eventTitle(event: WorkstationEvent): string {
  return `${event.kind} · ${event.task_id}`;
}

export default function WorkstationPage() {
  const [snapshot, setSnapshot] = useState<WorkstationResourcesResponse>(EMPTY);
  const [eventSnapshot, setEventSnapshot] = useState<WorkstationEventsResponse>(EMPTY_EVENTS);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true);
    try {
      const [next, nextEvents] = await Promise.all([
        api.getWorkstationResources(),
        api.getWorkstationEvents(),
      ]);
      setSnapshot(next);
      setEventSnapshot(nextEvents);
      setError(next.error ?? nextEvents.error);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void load(), 0);
    const timer = window.setInterval(() => void load(), 5000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [load]);

  const tasks = useMemo(
    () => snapshot.resources.filter((resource) => resource.resource_type === "browser_task"),
    [snapshot.resources],
  );
  const journals = useMemo(
    () => snapshot.resources.filter((resource) => resource.resource_type === "execution_journal"),
    [snapshot.resources],
  );
  const browser = snapshot.resources.find((resource) => resource.resource_type === "browser");

  return (
    <div className="flex flex-1 flex-col gap-6 overflow-auto p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <H2>Workstation</H2>
          <p className="mt-1 text-sm text-muted-foreground">
            A read-only projection of the canonical Electron browser runtime and execution journals.
          </p>
        </div>
        <Button ghost size="sm" onClick={() => void load(true)} disabled={refreshing}>
          <RefreshCw className={refreshing ? "animate-spin" : ""} />
          Refresh
        </Button>
      </div>

      {loading ? (
        <div className="flex min-h-40 items-center justify-center"><Spinner /></div>
      ) : (
        <>
          <Card>
            <CardContent className="flex flex-wrap items-center gap-4 py-4">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">Controller</span>
                <Badge tone={snapshot.available ? "success" : "warning"}>
                  {snapshot.available ? "connected" : "degraded"}
                </Badge>
              </div>
              <span className="text-sm text-muted-foreground">Runtime: {snapshot.runtime ?? "—"}</span>
              <span className="text-sm text-muted-foreground">Tasks: {tasks.length}</span>
              {browser && (
                <span className="text-sm text-muted-foreground">
                  Tabs: {stateText(browser, "tab_count")}
                </span>
              )}
            </CardContent>
          </Card>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Browser tasks</h3>
              <span className="text-xs text-muted-foreground">{tasks.length} canonical resource{tasks.length === 1 ? "" : "s"}</span>
            </div>
            {tasks.length === 0 ? (
              <Card><CardContent className="py-8 text-sm text-muted-foreground">No BrowserTask is currently registered.</CardContent></Card>
            ) : (
              <div className="grid gap-3 md:grid-cols-2">
                {tasks.map((resource) => (
                  <Card key={resource.resource_id}>
                    <CardContent className="space-y-2 py-4">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="font-medium">{resourceTitle(resource)}</div>
                          <div className="font-mono text-xs text-muted-foreground">{resource.resource_id}</div>
                        </div>
                        <Badge tone={stateText(resource, "execution_status") === "running" ? "success" : "warning"}>
                          {stateText(resource, "execution_status")}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                        <span>Browser: {stateText(resource, "browser_status")}</span>
                        <span>Tab: {stateText(resource, "tab_id")}</span>
                        <span>Session: {resource.session_id ?? "—"}</span>
                        <span>Run: {stateText(resource, "run_id")}</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Execution journals</h3>
              <span className="text-xs text-muted-foreground">Operational evidence summaries</span>
            </div>
            <Card>
              <CardContent className="divide-y divide-border py-1">
                {journals.length === 0 ? (
                  <p className="py-5 text-sm text-muted-foreground">No journal resource is available.</p>
                ) : journals.map((resource) => (
                  <div className="flex items-center gap-3 py-3 text-sm" key={resource.resource_id}>
                    <CircleDot className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="min-w-0 flex-1 truncate">{resource.task_id ?? resource.resource_id}</span>
                    <span className="text-xs text-muted-foreground">
                      {stateText(resource, "event_count")} events
                    </span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Recent events</h3>
              <span className="text-xs text-muted-foreground">
                {eventSnapshot.events.length} bounded event{eventSnapshot.events.length === 1 ? "" : "s"}
              </span>
            </div>
            <Card>
              <CardContent className="divide-y divide-border py-1">
                {eventSnapshot.events.length === 0 ? (
                  <p className="py-5 text-sm text-muted-foreground">No recent Workstation events are available.</p>
                ) : eventSnapshot.events.map((event) => (
                  <div className="flex items-start gap-3 py-3 text-sm" key={event.event_id}>
                    <ScrollText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <div className="font-medium">{eventTitle(event)}</div>
                      <div className="truncate text-xs text-muted-foreground">{event.message}</div>
                    </div>
                    <time className="shrink-0 text-xs text-muted-foreground" dateTime={event.timestamp}>
                      {event.timestamp}
                    </time>
                  </div>
                ))}
              </CardContent>
            </Card>
          </section>
        </>
      )}
    </div>
  );
}
