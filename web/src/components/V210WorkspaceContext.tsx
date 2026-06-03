import { useEffect, useState } from "react";
import { Layers, MapPin } from "lucide-react";
import { api } from "@/lib/api";
import type { V210Workspace, V210Session, V210AdaptersHealthResponse } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";

export function V210WorkspaceContext() {
  const [workspaces, setWorkspaces] = useState<V210Workspace[]>([]);
  const [sessions, setSessions] = useState<V210Session[]>([]);
  const [health, setHealth] = useState<V210AdaptersHealthResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetch = async () => {
      try {
        const [ws, ss, h] = await Promise.allSettled([
          api.v210ListWorkspaces(),
          api.v210ListSessions(),
          api.v210AdaptersHealth(),
        ]);
        if (!cancelled) {
          if (ws.status === "fulfilled") setWorkspaces(ws.value.workspaces);
          if (ss.status === "fulfilled") setSessions(ss.value.sessions);
          if (h.status === "fulfilled") setHealth(h.value);
        }
      } catch {
        // Silently swallow — this is supplementary context
      }
    };
    fetch();
    const interval = setInterval(fetch, 60000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (workspaces.length === 0 && sessions.length === 0 && !health) {
    return null;
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Layers className="h-4 w-4 text-muted-foreground" />
          v2.10 Context
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Mode</span>
          <Badge tone={health?.mode === "multi_entry" ? "success" : "secondary"}>
            {health?.mode === "multi_entry" ? "Multi-Entry" : "CLI Legacy"}
          </Badge>
        </div>
        {workspaces.length > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Workspaces</span>
            <span className="font-medium">{workspaces.length}</span>
          </div>
        )}
        {sessions.length > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Sessions</span>
            <span className="font-medium">{sessions.length}</span>
          </div>
        )}
        {health && health.registered_entrypoints.length > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Adapters</span>
            <div className="flex gap-1">
              {health.registered_entrypoints.map((ep) => (
                <Badge key={ep} tone="outline" className="text-xs">
                  <MapPin className="mr-1 h-3 w-3" />
                  {ep}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
