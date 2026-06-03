import { useEffect, useState } from "react";
import { Activity, WifiOff } from "lucide-react";
import { api } from "@/lib/api";
import type { V210AdaptersHealthResponse } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";

const ENTRYPOINT_LABELS: Record<string, string> = {
  feishu: "Feishu",
  discord: "Discord",
  web: "Web Console",
  cli: "CLI",
  mac_app: "Mac App",
};

function entrypointLabel(ep: string): string {
  return ENTRYPOINT_LABELS[ep] ?? ep;
}

type BadgeTone = "default" | "destructive" | "outline" | "secondary" | "success" | "warning";

function statusTone(status: string): BadgeTone {
  switch (status) {
    case "connected":
      return "success";
    case "disconnected":
    case "error":
      return "destructive";
    case "unknown":
      return "secondary";
    case "unregistered":
      return "outline";
    default:
      return "secondary";
  }
}

export function AdapterHealthCard() {
  const [health, setHealth] = useState<V210AdaptersHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetch = async () => {
      try {
        const data = await api.v210AdaptersHealth();
        if (!cancelled) {
          setHealth(data);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    };
    fetch();
    const interval = setInterval(fetch, 30000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (error) {
    return (
      <Card className="border-destructive/40">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <WifiOff className="h-4 w-4 text-destructive" />
            Adapter Health
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-destructive">
          Failed to load adapter health: {error}
        </CardContent>
      </Card>
    );
  }

  if (!health) {
    return null;
  }

  const isLegacy = health.mode === "cli_legacy";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Activity className="h-4 w-4 text-muted-foreground" />
          Adapter Health
          <Badge tone={isLegacy ? "secondary" : "success"}>
            {isLegacy ? "CLI Legacy" : "Multi-Entry"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLegacy ? (
          <p className="text-sm text-muted-foreground">
            No external adapters registered. System is in CLI-only legacy mode.
          </p>
        ) : (
          <div className="space-y-2">
            {health.registered_entrypoints.map((ep) => {
              const adapter = health.adapters[ep];
              const status = adapter?.status ?? "unknown";
              return (
                <div key={ep} className="flex items-center justify-between text-sm">
                  <span className="font-medium">{entrypointLabel(ep)}</span>
                  <Badge tone={statusTone(status)}>{status}</Badge>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
