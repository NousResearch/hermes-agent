import { useEffect, useState } from "react";
import { Circle, Wifi, WifiOff, AlertTriangle } from "lucide-react";
import { api } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";

type BackendStatus = "online" | "offline" | "degraded" | "mock";

interface BackendStatusState {
  status: BackendStatus;
  details: string | null;
  version: string | null;
  activeSessions: number;
}

export function BackendStatusIndicator({ compact = false }: { compact?: boolean }) {
  const [state, setState] = useState<BackendStatusState>({
    status: "offline",
    details: null,
    version: null,
    activeSessions: 0,
  });

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const data: StatusResponse = await api.getStatus();
        if (!mounted) return;

        const hasIssues =
          data.gateway_state === "startup_failed" ||
          Object.values(data.gateway_platforms ?? {}).some(
            (p) => p.state === "fatal"
          );

        setState({
          status: hasIssues ? "degraded" : "online",
          details: data.gateway_running
            ? `Gateway PID ${data.gateway_pid ?? "unknown"}`
            : "Gateway stopped",
          version: data.version,
          activeSessions: data.active_sessions,
        });
      } catch {
        if (!mounted) return;
        // Try to detect if it's a mock/local environment
        try {
          const res = await fetch("/api/status", { method: "GET" });
          if (!res.ok) throw new Error("not ok");
          // Got a response but failed to parse — degraded
          if (mounted) {
            setState((prev) => ({
              ...prev,
              status: "degraded",
              details: "Backend responded with error",
            }));
          }
        } catch {
          if (mounted) {
            setState({ status: "offline", details: null, version: null, activeSessions: 0 });
          }
        }
      }
    };

    load();
    const interval: ReturnType<typeof setInterval> = setInterval(load, 8000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const colorClass =
    state.status === "online"
      ? "text-success"
      : state.status === "degraded"
        ? "text-warning"
        : "text-muted-foreground";

  const label =
    state.status === "online"
      ? "Backend online"
      : state.status === "degraded"
        ? "Backend degraded"
        : state.status === "mock"
          ? "Mock mode"
          : "Backend offline";

  if (compact) {
    return (
      <div className={`flex items-center gap-1.5 ${colorClass}`} title={state.details ?? label}>
        <Circle
          className={`h-2 w-2 fill-current ${
            state.status === "online" ? "animate-pulse" : ""
          }`}
        />
        {!compact && (
          <span className="text-[10px] font-compressed tracking-widest uppercase">
            {label}
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {state.status === "online" ? (
        <Wifi className={`h-3.5 w-3.5 ${colorClass}`} />
      ) : state.status === "offline" ? (
        <WifiOff className={`h-3.5 w-3.5 ${colorClass}`} />
      ) : (
        <AlertTriangle className={`h-3.5 w-3.5 ${colorClass}`} />
      )}
      <div className="flex flex-col">
        <span className={`text-[10px] font-compressed tracking-widest uppercase ${colorClass}`}>
          {label}
        </span>
        {state.version && (
          <span className="text-[9px] text-muted-foreground font-mono">
            v{state.version}
          </span>
        )}
        {state.details && (
          <span className="text-[9px] text-muted-foreground hidden sm:inline">
            {state.details}
          </span>
        )}
      </div>
    </div>
  );
}
