/**
 * MissionControlPage — the root landing page for the dashboard.
 *
 * When the user hits /mission-control (or is redirected there from /),
 * they see a clean, header-less interface with the embedded chat terminal
 * as the primary interaction surface.
 *
 * Layout:
 *   - Full-height dark canvas
 *   - Top: "HERMES MISSION CONTROL" label (subtle, branding)
 *   - Center: Chat terminal (the xterm PTY embed from ChatPage)
 *   - Bottom: minimal status strip
 *
 * This page intentionally avoids:
 *   - The primary objective display (per user request)
 *   - JSON output in the Agents tab (cleaner display)
 *   - Command tab with separate readability issues
 */

import { useEffect, useState } from "react";
import { Terminal, Shield, Activity } from "lucide-react";
import { api } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { isDashboardEmbeddedChatEnabled } from "@/lib/dashboard-flags";

function StatusPill({
  icon: Icon,
  label,
  status,
}: {
  icon: React.ElementType;
  label: string;
  status: string;
}) {
  const colorMap: Record<string, string> = {
    running: "text-success",
    active: "text-success",
    online: "text-success",
    stopped: "text-destructive",
    error: "text-destructive",
  };
  const color = colorMap[status.toLowerCase()] ?? "text-muted-foreground";

  return (
    <div className="flex items-center gap-2 rounded-sm border border-border/50 px-3 py-1.5 bg-background/30">
      <Icon className={`h-3.5 w-3.5 shrink-0 ${color}`} />
      <div className="flex flex-col">
        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
          {label}
        </span>
        <span className={`text-xs font-mono-ui capitalize ${color}`}>
          {status}
        </span>
      </div>
    </div>
  );
}

function GreetingBanner() {
  const [hour, setHour] = useState(new Date().getHours());

  useEffect(() => {
    const timer = setInterval(() => setHour(new Date().getHours()), 60000);
    return () => clearInterval(timer);
  }, []);

  const timeGreeting = hour >= 5 && hour < 12 ? "Good morning" :
    hour >= 12 && hour < 17 ? "Good afternoon" :
    hour >= 17 && hour < 22 ? "Good evening" : "Good night";

  return (
    <div className="flex flex-col items-center gap-1 py-6 text-center">
      <div className="flex items-center gap-2 mb-1">
        <Terminal className="h-5 w-5 text-muted-foreground/60" />
        <h1 className="text-2xl font-bold tracking-[0.2em] uppercase text-foreground/90"
            style={{ fontFamily: "var(--font-brand)" }}>
          Mission Control
        </h1>
        <Shield className="h-5 w-5 text-muted-foreground/60" />
      </div>
      <p className="text-sm text-muted-foreground/60 tracking-wide">
        {timeGreeting}
      </p>
    </div>
  );
}

function SystemStrip({ status }: { status: StatusResponse | null }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-3 px-4 py-3 mt-2">
      <StatusPill
        icon={Activity}
        label="Gateway"
        status={status?.gateway_running ? "running" : "stopped"}
      />
      <Badge tone="outline" className="text-[10px] px-2 py-0.5">
        v{status?.version ?? "unknown"}
      </Badge>
    </div>
  );
}

/**
 * MissionControlPage is the main dashboard entry point.
 *
 * Key design decisions:
 * 1. NO page header — handled by App.tsx (isMissionControlRoute check)
 * 2. NO sidebar — the terminal IS the interface
 * 3. Clean greeting banner instead of primary objective
 * 4. System status strip at bottom (optional, collapses when terminal is active)
 *
 * The embedded chat terminal (ChatPage) is rendered by the App shell when
 * embeddedChat is enabled via the --tui flag. This page just provides the
 * layout container and context for the terminal.
 */
export default function MissionControlPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check if we're in embedded chat mode — the terminal renders outside
  // the <Routes> block in App.tsx, so this page just needs to exist
  // to claim the URL path and provide context.
  const embeddedChat = isDashboardEmbeddedChatEnabled();

  // Load status on mount
  useEffect(() => {
    api.getStatus()
      .then(setStatus)
      .catch((err) => setError(String(err)));
  }, []);

  // When embeddedChat is true, the terminal renders persistently in App.tsx
  // outside <Routes>. This page serves as a "sink" — it renders nothing
  // visually (the terminal is already mounted) but claims the /mission-control
  // URL path so the redirect from / works and the header stays hidden.
  if (embeddedChat) {
    return null;
  }

  // Fallback when embedded chat is NOT enabled: show a static page
  // telling the user to enable --tui mode for the terminal.
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col items-center justify-center p-6">
      <GreetingBanner />

      {error && (
        <div className="mb-4 px-4 py-2 bg-destructive/10 border border-destructive/20 rounded-sm text-sm text-destructive">
          Failed to load status: {error}
        </div>
      )}

      <SystemStrip status={status} />

      <div className="mt-8 text-center">
        <p className="text-muted-foreground mb-3">
          The embedded chat terminal requires the dashboard to run with the
          <code className="ml-1 mr-2 px-1.5 py-0.5 bg-border/50 rounded text-xs font-mono-ui">--tui</code>
          flag.
        </p>
        <p className="text-xs text-muted-foreground/50">
          Start the dashboard with:
          <code className="ml-1 mr-2 px-1.5 py-0.5 bg-border/50 rounded text-xs font-mono-ui">
            hermes dashboard --tui
          </code>
        </p>
      </div>

      <div className="mt-8 flex gap-3">
        <Button
          ghost
          onClick={() => (window.location.href = "/sessions")}
          className="text-xs tracking-wider uppercase opacity-60 hover:opacity-100"
        >
          Browse Sessions
        </Button>
      </div>
    </div>
  );
}
