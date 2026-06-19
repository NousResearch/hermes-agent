import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  Home,
  MessageSquare,
  Radio,
  RotateCw,
  WifiOff,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { MessagingPlatform } from "@/lib/api";
import { cn } from "@/lib/utils";

const stateTone: Record<string, "success" | "warning" | "destructive" | "secondary" | "outline"> = {
  connected: "success",
  disabled: "secondary",
  disconnected: "warning",
  fatal: "destructive",
  gateway_stopped: "warning",
  not_configured: "outline",
  pending_restart: "warning",
  startup_failed: "destructive",
};

function formatState(state: string): string {
  return state.replace(/_/g, " ");
}

export default function RoomHubsPage() {
  const [platforms, setPlatforms] = useState<MessagingPlatform[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getMessagingPlatforms()
      .then((res) => {
        if (!cancelled) setPlatforms(res.platforms);
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

  const hubs = useMemo(
    () => platforms.filter((platform) => platform.home_channel !== null),
    [platforms],
  );
  const configured = useMemo(
    () => platforms.filter((platform) => platform.configured),
    [platforms],
  );
  const connected = useMemo(
    () => platforms.filter((platform) => platform.state === "connected"),
    [platforms],
  );

  if (loading) {
    return (
      <div className="flex min-h-[18rem] items-center justify-center">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 pb-6">
      <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <Card className="border-midground/25 bg-card/70">
          <CardContent className="p-5 sm:p-6">
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div className="space-y-2">
                <Badge tone="outline" className="w-fit uppercase tracking-[0.16em]">
                  room hubs
                </Badge>
                <h2 className="font-expanded text-2xl font-bold tracking-[0.06em] text-midground sm:text-3xl">
                  Messaging rooms, kept deliberately off the main screen.
                </h2>
                <p className="max-w-3xl text-sm leading-6 text-muted-foreground">
                  This page collects home channels and configured platforms. The dashboard gets the quick signal; this page gets the room-level plumbing.
                </p>
              </div>
              <Link
                to="/channels"
                className="inline-flex shrink-0 items-center justify-center gap-2 rounded-sm border border-border px-3 py-2 text-sm font-medium uppercase tracking-[0.08em] text-foreground transition-colors hover:border-midground/50 hover:bg-midground/5"
              >
                <Radio className="h-4 w-4" />
                Configure channels
              </Link>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Hub summary</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-3 gap-2 xl:grid-cols-1">
            <Summary label="Home hubs" value={String(hubs.length)} icon={Home} />
            <Summary label="Connected" value={String(connected.length)} icon={CheckCircle2} />
            <Summary label="Configured" value={`${configured.length}/${platforms.length}`} icon={MessageSquare} />
          </CardContent>
        </Card>
      </section>

      {error ? (
        <Card className="border-destructive/50">
          <CardContent className="flex items-center gap-2 p-4 text-sm text-destructive">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            Failed to load hubs: {error}
          </CardContent>
        </Card>
      ) : null}

      <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Home className="h-4 w-4 text-midground" />
              Home hubs
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {hubs.length > 0 ? hubs.map((platform) => (
              <HubCard key={platform.id} platform={platform} prominent />
            )) : (
              <EmptyState
                icon={WifiOff}
                title="No home room yet"
                detail="Use /sethome from a messaging room, then this page becomes the map. Very cartographic."
              />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Radio className="h-4 w-4 text-midground" />
              Channel connectors
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {platforms.map((platform) => (
              <HubCard key={platform.id} platform={platform} />
            ))}
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

function HubCard({ platform, prominent = false }: { platform: MessagingPlatform; prominent?: boolean }) {
  const home = platform.home_channel;
  const missingRequired = platform.env_vars.filter((env) => env.required && !env.is_set);
  const badgeTone = stateTone[platform.state] ?? "outline";

  return (
    <div
      className={cn(
        "rounded-sm border bg-background/30 p-3 transition-colors",
        prominent ? "border-midground/35" : "border-border/70",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-foreground">
            {home?.name || platform.name}
          </p>
          <p className="mt-1 truncate text-xs text-muted-foreground">
            {platform.description}
          </p>
        </div>
        <Badge tone={badgeTone}>{formatState(platform.state)}</Badge>
      </div>

      <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
        <HubDetail label="Platform" value={platform.name} />
        <HubDetail label="Home" value={home ? "yes" : "not set"} />
        {home ? <HubDetail label="Room" value={home.chat_id} mono /> : null}
        {home?.thread_id ? <HubDetail label="Thread" value={home.thread_id} mono /> : null}
      </dl>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {platform.docs_url ? (
          <a
            href={platform.docs_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-sm border border-border px-2.5 py-1.5 text-xs font-medium uppercase tracking-[0.08em] text-foreground transition-colors hover:border-midground/50 hover:bg-midground/5"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            Docs
          </a>
        ) : null}
        {platform.state === "pending_restart" ? (
          <span className="inline-flex items-center gap-1 text-xs text-warning">
            <RotateCw className="h-3.5 w-3.5" /> Restart gateway to apply
          </span>
        ) : null}
        {missingRequired.length > 0 ? (
          <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
            <AlertTriangle className="h-3.5 w-3.5" /> Missing {missingRequired.length} required value{missingRequired.length === 1 ? "" : "s"}
          </span>
        ) : null}
      </div>
    </div>
  );
}

function HubDetail({ label, mono = false, value }: { label: string; mono?: boolean; value: string }) {
  return (
    <div className="min-w-0 rounded-sm border border-border/60 bg-card/30 px-2 py-1.5">
      <dt className="uppercase tracking-[0.14em] text-muted-foreground">{label}</dt>
      <dd className={cn("mt-0.5 truncate text-foreground", mono && "font-courier")}>{value}</dd>
    </div>
  );
}

function Summary({ icon: Icon, label, value }: { icon: typeof Home; label: string; value: string }) {
  return (
    <div className="rounded-sm border border-border/70 bg-background/30 p-3">
      <Icon className="h-4 w-4 text-midground" />
      <p className="mt-3 text-xs uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <p className="mt-1 text-lg font-semibold text-foreground">{value}</p>
    </div>
  );
}

function EmptyState({
  detail,
  icon: Icon,
  title,
}: {
  detail: string;
  icon: typeof WifiOff;
  title: string;
}) {
  return (
    <div className="rounded-sm border border-dashed border-border/70 p-5 text-center">
      <Icon className="mx-auto h-6 w-6 text-muted-foreground" />
      <p className="mt-3 text-sm font-medium text-foreground">{title}</p>
      <p className="mt-1 text-sm text-muted-foreground">{detail}</p>
    </div>
  );
}
