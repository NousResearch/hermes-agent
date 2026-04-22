import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  Download,
  ExternalLink,
  Flame,
  Gauge,
  Loader2,
  Newspaper,
  Radio,
  RotateCw,
  TrendingUp,
  Wifi,
  WifiOff,
  Wrench,
  X,
} from "lucide-react";
import { Cell, Grid } from "@nous-research/ui";
import { api } from "@/lib/api";
import type {
  ActionStatusResponse,
  PlatformStatus,
  SessionInfo,
  StatusResponse,
} from "@/lib/api";
import { cn, timeAgo, isoTimeAgo } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Markdown } from "@/components/Markdown";
import { Toast } from "@/components/Toast";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useI18n } from "@/i18n";

const ACTION_NAMES: Record<"restart" | "update", string> = {
  restart: "gateway-restart",
  update: "hermes-update",
};

const MARKET_SOURCE_TABS = [
  { key: "overview", label: "OVERVIEW", url: "https://crypto.kedaya.xyz/zh/md" },
  { key: "btc", label: "BTC", url: "https://crypto.kedaya.xyz/zh/BTC/USDT/md" },
  { key: "eth", label: "ETH", url: "https://crypto.kedaya.xyz/zh/ETH/USDT/md" },
  { key: "sol", label: "SOL", url: "https://crypto.kedaya.xyz/zh/SOL/USDT/md" },
  { key: "xrp", label: "XRP", url: "https://crypto.kedaya.xyz/zh/XRP/USDT/md" },
] as const;

type MarketSourceKey = (typeof MARKET_SOURCE_TABS)[number]["key"];

interface MarketSourceState {
  body: string;
  fetchedAt: number | null;
  loading: boolean;
  error: string | null;
}

const EMPTY_SOURCE_STATE: MarketSourceState = {
  body: "",
  fetchedAt: null,
  loading: false,
  error: null,
};

export default function StatusPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [pendingAction, setPendingAction] = useState<
    "restart" | "update" | null
  >(null);
  const [activeAction, setActiveAction] = useState<"restart" | "update" | null>(
    null,
  );
  const [actionStatus, setActionStatus] = useState<ActionStatusResponse | null>(
    null,
  );
  const [toast, setToast] = useState<ToastState | null>(null);
  const [marketSources, setMarketSources] = useState<
    Record<MarketSourceKey, MarketSourceState>
  >(() =>
    Object.fromEntries(
      MARKET_SOURCE_TABS.map((tab) => [tab.key, { ...EMPTY_SOURCE_STATE }]),
    ) as Record<MarketSourceKey, MarketSourceState>,
  );
  const logScrollRef = useRef<HTMLPreElement | null>(null);
  const { t } = useI18n();

  useEffect(() => {
    const load = () => {
      api
        .getStatus()
        .then(setStatus)
        .catch(() => {});
      api
        .getSessions(50)
        .then((resp) => setSessions(resp.sessions))
        .catch(() => {});
    };
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!activeAction) return;
    const name = ACTION_NAMES[activeAction];
    let cancelled = false;

    const poll = async () => {
      try {
        const resp = await api.getActionStatus(name);
        if (cancelled) return;
        setActionStatus(resp);
        if (!resp.running) {
          const ok = resp.exit_code === 0;
          setToast({
            type: ok ? "success" : "error",
            message: ok
              ? t.status.actionFinished
              : `${t.status.actionFailed} (exit ${resp.exit_code ?? "?"})`,
          });
          return;
        }
      } catch {
        // transient fetch error; keep polling
      }
      if (!cancelled) setTimeout(poll, 1500);
    };

    poll();
    return () => {
      cancelled = true;
    };
  }, [activeAction, t.status.actionFinished, t.status.actionFailed]);

  useEffect(() => {
    const el = logScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [actionStatus?.lines]);

  useEffect(() => {
    let cancelled = false;

    const loadMarketSources = async () => {
      setMarketSources((prev) => {
        const next = { ...prev };
        for (const tab of MARKET_SOURCE_TABS) {
          next[tab.key] = { ...prev[tab.key], loading: true, error: null };
        }
        return next;
      });

      await Promise.all(
        MARKET_SOURCE_TABS.map(async (tab) => {
          try {
            const res = await fetch(tab.url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const body = await res.text();
            if (cancelled) return;
            setMarketSources((prev) => ({
              ...prev,
              [tab.key]: {
                body,
                fetchedAt: Date.now(),
                loading: false,
                error: null,
              },
            }));
          } catch (err) {
            if (cancelled) return;
            setMarketSources((prev) => ({
              ...prev,
              [tab.key]: {
                ...prev[tab.key],
                loading: false,
                error: err instanceof Error ? err.message : String(err),
              },
            }));
          }
        }),
      );
    };

    loadMarketSources();
    const interval = setInterval(loadMarketSources, 60_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const runAction = async (action: "restart" | "update") => {
    setPendingAction(action);
    setActionStatus(null);
    try {
      if (action === "restart") {
        await api.restartGateway();
      } else {
        await api.updateHermes();
      }
      setActiveAction(action);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setToast({
        type: "error",
        message: `${t.status.actionFailed}: ${detail}`,
      });
    } finally {
      setPendingAction(null);
    }
  };

  const dismissLog = () => {
    setActiveAction(null);
    setActionStatus(null);
  };

  if (!status) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  const PLATFORM_STATE_BADGE: Record<
    string,
    { variant: "success" | "warning" | "destructive"; label: string }
  > = {
    connected: { variant: "success", label: t.status.connected },
    disconnected: { variant: "warning", label: t.status.disconnected },
    fatal: { variant: "destructive", label: t.status.error },
  };

  const GATEWAY_STATE_DISPLAY: Record<
    string,
    { badge: "success" | "warning" | "destructive" | "outline"; label: string }
  > = {
    running: { badge: "success", label: t.status.running },
    starting: { badge: "warning", label: t.status.starting },
    startup_failed: { badge: "destructive", label: t.status.failed },
    stopped: { badge: "outline", label: t.status.stopped },
  };

  function gatewayValue(): string {
    if (status!.gateway_running && status!.gateway_health_url)
      return status!.gateway_health_url;
    if (status!.gateway_running && status!.gateway_pid)
      return `${t.status.pid} ${status!.gateway_pid}`;
    if (status!.gateway_running) return t.status.runningRemote;
    if (status!.gateway_state === "startup_failed") return t.status.startFailed;
    return t.status.notRunning;
  }

  function gatewayBadge() {
    const info = status!.gateway_state
      ? GATEWAY_STATE_DISPLAY[status!.gateway_state]
      : null;
    if (info) return info;
    return status!.gateway_running
      ? { badge: "success" as const, label: t.status.running }
      : { badge: "outline" as const, label: t.common.off };
  }

  const gwBadge = gatewayBadge();
  const platforms = Object.entries(status.gateway_platforms ?? {});
  const healthyPlatforms = platforms.filter(
    ([, info]) => info.state === "connected",
  ).length;
  const sourceCoverage = Object.values(marketSources).filter(
    (item) => item.body.trim().length > 0,
  ).length;
  const liveIntelCount = extractSignalBullets(
    marketSources.overview.body || marketSources.btc.body,
  ).length;

  const items = [
    {
      icon: Cpu,
      label: t.status.agent,
      value: `v${status.version}`,
      badgeText: t.common.live,
      badgeVariant: "success" as const,
    },
    {
      icon: Radio,
      label: t.status.gateway,
      value: gatewayValue(),
      badgeText: gwBadge.label,
      badgeVariant: gwBadge.badge,
    },
    {
      icon: Activity,
      label: t.status.activeSessions,
      value:
        status.active_sessions > 0
          ? `${status.active_sessions} ${t.status.running.toLowerCase()}`
          : t.status.noneRunning,
      badgeText: status.active_sessions > 0 ? t.common.live : t.common.off,
      badgeVariant: (status.active_sessions > 0 ? "success" : "outline") as
        | "success"
        | "outline",
    },
    {
      icon: Gauge,
      label: "Platforms online",
      value: `${healthyPlatforms}/${Math.max(platforms.length, 1)}`,
      badgeText: healthyPlatforms === platforms.length ? "stable" : "check",
      badgeVariant: (healthyPlatforms === platforms.length
        ? "success"
        : "warning") as "success" | "warning",
    },
  ];

  const activeSessions = sessions.filter((s) => s.is_active);
  const recentSessions = sessions.filter((s) => !s.is_active).slice(0, 5);

  // Collect alerts that need attention
  const alerts: { message: string; detail?: string }[] = [];
  if (status.gateway_state === "startup_failed") {
    alerts.push({
      message: t.status.gatewayFailedToStart,
      detail: status.gateway_exit_reason ?? undefined,
    });
  }
  const failedPlatforms = platforms.filter(
    ([, info]) => info.state === "fatal" || info.state === "disconnected",
  );
  for (const [name, info] of failedPlatforms) {
    const stateLabel =
      info.state === "fatal"
        ? t.status.platformError
        : t.status.platformDisconnected;
    alerts.push({
      message: `${name.charAt(0).toUpperCase() + name.slice(1)} ${stateLabel}`,
      detail: info.error_message ?? undefined,
    });
  }

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      {alerts.length > 0 && (
        <div className="border border-destructive/30 bg-destructive/[0.06] p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
            <div className="flex flex-col gap-2 min-w-0">
              {alerts.map((alert, i) => (
                <div key={i}>
                  <p className="text-sm font-medium text-destructive">
                    {alert.message}
                  </p>
                  {alert.detail && (
                    <p className="text-xs text-destructive/70 mt-0.5">
                      {alert.detail}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <Grid className="border-b md:!grid-cols-2 lg:!grid-cols-4">
        {items.map(({ icon: Icon, label, value, badgeText, badgeVariant }) => (
          <Cell
            key={label}
            className="flex min-w-0 flex-col gap-2 overflow-hidden"
          >
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">{label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </div>

            <div
              className="truncate text-2xl font-bold font-mondwest"
              title={value}
            >
              {value}
            </div>

            {badgeText && (
              <Badge variant={badgeVariant} className="self-start">
                {badgeVariant === "success" && (
                  <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                )}
                {badgeText}
              </Badge>
            )}
          </Cell>
        ))}

        <Cell className="flex min-w-0 flex-col gap-2 overflow-hidden">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium">
              {t.status.actions}
            </CardTitle>
            <Wrench className="h-4 w-4 text-muted-foreground" />
          </div>

          <div className="flex gap-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => runAction("restart")}
              disabled={
                pendingAction !== null ||
                (activeAction !== null && actionStatus?.running !== false)
              }
              className="flex-1 min-w-0"
            >
              <RotateCw
                className={cn(
                  "h-3.5 w-3.5",
                  (pendingAction === "restart" ||
                    (activeAction === "restart" && actionStatus?.running)) &&
                    "animate-spin",
                )}
              />

              {activeAction === "restart" && actionStatus?.running
                ? t.status.restartingGateway
                : t.status.restartGateway}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => runAction("update")}
              disabled={
                pendingAction !== null ||
                (activeAction !== null && actionStatus?.running !== false)
              }
              className="flex-1 min-w-0"
            >
              <Download
                className={cn(
                  "h-3.5 w-3.5",
                  (pendingAction === "update" ||
                    (activeAction === "update" && actionStatus?.running)) &&
                    "animate-pulse",
                )}
              />

              {activeAction === "update" && actionStatus?.running
                ? t.status.updatingHermes
                : t.status.updateHermes}
            </Button>
          </div>
        </Cell>
      </Grid>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_1.7fr]">
        <Card className="overflow-hidden border-primary/15 bg-gradient-to-br from-card via-card to-primary/[0.04] shadow-[0_0_0_1px_rgba(255,255,255,0.03),0_24px_80px_rgba(0,0,0,0.22)]">
          <CardHeader className="gap-3 border-b border-border/80 bg-black/10">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                <CardTitle className="text-base tracking-[0.14em]">Trade cockpit</CardTitle>
              </div>
              <Badge variant="success">live</Badge>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <TerminalMetric
                icon={Flame}
                label="Active alerts"
                value={String(alerts.length)}
                tone={alerts.length > 0 ? "warn" : "ok"}
              />
              <TerminalMetric
                icon={Newspaper}
                label="Intel feeds"
                value={`${sourceCoverage}/5`}
                tone={sourceCoverage >= 4 ? "ok" : "warn"}
              />
              <TerminalMetric
                icon={Gauge}
                label="Signal bullets"
                value={String(liveIntelCount)}
                tone={liveIntelCount > 0 ? "ok" : "neutral"}
              />
            </div>
          </CardHeader>
          <CardContent className="grid gap-4 p-4">
            <div className="grid gap-3 md:grid-cols-2">
              <IntelSummaryCard
                title="System state"
                tone={status.gateway_running ? "ok" : "warn"}
                bullets={[
                  `Gateway: ${gwBadge.label}`,
                  `Platforms online: ${healthyPlatforms}/${platforms.length}`,
                  `Active sessions: ${status.active_sessions}`,
                ]}
              />
              <IntelSummaryCard
                title="Market intel"
                tone={sourceCoverage >= 4 ? "ok" : "warn"}
                bullets={extractSignalBullets(
                  marketSources.overview.body || marketSources.btc.body,
                ).slice(0, 3)}
              />
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/60 p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  Operator note
                </div>
                <div className="text-[11px] text-muted-foreground">refresh 60s</div>
              </div>
              <p className="text-sm leading-6 text-foreground/90">
                保留系统状态 + 行情情报双视角。现在首页已经不是纯状态页，而是可直接看平台健康、信号密度、外部情报摘要的交易终端首屏。
              </p>
            </div>
          </CardContent>
        </Card>

        <MarketIntelPanel marketSources={marketSources} />
      </div>

      {activeAction && (
        <div className="border border-border bg-background-base/50">
          <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
            <div className="flex items-center gap-2 min-w-0">
              {actionStatus?.running ? (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-warning" />
              ) : actionStatus?.exit_code === 0 ? (
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-success" />
              ) : actionStatus !== null ? (
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-destructive" />
              ) : (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />
              )}

              <span className="text-xs font-mondwest tracking-[0.12em] truncate">
                {activeAction === "restart"
                  ? t.status.restartGateway
                  : t.status.updateHermes}
              </span>

              <Badge
                variant={
                  actionStatus?.running
                    ? "warning"
                    : actionStatus?.exit_code === 0
                      ? "success"
                      : actionStatus
                        ? "destructive"
                        : "outline"
                }
                className="text-[10px] shrink-0"
              >
                {actionStatus?.running
                  ? t.status.running
                  : actionStatus?.exit_code === 0
                    ? t.status.actionFinished
                    : actionStatus
                      ? `${t.status.actionFailed} (${actionStatus.exit_code ?? "?"})`
                      : t.common.loading}
              </Badge>
            </div>

            <button
              type="button"
              onClick={dismissLog}
              className="shrink-0 opacity-60 hover:opacity-100 cursor-pointer"
              aria-label={t.common.close}
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          <pre
            ref={logScrollRef}
            className="max-h-72 overflow-auto px-3 py-2 font-mono-ui text-[11px] leading-relaxed whitespace-pre-wrap break-all"
          >
            {actionStatus?.lines && actionStatus.lines.length > 0
              ? actionStatus.lines.join("\n")
              : t.status.waitingForOutput}
          </pre>
        </div>
      )}

      {platforms.length > 0 && (
        <PlatformsCard
          platforms={platforms}
          platformStateBadge={PLATFORM_STATE_BADGE}
        />
      )}

      {activeSessions.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-success" />
              <CardTitle className="text-base">
                {t.status.activeSessions}
              </CardTitle>
            </div>
          </CardHeader>

          <CardContent className="grid gap-3">
            {activeSessions.map((s) => (
              <div
                key={s.id}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border border-border p-3 w-full"
              >
                <div className="flex flex-col gap-1 min-w-0 w-full">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm truncate">
                      {s.title ?? t.common.untitled}
                    </span>

                    <Badge variant="success" className="text-[10px] shrink-0">
                      <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                      {t.common.live}
                    </Badge>
                  </div>

                  <span className="text-xs text-muted-foreground truncate">
                    <span className="font-mono-ui">
                      {(s.model ?? t.common.unknown).split("/").pop()}
                    </span>{" "}
                    · {s.message_count} {t.common.msgs} ·{" "}
                    {timeAgo(s.last_active)}
                  </span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {recentSessions.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-base">
                {t.status.recentSessions}
              </CardTitle>
            </div>
          </CardHeader>

          <CardContent className="grid gap-3">
            {recentSessions.map((s) => (
              <div
                key={s.id}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border border-border p-3 w-full"
              >
                <div className="flex flex-col gap-1 min-w-0 w-full">
                  <span className="font-medium text-sm truncate">
                    {s.title ?? t.common.untitled}
                  </span>

                  <span className="text-xs text-muted-foreground truncate">
                    <span className="font-mono-ui">
                      {(s.model ?? t.common.unknown).split("/").pop()}
                    </span>{" "}
                    · {s.message_count} {t.common.msgs} ·{" "}
                    {timeAgo(s.last_active)}
                  </span>

                  {s.preview && (
                    <span className="text-xs text-muted-foreground/70 truncate">
                      {s.preview}
                    </span>
                  )}
                </div>

                <Badge
                  variant="outline"
                  className="text-[10px] shrink-0 self-start sm:self-center"
                >
                  <Database className="mr-1 h-3 w-3" />
                  {s.source ?? "local"}
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function MarketIntelPanel({
  marketSources,
}: {
  marketSources: Record<MarketSourceKey, MarketSourceState>;
}) {
  const overviewSignals = extractSignalBullets(
    marketSources.overview.body || marketSources.btc.body,
  ).slice(0, 6);

  return (
    <Card className="overflow-hidden border-border/80 shadow-[0_0_0_1px_rgba(255,255,255,0.02),0_24px_80px_rgba(0,0,0,0.18)]">
      <CardHeader className="gap-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Newspaper className="h-5 w-5 text-primary" />
            <CardTitle className="text-base tracking-[0.14em]">
              Market intelligence
            </CardTitle>
          </div>
          <Badge variant="outline">5 feeds</Badge>
        </div>
        {overviewSignals.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {overviewSignals.slice(0, 4).map((signal, index) => (
              <Badge
                key={`${signal}-${index}`}
                variant="secondary"
                className="max-w-full normal-case tracking-normal"
              >
                {signal}
              </Badge>
            ))}
          </div>
        )}
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <Tabs defaultValue="overview">
          {(active, setActive) => {
            const selected = MARKET_SOURCE_TABS.find((tab) => tab.key === active) ??
              MARKET_SOURCE_TABS[0];
            const state = marketSources[selected.key];
            const summary = extractSignalBullets(state.body).slice(0, 5);

            return (
              <div className="flex flex-col gap-4">
                <TabsList className="h-auto flex-wrap gap-2 border-none">
                  {MARKET_SOURCE_TABS.map((tab) => (
                    <TabsTrigger
                      key={tab.key}
                      active={active === tab.key}
                      value={tab.key}
                      onClick={() => setActive(tab.key)}
                      className={
                        active === tab.key
                          ? "rounded-full border border-primary/30 bg-primary/10 px-3 py-2 text-primary after:hidden"
                          : "rounded-full border border-border bg-background/60 px-3 py-2 text-muted-foreground hover:border-primary/20 hover:bg-primary/[0.06]"
                      }
                    >
                      {tab.label}
                    </TabsTrigger>
                  ))}
                </TabsList>

                <div className="grid gap-4 lg:grid-cols-[0.9fr_1.6fr]">
                  <div className="space-y-3 rounded-2xl border border-border/70 bg-background/50 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                          Quick brief
                        </div>
                        <div className="mt-1 text-sm font-semibold text-foreground">
                          {selected.label}
                        </div>
                      </div>
                      <a
                        href={selected.url}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                      >
                        open
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    </div>

                    <div className="flex flex-wrap gap-2 text-[11px]">
                      <Badge variant={state.error ? "destructive" : state.loading ? "warning" : "success"}>
                        {state.error ? "error" : state.loading ? "loading" : "ready"}
                      </Badge>
                      {state.fetchedAt && (
                        <Badge variant="outline">
                          {new Date(state.fetchedAt).toLocaleTimeString()}
                        </Badge>
                      )}
                    </div>

                    {summary.length > 0 ? (
                      <ul className="space-y-2 text-sm leading-6 text-foreground/90">
                        {summary.map((item, index) => (
                          <li key={`${item}-${index}`} className="flex gap-2">
                            <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="rounded-xl border border-dashed border-border px-3 py-4 text-sm text-muted-foreground">
                        {state.error ?? (state.loading ? "正在拉取情报源…" : "暂无可提炼摘要")}
                      </div>
                    )}
                  </div>

                  <div className="min-h-[420px] rounded-2xl border border-border/70 bg-background/40 p-4">
                    {state.body ? (
                      <div className="max-h-[620px] overflow-auto pr-1">
                        <Markdown content={state.body} />
                      </div>
                    ) : (
                      <div className="flex h-full min-h-[320px] items-center justify-center text-sm text-muted-foreground">
                        {state.error ?? (state.loading ? "加载中…" : "暂无数据")}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          }}
        </Tabs>
      </CardContent>
    </Card>
  );
}

function TerminalMetric({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  tone: "ok" | "warn" | "neutral";
}) {
  const toneClass =
    tone === "ok"
      ? "border-emerald-500/20 bg-emerald-500/[0.08] text-emerald-300"
      : tone === "warn"
        ? "border-amber-500/20 bg-amber-500/[0.08] text-amber-200"
        : "border-border/80 bg-background/60 text-foreground";

  return (
    <div className={`rounded-2xl border p-3 ${toneClass}`}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-[11px] uppercase tracking-[0.18em] opacity-70">
          {label}
        </span>
        <Icon className="h-4 w-4 opacity-80" />
      </div>
      <div className="text-2xl font-semibold tracking-tight text-foreground">
        {value}
      </div>
    </div>
  );
}

function IntelSummaryCard({
  title,
  bullets,
  tone,
}: {
  title: string;
  bullets: string[];
  tone: "ok" | "warn";
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border p-3",
        tone === "ok"
          ? "border-emerald-500/20 bg-emerald-500/[0.05]"
          : "border-amber-500/20 bg-amber-500/[0.05]",
      )}
    >
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
        {title}
      </div>
      <div className="space-y-2 text-sm leading-6">
        {bullets.length > 0 ? (
          bullets.map((bullet, index) => (
            <div key={`${bullet}-${index}`} className="flex gap-2 text-foreground/90">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
              <span>{bullet}</span>
            </div>
          ))
        ) : (
          <div className="text-muted-foreground">暂无可提炼要点</div>
        )}
      </div>
    </div>
  );
}

function extractSignalBullets(markdown: string): string[] {
  return markdown
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter(
      (line) =>
        /^[-*+]\s+/.test(line) ||
        /^\d+[.)]\s+/.test(line) ||
        /^#{1,4}\s+/.test(line) ||
        /BTC|ETH|SOL|XRP|偏强|偏弱|突破|回踩|支撑|阻力|funding|OI|费率/i.test(line),
    )
    .map((line) =>
      line
        .replace(/^[-*+]\s+/, "")
        .replace(/^\d+[.)]\s+/, "")
        .replace(/^#{1,4}\s+/, "")
        .trim(),
    )
    .filter((line, index, arr) => arr.indexOf(line) === index)
    .slice(0, 12);
}

function PlatformsCard({ platforms, platformStateBadge }: PlatformsCardProps) {
  const { t } = useI18n();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Radio className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-base">
            {t.status.connectedPlatforms}
          </CardTitle>
        </div>
      </CardHeader>

      <CardContent className="grid gap-3">
        {platforms.map(([name, info]) => {
          const display = platformStateBadge[info.state] ?? {
            variant: "outline" as const,
            label: info.state,
          };
          const IconComponent =
            info.state === "connected"
              ? Wifi
              : info.state === "fatal"
                ? AlertTriangle
                : WifiOff;

          return (
            <div
              key={name}
              className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border border-border p-3 w-full"
            >
              <div className="flex items-center gap-3 min-w-0 w-full">
                <IconComponent
                  className={`h-4 w-4 shrink-0 ${
                    info.state === "connected"
                      ? "text-success"
                      : info.state === "fatal"
                        ? "text-destructive"
                        : "text-warning"
                  }`}
                />

                <div className="flex flex-col gap-0.5 min-w-0">
                  <span className="text-sm font-medium capitalize truncate">
                    {name}
                  </span>

                  {info.error_message && (
                    <span className="text-xs text-destructive">
                      {info.error_message}
                    </span>
                  )}

                  {info.updated_at && (
                    <span className="text-xs text-muted-foreground">
                      {t.status.lastUpdate}: {isoTimeAgo(info.updated_at)}
                    </span>
                  )}
                </div>
              </div>

              <Badge
                variant={display.variant}
                className="shrink-0 self-start sm:self-center"
              >
                {display.variant === "success" && (
                  <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                )}
                {display.label}
              </Badge>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

interface ToastState {
  message: string;
  type: "success" | "error";
}

interface PlatformsCardProps {
  platforms: [string, PlatformStatus][];
  platformStateBadge: Record<
    string,
    { variant: "success" | "warning" | "destructive"; label: string }
  >;
}
