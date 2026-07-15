import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, Copy, ExternalLink, RefreshCw, Server, Wallet } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api, type OpsService, type WebhubOpsSummaryResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

function groupServices(services: OpsService[]): Array<[string, OpsService[]]> {
  const grouped = new Map<string, OpsService[]>();
  for (const service of services) {
    const bucket = grouped.get(service.group) ?? [];
    bucket.push(service);
    grouped.set(service.group, bucket);
  }
  return [...grouped.entries()];
}

function serviceAccent(service: OpsService): string {
  const key = `${service.id} ${service.title}`.toLowerCase();
  if (key.includes("payment")) return "from-[#f2d2a2]/30 via-[#8f5f2c]/15 to-transparent";
  if (key.includes("grafana") || key.includes("metric")) return "from-[#d86f45]/30 via-[#58332d]/10 to-transparent";
  if (key.includes("chat")) return "from-[#9bc9c2]/30 via-[#2e5e5b]/10 to-transparent";
  return "from-[#c1d8d0]/20 via-[#34453f]/10 to-transparent";
}

function serviceIcon(service: OpsService) {
  const key = `${service.id} ${service.title}`.toLowerCase();
  if (key.includes("payment")) return Wallet;
  return Server;
}

function formatUsd(value: number): string {
  return `$${value.toFixed(value >= 10 ? 2 : 4)}`;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export default function OpsPage() {
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();
  const [services, setServices] = useState<OpsService[]>([]);
  const [webhub, setWebhub] = useState<WebhubOpsSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = async (mode: "initial" | "refresh" = "initial") => {
    if (mode === "refresh") setRefreshing(true);
    try {
      const [servicesResponse, webhubResponse] = await Promise.all([
        api.getOpsServices(),
        api.getWebhubOpsSummary(),
      ]);
      setServices(servicesResponse.services);
      setWebhub(webhubResponse);
    } catch (error) {
      showToast(`Could not load ops data: ${error}`, "error");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    setEnd(
      <Button ghost type="button" onClick={() => void load("refresh")} disabled={refreshing}>
        {refreshing ? <Spinner /> : <RefreshCw className="h-4 w-4" />}
        Refresh
      </Button>,
    );
    return () => setEnd(null);
  }, [refreshing, setEnd]);

  const grouped = useMemo(() => groupServices(services), [services]);
  const origin = typeof window !== "undefined" ? window.location.origin : "";

  const copyUrl = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      showToast("Copied link", "success");
    } catch (error) {
      showToast(`Could not copy link: ${error}`, "error");
    }
  };

  const openUrl = (value: string) => {
    window.open(value, "_blank", "noopener,noreferrer");
  };

  if (loading) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-8">
      <Toast toast={toast} />
      <section className="relative overflow-hidden rounded-[28px] border border-border/70 bg-background/95 p-6 shadow-[0_24px_90px_rgba(0,0,0,0.18)]">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(211,190,149,0.16),transparent_38%),radial-gradient(circle_at_bottom_right,rgba(112,163,154,0.12),transparent_42%)]" />
        <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl space-y-3">
            <p className="font-mondwest text-[0.72rem] uppercase tracking-[0.24em] text-muted-foreground">
              Tailnet Ops
            </p>
            <h1 className="font-mondwest text-3xl uppercase tracking-[0.08em] text-foreground">
              Mac Mini Service Directory
            </h1>
            <p className="max-w-xl text-sm text-muted-foreground">
              One jump page for the Hermes surfaces and nearby services you use from your
              Mac mini, MacBook, phone, and iPad over Tailscale.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm text-muted-foreground sm:grid-cols-3">
            <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
              <p className="font-mondwest text-[0.65rem] uppercase tracking-[0.18em]">Services</p>
              <p className="mt-2 text-xl text-foreground">{services.length}</p>
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
              <p className="font-mondwest text-[0.65rem] uppercase tracking-[0.18em]">Host</p>
              <p className="mt-2 truncate text-foreground">{origin.replace(/^https?:\/\//, "") || "Unknown"}</p>
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
              <p className="font-mondwest text-[0.65rem] uppercase tracking-[0.18em]">Mode</p>
              <p className="mt-2 text-foreground">Tailscale only</p>
            </div>
          </div>
        </div>
      </section>

      {webhub && (
        <section className="space-y-3">
          <div className="flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="font-mondwest text-sm uppercase tracking-[0.18em] text-foreground">
                Webhub Observability
              </p>
              <p className="text-sm text-muted-foreground">
                OpenRouter cost, throughput, and latency feeding the dashboard, Prometheus,
                and the rolling morning-briefing snippet.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge tone={webhub.status.plugin_enabled ? "secondary" : "outline"}>
                Plugin {webhub.status.plugin_enabled ? "enabled" : "disabled"}
              </Badge>
              <Badge tone={webhub.status.openrouter_configured ? "secondary" : "outline"}>
                OpenRouter {webhub.status.openrouter_configured ? "ready" : "missing key"}
              </Badge>
              <Badge tone={webhub.status.slack_configured ? "secondary" : "outline"}>
                Slack {webhub.status.slack_configured ? "ready" : "configure in Channels"}
              </Badge>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Card className="border-border/70 bg-background/90">
              <CardContent className="p-5">
                <p className="font-mondwest text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground">
                  Estimated Cost
                </p>
                <p className="mt-2 text-2xl text-foreground">
                  {formatUsd(webhub.summary.cost.estimated_usd)}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Last {webhub.summary.window_hours}h
                </p>
              </CardContent>
            </Card>
            <Card className="border-border/70 bg-background/90">
              <CardContent className="p-5">
                <p className="font-mondwest text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground">
                  Throughput
                </p>
                <p className="mt-2 text-2xl text-foreground">
                  {webhub.summary.requests.throughput_per_hour.toFixed(2)}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  requests/hour
                </p>
              </CardContent>
            </Card>
            <Card className="border-border/70 bg-background/90">
              <CardContent className="p-5">
                <p className="font-mondwest text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground">
                  Success Rate
                </p>
                <p className="mt-2 text-2xl text-foreground">
                  {formatPercent(webhub.summary.requests.success_rate)}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {webhub.summary.requests.ok} ok / {webhub.summary.requests.error} errors
                </p>
              </CardContent>
            </Card>
            <Card className="border-border/70 bg-background/90">
              <CardContent className="p-5">
                <p className="font-mondwest text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground">
                  Avg Latency
                </p>
                <p className="mt-2 text-2xl text-foreground">
                  {webhub.summary.latency.avg_ms.toFixed(1)} ms
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {webhub.summary.requests.total} requests observed
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
            <Card className="border-border/70 bg-background/90">
              <CardContent className="space-y-4 p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h2 className="font-mondwest text-lg uppercase tracking-[0.08em] text-foreground">
                      Dashboard Feed
                    </h2>
                    <p className="text-sm text-muted-foreground">
                      Rolling summary written to <span className="font-mono">{webhub.status.briefing_file}</span>
                    </p>
                  </div>
                  <Button ghost type="button" onClick={() => openUrl(webhub.links.prometheus_url)}>
                    Prometheus
                    <ExternalLink className="h-4 w-4" />
                  </Button>
                </div>
                <pre className="overflow-x-auto rounded-2xl border border-border/70 bg-background/85 p-4 text-xs text-muted-foreground">
                  {webhub.summary.briefing_markdown}
                </pre>
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-background/90">
              <CardContent className="space-y-4 p-5">
                <div>
                  <h2 className="font-mondwest text-lg uppercase tracking-[0.08em] text-foreground">
                    Top Models
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    Highest request volume across the rolling window.
                  </p>
                </div>
                <div className="space-y-3">
                  {webhub.summary.top_models.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      No OpenRouter traffic captured yet.
                    </p>
                  ) : (
                    webhub.summary.top_models.map((item) => (
                      <div
                        key={item.model}
                        className="rounded-2xl border border-border/70 bg-background/85 px-4 py-3"
                      >
                        <p className="truncate font-mono text-sm text-foreground">{item.model}</p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {item.requests} requests • {formatUsd(item.estimated_cost_usd)} •{" "}
                          {item.avg_latency_ms.toFixed(1)} ms
                        </p>
                      </div>
                    ))
                  )}
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button type="button" onClick={() => openUrl(webhub.links.grafana_url)}>
                    Open Grafana
                    <ArrowUpRight className="h-4 w-4" />
                  </Button>
                  <Button ghost type="button" onClick={() => openUrl(webhub.links.channels_url)}>
                    Slack / Channels
                    <ExternalLink className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {webhub.summary.recent_errors.length > 0 && (
            <Card className="border-border/70 bg-background/90">
              <CardContent className="space-y-3 p-5">
                <div>
                  <h2 className="font-mondwest text-lg uppercase tracking-[0.08em] text-foreground">
                    Recent Errors
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    Latest OpenRouter failures captured by the webhub sink.
                  </p>
                </div>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {webhub.summary.recent_errors.map((item, index) => (
                    <div
                      key={`${item.observed_at}-${item.model}-${index}`}
                      className="rounded-2xl border border-border/70 bg-background/85 px-4 py-3"
                    >
                      <p className="truncate font-mono text-xs text-foreground">{item.model || "unknown model"}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{item.error_type || "request_error"}</p>
                      <p className="mt-2 line-clamp-3 text-sm text-muted-foreground">
                        {item.error_message || "No error message captured."}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </section>
      )}

      {grouped.map(([group, items]) => (
        <section key={group} className="space-y-3">
          <div>
            <p className="font-mondwest text-sm uppercase tracking-[0.18em] text-foreground">
              {group}
            </p>
            <p className="text-sm text-muted-foreground">
              {group === "Hermes"
                ? "Primary Hermes surfaces on the Mac mini."
                : "Adjacent infrastructure and observability endpoints."}
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {items.map((service) => {
              const Icon = serviceIcon(service);
              return (
                <Card
                  key={service.id}
                  className={cn(
                    "overflow-hidden border-border/70 bg-background/90",
                    "shadow-[0_12px_40px_rgba(0,0,0,0.12)]",
                  )}
                >
                  <CardContent className="relative p-0">
                    <div className={cn("absolute inset-0 bg-gradient-to-br", serviceAccent(service))} />
                    <div className="relative flex h-full flex-col gap-4 p-5">
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-foreground">
                            <Icon className="h-4 w-4" />
                            <h2 className="font-mondwest text-lg uppercase tracking-[0.08em]">
                              {service.title}
                            </h2>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {service.description || "No description configured."}
                          </p>
                        </div>
                        <Badge tone="outline">{service.group}</Badge>
                      </div>

                      <div className="rounded-2xl border border-border/70 bg-background/85 px-3 py-3">
                        <p className="line-clamp-2 break-all font-mono text-xs text-muted-foreground">
                          {service.url}
                        </p>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        {service.tags.map((tag) => (
                          <Badge key={tag} tone="secondary">
                            {tag}
                          </Badge>
                        ))}
                      </div>

                      <div className="mt-auto flex items-center gap-2">
                        <Button type="button" className="flex-1" onClick={() => openUrl(service.url)}>
                          Open
                          <ArrowUpRight className="h-4 w-4" />
                        </Button>
                        <Button
                          ghost
                          type="button"
                          onClick={() => void copyUrl(service.url)}
                          aria-label={`Copy ${service.title} URL`}
                        >
                          <Copy className="h-4 w-4" />
                        </Button>
                        <Button
                          ghost
                          type="button"
                          aria-label={`Open ${service.title} in a new tab`}
                          onClick={() => openUrl(service.url)}
                        >
                          <ExternalLink className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
