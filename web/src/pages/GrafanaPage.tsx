import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  AlertTriangle,
  BarChart3,
  ExternalLink,
  LayoutGrid,
  RefreshCw,
  ShieldAlert,
} from "lucide-react";
import { api } from "@/lib/api";
import type { GrafanaConfig, GrafanaPanelConfig } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useI18n } from "@/i18n";

interface IncidentContext {
  incident_id: string;
  service: string;
  namespace: string;
  workload: string;
  from: string;
  to: string;
}

interface NormalizedGrafanaPanel {
  id: string;
  title: string;
  description: string;
  height: number;
  span: number;
}

const DEFAULT_CONTEXT: IncidentContext = {
  incident_id: "INC-2026-001",
  service: "checkout-api",
  namespace: "production",
  workload: "checkout-api",
  from: "now-6h",
  to: "now",
};

const CONTEXT_FIELDS: Array<keyof IncidentContext> = [
  "incident_id",
  "service",
  "namespace",
  "workload",
  "from",
  "to",
];

const CONTEXT_LABELS: Record<keyof IncidentContext, string> = {
  incident_id: "Incident",
  service: "Service",
  namespace: "Namespace",
  workload: "Workload",
  from: "From",
  to: "To",
};

export default function GrafanaPage() {
  const { t } = useI18n();
  const [config, setConfig] = useState<GrafanaConfig | null>(null);
  const [context, setContext] = useState<IncidentContext>(DEFAULT_CONTEXT);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadConfig = () => {
    setLoading(true);
    setError(null);
    api.getGrafanaConfig()
      .then((nextConfig) => {
        setConfig(nextConfig);
        setContext((current) => ({
          ...current,
          from: nextConfig.default_from || current.from,
          to: nextConfig.default_to || current.to,
        }));
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    const timer = window.setTimeout(loadConfig, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const panels = useMemo(() => normalizePanels(config?.panels), [config?.panels]);
  const configured = Boolean(
    config?.enabled
      && config.base_url
      && config.dashboard_uid
      && panels.length > 0
      && panels.some((panel) => panel.id),
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-muted-foreground" />
            <h1 className="font-expanded text-lg font-bold uppercase tracking-[0.08em]">
              {t.grafana.title}
            </h1>
          </div>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            {t.grafana.subtitle}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant={configured ? "success" : "warning"}>
            {configured ? t.common.configured : t.grafana.notConfiguredBadge}
          </Badge>
          <Button variant="outline" size="sm" onClick={loadConfig}>
            <RefreshCw className="h-3.5 w-3.5" />
            {t.common.refresh}
          </Button>
        </div>
      </div>

      {error && (
        <div className="border border-destructive/30 bg-destructive/[0.06] p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-destructive">{t.grafana.failedToLoad}</p>
              <p className="mt-1 break-words text-xs text-destructive/70">{error}</p>
            </div>
          </div>
        </div>
      )}

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <LayoutGrid className="h-5 w-5 text-muted-foreground" />
            <CardTitle>{t.grafana.contextTitle}</CardTitle>
          </div>
          <CardDescription>{t.grafana.contextDescription}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          {CONTEXT_FIELDS.map((field) => (
            <label key={field} className="flex min-w-0 flex-col gap-1">
              <span className="font-display text-xs uppercase tracking-[0.08em] text-muted-foreground">
                {CONTEXT_LABELS[field]}
              </span>
              <input
                className="h-9 w-full border border-input bg-background px-3 font-mono-ui text-xs text-foreground outline-none transition-colors focus:border-foreground"
                value={context[field]}
                onChange={(event) => setContext({ ...context, [field]: event.target.value })}
                aria-label={CONTEXT_LABELS[field]}
              />
            </label>
          ))}
        </CardContent>
      </Card>

      {!configured && (
        <EmptyConfigState config={config} />
      )}

      {configured && config && (
        <div
          className="grid gap-4"
          style={{ gridTemplateColumns: "repeat(12, minmax(0, 1fr))" }}
        >
          {panels.map((panel, index) => {
            const url = buildGrafanaPanelUrl(config, panel, context);
            const span = clampSpan(panel.span);
            return (
              <GrafanaPanelCard
                key={`${panel.id}-${index}-${url}`}
                panel={panel}
                url={url}
                span={span}
                fallbackText={config.fallback_text || t.grafana.defaultFallbackText}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function EmptyConfigState({ config }: { config: GrafanaConfig | null }) {
  const { t } = useI18n();
  const missing = [
    !config?.enabled ? "aiops.grafana.enabled" : null,
    !config?.base_url ? "aiops.grafana.base_url" : null,
    !config?.dashboard_uid ? "aiops.grafana.dashboard_uid" : null,
    !normalizePanels(config?.panels).some((panel) => panel.id) ? "aiops.grafana.panels" : null,
  ].filter(Boolean);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-5 w-5 text-warning" />
          <CardTitle>{t.grafana.emptyTitle}</CardTitle>
        </div>
        <CardDescription>{t.grafana.emptyDescription}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {missing.map((key) => (
            <div key={key} className="border border-border bg-background/50 p-3">
              <span className="font-mono-ui text-xs text-muted-foreground">{key}</span>
            </div>
          ))}
        </div>
        <p className="max-w-4xl text-xs text-muted-foreground">
          {t.grafana.securityNote}
        </p>
      </CardContent>
    </Card>
  );
}

function GrafanaPanelCard({
  panel,
  url,
  span,
  fallbackText,
}: {
  panel: NormalizedGrafanaPanel;
  url: string;
  span: number;
  fallbackText: string;
}) {
  const { t } = useI18n();
  const [failed, setFailed] = useState(false);
  const loadedRef = useRef(false);

  useEffect(() => {
    loadedRef.current = false;
    const timer = window.setTimeout(() => {
      if (!loadedRef.current) setFailed(true);
    }, 8000);
    return () => window.clearTimeout(timer);
  }, [url]);

  return (
    <Card
      className="grafana-panel-card min-w-0 overflow-hidden"
      style={{ "--panel-span": String(span) } as CSSProperties}
    >
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div className="min-w-0">
          <CardTitle className="truncate">{panel.title || `${t.grafana.panel} ${panel.id}`}</CardTitle>
          {panel.description && (
            <CardDescription className="mt-1 line-clamp-2">{panel.description}</CardDescription>
          )}
        </div>
        <a
          className="inline-flex h-8 w-8 shrink-0 items-center justify-center border border-border text-muted-foreground transition-colors hover:bg-foreground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          href={url}
          target="_blank"
          rel="noreferrer"
          aria-label={t.grafana.openPanel}
          title={t.grafana.openPanel}
        >
          <ExternalLink className="h-4 w-4" />
        </a>
      </CardHeader>
      <CardContent className="p-0">
        <div
          className="relative w-full bg-background"
          style={{ height: Math.max(panel.height, 220) }}
        >
          {failed && (
            <div className="absolute inset-0 z-10 flex items-center justify-center border-t border-border bg-card/95 p-4">
              <div className="max-w-xl text-center">
                <AlertTriangle className="mx-auto h-5 w-5 text-warning" />
                <p className="mt-2 text-sm font-medium">{t.grafana.embedBlocked}</p>
                <p className="mt-1 text-xs text-muted-foreground">{fallbackText}</p>
                <a
                  className="mt-3 inline-flex items-center gap-2 border border-border px-3 py-2 text-xs uppercase tracking-[0.08em] transition-colors hover:bg-foreground/10"
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  {t.grafana.openPanel}
                </a>
              </div>
            </div>
          )}
          <iframe
            title={panel.title || `${t.grafana.panel} ${panel.id}`}
            src={url}
            className="h-full w-full border-0"
            loading="lazy"
            referrerPolicy="no-referrer-when-downgrade"
            onLoad={() => {
              loadedRef.current = true;
              setFailed(false);
            }}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function normalizePanels(panels: GrafanaConfig["panels"] | undefined): NormalizedGrafanaPanel[] {
  if (!Array.isArray(panels)) return [];

  return panels
    .filter((panel): panel is GrafanaPanelConfig => Boolean(panel && typeof panel === "object"))
    .map((panel, index) => ({
      id: String(panel.id ?? ""),
      title: String(panel.title || `Panel ${index + 1}`),
      description: String(panel.description || ""),
      height: coerceNumber(panel.height, 320),
      span: coerceNumber(panel.span, 6),
    }));
}

function buildGrafanaPanelUrl(
  config: GrafanaConfig,
  panel: GrafanaPanelConfig,
  context: IncidentContext,
): string {
  const baseUrl = config.base_url.replace(/\/+$/, "");
  const dashboardUid = encodeURIComponent(config.dashboard_uid);
  const slug = encodeURIComponent(config.dashboard_slug || "aiops");
  const normalizedBaseUrl = /^https?:\/\//i.test(baseUrl) ? baseUrl : `https://${baseUrl}`;
  const url = new URL(`${normalizedBaseUrl}/d-solo/${dashboardUid}/${slug}`);
  const params = url.searchParams;

  params.set("panelId", String(panel.id));
  params.set("from", context.from || config.default_from || DEFAULT_CONTEXT.from);
  params.set("to", context.to || config.default_to || DEFAULT_CONTEXT.to);

  if (config.org_id) params.set("orgId", config.org_id);
  if (config.theme) params.set("theme", config.theme);
  if (config.timezone) params.set("timezone", config.timezone);
  if (config.kiosk) params.set("kiosk", "tv");

  addGrafanaVariable(params, config.variable_map?.service, context.service);
  addGrafanaVariable(params, config.variable_map?.namespace, context.namespace);
  addGrafanaVariable(params, config.variable_map?.workload, context.workload);
  addGrafanaVariable(params, config.variable_map?.incident_id, context.incident_id);

  return url.toString();
}

function addGrafanaVariable(params: URLSearchParams, name: string | undefined, value: string) {
  if (!name || !value) return;
  params.set(`var-${name}`, value);
}

function coerceNumber(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function clampSpan(value: number): number {
  if (value <= 4) return 4;
  if (value >= 12) return 12;
  return 6;
}
