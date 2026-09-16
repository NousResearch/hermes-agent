import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { Gauge, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api, type UsageQuotaResponse, type UsageQuotaRoute, type UsageQuotaSnapshot } from "@/lib/api";

function formatDate(value: string | null): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatMetric(value: number | null, unit: string | null): string {
  if (value == null || !Number.isFinite(value)) return "Unknown";
  const rendered = new Intl.NumberFormat().format(value);
  return unit ? `${rendered} ${unit}` : `${rendered} (unit unknown)`;
}

function formatReset(value: string | null): string {
  if (!value) return "Reset unknown";
  const resetAt = new Date(value);
  if (Number.isNaN(resetAt.getTime())) return "Reset unknown";

  const seconds = Math.floor((resetAt.getTime() - Date.now()) / 1000);
  if (seconds <= 0) return "Resets now";
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  if (days > 0) return `Resets in ${days}d ${hours % 24}h`;
  if (hours > 0) return `Resets in ${hours}h ${minutes % 60}m`;
  return `Resets in ${Math.max(1, minutes)}m`;
}

function validRemainingPercent(route: UsageQuotaRoute): number | null {
  const value = route.remaining_percent;
  if (value == null || !Number.isFinite(value) || value < 0 || value > 100) return null;
  return Math.round(value * 10) / 10;
}

function formatPercent(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);
}

function routeStatusLabel(route: UsageQuotaRoute): string {
  if (route.status === "reported") return "Reported";
  if (route.status === "unavailable") return "Unavailable";
  return "Unknown";
}

function routeStatusClass(route: UsageQuotaRoute): string {
  if (route.status === "reported") return "bg-emerald-500";
  if (route.status === "unknown") return "bg-amber-500";
  return "bg-muted-foreground";
}

function routeStatusTextClass(route: UsageQuotaRoute): string {
  if (route.status === "reported") return "text-emerald-600";
  if (route.status === "unknown") return "text-amber-600";
  return "text-muted-foreground";
}

function RouteRow({ route }: { route: UsageQuotaRoute }) {
  const status = routeStatusLabel(route);
  const remainingPercent = validRemainingPercent(route);
  const hasGauge = route.status === "reported" && remainingPercent != null;
  const gaugeColor = remainingPercent != null && remainingPercent <= 20
    ? "bg-destructive"
    : remainingPercent != null && remainingPercent <= 50
      ? "bg-amber-500"
      : "bg-emerald-500";

  return (
    <div data-testid="usage-quota-route" className="space-y-3 rounded-lg border border-border/80 bg-muted/20 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${routeStatusClass(route)}`} aria-hidden="true" />
          <span className="truncate font-medium">{route.route}</span>
        </div>
        <span className={`text-xs ${routeStatusTextClass(route)}`}>{status}</span>
      </div>

      {hasGauge && (
        <div className="space-y-2">
          <div
            className="h-2.5 overflow-hidden rounded-full bg-muted"
            role="progressbar"
            aria-label={`${route.route} quota remaining`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={remainingPercent}
          >
            <div className={`h-full rounded-full transition-[width] ${gaugeColor}`} style={{ width: `${remainingPercent}%` }} />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="text-muted-foreground">
              {formatMetric(route.usage, route.unit)} used of {formatMetric(route.limit, route.unit)}
            </span>
            <span className="font-medium text-foreground">{formatPercent(remainingPercent)}% remaining</span>
          </div>
        </div>
      )}

      <div className="grid gap-x-4 gap-y-1 text-xs text-muted-foreground sm:grid-cols-3">
        <span>Used {formatMetric(route.usage, route.unit)}</span>
        <span>Limit {formatMetric(route.limit, route.unit)}</span>
        <span>Remaining {formatMetric(route.remaining, route.unit)}</span>
      </div>

      <div className="flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
        <span>{formatReset(route.reset_at)}</span>
        {route.detail && <span>{route.detail}</span>}
      </div>
    </div>
  );
}

type AccountGroup = {
  account: string;
  provider: string;
  routes: UsageQuotaRoute[];
};

const ANTIGRAVITY_PROVIDER = "antigravity";
const ANTIGRAVITY_FAMILY_LABELS = [
  "gemini_5hr",
  "claude_gpt_5hr",
  "gemini_weekly",
  "claude_gpt_weekly",
] as const;

type AntigravityFamilyLabel = (typeof ANTIGRAVITY_FAMILY_LABELS)[number];

type AntigravityFamilyConfig = {
  label: AntigravityFamilyLabel;
  matchesModel?: (name: string) => boolean;
};

const ANTIGRAVITY_FAMILIES: readonly AntigravityFamilyConfig[] = [
  {
    label: "gemini_5hr",
    matchesModel: (name: string) => name.startsWith("gemini-") || name.startsWith("gemini_"),
  },
  {
    label: "claude_gpt_5hr",
    matchesModel: (name: string) =>
      name.startsWith("claude-") || name.startsWith("claude_") || name.startsWith("gpt-") || name.startsWith("gpt_"),
  },
  {
    label: "gemini_weekly",
  },
  {
    label: "claude_gpt_weekly",
  },
] as const;

function isAntigravityProvider(provider: string): boolean {
  return provider.trim().toLowerCase() === ANTIGRAVITY_PROVIDER;
}

function isAntigravityFamilyLabel(name: string): boolean {
  return (ANTIGRAVITY_FAMILY_LABELS as readonly string[]).includes(name);
}

function quotaName(route: UsageQuotaRoute): string {
  const name = route.route.trim().toLowerCase();
  return name;
}

function hasSameQuotaWindowValues(left: UsageQuotaRoute, right: UsageQuotaRoute): boolean {
  return left.usage === right.usage
    && left.limit === right.limit
    && (left.remaining ?? null) === (right.remaining ?? null)
    && (left.remaining_percent ?? null) === (right.remaining_percent ?? null)
    && left.unit === right.unit
    && left.reset_at === right.reset_at
    && left.status === right.status;
}

function isValidReportedRoute(route: UsageQuotaRoute): boolean {
  if (route.status !== "reported") return false;
  if (route.limit == null || !Number.isFinite(route.limit) || route.limit <= 0) return false;
  if (route.usage == null || !Number.isFinite(route.usage) || route.usage < 0 || route.usage > route.limit) return false;
  if (route.remaining != null) {
    if (!Number.isFinite(route.remaining) || route.remaining < 0 || route.remaining > route.limit) return false;
    if (Math.abs(route.usage + route.remaining - route.limit) > 1e-4) return false;
  } else if (route.remaining_percent == null) {
    return false;
  }
  if (route.remaining_percent != null) {
    if (!Number.isFinite(route.remaining_percent) || route.remaining_percent < 0 || route.remaining_percent > 100) {
      return false;
    }
    const expectedPercent = route.remaining != null
      ? (route.remaining / route.limit) * 100
      : ((route.limit - route.usage) / route.limit) * 100;
    if (Math.abs(route.remaining_percent - expectedPercent) > 0.6) {
      return false;
    }
  }
  if (route.reset_at != null) {
    if (typeof route.reset_at !== "string" || !route.reset_at.trim()) return false;
    const timestamp = Date.parse(route.reset_at);
    if (Number.isNaN(timestamp)) return false;
  }
  return true;
}

function unknownFamilyRoute(label: string, base: UsageQuotaRoute | null, detail: string): UsageQuotaRoute {
  return {
    route: label,
    provider: base?.provider || ANTIGRAVITY_PROVIDER,
    account: base?.account || null,
    usage: null,
    limit: null,
    remaining: null,
    remaining_percent: null,
    unit: null,
    reset_at: null,
    status: "unknown",
    source: base?.source || "",
    detail,
  };
}

function resolveAntigravityFamilyRoute(
  family: AntigravityFamilyConfig,
  routes: UsageQuotaRoute[],
  fallback: UsageQuotaRoute | null,
): UsageQuotaRoute {
  const { label, matchesModel } = family;
  const explicitRows = routes.filter((route) => quotaName(route) === label);
  const modelRoutes = matchesModel
    ? routes.filter(
        (route) =>
          !isAntigravityFamilyLabel(quotaName(route)) && matchesModel(quotaName(route)),
      )
    : [];

  if (explicitRows.length > 1) {
    return unknownFamilyRoute(
      label,
      explicitRows[0] || fallback,
      `Multiple explicit quota windows were reported for ${label}; the quota is unavailable.`,
    );
  }

  if (explicitRows.length === 1) {
    const explicitRow = explicitRows[0];
    const allRelevantRows = [explicitRow, ...modelRoutes];
    const allEqual = allRelevantRows.every((route) => hasSameQuotaWindowValues(route, explicitRow));
    if (!allEqual || !isValidReportedRoute(explicitRow)) {
      const detail = !allEqual
        ? `Antigravity quota windows for ${label} were inconsistent; the quota is unavailable.`
        : `Antigravity reported unusable quota data for ${label}; the quota is unavailable.`;
      return unknownFamilyRoute(label, explicitRow, detail);
    }
    return { ...explicitRow, route: label };
  }

  if (modelRoutes.length === 0) {
    return unknownFamilyRoute(label, fallback, `Antigravity did not report the ${label} quota window.`);
  }

  const representative = modelRoutes[0];
  const allEqual = modelRoutes.every((route) => hasSameQuotaWindowValues(route, representative));
  if (!allEqual || !isValidReportedRoute(representative)) {
    const detail = !allEqual
      ? `Antigravity model quota windows for ${label} were inconsistent; the quota is unavailable.`
      : `Antigravity reported unusable quota data for ${label}; the quota is unavailable.`;
    return unknownFamilyRoute(label, representative, detail);
  }

  // The model rows represent one shared account window. Copy the common row;
  // never sum model limits or usage into an invented aggregate.
  return { ...representative, route: label };
}

function displayRoutesForAccount(provider: string, routes: UsageQuotaRoute[]): UsageQuotaRoute[] {
  if (!isAntigravityProvider(provider)) return routes;

  const fallback = routes[0] || null;
  return ANTIGRAVITY_FAMILIES.map((family) =>
    resolveAntigravityFamilyRoute(family, routes, fallback),
  );
}

function groupRoutesByAccount(routes: UsageQuotaRoute[]): AccountGroup[] {
  const providerGroups = new Map<string, Map<string, AccountGroup>>();
  routes.forEach((route) => {
    const account = route.account?.trim() || route.provider?.trim() || "Account unknown";
    const provider = route.provider?.trim() || "Provider unknown";
    let accountMap = providerGroups.get(provider);
    if (!accountMap) {
      accountMap = new Map<string, AccountGroup>();
      providerGroups.set(provider, accountMap);
    }
    const existing = accountMap.get(account);
    if (existing) {
      existing.routes.push(route);
      return;
    }
    accountMap.set(account, { account, provider, routes: [route] });
  });
  const result: AccountGroup[] = [];
  for (const accountMap of providerGroups.values()) {
    for (const group of accountMap.values()) {
      result.push({
        ...group,
        routes: displayRoutesForAccount(group.provider, group.routes),
      });
    }
  }
  return result;
}

function AccountCard({ group }: { group: AccountGroup }) {
  const quotaCount = group.routes.length;
  return (
    <details
      data-testid="usage-quota-account-card"
      open
      className="space-y-3 rounded-xl border border-border bg-card/70 p-4 shadow-sm"
    >
      <summary className="flex cursor-pointer list-none items-start justify-between gap-3 rounded-lg [&::-webkit-details-marker]:hidden">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold">{group.account}</h3>
          <p className="mt-1 font-mono text-[11px] text-muted-foreground">{group.provider}</p>
        </div>
        <span className="rounded-full border border-border px-2 py-1 text-[11px] text-muted-foreground">
          {quotaCount} {quotaCount === 1 ? "quota" : "quotas"}
        </span>
      </summary>
      <div className="space-y-2">
        {group.routes.map((route, index) => (
          <RouteRow key={`${route.route}:${route.source}:${index}`} route={route} />
        ))}
      </div>
    </details>
  );
}

function PercentageWindow({ window }: { window: UsageQuotaSnapshot["windows"][number] }) {
  const used = window.used_percent;
  const remaining = used == null ? null : Math.max(0, Math.round(100 - used));
  return (
    <div className="space-y-2 rounded-lg border border-border/80 bg-muted/20 p-4">
      <div className="flex justify-between text-sm">
        <span>{window.label}</span>
        <span className="font-mono text-muted-foreground">
          {remaining == null ? "Remaining unknown" : `${remaining}% remaining`}
        </span>
      </div>
      {remaining != null && (
        <div
          className="h-2.5 overflow-hidden rounded-full bg-muted"
          role="progressbar"
          aria-label={`${window.label} quota remaining`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={remaining}
        >
          <div className={`h-full rounded-full ${remaining <= 20 ? "bg-destructive" : remaining <= 50 ? "bg-amber-500" : "bg-emerald-500"}`} style={{ width: `${remaining}%` }} />
        </div>
      )}
      <div className="flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
        <span>{used == null ? "Usage unknown" : `${Math.round(used)}% used`}</span>
        <span>{window.reset_at ? `Resets ${formatDate(window.reset_at)}` : (window.detail || "Reset unknown")}</span>
      </div>
    </div>
  );
}

function ProviderCard({ snapshot }: { snapshot: UsageQuotaSnapshot }) {
  const status = snapshot.partial ? "Partial" : snapshot.available ? "Reported" : "Unavailable";
  const statusClass = snapshot.available && !snapshot.partial ? "text-emerald-600" : "text-muted-foreground";
  const routes = snapshot.routes ?? [];
  const accountGroups = groupRoutesByAccount(routes);
  const providerName = snapshot.provider === "9router" ? "9Router" : snapshot.title || snapshot.provider;
  const isNineRouter = snapshot.provider === "9router";

  return (
    <Card className={isNineRouter ? "col-span-full border-0 bg-transparent p-0 shadow-none" : "overflow-hidden"}>
      <CardHeader className={isNineRouter ? "px-0" : "border-b border-border/70 bg-muted/10"}>
        <div className="flex items-start justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-border bg-card">
              <Gauge className="h-5 w-5 text-muted-foreground" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <CardTitle className="truncate text-base">{providerName}</CardTitle>
              <p className="mt-1 text-xs text-muted-foreground">{snapshot.title || "Provider account limits"}</p>
            </div>
          </div>
          <span className={`shrink-0 text-xs ${statusClass}`}>{status}</span>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
          <span className="font-mono">{snapshot.provider}</span>
          <span>Read-only provider data</span>
        </div>
      </CardHeader>
      <CardContent className={isNineRouter ? "space-y-4 p-0" : "space-y-4 pt-5"}>
        {snapshot.plan && <p className="text-sm text-muted-foreground">Plan: {snapshot.plan}</p>}
        {routes.length > 0 && (
          <div data-testid="usage-quota-route-breakdown" className="space-y-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-sm font-medium">Accounts &amp; route quotas</h2>
              <span className="text-xs text-muted-foreground">No combined total</span>
            </div>
            <div className={isNineRouter ? "grid gap-4 md:grid-cols-2" : "space-y-3"}>
              {accountGroups.map((group) => <AccountCard key={JSON.stringify([group.provider, group.account])} group={group} />)}
            </div>
          </div>
        )}
        {snapshot.windows.length > 0 ? snapshot.windows.map((window) => <PercentageWindow key={window.label} window={window} />) : routes.length === 0 && (
          <p className="text-sm text-muted-foreground">
            {snapshot.provider === "9router"
              ? "Route breakdown is unavailable until the 9Router management session is authorized."
              : "No percentage quota window was reported by this provider."}
          </p>
        )}
        {snapshot.details.length > 0 && <div className="space-y-1 border-t border-border pt-3 text-sm text-muted-foreground">{snapshot.details.map((detail) => <p key={detail}>{detail}</p>)}</div>}
        {snapshot.unavailable_reason && <p className="text-sm text-muted-foreground">{snapshot.unavailable_reason}</p>}
        <p className="border-t border-border pt-3 text-xs text-muted-foreground">
          Source: {snapshot.source || "Unknown"} · Updated {formatDate(snapshot.fetched_at)}
          {snapshot.scope ? ` · Scope: ${snapshot.scope}` : ""}
          {snapshot.stale ? " · Stale data" : ""}
          {snapshot.partial ? " · Partial data" : ""}
        </p>
      </CardContent>
    </Card>
  );
}

export default function UsageQuotaPage() {
  const [data, setData] = useState<UsageQuotaResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setAfterTitle, setEnd } = usePageHeader();
  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.getUsageQuota().then(setData).catch((err) => setError(String(err))).finally(() => setLoading(false));
  }, []);
  useLayoutEffect(() => {
    setAfterTitle(null);
    setEnd(<Button type="button" ghost size="icon" onClick={load} disabled={loading} aria-label="Refresh quota">{loading ? <Spinner /> : <RefreshCw />}</Button>);
    return () => { setAfterTitle(null); setEnd(null); };
  }, [load, loading, setAfterTitle, setEnd]);
  useEffect(() => { load(); }, [load]);
  return (
    <div className="flex flex-col gap-6">
      <p className="text-sm text-muted-foreground">Provider account limits from official provider APIs, including 9Router management data when authorized. Values are not estimated and are not local analytics or billing data. 9Router accounts and routes are shown independently; no combined total is reported until units and reset windows are confirmed.</p>
      {loading && !data && <div className="flex justify-center py-24"><Spinner className="text-2xl text-primary" /></div>}
      {error && <Card><CardContent className="py-6"><p className="text-center text-sm text-destructive">{error}</p></CardContent></Card>}
      {data && data.providers.length === 0 && <Card><CardContent className="py-12 text-center text-sm text-muted-foreground">No supported providers are configured.</CardContent></Card>}
      {data && (
        <div className="grid gap-6 lg:grid-cols-2">
          {data.providers.map((snapshot) => <ProviderCard key={snapshot.provider} snapshot={snapshot} />)}
        </div>
      )}
    </div>
  );
}
