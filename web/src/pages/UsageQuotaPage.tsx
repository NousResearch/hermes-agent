import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { Gauge, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api, type UsageQuotaResponse, type UsageQuotaSnapshot } from "@/lib/api";

function formatDate(value: string | null): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function ProviderCard({ snapshot }: { snapshot: UsageQuotaSnapshot }) {
  const status = snapshot.available ? "Reported" : "Unavailable";
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-2">
            <Gauge className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-base">{snapshot.title || snapshot.provider}</CardTitle>
          </div>
          <span className={`text-xs ${snapshot.available ? "text-emerald-600" : "text-muted-foreground"}`}>
            {status}
          </span>
        </div>
        <p className="font-mono text-xs text-muted-foreground">{snapshot.provider}</p>
      </CardHeader>
      <CardContent className="space-y-4">
        {snapshot.plan && <p className="text-sm text-muted-foreground">Plan: {snapshot.plan}</p>}
        {snapshot.windows.length > 0 ? snapshot.windows.map((window) => {
          const used = window.used_percent;
          const remaining = used == null ? null : Math.max(0, Math.round(100 - used));
          return (
            <div key={window.label} className="space-y-1.5">
              <div className="flex justify-between text-sm">
                <span>{window.label}</span>
                <span className="font-mono text-muted-foreground">
                  {remaining == null ? "Remaining unknown" : `${remaining}% remaining`}
                </span>
              </div>
              {remaining != null && (
                <div
                  className="h-2 overflow-hidden rounded-full bg-muted"
                  role="progressbar"
                  aria-label={`${window.label} quota remaining`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={remaining}
                >
                  <div className={`h-full ${remaining <= 20 ? "bg-destructive" : remaining <= 50 ? "bg-amber-500" : "bg-emerald-500"}`} style={{ width: `${remaining}%` }} />
                </div>
              )}
              <div className="flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
                <span>{used == null ? "Usage unknown" : `${Math.round(used)}% used`}</span>
                <span>{window.reset_at ? `Resets ${formatDate(window.reset_at)}` : (window.detail || "Reset unknown")}</span>
              </div>
            </div>
          );
        }) : <p className="text-sm text-muted-foreground">No percentage quota window was reported by this provider.</p>}
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
      <p className="text-sm text-muted-foreground">Provider account limits from official provider APIs. Values are not estimated and are not local analytics or billing data.</p>
      {loading && !data && <div className="flex justify-center py-24"><Spinner className="text-2xl text-primary" /></div>}
      {error && <Card><CardContent className="py-6"><p className="text-center text-sm text-destructive">{error}</p></CardContent></Card>}
      {data && data.providers.length === 0 && <Card><CardContent className="py-12 text-center text-sm text-muted-foreground">No supported providers are configured.</CardContent></Card>}
      {data && data.providers.length > 0 && <div className="grid gap-6 lg:grid-cols-2">{data.providers.map((snapshot) => <ProviderCard key={snapshot.provider} snapshot={snapshot} />)}</div>}
    </div>
  );
}
