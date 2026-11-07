import { useCallback, useEffect, useState } from "react";
import { RefreshCw, Wallet } from "lucide-react";
import { api } from "@/lib/api";
import type { ModelBalanceResponse, ProviderBalance } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { useI18n } from "@/i18n";

/** Currency/unit symbol for a probe's unit code; unknown units print as-is. */
function unitLabel(unit: string, amount: number): string {
  const code = (unit || "").toUpperCase();
  if (code === "CNY") return `¥${amount.toFixed(2)}`;
  if (code === "USD") return `$${amount.toFixed(2)}`;
  if (!code) return amount.toFixed(2);
  return `${amount.toFixed(2)} ${unit}`;
}

function formatStamp(value?: string): string {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * One provider's row: the headline number when a probe answered, otherwise the short
 * reason there is none. Quota-style providers (a daily cap) also get a used/limit bar.
 */
function BalanceRow({ row }: { row: ProviderBalance }) {
  const { t } = useI18n();
  const quota = row.quota || {};
  const hasBar =
    typeof quota.used === "number" &&
    typeof quota.limit === "number" &&
    quota.limit > 0;
  const pct = hasBar
    ? Math.min(100, Math.max(0, (Number(quota.used) / Number(quota.limit)) * 100))
    : 0;

  const reason =
    row.status === "no_key"
      ? t.models.balanceNoKey
      : row.status === "unsupported"
        ? t.models.balanceNoApi
        : row.status === "error"
          ? `${t.models.balanceError}${row.error ? `: ${row.error}` : ""}`
          : "";

  return (
    <div className="flex min-w-0 flex-col gap-2 border-b border-border/60 py-2.5 last:border-b-0">
      <div className="flex min-w-0 items-baseline justify-between gap-3">
        <span className="truncate text-sm font-medium">{row.label || row.id}</span>
        {row.status === "ok" && row.primary ? (
          <span className="shrink-0 font-mono-ui text-sm">
            {unitLabel(row.primary.unit, Number(row.primary.amount) || 0)}
          </span>
        ) : (
          <span className="shrink-0 text-right text-xs text-muted-foreground">
            {reason}
          </span>
        )}
      </div>

      {row.status === "ok" && hasBar && (
        <div className="flex min-w-0 flex-col gap-1">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-[width] duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="flex min-w-0 justify-between gap-2 text-[11px] text-muted-foreground">
            <span className="truncate">
              {t.models.balanceUsed} {Number(quota.used).toFixed(2)} /{" "}
              {Number(quota.limit).toFixed(2)}
            </span>
            {quota.reset_at && (
              <span className="shrink-0">
                {t.models.balanceResets} {formatStamp(quota.reset_at)}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Provider balance / quota snapshot for the Models page.
 *
 * The list is whatever the backend reports for the providers this profile has configured,
 * so a provider with no documented balance endpoint shows up with a one-line reason instead
 * of a fabricated number. `?refresh=1` bypasses the backend's 60 s cache.
 */
export function BalanceCard() {
  const { t } = useI18n();
  const [data, setData] = useState<ModelBalanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState<string | null>(null);

  const load = useCallback(async (refresh: boolean) => {
    // Only a user-driven refresh flips the spinner synchronously; the first load already
    // starts in the loading state (a synchronous setState in an effect cascades renders).
    if (refresh) setLoading(true);
    try {
      setData(await api.getModelBalance(refresh));
      setFailed(null);
    } catch (err) {
      setFailed(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(false);
  }, [load]);

  const rows = data?.providers ?? [];

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3 pb-3">
        <div className="flex min-w-0 items-center gap-2">
          <Wallet className="h-4 w-4 text-muted-foreground" />
          <CardTitle className="text-base">{t.models.balance}</CardTitle>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {data?.fetched_at && !loading && (
            <span className="text-[11px] text-muted-foreground">
              {t.models.balanceUpdated} {formatStamp(data.fetched_at)}
            </span>
          )}
          <Button
            size="sm"
            outlined
            aria-label={t.models.balance}
            disabled={loading}
            onClick={() => void load(true)}
          >
            {loading ? (
              <Spinner className="h-3.5 w-3.5" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {failed ? (
          <p className="text-xs text-destructive">{failed}</p>
        ) : rows.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            {loading ? "…" : t.models.balanceEmpty}
          </p>
        ) : (
          <div className="flex min-w-0 flex-col">
            {rows.map((row) => (
              <BalanceRow key={row.id} row={row} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
