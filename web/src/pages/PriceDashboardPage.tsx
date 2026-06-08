import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import {
  CreditCard,
  Download,
  FileText,
  Globe,
  RefreshCw,
  TrendingUp,
} from "lucide-react";
import { api, HERMES_BASE_PATH } from "@/lib/api";
import type { PriceScraperResult } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Stats } from "@nous-research/ui/ui/components/stats";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { usePageHeader } from "@/contexts/usePageHeader";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso: string): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function categoryColor(cat: string): string {
  switch (cat) {
    case "price": return "text-emerald-400";
    case "percentage": return "text-amber-400";
    case "quantity": return "text-sky-400";
    case "rating": return "text-violet-400";
    default: return "text-muted-foreground";
  }
}


// ---------------------------------------------------------------------------
// Stripe banner
// ---------------------------------------------------------------------------

function StripeBanner({ link }: { link: string }) {
  return (
    <Card className="border-emerald-500/40 bg-emerald-500/5">
      <CardContent className="py-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <CreditCard className="h-5 w-5 text-emerald-400 shrink-0" />
            <div>
              <p className="text-sm font-medium text-foreground">Subscribe for full access</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                Unlock unlimited scrapes, historical tracking, and CSV/PDF exports.
              </p>
            </div>
          </div>
          <a href={link} target="_blank" rel="noopener noreferrer">
            <Button size="sm" className="whitespace-nowrap">
              <CreditCard className="h-3.5 w-3.5 mr-1.5" />
              Subscribe with Stripe
            </Button>
          </a>
        </div>
      </CardContent>
    </Card>
  );
}

function StripeSetupHint() {
  return (
    <Card className="border-border/50">
      <CardContent className="py-4">
        <div className="flex items-center gap-3">
          <CreditCard className="h-5 w-5 text-muted-foreground shrink-0" />
          <p className="text-sm text-muted-foreground">
            Set <span className="font-mono text-xs bg-muted/50 px-1.5 py-0.5 rounded">HERMES_STRIPE_PAYMENT_LINK</span> in your{" "}
            <a href="/env" className="underline hover:text-foreground transition-colors">environment keys</a>{" "}
            to show a subscription button here.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Results table
// ---------------------------------------------------------------------------

function ResultsTable({ results }: { results: PriceScraperResult[] }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set([0]));

  const toggle = (i: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });

  if (results.length === 0) {
    return (
      <Card>
        <CardContent className="py-16">
          <div className="flex flex-col items-center gap-3 text-muted-foreground">
            <TrendingUp className="h-10 w-10 opacity-30" />
            <p className="text-sm font-medium">No scraped data yet</p>
            <p className="text-xs text-text-tertiary text-center max-w-xs">
              Ask Hermes to scrape a URL using the <span className="font-mono">web-price-scraper</span> skill.
              Results will appear here automatically.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {results.map((r, i) => (
        <Card key={`${r.url}-${r.scraped_at}`}>
          <CardHeader
            className="cursor-pointer select-none"
            onClick={() => toggle(i)}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <Globe className="h-4 w-4 text-muted-foreground shrink-0" />
                <div className="min-w-0">
                  <CardTitle className="text-sm font-medium truncate">{r.domain}</CardTitle>
                  <p className="text-xs text-muted-foreground truncate mt-0.5">{r.url}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Badge tone="secondary" className="text-xs hidden sm:inline-flex">
                  {r.item_count ?? r.items?.length ?? 0} items
                </Badge>
                <span className="text-xs text-muted-foreground hidden md:block whitespace-nowrap">
                  {formatDate(r.scraped_at)}
                </span>
              </div>
            </div>
          </CardHeader>

          {expanded.has(i) && r.items && r.items.length > 0 && (
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full font-mondwest normal-case text-sm">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground text-xs">
                      <th className="text-left py-2 pr-4 font-medium">Label</th>
                      <th className="text-right py-2 px-4 font-medium">Value</th>
                      <th className="text-right py-2 px-4 font-medium hidden sm:table-cell">Raw</th>
                      <th className="text-right py-2 px-4 font-medium hidden md:table-cell">Unit</th>
                      <th className="text-right py-2 pl-4 font-medium hidden sm:table-cell">Category</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.items.map((item, j) => (
                      <tr key={j} className="border-b border-border/50 hover:bg-secondary/20 transition-colors">
                        <td className="py-2 pr-4 max-w-[200px] truncate">{item.label || "—"}</td>
                        <td className={`text-right py-2 px-4 font-medium ${categoryColor(item.category)}`}>
                          {String(item.value)}
                        </td>
                        <td className="text-right py-2 px-4 text-muted-foreground hidden sm:table-cell">
                          {item.raw || "—"}
                        </td>
                        <td className="text-right py-2 px-4 text-muted-foreground hidden md:table-cell">
                          {item.unit || "—"}
                        </td>
                        <td className="text-right py-2 pl-4 hidden sm:table-cell">
                          <Badge tone="secondary" className="text-xs capitalize">
                            {item.category || "other"}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          )}

          {expanded.has(i) && (!r.items || r.items.length === 0) && (
            <CardContent>
              <p className="text-sm text-muted-foreground text-center py-4">No items in this result.</p>
            </CardContent>
          )}
        </Card>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function PriceDashboardPage() {
  const [results, setResults] = useState<PriceScraperResult[]>([]);
  const [stripeLink, setStripeLink] = useState<string>("");
  const [stripeConfigured, setStripeConfigured] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setEnd } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.getPriceScraperResults(),
      api.getPriceScraperConfig(),
    ])
      .then(([res, cfg]) => {
        setResults(res.results);
        setStripeLink(cfg.stripe_payment_link);
        setStripeConfigured(cfg.configured);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  useLayoutEffect(() => {
    setEnd(
      <div className="flex items-center gap-2">
        <Button
          type="button"
          ghost
          size="icon"
          className="text-muted-foreground hover:text-foreground"
          onClick={load}
          disabled={loading}
          aria-label="Refresh"
        >
          {loading ? <Spinner /> : <RefreshCw />}
        </Button>
        <a
          href={`${HERMES_BASE_PATH}/api/price-scraper/export/csv`}
          download="price-scraper-export.csv"
        >
          <Button type="button" size="sm" outlined>
            <Download className="h-3.5 w-3.5 mr-1.5" />
            CSV
          </Button>
        </a>
        <Button
          type="button"
          size="sm"
          outlined
          onClick={() => window.print()}
        >
          <FileText className="h-3.5 w-3.5 mr-1.5" />
          PDF
        </Button>
      </div>,
    );
    return () => setEnd(null);
  }, [results, loading, load, setEnd]);

  const totalItems = results.reduce((sum, r) => sum + (r.item_count ?? r.items?.length ?? 0), 0);
  const domainCount = new Set(results.map((r) => r.domain)).size;
  const latest = results[0]?.scraped_at ?? null;

  return (
    <div className="flex flex-col gap-6">
      {/* Stripe banner */}
      {stripeConfigured === true && stripeLink && <StripeBanner link={stripeLink} />}
      {stripeConfigured === false && <StripeSetupHint />}

      {/* Stats */}
      {!loading && !error && (
        <Card>
          <CardContent className="py-6">
            <Stats
              items={[
                { label: "Total Items", value: String(totalItems) },
                { label: "Domains Scraped", value: String(domainCount) },
                { label: "Scrape Records", value: String(results.length) },
                { label: "Last Scraped", value: latest ? formatDate(latest) : "Never" },
              ]}
            />
          </CardContent>
        </Card>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-24">
          <Spinner className="text-2xl text-primary" />
        </div>
      )}

      {/* Error */}
      {error && (
        <Card>
          <CardContent className="py-6">
            <p className="text-sm text-destructive text-center">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {!loading && !error && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-medium text-foreground">Scraped Data</h2>
            {results.length > 0 && (
              <Badge tone="secondary" className="text-xs">{results.length} records</Badge>
            )}
          </div>
          <ResultsTable results={results} />
        </div>
      )}
    </div>
  );
}
