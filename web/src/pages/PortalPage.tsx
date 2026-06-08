import { useCallback, useState } from "react";
import { Globe, Plus, Trash2, TrendingUp, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { fetchJSON } from "@/lib/api";

interface SubscriberStatus {
  active: boolean;
  plan: string;
  url_limit: number;
  url_count: number;
  status: string;
  message?: string;
}

interface UrlList {
  urls: string[];
  url_limit: number;
}

export default function PortalPage() {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<SubscriberStatus | null>(null);
  const [urls, setUrls] = useState<string[]>([]);
  const [urlLimit, setUrlLimit] = useState(5);
  const [newUrl, setNewUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const lookup = useCallback(async (e?: string) => {
    const addr = (e ?? email).trim().toLowerCase();
    if (!addr) return;
    setLoading(true);
    setError(null);
    setSuccessMsg(null);
    try {
      const [s, u] = await Promise.all([
        fetchJSON<SubscriberStatus>(`/api/subscribe/status?email=${encodeURIComponent(addr)}`),
        fetchJSON<UrlList>(`/api/subscribe/urls?email=${encodeURIComponent(addr)}`).catch(() => null),
      ]);
      setStatus(s);
      if (u) { setUrls(u.urls); setUrlLimit(u.url_limit); }
      setSubmitted(true);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [email]);

  const addUrl = async () => {
    const url = newUrl.trim();
    if (!url) return;
    setError(null);
    setSuccessMsg(null);
    try {
      await fetchJSON("/api/subscribe/add-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim().toLowerCase(), url }),
      });
      setNewUrl("");
      setSuccessMsg("URL added successfully.");
      lookup(email);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg.replace(/^\d+:\s*/, ""));
    }
  };

  const removeUrl = async (url: string) => {
    setError(null);
    setSuccessMsg(null);
    try {
      await fetchJSON("/api/subscribe/remove-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim().toLowerCase(), url }),
      });
      setSuccessMsg("URL removed.");
      lookup(email);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg.replace(/^\d+:\s*/, ""));
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Nav */}
      <nav className="border-b border-border/50 px-6 py-4 flex items-center justify-between">
        <a href="/subscribe" className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-emerald-400" />
          <span className="font-mondwest text-display tracking-wider text-foreground">
            Price Tracker
          </span>
        </a>
        <a href="/subscribe">
          <Button type="button" size="sm" outlined>← Plans</Button>
        </a>
      </nav>

      <main className="flex-1 flex flex-col items-center px-6 pt-16 pb-16">
        <div className="max-w-lg w-full flex flex-col gap-6">
          <div className="text-center">
            <h1 className="font-mondwest text-display text-2xl tracking-wider text-foreground">
              Subscriber Portal
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Enter your email to manage your tracked URLs
            </p>
          </div>

          {/* Email lookup */}
          <Card>
            <CardContent className="py-5 flex flex-col gap-3">
              <Label htmlFor="email-input" className="text-sm">Your email address</Label>
              <div className="flex gap-2">
                <Input
                  id="email-input"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && lookup()}
                  className="flex-1"
                />
                <Button
                  type="button"
                  onClick={() => lookup()}
                  disabled={loading || !email.trim()}
                >
                  {loading ? <Spinner /> : "Look up"}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Error */}
          {error && (
            <p className="text-sm text-destructive text-center">{error}</p>
          )}
          {successMsg && (
            <p className="text-sm text-emerald-400 text-center">{successMsg}</p>
          )}

          {/* No subscription */}
          {submitted && status && !status.active && (
            <Card>
              <CardContent className="py-8 text-center flex flex-col gap-4">
                <p className="text-sm text-muted-foreground">
                  {status.message ?? "No active subscription found for that email."}
                </p>
                <a href="/subscribe">
                  <Button type="button">View Plans →</Button>
                </a>
              </CardContent>
            </Card>
          )}

          {/* Active subscription */}
          {submitted && status?.active && (
            <>
              {/* Subscription status */}
              <Card>
                <CardContent className="py-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Badge tone="success" className="capitalize">{status.plan}</Badge>
                    <span className="text-sm text-foreground">{email}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {urls.length}/{urlLimit} URLs
                    </span>
                    <Button
                      type="button"
                      ghost
                      size="icon"
                      onClick={() => lookup()}
                      disabled={loading}
                    >
                      {loading ? <Spinner /> : <RefreshCw className="h-3.5 w-3.5" />}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Add URL */}
              {urls.length < urlLimit && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Plus className="h-4 w-4" /> Add a URL to track
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex gap-2">
                    <Input
                      type="url"
                      placeholder="https://example.com/pricing"
                      value={newUrl}
                      onChange={(e) => setNewUrl(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && addUrl()}
                      className="flex-1"
                    />
                    <Button type="button" onClick={addUrl} disabled={!newUrl.trim()}>
                      Add
                    </Button>
                  </CardContent>
                </Card>
              )}

              {/* URL list */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Globe className="h-4 w-4" /> Tracked URLs
                    <Badge tone="secondary" className="text-xs ml-auto">
                      {urls.length} / {urlLimit}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {urls.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No URLs yet. Add one above to start tracking.
                    </p>
                  ) : (
                    <ul className="flex flex-col gap-2">
                      {urls.map((url) => (
                        <li
                          key={url}
                          className="flex items-center justify-between gap-2 text-sm border border-border/50 rounded px-3 py-2"
                        >
                          <span className="truncate text-muted-foreground">{url}</span>
                          <Button
                            type="button"
                            ghost
                            size="icon"
                            className="text-muted-foreground hover:text-destructive shrink-0"
                            onClick={() => removeUrl(url)}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </li>
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>

              {urls.length >= urlLimit && (
                <p className="text-xs text-center text-muted-foreground">
                  You've reached your {urlLimit}-URL limit.{" "}
                  <a href="/subscribe" className="underline hover:text-foreground">
                    Upgrade to Pro
                  </a>{" "}
                  for unlimited URLs.
                </p>
              )}
            </>
          )}
        </div>
      </main>

      <footer className="border-t border-border/50 px-6 py-4 text-center text-xs text-text-tertiary">
        Powered by Hermes Agent · Secure payments via Stripe
      </footer>
    </div>
  );
}
